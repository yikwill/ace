#!/usr/bin/env python3
"""Build ACE-compatible 6-hourly emission / external-forcing zarr stores.

Per-species surface and column-integrated elevated fields from
/p/lustre5/yik1/datasets/emis/, regridded to the E3SM aerosol Gaussian grid,
with monthly climatologies expanded to 6-hourly 1945–1980 (noleap).

PI MAM4/SOAG: year 1850 from the 1850–2014 transient files (repeating clim).
PD MAM4/SOAG: 2010_clim.
Chem/DMS/NO2: 2010-as-1850 (PI) or 2010_clim (PD); fluxes identical across eras.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)

EMIS_ROOT = Path("/p/lustre5/yik1/datasets/emis")
TARGET_ZARR = Path("/p/lustre5/yik1/datasets/e3sm-aerosol-PD-1945-1980.zarr")
OUT_DIR = Path("/p/lustre5/yik1/datasets")

NOLEAP_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
KM_TO_CM = 1.0e5

SKIP_COORDS = {
    "date",
    "datesec",
    "time_bnds",
    "lat_bnds",
    "lon_bnds",
    "gw",
    "area",
    "time_bound",
    "altitude",
    "altitude_int",
}


@dataclass(frozen=True)
class EmisSpec:
    species: str
    kind: str  # "sfc" | "elev"
    relative_path: str
    extract_year: int | None = None  # for 1850-2014 files


# Paths relative to EMIS_ROOT. ISOP_VBS omitted (duplicate of ISOP).
ELEV_SPECS: dict[str, dict[str, EmisSpec]] = {
    "PI": {
        "NO2": EmisSpec(
            "NO2",
            "elev",
            "chem_gases/2degrees/emissions-cmip6_NO2_aircraft_vertical_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "SO2": EmisSpec(
            "SO2",
            "elev",
            "DECK_ne30/cmip6_mam4_so2_elev_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "SOAG0": EmisSpec(
            "SOAG0",
            "elev",
            "DECK_ne30/emissions-cmip6_e3sm_SOAG0_elev_1850-2014_1.9x2.5_c20230201.nc",
            extract_year=1850,
        ),
        "bc_a4": EmisSpec(
            "bc_a4",
            "elev",
            "DECK_ne30/cmip6_mam4_bc_a4_elev_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "pom_a4": EmisSpec(
            "pom_a4",
            "elev",
            "DECK_ne30/cmip6_mam4_pom_a4_elev_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "so4_a1": EmisSpec(
            "so4_a1",
            "elev",
            "DECK_ne30/cmip6_mam4_so4_a1_elev_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "so4_a2": EmisSpec(
            "so4_a2",
            "elev",
            "DECK_ne30/cmip6_mam4_so4_a2_elev_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "num_a1": EmisSpec(
            "num_a1",
            "elev",
            "DECK_ne30/cmip6_mam4_num_a1_elev_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "num_a2": EmisSpec(
            "num_a2",
            "elev",
            "DECK_ne30/cmip6_mam4_num_a2_elev_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "num_a4": EmisSpec(
            "num_a4",
            "elev",
            "DECK_ne30/cmip6_mam4_num_a4_elev_1850-2014_c180205.nc",
            extract_year=1850,
        ),
    },
    "PD": {
        "NO2": EmisSpec(
            "NO2",
            "elev",
            "chem_gases/2degrees/emissions-cmip6_NO2_aircraft_vertical_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "SO2": EmisSpec(
            "SO2", "elev", "DECK_ne30/cmip6_mam4_so2_elev_1x1_2010_clim_c20190821.nc"
        ),
        "SOAG0": EmisSpec(
            "SOAG0",
            "elev",
            "DECK_ne30/emissions-cmip6_e3sm_SOAG0_elev_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "bc_a4": EmisSpec(
            "bc_a4",
            "elev",
            "DECK_ne30/cmip6_mam4_bc_a4_elev_1x1_2010_clim_c20190821.nc",
        ),
        "pom_a4": EmisSpec(
            "pom_a4",
            "elev",
            "DECK_ne30/cmip6_mam4_pom_a4_elev_1x1_2010_clim_c20190821.nc",
        ),
        "so4_a1": EmisSpec(
            "so4_a1",
            "elev",
            "DECK_ne30/cmip6_mam4_so4_a1_elev_1x1_2010_clim_c20190821.nc",
        ),
        "so4_a2": EmisSpec(
            "so4_a2",
            "elev",
            "DECK_ne30/cmip6_mam4_so4_a2_elev_1x1_2010_clim_c20190821.nc",
        ),
        "num_a1": EmisSpec(
            "num_a1",
            "elev",
            "DECK_ne30/cmip6_mam4_num_a1_elev_1x1_2010_clim_c20190821.nc",
        ),
        "num_a2": EmisSpec(
            "num_a2",
            "elev",
            "DECK_ne30/cmip6_mam4_num_a2_elev_1x1_2010_clim_c20190821.nc",
        ),
        "num_a4": EmisSpec(
            "num_a4",
            "elev",
            "DECK_ne30/cmip6_mam4_num_a4_elev_1x1_2010_clim_c20190821.nc",
        ),
    },
}

SFC_SPECS: dict[str, dict[str, EmisSpec]] = {
    "PI": {
        "C2H4": EmisSpec(
            "C2H4",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_C2H4_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "C2H6": EmisSpec(
            "C2H6",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_C2H6_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "C3H8": EmisSpec(
            "C3H8",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_C3H8_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "CH2O": EmisSpec(
            "CH2O",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_CH2O_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "CH3CHO": EmisSpec(
            "CH3CHO",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_CH3CHO_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "CH3COCH3": EmisSpec(
            "CH3COCH3",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_CH3COCH3_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "CO": EmisSpec(
            "CO",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_CO_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "NO": EmisSpec(
            "NO",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_NO_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "ISOP": EmisSpec(
            "ISOP",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_ISOP_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "C10H16": EmisSpec(
            "C10H16",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_MTERP_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "E90": EmisSpec(
            "E90",
            "sfc",
            "chem_gases/2degrees/emissions_E90_surface_2010-as-1850_clim_1.9x2.5_c20230213.nc",
        ),
        "DMS": EmisSpec(
            "DMS",
            "sfc",
            "DMSflux.2010-as-1850.1deg_latlon_conserv.POPmonthlyClimFromACES4BGC_c20190220.nc",
        ),
        "SO2": EmisSpec(
            "SO2",
            "sfc",
            "DECK_ne30/cmip6_mam4_so2_surf_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "SOAG0": EmisSpec(
            "SOAG0",
            "sfc",
            "DECK_ne30/emissions-cmip6_e3sm_SOAG0_surf_1850-2014_1.9x2.5_c20230201.nc",
            extract_year=1850,
        ),
        "bc_a4": EmisSpec(
            "bc_a4",
            "sfc",
            "DECK_ne30/cmip6_mam4_bc_a4_surf_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "pom_a4": EmisSpec(
            "pom_a4",
            "sfc",
            "DECK_ne30/cmip6_mam4_pom_a4_surf_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "so4_a1": EmisSpec(
            "so4_a1",
            "sfc",
            "DECK_ne30/cmip6_mam4_so4_a1_surf_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "so4_a2": EmisSpec(
            "so4_a2",
            "sfc",
            "DECK_ne30/cmip6_mam4_so4_a2_surf_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "num_a1": EmisSpec(
            "num_a1",
            "sfc",
            "DECK_ne30/cmip6_mam4_num_a1_surf_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "num_a2": EmisSpec(
            "num_a2",
            "sfc",
            "DECK_ne30/cmip6_mam4_num_a2_surf_1850-2014_c180205.nc",
            extract_year=1850,
        ),
        "num_a4": EmisSpec(
            "num_a4",
            "sfc",
            "DECK_ne30/cmip6_mam4_num_a4_surf_1850-2014_c180205.nc",
            extract_year=1850,
        ),
    },
    "PD": {
        "C2H4": EmisSpec(
            "C2H4",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_C2H4_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "C2H6": EmisSpec(
            "C2H6",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_C2H6_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "C3H8": EmisSpec(
            "C3H8",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_C3H8_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "CH2O": EmisSpec(
            "CH2O",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_CH2O_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "CH3CHO": EmisSpec(
            "CH3CHO",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_CH3CHO_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "CH3COCH3": EmisSpec(
            "CH3COCH3",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_CH3COCH3_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "CO": EmisSpec(
            "CO",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_CO_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "NO": EmisSpec(
            "NO",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_NO_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "ISOP": EmisSpec(
            "ISOP",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_ISOP_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "C10H16": EmisSpec(
            "C10H16",
            "sfc",
            "chem_gases/2degrees/emissions-cmip6_e3sm_MTERP_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "E90": EmisSpec(
            "E90",
            "sfc",
            "chem_gases/2degrees/emissions_E90_surface_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "DMS": EmisSpec(
            "DMS",
            "sfc",
            "DMSflux.2010.1deg_latlon_conserv.POPmonthlyClimFromACES4BGC_c20190220.nc",
        ),
        "SO2": EmisSpec(
            "SO2", "sfc", "DECK_ne30/cmip6_mam4_so2_surf_1x1_2010_clim_c20190821.nc"
        ),
        "SOAG0": EmisSpec(
            "SOAG0",
            "sfc",
            "DECK_ne30/emissions-cmip6_e3sm_SOAG0_surf_2010_clim_1.9x2.5_c20230213.nc",
        ),
        "bc_a4": EmisSpec(
            "bc_a4", "sfc", "DECK_ne30/cmip6_mam4_bc_a4_surf_1x1_2010_clim_c20190821.nc"
        ),
        "pom_a4": EmisSpec(
            "pom_a4",
            "sfc",
            "DECK_ne30/cmip6_mam4_pom_a4_surf_1x1_2010_clim_c20190821.nc",
        ),
        "so4_a1": EmisSpec(
            "so4_a1",
            "sfc",
            "DECK_ne30/cmip6_mam4_so4_a1_surf_1x1_2010_clim_c20190821.nc",
        ),
        "so4_a2": EmisSpec(
            "so4_a2",
            "sfc",
            "DECK_ne30/cmip6_mam4_so4_a2_surf_1x1_2010_clim_c20190821.nc",
        ),
        "num_a1": EmisSpec(
            "num_a1",
            "sfc",
            "DECK_ne30/cmip6_mam4_num_a1_surf_1x1_2010_clim_c20190821.nc",
        ),
        "num_a2": EmisSpec(
            "num_a2",
            "sfc",
            "DECK_ne30/cmip6_mam4_num_a2_surf_1x1_2010_clim_c20190821.nc",
        ),
        "num_a4": EmisSpec(
            "num_a4",
            "sfc",
            "DECK_ne30/cmip6_mam4_num_a4_surf_1x1_2010_clim_c20190821.nc",
        ),
    },
}


def month_centroid_month_frac(month: int) -> float:
    return (month - 1) + (NOLEAP_DAYS[month - 1] - 1) / (2 * NOLEAP_DAYS[month - 1])


def cftime_to_month_frac(t) -> float:
    hour = t.hour + t.minute / 60 + t.second / 3600
    return (t.month - 1) + ((t.day - 1) + hour / 24) / NOLEAP_DAYS[t.month - 1]


def out_var_name(species: str, kind: str) -> str:
    return f"emis_{species}_{kind}"


def sector_var_names(ds: xr.Dataset) -> list[str]:
    names = []
    for v in ds.data_vars:
        if v in SKIP_COORDS:
            continue
        da = ds[v]
        if not np.issubdtype(da.dtype, np.number):
            continue
        if da.ndim < 2:
            continue
        # keep spacetime / vertical emission fields only
        dims = set(da.dims)
        if "lat" in dims and "lon" in dims:
            names.append(v)
    return names


def column_integrate(da: xr.DataArray, altitude_int: xr.DataArray) -> xr.DataArray:
    """Integrate molecules/cm3/s over altitude using interface thickness (km→cm)."""
    dz_km = np.diff(np.asarray(altitude_int.values, dtype=np.float64))
    if dz_km.shape[0] != da.sizes["altitude"]:
        raise ValueError(
            f"altitude_int length {altitude_int.size} incompatible with "
            f"altitude size {da.sizes['altitude']}"
        )
    dz_cm = xr.DataArray(
        dz_km * KM_TO_CM,
        dims=("altitude",),
        coords={"altitude": da["altitude"]},
    )
    out = (da * dz_cm).sum(dim="altitude")
    units = str(da.attrs.get("units", ""))
    if "cm3" in units:
        out.attrs["units"] = units.replace("cm3", "cm2")
    else:
        out.attrs["units"] = "molecules/cm2/s"
    out.attrs["long_name"] = (
        da.attrs.get("long_name", "emission") + " (column integrated)"
    )
    out.attrs["column_integration"] = (
        "sum(rate * dz); dz = diff(altitude_int) converted km→cm"
    )
    return out


def load_monthly_clim(spec: EmisSpec, emis_root: Path) -> xr.DataArray:
    path = emis_root / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path, decode_times=False)
    try:
        if spec.extract_year is not None:
            if "date" not in ds:
                raise ValueError(f"{path} missing date for year extract")
            years = np.asarray(ds["date"].values).ravel() // 10000
            idx = np.where(years == spec.extract_year)[0]
            if idx.size != 12:
                raise ValueError(
                    f"{path}: expected 12 months for year {spec.extract_year}, "
                    f"got {idx.size}"
                )
            ds = ds.isel(time=idx)

        if ds.sizes.get("time", 0) != 12:
            raise ValueError(f"{path}: expected time=12 after load, got {ds.sizes}")

        sectors = sector_var_names(ds)
        if not sectors:
            raise ValueError(f"{path}: no emission sector variables found")

        total = None
        units = None
        for name in sectors:
            da = ds[name]
            if "altitude" in da.dims:
                if "altitude_int" not in ds:
                    raise ValueError(f"{path}: elevated field missing altitude_int")
                da = column_integrate(da, ds["altitude_int"])
            if units is None:
                units = da.attrs.get("units", "molecules/cm2/s")
            total = (
                da.astype(np.float64)
                if total is None
                else total + da.astype(np.float64)
            )

        assert total is not None
        total = total.transpose("time", "lat", "lon")
        total.name = out_var_name(spec.species, spec.kind)
        kind_label = "surface" if spec.kind == "sfc" else "elevated column"
        total.attrs = {
            "units": units or "molecules/cm2/s",
            "long_name": f"{spec.species} {kind_label} emission (all sectors summed)",
            "species": spec.species,
            "emission_kind": spec.kind,
            "source_file": str(path),
            "sectors_summed": ",".join(sectors),
        }
        if spec.extract_year is not None:
            total.attrs["extract_year"] = spec.extract_year
        # Drop non-lat/lon coords that confuse regrid
        total = total.reset_coords(drop=True)
        return total.load()
    finally:
        ds.close()


def get_regridder(lat_in, lon_in, lat_out, lon_out, method: str = "conservative"):
    import xesmf as xe

    ds_in = xr.Dataset(
        {
            "lat": ("lat", np.asarray(lat_in, dtype=np.float64)),
            "lon": ("lon", np.asarray(lon_in, dtype=np.float64)),
        }
    )
    ds_out = xr.Dataset(
        {
            "lat": ("lat", np.asarray(lat_out, dtype=np.float64)),
            "lon": ("lon", np.asarray(lon_out, dtype=np.float64)),
        }
    )
    return xe.Regridder(ds_in, ds_out, method=method, periodic=True)


def regrid_field(
    da: xr.DataArray,
    lat_out,
    lon_out,
    regridder_cache: dict,
    method: str = "conservative",
) -> xr.DataArray:
    key = (
        tuple(np.round(np.asarray(da["lat"].values), 6)),
        tuple(np.round(np.asarray(da["lon"].values), 6)),
        method,
    )
    if key not in regridder_cache:
        logger.info(
            "Building xesmf regridder %s → (%d, %d) method=%s",
            (da.sizes["lat"], da.sizes["lon"]),
            len(lat_out),
            len(lon_out),
            method,
        )
        regridder_cache[key] = get_regridder(
            da["lat"], da["lon"], lat_out, lon_out, method=method
        )
    regridder = regridder_cache[key]
    out = regridder(da, keep_attrs=True)
    nan_frac = float(np.isnan(out.values).mean()) if out.size else 0.0
    if nan_frac > 0.0 and method != "bilinear":
        logger.warning(
            "Regrid method=%s left nan_frac=%.4g; falling back to bilinear",
            method,
            nan_frac,
        )
        return regrid_field(da, lat_out, lon_out, regridder_cache, method="bilinear")
    # Residual NaNs (e.g. uncovered cells) → zero emission
    out = out.fillna(0.0)
    out = out.assign_coords(lat=lat_out, lon=lon_out)
    return out


def interpolate_clim_to_times(clim: xr.Dataset, target_times: np.ndarray) -> xr.Dataset:
    """Periodic cubic spline on monthly centroids → target times (same year cycle)."""
    x_knot = np.array([month_centroid_month_frac(m) for m in range(1, 13)])
    x_ext = np.concatenate([[x_knot[-1] - 12], x_knot, [x_knot[0] + 12]])
    target_x = np.array([cftime_to_month_frac(t) for t in target_times])

    out_vars = {}
    for var in clim.data_vars:
        y = np.asarray(clim[var].values, dtype=np.float64)
        y_ext = np.concatenate([y[-1:], y, y[:1]], axis=0)
        spline = CubicSpline(x_ext, y_ext, axis=0, bc_type="natural")
        values = spline(target_x).astype(np.float32)
        out_vars[var] = xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={"time": target_times, "lat": clim["lat"], "lon": clim["lon"]},
            attrs=clim[var].attrs,
        )
    return xr.Dataset(out_vars)


def load_target_grid_and_times(
    target_zarr: Path,
    year_start: int,
    year_end: int,
) -> tuple[xr.DataArray, xr.DataArray, np.ndarray]:
    ds = xr.open_zarr(target_zarr, consolidated=True)
    lat = ds["lat"].load()
    lon = ds["lon"].load()
    time = ds["time"].load()
    # select years via cftime
    years = np.array([t.year for t in time.values])
    mask = (years >= year_start) & (years <= year_end)
    times = time.values[mask]
    ds.close()
    if times.size == 0:
        raise ValueError(f"No times in {year_start}–{year_end} from {target_zarr}")
    return lat, lon, times


def build_monthly_clim_dataset(
    era: str,
    emis_root: Path,
    lat_out,
    lon_out,
    species_filter: set[str] | None,
    regrid_method: str,
) -> tuple[xr.Dataset, dict]:
    specs: list[EmisSpec] = []
    for kind_map in (SFC_SPECS[era], ELEV_SPECS[era]):
        for sp, spec in kind_map.items():
            if species_filter is not None and sp not in species_filter:
                continue
            specs.append(spec)

    if not specs:
        raise ValueError("No species selected")

    regridder_cache: dict = {}
    conservation: dict[str, dict] = {}
    data_vars: dict[str, xr.DataArray] = {}

    for spec in specs:
        t0 = time.time()
        logger.info(
            "Loading %s %s from %s",
            era,
            out_var_name(spec.species, spec.kind),
            spec.relative_path,
        )
        da = load_monthly_clim(spec, emis_root)
        # area-weight integral before regrid (approx with cos(lat))
        lat_rad = np.deg2rad(np.asarray(da["lat"].values))
        w = np.cos(lat_rad)
        w = w / w.mean()
        pre = float((da * w[:, None]).mean().values)

        da_rg = regrid_field(
            da, lat_out, lon_out, regridder_cache, method=regrid_method
        )
        lat_rad_o = np.deg2rad(np.asarray(lat_out))
        wo = np.cos(lat_rad_o)
        wo = wo / wo.mean()
        post = float((da_rg * wo[:, None]).mean().values)
        name = out_var_name(spec.species, spec.kind)
        conservation[name] = {
            "pre_regrid_mean_coslat": pre,
            "post_regrid_mean_coslat": post,
            "rel_diff": (post - pre) / pre if pre != 0 else float("nan"),
        }
        data_vars[name] = da_rg.astype(np.float32)
        logger.info(
            "  %s done in %.1fs; regrid rel_diff=%.4g",
            name,
            time.time() - t0,
            conservation[name]["rel_diff"],
        )

    clim = xr.Dataset(data_vars)
    # synthetic month index 0..11 (Jan..Dec); drop native time for clarity
    clim = clim.assign_coords(time=np.arange(12))
    clim.attrs["era"] = era
    clim.attrs["interpolation_method"] = (
        "cubic spline on monthly centroids with periodic boundary (Dec/Jan wrap)"
    )
    clim.attrs["regrid_method"] = regrid_method
    return clim, conservation


def _zarr_encoding(ds: xr.Dataset, time_chunk: int) -> dict:
    """Chunk encoding compatible with zarr v3 + xarray (no legacy numcodecs)."""
    encoding = {}
    for name, da in ds.data_vars.items():
        chunks = tuple(time_chunk if d == "time" else da.sizes[d] for d in da.dims)
        encoding[name] = {"chunks": chunks}
    return encoding


def write_emis_zarr(
    clim: xr.Dataset,
    target_times: np.ndarray,
    output_zarr: Path,
    time_chunk: int = 124,
) -> None:
    """Expand repeating monthly clim to full 6h series and write zarr year-by-year.

    Because the seasonal cycle repeats, one noleap year is interpolated once and
    the spatial fields are reused for every year with updated time coordinates.
    """
    if output_zarr.exists():
        import shutil

        logger.warning("Removing existing %s", output_zarr)
        shutil.rmtree(output_zarr)

    years = sorted({t.year for t in target_times})
    # Build template from the first year's timestamps (noleap → identical month fracs)
    first_year = years[0]
    template_times = np.array([t for t in target_times if t.year == first_year])
    logger.info(
        "Interpolating reference year %d (%d steps)", first_year, template_times.size
    )
    template = interpolate_clim_to_times(clim, template_times)
    # Drop time coord from data arrays so we can reassign per year
    spatial = {name: template[name].values for name in template.data_vars}
    var_attrs = {name: dict(template[name].attrs) for name in template.data_vars}
    lat = template["lat"]
    lon = template["lon"]

    initialized = False
    encoding = None
    time_dep = list(template.data_vars)

    for year in years:
        year_times = np.array([t for t in target_times if t.year == year])
        if year_times.size != template_times.size:
            raise ValueError(
                f"Year {year} has {year_times.size} steps; "
                f"expected {template_times.size}"
            )
        t0 = time.time()
        data_vars = {
            name: xr.DataArray(
                spatial[name],
                dims=("time", "lat", "lon"),
                coords={"time": year_times, "lat": lat, "lon": lon},
                attrs=var_attrs[name],
            )
            for name in time_dep
        }
        ds = xr.Dataset(data_vars, attrs=dict(clim.attrs))
        ds.attrs["source_climatology"] = (
            "PI: 1850 (MAM4/SOAG) or 2010-as-1850 clim; PD: 2010_clim; "
            "cubic spline monthly centroids with Dec/Jan wrap, tiled over years"
        )
        if encoding is None:
            encoding = _zarr_encoding(ds, time_chunk=time_chunk)

        if not initialized:
            ds.to_zarr(
                str(output_zarr),
                mode="w",
                encoding=encoding,
                zarr_format=3,
                consolidated=True,
            )
            initialized = True
        else:
            ds[time_dep].to_zarr(
                str(output_zarr),
                mode="a",
                append_dim="time",
                zarr_format=3,
                consolidated=False,
            )
        logger.info(
            "Wrote year %d (%d steps) to %s in %.1fs",
            year,
            year_times.size,
            output_zarr.name,
            time.time() - t0,
        )

    zarr.consolidate_metadata(str(output_zarr))
    logger.info("Consolidated metadata for %s", output_zarr)


def build_era(
    era: str,
    *,
    emis_root: Path,
    target_zarr: Path,
    out_dir: Path,
    year_start: int,
    year_end: int,
    species: set[str] | None,
    regrid_method: str,
    output_suffix: str,
    time_chunk: int,
) -> Path:
    lat, lon, times = load_target_grid_and_times(target_zarr, year_start, year_end)
    clim, conservation = build_monthly_clim_dataset(
        era, emis_root, lat.values, lon.values, species, regrid_method
    )
    out_path = out_dir / (
        f"e3sm-aerosol-{era}-emis-forcing-6hourly-{year_start}-{year_end}{output_suffix}.zarr"
    )
    cons_path = out_path.with_suffix(".conservation.json")
    with open(cons_path, "w") as f:
        json.dump(conservation, f, indent=2)
    logger.info("Wrote conservation report %s", cons_path)

    # also save monthly clim for debugging
    clim_nc = out_path.with_name(out_path.name.replace(".zarr", "-monthly-clim.nc"))
    clim.to_netcdf(clim_nc)
    logger.info("Wrote monthly clim %s", clim_nc)

    write_emis_zarr(clim, times, out_path, time_chunk=time_chunk)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", choices=["PI", "PD", "both"], default="both")
    p.add_argument("--emis-root", type=Path, default=EMIS_ROOT)
    p.add_argument("--target-zarr", type=Path, default=TARGET_ZARR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--year-start", type=int, default=1945)
    p.add_argument("--year-end", type=int, default=1980)
    p.add_argument(
        "--species",
        type=str,
        default=None,
        help="Comma-separated species filter (e.g. SO2,DMS)",
    )
    p.add_argument("--regrid-method", default="conservative")
    p.add_argument("--output-suffix", default="", help="e.g. -smoke")
    p.add_argument("--time-chunk", type=int, default=124)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    species = (
        {s.strip() for s in args.species.split(",") if s.strip()}
        if args.species
        else None
    )
    eras = ["PI", "PD"] if args.era == "both" else [args.era]
    for era in eras:
        path = build_era(
            era,
            emis_root=args.emis_root,
            target_zarr=args.target_zarr,
            out_dir=args.out_dir,
            year_start=args.year_start,
            year_end=args.year_end,
            species=species,
            regrid_method=args.regrid_method,
            output_suffix=args.output_suffix,
            time_chunk=args.time_chunk,
        )
        logger.info("Finished %s → %s", era, path)


if __name__ == "__main__":
    main()
