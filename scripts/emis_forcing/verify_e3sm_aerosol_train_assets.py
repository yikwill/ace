#!/usr/bin/env python3
"""Verify e3sm-aerosol train configs use coherent zarr datasets and stats bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

import xarray as xr

DATA_ROOT = Path("/global/cfs/projectdirs/e3sm/yikwill/datasets")
AEROSOL_VARS = ("aerindexall", "colccn.3")


class ConfigExpectation(TypedDict):
    zarrs: list[str]
    stats_dir: str
    aerosol_stats_source: str
    expect_emis: bool
    clim_zarr: NotRequired[str]


CONFIG_EXPECTATIONS: dict[str, ConfigExpectation] = {
    "config-train-PI-1945-1980-aerosol-clim-forcing.yaml": {
        "zarrs": [
            "e3sm-aerosol-PI-aerosol-clim-6hourly-1945-1980.zarr",
            "e3sm-aerosol-PI-1945-1980.zarr",
        ],
        "stats_dir": "e3sm-aerosol-stats-aerosol-clim-forcing/PI",
        "aerosol_stats_source": "clim",
        "clim_zarr": "e3sm-aerosol-PI-aerosol-clim-6hourly-1945-1980.zarr",
        "expect_emis": False,
    },
    "config-train-PD-1945-1980-aerosol-clim-forcing.yaml": {
        "zarrs": [
            "e3sm-aerosol-PD-aerosol-clim-6hourly-1945-1980.zarr",
            "e3sm-aerosol-PD-1945-1980.zarr",
        ],
        "stats_dir": "e3sm-aerosol-stats-aerosol-clim-forcing/PD",
        "aerosol_stats_source": "clim",
        "clim_zarr": "e3sm-aerosol-PD-aerosol-clim-6hourly-1945-1980.zarr",
        "expect_emis": False,
    },
    "config-train-PI-PD-1945-1980-aerosol-clim-forcing.yaml": {
        "zarrs": [
            "e3sm-aerosol-PI-aerosol-clim-6hourly-1945-1980.zarr",
            "e3sm-aerosol-PD-aerosol-clim-6hourly-1945-1980.zarr",
            "e3sm-aerosol-PI-1945-1980.zarr",
            "e3sm-aerosol-PD-1945-1980.zarr",
        ],
        "stats_dir": "e3sm-aerosol-stats-aerosol-clim-forcing/combined",
        "aerosol_stats_source": "clim-combined",
        "expect_emis": False,
    },
    "config-train-PI-PD-1945-1980-aerosol-prognostic.yaml": {
        "zarrs": [
            "e3sm-aerosol-PI-1945-1980.zarr",
            "e3sm-aerosol-PD-1945-1980.zarr",
            "e3sm-aerosol-PI-emis-forcing-6hourly-1945-1980.zarr",
            "e3sm-aerosol-PD-emis-forcing-6hourly-1945-1980.zarr",
        ],
        "stats_dir": "e3sm-aerosol-stats-aerosol-prognostic/combined",
        "aerosol_stats_source": "sim-combined",
        "expect_emis": True,
    },
    "config-train-PI-PD-1945-1980-aerosol-prognostic-emis-reduced.yaml": {
        "zarrs": [
            "e3sm-aerosol-PI-1945-1980.zarr",
            "e3sm-aerosol-PD-1945-1980.zarr",
            "e3sm-aerosol-PI-emis-forcing-6hourly-1945-1980.zarr",
            "e3sm-aerosol-PD-emis-forcing-6hourly-1945-1980.zarr",
        ],
        "stats_dir": "e3sm-aerosol-stats-aerosol-prognostic/combined",
        "aerosol_stats_source": "sim-combined",
        "expect_emis": True,
    },
}


def _stat(path: Path, var: str, kind: str) -> float:
    return float(xr.open_dataset(path / f"{kind}.nc")[var].values)


def verify_zarrs(zarrs: list[str]) -> list[str]:
    errors: list[str] = []
    for name in zarrs:
        path = DATA_ROOT / name
        if not path.is_dir():
            errors.append(f"missing zarr: {path}")
            continue
        ds = xr.open_zarr(path, consolidated=True)
        if "ccn.3bl" in ds.data_vars:
            errors.append(f"ccn.3bl still present in {path}")
        ds.close()
    return errors


def verify_stats_dir(stats_rel: str) -> list[str]:
    errors: list[str] = []
    stats = DATA_ROOT / stats_rel
    for name in (
        "centering.nc",
        "scaling-full-field.nc",
        "scaling-residual.nc",
        "time-mean.nc",
    ):
        if not (stats / name).exists():
            errors.append(f"missing stats file: {stats / name}")
    if not (stats / "centering.nc").exists():
        return errors
    centering = xr.open_dataset(stats / "centering.nc")
    if "ccn.3bl" in centering.data_vars:
        errors.append(f"ccn.3bl present in {stats}/centering.nc")
    centering.close()
    prect = _stat(stats, "PRECT", "scaling-full-field")
    # kg/m^2/s scale is ~1e-5; raw m/s would be ~1e-8
    if prect < 1e-6:
        errors.append(
            f"PRECT full-field std looks like m/s not kg/m^2/s: {prect:g} in {stats}"
        )
    return errors


def _stream_mean_std(zarr_rel: str, var: str, n_times: int = 40) -> tuple[float, float]:
    """Cheap sample check: mean/std over first n_times (not full-dataset equality)."""
    import numpy as np

    ds = xr.open_zarr(DATA_ROOT / zarr_rel, consolidated=True)
    da = ds[var].isel(time=slice(0, n_times)).values.astype(np.float64)
    ds.close()
    return float(da.mean()), float(da.std())


def verify_aerosol_provenance(
    stats_rel: str, source: str, clim_zarr: str | None
) -> list[str]:
    """Check aerosol stats are finite; clim vs sim residual should differ."""
    errors: list[str] = []
    stats = DATA_ROOT / stats_rel
    for var in AEROSOL_VARS:
        try:
            mean = _stat(stats, var, "centering")
            res = _stat(stats, var, "scaling-residual")
        except Exception as exc:
            errors.append(f"missing {var} in {stats_rel}: {exc}")
            continue
        if not (mean == mean) or res <= 0:
            errors.append(f"bad {var} stats in {stats_rel}: mean={mean} res={res}")

    # Clim-forcing residual for aerindexall should be much smaller than prognostic
    if source == "clim" and clim_zarr:
        clim_res = _stat(stats, "aerindexall", "scaling-residual")
        prog = (
            DATA_ROOT / "e3sm-aerosol-stats-aerosol-prognostic" / Path(stats_rel).name
        )
        if (prog / "scaling-residual.nc").exists():
            sim_res = _stat(prog, "aerindexall", "scaling-residual")
            if clim_res >= sim_res:
                errors.append(
                    (
                        f"clim residual aerindexall ({clim_res:g}) "
                        f"should be < sim ({sim_res:g})"
                    )
                )
    return errors


def verify_emis(stats_rel: str, expect_emis: bool) -> list[str]:
    errors: list[str] = []
    stats = DATA_ROOT / stats_rel
    centering = xr.open_dataset(stats / "centering.nc")
    emis_vars = [v for v in centering.data_vars if str(v).startswith("emis_")]
    centering.close()
    if expect_emis and not emis_vars:
        errors.append(f"no emis_* variables in {stats}")
    if not expect_emis and emis_vars:
        errors.append(f"unexpected emis_* variables in {stats}: {emis_vars[:3]}...")
    return errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", action="append", default=list(CONFIG_EXPECTATIONS))
    args = p.parse_args()

    all_errors: list[str] = []
    for cfg in args.config:
        spec = CONFIG_EXPECTATIONS[cfg]
        print(f"=== {cfg} ===")
        errs = verify_zarrs(spec["zarrs"])
        errs += verify_stats_dir(spec["stats_dir"])
        errs += verify_aerosol_provenance(
            spec["stats_dir"],
            spec["aerosol_stats_source"],
            spec.get("clim_zarr"),
        )
        errs += verify_emis(spec["stats_dir"], spec["expect_emis"])
        if errs:
            print("  FAIL")
            for e in errs:
                print(f"    - {e}")
            all_errors.extend(errs)
        else:
            print("  OK")

    if all_errors:
        print(f"\n{len(all_errors)} issue(s) found.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
