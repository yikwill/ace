#!/usr/bin/env python3
"""Smoke-validate PI/PD emis forcing zarrs against the plan checklist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_PI = Path(
    "/p/lustre5/yik1/datasets/e3sm-aerosol-PI-emis-forcing-6hourly-1945-1980.zarr"
)
DEFAULT_PD = Path(
    "/p/lustre5/yik1/datasets/e3sm-aerosol-PD-emis-forcing-6hourly-1945-1980.zarr"
)
TARGET = Path("/p/lustre5/yik1/datasets/e3sm-aerosol-PD-1945-1980.zarr")


def open_z(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=True)


def check_grid_time(ds: xr.Dataset, target: xr.Dataset, label: str) -> list[str]:
    errs = []
    if (
        ds.sizes.get("lat") != target.sizes["lat"]
        or ds.sizes.get("lon") != target.sizes["lon"]
    ):
        errs.append(
            f"{label}: lat/lon size mismatch {dict(ds.sizes)} vs {dict(target.sizes)}"
        )
    if not np.allclose(ds.lat.values, target.lat.values):
        errs.append(f"{label}: lat values differ from target")
    if not np.allclose(ds.lon.values, target.lon.values):
        errs.append(f"{label}: lon values differ from target")
    if ds.sizes.get("time") != target.sizes["time"]:
        got = ds.sizes.get("time")
        want = target.sizes["time"]
        errs.append(f"{label}: time length {got} != target {want}")
    else:
        if str(ds.time.values[0]) != str(target.time.values[0]) or str(
            ds.time.values[-1]
        ) != str(target.time.values[-1]):
            errs.append(
                f"{label}: time endpoints {ds.time.values[0]}…{ds.time.values[-1]} "
                f"vs {target.time.values[0]}…{target.time.values[-1]}"
            )
    cal = getattr(ds.time.values[0], "calendar", None)
    if cal not in (None, "noleap", "365_day"):
        errs.append(f"{label}: unexpected calendar {cal}")
    return errs


def check_repeating_clim(ds: xr.Dataset, var: str, label: str) -> list[str]:
    errs = []
    if var not in ds:
        errs.append(f"{label}: missing {var}")
        return errs
    # compare first timestep of 1946 vs 1945 same month-day-hour
    t = ds.time.values
    years = np.array([x.year for x in t])
    i0 = np.where(years == 1945)[0]
    i1 = np.where(years == 1946)[0]
    if i0.size == 0 or i1.size == 0:
        errs.append(f"{label}: missing 1945/1946 for repeat check")
        return errs
    a = ds[var].isel(time=i0).values
    b = ds[var].isel(time=i1).values
    if a.shape != b.shape:
        errs.append(f"{label}: year length mismatch for {var}")
        return errs
    max_abs = float(np.max(np.abs(a - b)))
    if max_abs > 1e-4 * max(float(np.max(np.abs(a))), 1.0):
        errs.append(
            f"{label}: {var} 1945 vs 1946 max_abs={max_abs:g} (expected near 0)"
        )
    else:
        print(f"OK {label} {var} repeating clim max_abs={max_abs:g}")
    return errs


def check_pi_pd_differ(pi: xr.Dataset, pd: xr.Dataset, var: str) -> list[str]:
    errs = []
    if var not in pi or var not in pd:
        errs.append(f"missing {var} in PI or PD")
        return errs
    # mean over first year
    a = pi[var].isel(time=slice(0, 1460)).mean().values
    b = pd[var].isel(time=slice(0, 1460)).mean().values
    rel = abs(float(a) - float(b)) / max(abs(float(b)), 1e-30)
    if rel < 1e-3:
        errs.append(f"{var}: PI≈PD (rel={rel:g}); expected MAM4 difference")
    else:
        print(
            f"OK {var} PI vs PD differ: "
            f"PI_mean={float(a):g} PD_mean={float(b):g} rel={rel:g}"
        )
    return errs


def check_pi_pd_match(pi: xr.Dataset, pd: xr.Dataset, var: str) -> list[str]:
    errs = []
    if var not in pi or var not in pd:
        errs.append(f"missing {var} in PI or PD")
        return errs
    a = pi[var].isel(time=0).values
    b = pd[var].isel(time=0).values
    max_abs = float(np.max(np.abs(a - b)))
    if max_abs > 1e-4 * max(float(np.max(np.abs(a))), 1.0):
        errs.append(f"{var}: PI vs PD should match for chem/DMS; max_abs={max_abs:g}")
    else:
        print(f"OK {var} PI matches PD (chem/DMS) max_abs={max_abs:g}")
    return errs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pi", type=Path, default=DEFAULT_PI)
    p.add_argument("--pd", type=Path, default=DEFAULT_PD)
    p.add_argument("--target", type=Path, default=TARGET)
    args = p.parse_args()

    errs: list[str] = []
    target = open_z(args.target)
    pi = open_z(args.pi)
    pd = open_z(args.pd)

    print("PI vars:", sorted(pi.data_vars))
    print("PD vars:", sorted(pd.data_vars))
    errs += check_grid_time(pi, target, "PI")
    errs += check_grid_time(pd, target, "PD")
    errs += check_repeating_clim(pi, "emis_SO2_sfc", "PI")
    errs += check_repeating_clim(pd, "emis_SO2_sfc", "PD")
    errs += check_pi_pd_differ(pi, pd, "emis_SO2_sfc")
    errs += check_pi_pd_differ(pi, pd, "emis_bc_a4_elev")
    errs += check_pi_pd_match(pi, pd, "emis_DMS_sfc")
    errs += check_pi_pd_match(pi, pd, "emis_CO_sfc")
    errs += check_pi_pd_match(pi, pd, "emis_NO2_elev")

    # conservation reports if present
    for era, path in [("PI", args.pi), ("PD", args.pd)]:
        cons = path.with_suffix(".conservation.json")
        if cons.exists():
            print(f"Conservation report present: {cons}")
        else:
            print(f"NOTE: no conservation json at {cons}")

    target.close()
    pi.close()
    pd.close()

    if errs:
        print("FAILURES:")
        for e in errs:
            print(" -", e)
        return 1
    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
