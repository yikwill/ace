#!/usr/bin/env python3
"""Compute emis forcing stats from 6-hourly zarrs and merge into ACE scaling files.

Atmospheric stats (including prognostic ``aerindexall`` / ``colccn.3``) are taken
unchanged from the PI+PD native 6-hourly simulation stats. Emission-forcing
channels are computed from the 6-hourly interpolated emis zarrs using the same
reductions as ``get_stats.py``:

* centering: mean over (time, lat, lon)
* scaling-full-field: std over (time, lat, lon)
* scaling-residual: std of diff(time) over (time, lat, lon)

Emis variables are input-only forcings; ACE loss uses full-field scaling for
them, but residual entries are still written for completeness.

Prefer ``build_aerosol_stats_trees.py`` for the canonical clim-forcing /
prognostic stats trees. This script remains for ad-hoc emis merges.

Example:
  python merge_emis_into_stats.py \\
    --base-centering .../combined/centering.nc \
    --base-scaling-full .../combined/scaling-full-field.nc \
    --base-scaling-residual .../combined/scaling-residual.nc \
    --out-dir .../e3sm-aerosol-stats-aerosol-prognostic/combined
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_DATA_ROOT = Path("/global/cfs/projectdirs/e3sm/yikwill/datasets")
DEFAULT_PI_ZARR = (
    DEFAULT_DATA_ROOT / "e3sm-aerosol-PI-emis-forcing-6hourly-1945-1980.zarr"
)
DEFAULT_PD_ZARR = (
    DEFAULT_DATA_ROOT / "e3sm-aerosol-PD-emis-forcing-6hourly-1945-1980.zarr"
)
# Legacy helper: prefer build_aerosol_stats_trees.py for new trees.
DEFAULT_BASE = DEFAULT_DATA_ROOT / "e3sm-aerosol-stats-aerosol-prognostic/combined"
DEFAULT_OUT = DEFAULT_DATA_ROOT / "e3sm-aerosol-stats-aerosol-prognostic/combined"

PROGNOSTIC_AEROSOL_VARS = ("aerindexall", "colccn.3")


def _combine_mean_var(
    count: int, mean: float, m2: float, batch: np.ndarray
) -> tuple[int, float, float]:
    """Chan-style parallel mean/variance update (population variance)."""
    flat = np.asarray(batch, dtype=np.float64).ravel()
    n = flat.size
    if n == 0:
        return count, mean, m2
    bmean = float(flat.mean())
    bvar = float(flat.var())
    if count == 0:
        return n, bmean, bvar * n
    total = count + n
    delta = bmean - mean
    mean = mean + delta * n / total
    m2 = m2 + bvar * n + delta**2 * count * n / total
    return total, mean, m2


def _finalize_std(count: int, m2: float) -> float:
    if count == 0:
        return 1.0
    std = float(np.sqrt(max(m2 / count, 0.0)))
    if not np.isfinite(std) or std == 0.0:
        return 1.0
    return std


def streaming_stats(
    paths: list[Path], var: str, time_chunk: int = 124
) -> tuple[float, float, float]:
    """Mean, full-field std, and 6-hour residual std for one zarr variable."""
    count = 0
    mean = 0.0
    m2 = 0.0
    res_count = 0
    res_mean = 0.0
    res_m2 = 0.0
    prev: np.ndarray | None = None
    skipped = 0

    for path in paths:
        ds = xr.open_zarr(path, consolidated=True, chunks={"time": time_chunk})
        da = ds[var]
        n_time = int(da.sizes["time"])
        for start in range(0, n_time, time_chunk):
            stop = min(start + time_chunk, n_time)
            try:
                block = (
                    da.isel(time=slice(start, stop)).compute().values.astype(np.float64)
                )
            except RuntimeError as exc:
                warnings.warn(
                    f"Skipping {path.name} {var} time {start}:{stop}: {exc}",
                    stacklevel=2,
                )
                skipped += stop - start
                prev = None
                continue
            for t in range(block.shape[0]):
                cur = block[t]
                count, mean, m2 = _combine_mean_var(count, mean, m2, cur)
                if prev is not None:
                    res_count, res_mean, res_m2 = _combine_mean_var(
                        res_count, res_mean, res_m2, cur - prev
                    )
                prev = cur
        ds.close()

    if skipped:
        warnings.warn(f"{var}: skipped {skipped} timesteps due to read errors")

    if not np.isfinite(mean):
        mean = 0.0
    return mean, _finalize_std(count, m2), _finalize_std(res_count, res_m2)


def emis_stats_from_zarrs(pi_zarr: Path, pd_zarr: Path) -> tuple[dict, dict, dict]:
    """Compute ACE stats for all emis_* variables in the PI/PD zarr stores."""
    pi_ds = xr.open_zarr(pi_zarr, consolidated=True)
    pd_ds = xr.open_zarr(pd_zarr, consolidated=True)
    pi_vars = sorted(v for v in pi_ds.data_vars if v.startswith("emis_"))
    pd_vars = sorted(v for v in pd_ds.data_vars if v.startswith("emis_"))
    pi_ds.close()
    pd_ds.close()
    if pi_vars != pd_vars:
        raise ValueError(
            f"emis variable mismatch: only_in_pi={set(pi_vars)-set(pd_vars)} "
            f"only_in_pd={set(pd_vars)-set(pi_vars)}"
        )

    means: dict[str, float] = {}
    full_stds: dict[str, float] = {}
    residual_stds: dict[str, float] = {}
    for name in pi_vars:
        mean, ff_std, res_std = streaming_stats([pi_zarr, pd_zarr], name)
        means[name] = mean
        full_stds[name] = ff_std
        residual_stds[name] = res_std
        print(
            f"{name}: mean={mean:g} full_field_std={ff_std:g} residual_std={res_std:g}"
        )
    return means, full_stds, residual_stds


def merge_into_nc(
    base_path: Path | None,
    extras: dict[str, float],
    out_path: Path,
) -> None:
    if base_path is not None and base_path.exists():
        ds = xr.open_dataset(base_path)
        data = {k: ds[k] for k in ds.data_vars}
        ds.close()
    else:
        data = {}
    for k, v in extras.items():
        data[k] = xr.DataArray(np.float32(v), name=k)
    out = xr.Dataset(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    print(f"Wrote {out_path} ({len(out.data_vars)} vars)")


def verify_prognostic_stats(out_dir: Path, base_dir: Path) -> None:
    """Assert aerindexall/colccn.3 match native 6-hourly PI+PD simulation stats."""
    for name in ("centering.nc", "scaling-full-field.nc", "scaling-residual.nc"):
        base = xr.open_dataset(base_dir / name)
        out = xr.open_dataset(out_dir / name)
        for var in PROGNOSTIC_AEROSOL_VARS:
            b = float(base[var].values)
            o = float(out[var].values)
            if not np.isclose(b, o, rtol=0, atol=0):
                raise AssertionError(
                    f"{var} in {name}: out={o:g} != base={b:g} "
                    "(prognostic aerosol stats must come from PI+PD simulation stats)"
                )
        base.close()
        out.close()
    print(
        "Verified aerindexall/colccn.3 match " f"{base_dir} in all three scaling files."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pi-zarr", type=Path, default=DEFAULT_PI_ZARR)
    p.add_argument("--pd-zarr", type=Path, default=DEFAULT_PD_ZARR)
    p.add_argument(
        "--base-centering",
        type=Path,
        default=DEFAULT_BASE / "centering.nc",
    )
    p.add_argument(
        "--base-scaling-full",
        type=Path,
        default=DEFAULT_BASE / "scaling-full-field.nc",
    )
    p.add_argument(
        "--base-scaling-residual",
        type=Path,
        default=DEFAULT_BASE / "scaling-residual.nc",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify prognostic aerosol stats in --out-dir; do not rewrite.",
    )
    args = p.parse_args()

    if args.verify_only:
        verify_prognostic_stats(args.out_dir, args.base_scaling_residual.parent)
        return

    means, full_stds, residual_stds = emis_stats_from_zarrs(args.pi_zarr, args.pd_zarr)
    merge_into_nc(args.base_centering, means, args.out_dir / "centering.nc")
    merge_into_nc(
        args.base_scaling_full, full_stds, args.out_dir / "scaling-full-field.nc"
    )
    merge_into_nc(
        args.base_scaling_residual,
        residual_stds,
        args.out_dir / "scaling-residual.nc",
    )
    verify_prognostic_stats(args.out_dir, args.base_scaling_residual.parent)


if __name__ == "__main__":
    main()
