#!/usr/bin/env python3
"""Build aerosol-clim-forcing and aerosol-prognostic stats trees from 6h zarrs.

Writes PRECT-corrected (×1000) ACE stats under::

  e3sm-aerosol-stats-aerosol-clim-forcing/{PI,PD,combined}/
  e3sm-aerosol-stats-aerosol-prognostic/{PI,PD,combined}/

Each directory contains centering.nc, scaling-full-field.nc,
scaling-residual.nc, and time-mean.nc.

Philosophy:
- Atmospheric prognostics from native sim zarrs (PRECT ×1000).
- Clim-forcing tree: aerindexall/colccn.3 from aerosol clim zarrs.
- Prognostic tree: aerindexall/colccn.3 from sim zarrs; emis_* from emis zarrs.
- Never includes ccn.3bl.

Example (on an interactive node, many CPU workers)::

  python build_aerosol_stats_trees.py --workers 32
"""

from __future__ import annotations

import argparse
import logging
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

DATA_ROOT = Path("/global/cfs/projectdirs/e3sm/yikwill/datasets")
AEROSOL_VARS = ("aerindexall", "colccn.3")
EXCLUDE_VARS = frozenset({"ccn.3bl"})
PRECT_SCALE = 1000.0
STAT_FILES = (
    "centering.nc",
    "scaling-full-field.nc",
    "scaling-residual.nc",
    "time-mean.nc",
)


@dataclass(frozen=True)
class VarStats:
    mean: float
    full_field_std: float
    residual_std: float
    time_mean: np.ndarray  # (lat, lon)
    n_time: int
    lat: np.ndarray
    lon: np.ndarray


def _combine_mean_var(
    count: int, mean: float, m2: float, batch: np.ndarray
) -> tuple[int, float, float]:
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


def spacetime_vars(zarr_path: Path) -> list[str]:
    ds = xr.open_zarr(zarr_path, consolidated=True)
    try:
        names = []
        for name, da in ds.data_vars.items():
            if name in EXCLUDE_VARS:
                continue
            if tuple(da.dims) != ("time", "lat", "lon"):
                continue
            names.append(str(name))
        return sorted(names)
    finally:
        ds.close()


def compute_var_stats(
    zarr_path: str,
    var: str,
    time_chunk: int = 124,
    prect_scale: float = PRECT_SCALE,
) -> tuple[str, VarStats]:
    """Worker: stream one variable from one zarr. Returns (var, stats)."""
    path = Path(zarr_path)
    count = 0
    mean = 0.0
    m2 = 0.0
    res_count = 0
    res_mean = 0.0
    res_m2 = 0.0
    prev: np.ndarray | None = None
    time_sum: np.ndarray | None = None
    n_time = 0
    lat = lon = None
    scale = prect_scale if var == "PRECT" else 1.0

    ds = xr.open_zarr(path, consolidated=True, chunks={"time": time_chunk})
    da = ds[var]
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    n_total = int(da.sizes["time"])
    for start in range(0, n_total, time_chunk):
        stop = min(start + time_chunk, n_total)
        try:
            block = da.isel(time=slice(start, stop)).compute().values.astype(np.float64)
        except RuntimeError as exc:
            warnings.warn(f"Skipping {path.name} {var} time {start}:{stop}: {exc}")
            prev = None
            continue
        if scale != 1.0:
            block = block * scale
        for t in range(block.shape[0]):
            cur = block[t]
            count, mean, m2 = _combine_mean_var(count, mean, m2, cur)
            if prev is not None:
                res_count, res_mean, res_m2 = _combine_mean_var(
                    res_count, res_mean, res_m2, cur - prev
                )
            prev = cur
            if time_sum is None:
                time_sum = np.zeros_like(cur, dtype=np.float64)
            time_sum += cur
            n_time += 1
    ds.close()

    if time_sum is None or n_time == 0:
        raise RuntimeError(f"No readable timesteps for {var} in {path}")

    if not np.isfinite(mean):
        mean = 0.0
    return var, VarStats(
        mean=mean,
        full_field_std=_finalize_std(count, m2),
        residual_std=_finalize_std(res_count, res_m2),
        time_mean=(time_sum / n_time).astype(np.float32),
        n_time=n_time,
        lat=lat,
        lon=lon,
    )


def compute_zarr_stats(
    zarr_path: Path,
    variables: list[str] | None,
    workers: int,
    time_chunk: int,
) -> dict[str, VarStats]:
    vars_ = variables if variables is not None else spacetime_vars(zarr_path)
    vars_ = [v for v in vars_ if v not in EXCLUDE_VARS]
    logger.info(
        "Computing %d vars from %s with %d workers", len(vars_), zarr_path.name, workers
    )
    out: dict[str, VarStats] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(compute_var_stats, str(zarr_path), v, time_chunk): v
            for v in vars_
        }
        for fut in as_completed(futures):
            var, stats = fut.result()
            out[var] = stats
            logger.info(
                "  %s: mean=%.6g ff=%.6g res=%.6g",
                var,
                stats.mean,
                stats.full_field_std,
                stats.residual_std,
            )
    return out


def write_stats_dir(
    out_dir: Path,
    stats: dict[str, VarStats],
    *,
    history: str,
    force: bool,
) -> None:
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists; pass --force to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    any_stats = next(iter(stats.values()))
    n_time = any_stats.n_time
    lat = any_stats.lat
    lon = any_stats.lon

    centering = {}
    full = {}
    resid = {}
    tmean = {}
    for name, s in sorted(stats.items()):
        centering[name] = xr.DataArray(np.float32(s.mean), name=name)
        full[name] = xr.DataArray(np.float32(s.full_field_std), name=name)
        resid[name] = xr.DataArray(np.float32(s.residual_std), name=name)
        tmean[name] = xr.DataArray(
            s.time_mean,
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
            name=name,
        )

    attrs = {"history": history, "input_samples": np.int64(n_time)}
    xr.Dataset(centering, attrs=attrs).to_netcdf(out_dir / "centering.nc")
    xr.Dataset(full, attrs=attrs).to_netcdf(out_dir / "scaling-full-field.nc")
    xr.Dataset(resid, attrs=attrs).to_netcdf(out_dir / "scaling-residual.nc")
    xr.Dataset(tmean, attrs=attrs).to_netcdf(out_dir / "time-mean.nc")
    logger.info("Wrote %s (%d vars)", out_dir, len(stats))


def overlay_vars(
    base: dict[str, VarStats], overlays: dict[str, VarStats]
) -> dict[str, VarStats]:
    out = dict(base)
    out.update(overlays)
    return out


def combine_two_stats_dirs(
    pi_dir: Path,
    pd_dir: Path,
    out_dir: Path,
    *,
    history: str,
    force: bool,
) -> None:
    """Combine PI+PD stats with the same reductions as combine_stats.py."""
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists; pass --force to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    pi_c = xr.open_dataset(pi_dir / "centering.nc")
    pd_c = xr.open_dataset(pd_dir / "centering.nc")
    pi_ff = xr.open_dataset(pi_dir / "scaling-full-field.nc")
    pd_ff = xr.open_dataset(pd_dir / "scaling-full-field.nc")
    pi_res = xr.open_dataset(pi_dir / "scaling-residual.nc")
    pd_res = xr.open_dataset(pd_dir / "scaling-residual.nc")
    pi_tm = xr.open_dataset(pi_dir / "time-mean.nc")
    pd_tm = xr.open_dataset(pd_dir / "time-mean.nc")

    common = sorted((set(pi_c.data_vars) & set(pd_c.data_vars)) - EXCLUDE_VARS)
    samples = xr.DataArray(
        [
            float(pi_c.attrs.get("input_samples", 1)),
            float(pd_c.attrs.get("input_samples", 1)),
        ],
        dims=["run"],
    )
    total_samples = int(samples.sum())

    def _avg_mean(a: xr.Dataset, b: xr.Dataset) -> xr.Dataset:
        comb = xr.concat([a[common], b[common]], dim="run")
        return comb.weighted(samples).mean(dim="run")

    def _avg_std(a: xr.Dataset, b: xr.Dataset) -> xr.Dataset:
        comb = xr.concat([a[common], b[common]], dim="run")
        return (comb**2).weighted(samples).mean(dim="run") ** 0.5

    centering = _avg_mean(pi_c, pd_c)
    residual = _avg_std(pi_res, pd_res)
    # Full-field: include between-run mean variance (combine_stats.get_combined_stats)
    comb_ff = xr.concat([pi_ff[common], pd_ff[common]], dim="run")
    comb_c = xr.concat([pi_c[common], pd_c[common]], dim="run")
    avg_c = comb_c.weighted(samples).mean(dim="run")
    full = ((comb_c - avg_c) ** 2 + comb_ff**2).weighted(samples).mean(dim="run") ** 0.5
    tmean = _avg_mean(pi_tm, pd_tm)

    attrs = {"history": history, "input_samples": np.int64(total_samples)}
    for ds, name in [
        (centering, "centering.nc"),
        (full, "scaling-full-field.nc"),
        (residual, "scaling-residual.nc"),
        (tmean, "time-mean.nc"),
    ]:
        ds = ds.load()
        ds.attrs.update(attrs)
        # Drop run-only artifacts; keep float32 scalars / fields
        out = xr.Dataset({k: ds[k].astype(np.float32) for k in common}, attrs=attrs)
        out.to_netcdf(out_dir / name)

    for ds in (pi_c, pd_c, pi_ff, pd_ff, pi_res, pd_res, pi_tm, pd_tm):
        ds.close()
    logger.info("Wrote combined %s (%d vars)", out_dir, len(common))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--time-chunk", type=int, default=124)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--skip-compute",
        action="store_true",
        help="Only assemble/combine from cached per-source stats under --cache-dir",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Scratch dir for per-source intermediate stats "
            "(default: data-root/_stats_cache)"
        ),
    )
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    root = args.data_root
    cache = args.cache_dir or (root / "_aerosol_stats_cache")
    cache.mkdir(parents=True, exist_ok=True)

    sim = {
        "PI": root / "e3sm-aerosol-PI-1945-1980.zarr",
        "PD": root / "e3sm-aerosol-PD-1945-1980.zarr",
    }
    clim = {
        "PI": root / "e3sm-aerosol-PI-aerosol-clim-6hourly-1945-1980.zarr",
        "PD": root / "e3sm-aerosol-PD-aerosol-clim-6hourly-1945-1980.zarr",
    }
    emis = {
        "PI": root / "e3sm-aerosol-PI-emis-forcing-6hourly-1945-1980.zarr",
        "PD": root / "e3sm-aerosol-PD-emis-forcing-6hourly-1945-1980.zarr",
    }
    for path in [*sim.values(), *clim.values(), *emis.values()]:
        if not path.is_dir():
            raise FileNotFoundError(path)

    clim_tree = root / "e3sm-aerosol-stats-aerosol-clim-forcing"
    prog_tree = root / "e3sm-aerosol-stats-aerosol-prognostic"

    # --- compute per-source ---
    sim_stats: dict[str, dict[str, VarStats]] = {}
    clim_stats: dict[str, dict[str, VarStats]] = {}
    emis_stats: dict[str, dict[str, VarStats]] = {}

    for era in ("PI", "PD"):
        sim_cache = cache / f"sim-{era}"
        clim_cache = cache / f"clim-{era}"
        emis_cache = cache / f"emis-{era}"

        if args.skip_compute and (sim_cache / "centering.nc").exists():
            logger.info("Loading cached %s", sim_cache)
        else:
            s = compute_zarr_stats(sim[era], None, args.workers, args.time_chunk)
            write_stats_dir(
                sim_cache,
                s,
                history=f"Sim 6h zarr stats from {sim[era].name}; PRECT×{PRECT_SCALE}",
                force=True,
            )
            sim_stats[era] = s

        if args.skip_compute and (clim_cache / "centering.nc").exists():
            logger.info("Loading cached %s", clim_cache)
        else:
            c = compute_zarr_stats(
                clim[era], list(AEROSOL_VARS), args.workers, args.time_chunk
            )
            write_stats_dir(
                clim_cache,
                c,
                history=f"Aerosol clim 6h zarr stats from {clim[era].name}",
                force=True,
            )
            clim_stats[era] = c

        if args.skip_compute and (emis_cache / "centering.nc").exists():
            logger.info("Loading cached %s", emis_cache)
        else:
            emis_vars = [v for v in spacetime_vars(emis[era]) if v.startswith("emis_")]
            e = compute_zarr_stats(emis[era], emis_vars, args.workers, args.time_chunk)
            write_stats_dir(
                emis_cache,
                e,
                history=f"Emis 6h zarr stats from {emis[era].name}",
                force=True,
            )
            emis_stats[era] = e

        # Prefer in-memory if just computed; else reload from cache files into VarStats
        if era not in sim_stats:
            sim_stats[era] = _load_var_stats(sim_cache)
        if era not in clim_stats:
            clim_stats[era] = _load_var_stats(clim_cache)
        if era not in emis_stats:
            emis_stats[era] = _load_var_stats(emis_cache)

    # --- assemble era bundles ---
    for era in ("PI", "PD"):
        forcing = overlay_vars(sim_stats[era], clim_stats[era])
        write_stats_dir(
            clim_tree / era,
            forcing,
            history=(
                f"aerosol-clim-forcing {era}: atmos from {sim[era].name}; "
                f"aerindexall/colccn.3 from {clim[era].name}; PRECT×{PRECT_SCALE}"
            ),
            force=True,
        )
        prognostic = overlay_vars(sim_stats[era], emis_stats[era])
        write_stats_dir(
            prog_tree / era,
            prognostic,
            history=(
                f"aerosol-prognostic {era}: atmos+aerosol from {sim[era].name}; "
                f"emis from {emis[era].name}; PRECT×{PRECT_SCALE}"
            ),
            force=True,
        )

    combine_two_stats_dirs(
        clim_tree / "PI",
        clim_tree / "PD",
        clim_tree / "combined",
        history="aerosol-clim-forcing combined PI+PD",
        force=True,
    )
    combine_two_stats_dirs(
        prog_tree / "PI",
        prog_tree / "PD",
        prog_tree / "combined",
        history="aerosol-prognostic combined PI+PD",
        force=True,
    )
    logger.info("Done.")


def _load_var_stats(stats_dir: Path) -> dict[str, VarStats]:
    c = xr.open_dataset(stats_dir / "centering.nc")
    ff = xr.open_dataset(stats_dir / "scaling-full-field.nc")
    res = xr.open_dataset(stats_dir / "scaling-residual.nc")
    tm = xr.open_dataset(stats_dir / "time-mean.nc")
    n_time = int(c.attrs.get("input_samples", 0))
    out: dict[str, VarStats] = {}
    for name in c.data_vars:
        if name in EXCLUDE_VARS:
            continue
        out[str(name)] = VarStats(
            mean=float(c[name].values),
            full_field_std=float(ff[name].values),
            residual_std=float(res[name].values),
            time_mean=np.asarray(tm[name].values, dtype=np.float32),
            n_time=n_time,
            lat=np.asarray(tm["lat"].values),
            lon=np.asarray(tm["lon"].values),
        )
    for ds in (c, ff, res, tm):
        ds.close()
    return out


if __name__ == "__main__":
    main()
