"""Convert monthly netCDF files to a single ACE-compatible Zarr store."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Sequence

import click
import pandas as pd
import xarray as xr
import zarr
from xarray.coding.times import CFDatetimeCoder

coder = CFDatetimeCoder(use_cftime=True)

# Matches config-train-PI-PD-1945-1980-aerosol-clim-forcing.yaml I/O names.
# For clim-only stores used as the first merge source, include ak_*/bk_* so
# MergedXarrayDataset can provide hybrid vertical coordinates for conserve_dry_air.
DEFAULT_E3SM_AEROSOL_VARS: tuple[str, ...] = tuple(
    sorted(
        {
            "LANDFRAC",
            "OCNFRAC",
            "ICEFRAC",
            "SOLIN",
            "aerindexall",
            "colccn.3",
            "PHIS",
            "PS",
            "TS",
            "Tat2m",
            "STWat2m",
            "Uat10m",
            "Vat10m",
            "PRECT",
            "FLDS",
            "FLUS",
            "FSDS",
            "FSUS",
            "FSUTOA",
            "FLUT",
            "DTENDTTW",
            "LHFLX",
            "SHFLX",
            "cdnc",
            "lcc",
            "lwp",
            *[f"T_{i}" for i in range(8)],
            *[f"STW_{i}" for i in range(8)],
            *[f"U_{i}" for i in range(8)],
            *[f"V_{i}" for i in range(8)],
            *[f"ak_{i}" for i in range(9)],
            *[f"bk_{i}" for i in range(9)],
        }
    )
)

DROP_VARIABLES = (
    "time_bnds",
    "lat_bnds",
    "lon_bnds",
    "area",
    "gw",
    "hyam",
    "hybm",
    "hyai",
    "hybi",
    "P0",
)


def _allowed_dims(dims: tuple[str, ...]) -> bool:
    return dims == () or set(dims) <= {"time", "lat", "lon"}


def _select_variables(ds: xr.Dataset, variable_names: Sequence[str]) -> xr.Dataset:
    missing = sorted(set(variable_names) - set(ds.variables))
    if missing:
        raise ValueError(f"Missing variables in first file: {missing}")
    selected = ds[variable_names]
    disallowed = [
        name
        for name in selected.data_vars
        if not _allowed_dims(tuple(selected[name].dims))
    ]
    if disallowed:
        raise ValueError(
            "Variables with unsupported dimensions (expected scalar or time/lat/lon): "
            f"{disallowed}"
        )
    return selected


def _drop_variables_for(path: str, variable_names: Sequence[str]) -> list[str]:
    keep = set(variable_names) | {"time", "lat", "lon"}
    with xr.open_dataset(path, decode_times=False) as ds:
        return sorted(v for v in ds.variables if v not in keep)


def _encoding(ds: xr.Dataset, time_chunk: int) -> dict:
    encoding: dict = {}
    for name, da in ds.data_vars.items():
        if "time" in da.dims:
            chunks = tuple(
                time_chunk if dim == "time" else ds.sizes[dim] for dim in da.dims
            )
            encoding[name] = {"chunks": chunks}
    return encoding


def _time_dependent_names(ds: xr.Dataset) -> list[str]:
    return [name for name in ds.data_vars if "time" in ds[name].dims]


def _filter_paths_by_date(
    paths: list[str],
    start_date: str | None,
    end_date: str | None,
) -> list[str]:
    if not start_date and not end_date:
        return paths
    start = pd.Timestamp(start_date) if start_date else None
    end = pd.Timestamp(end_date) if end_date else None
    kept: list[str] = []
    for path in paths:
        stem = os.path.basename(path).split(".")[-2]  # e.g. 1945-01
        year, month = map(int, stem.split("-"))
        month_start = pd.Timestamp(year=year, month=month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)
        if start is not None and month_end < start:
            continue
        if end is not None and month_start > end:
            continue
        kept.append(path)
    return kept


def _batch_paths(paths: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        return [paths]
    return [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]


def _read_month(
    path: str,
    variable_names: Sequence[str],
    drop_variables: Sequence[str],
) -> xr.Dataset:
    with xr.open_dataset(
        path,
        decode_times=coder,
        drop_variables=list(drop_variables),
    ) as raw:
        return _select_variables(raw, variable_names).load()


def _open_batch(
    paths: Sequence[str],
    variable_names: Sequence[str],
    drop_variables: Sequence[str],
    time_dep_names: Sequence[str],
    *,
    workers: int,
) -> xr.Dataset:
    if len(paths) == 1 or workers <= 1:
        parts = [_read_month(path, variable_names, drop_variables) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(
                pool.map(
                    lambda path: _read_month(path, variable_names, drop_variables),
                    paths,
                )
            )
    ds = xr.concat(
        [part[time_dep_names] for part in parts],
        dim="time",
        data_vars="minimal",
        coords="minimal",
    )
    static_names = [
        name for name in parts[0].data_vars if "time" not in parts[0][name].dims
    ]
    if static_names:
        ds = xr.merge(
            [ds, parts[0][static_names]],
            compat="override",
            join="override",
        )
    return ds


def _write_batch(
    ds: xr.Dataset,
    output_zarr: str,
    *,
    initialized: bool,
    encoding: dict,
    time_dep_names: list[str],
    consolidate: bool,
) -> None:
    if not initialized:
        ds.to_zarr(
            output_zarr,
            mode="w",
            encoding=encoding,
            zarr_format=3,
            consolidated=consolidate,
        )
        return
    ds[time_dep_names].to_zarr(
        output_zarr,
        mode="a",
        append_dim="time",
        zarr_format=3,
        consolidated=False,
    )


def _convert_legacy_monthly(
    paths: list[str],
    output_zarr: str,
    variable_names: list[str],
    start_date: str | None,
    end_date: str | None,
    time_chunk: int,
    consolidate: bool,
    drop_variables: list[str],
) -> int:
    encoding: dict | None = None
    time_dep_names: list[str] | None = None
    total_steps = 0
    initialized = False
    for path in paths:
        with xr.open_dataset(
            path,
            decode_times=coder,
            drop_variables=drop_variables,
        ) as raw:
            ds = _select_variables(raw, variable_names)
            if start_date or end_date:
                ds = ds.sel(time=slice(start_date, end_date))
            if ds.sizes.get("time", 0) == 0:
                continue
            if encoding is None:
                encoding = _encoding(ds, time_chunk=time_chunk)
                time_dep_names = _time_dependent_names(ds)
            assert encoding is not None and time_dep_names is not None
            _write_batch(
                ds,
                output_zarr,
                initialized=initialized,
                encoding=encoding,
                time_dep_names=time_dep_names,
                consolidate=False,
            )
            initialized = True
            total_steps += ds.sizes["time"]
            print(f"Processed {os.path.basename(path)} ({ds.sizes['time']} steps)")
    if consolidate and initialized:
        zarr.consolidate_metadata(output_zarr)
    return total_steps


def _convert_batched(
    paths: list[str],
    output_zarr: str,
    variable_names: list[str],
    start_date: str | None,
    end_date: str | None,
    time_chunk: int,
    batch_months: int,
    workers: int,
    consolidate: bool,
) -> int:
    drop_variables = _drop_variables_for(paths[0], variable_names)
    drop_variables = sorted(set(drop_variables) | set(DROP_VARIABLES))

    with xr.open_dataset(
        paths[0],
        decode_times=coder,
        drop_variables=drop_variables,
    ) as first_raw:
        first = _select_variables(first_raw, variable_names)
        encoding = _encoding(first, time_chunk=time_chunk)
        time_dep_names = _time_dependent_names(first)

    total_steps = 0
    initialized = False
    batches = _batch_paths(paths, batch_months)
    for batch_idx, batch in enumerate(batches, start=1):
        t0 = time.time()
        ds = _open_batch(
            batch,
            variable_names,
            drop_variables,
            time_dep_names,
            workers=workers,
        )
        if start_date or end_date:
            ds = ds.sel(time=slice(start_date, end_date))
        if ds.sizes.get("time", 0) == 0:
            continue

        _write_batch(
            ds,
            output_zarr,
            initialized=initialized,
            encoding=encoding,
            time_dep_names=time_dep_names,
            consolidate=False,
        )
        initialized = True
        batch_steps = ds.sizes["time"]
        total_steps += batch_steps
        elapsed = time.time() - t0
        print(
            f"Batch {batch_idx}/{len(batches)}: "
            f"{os.path.basename(batch[0])} .. {os.path.basename(batch[-1])} "
            f"({batch_steps} steps, {elapsed:.1f}s)"
        )

    if consolidate and initialized:
        zarr.consolidate_metadata(output_zarr)
    return total_steps


@click.command()
@click.argument("input_directory")
@click.argument("output_zarr")
@click.option(
    "--file-pattern",
    default="*.nc",
    show_default=True,
    help="Glob pattern for monthly netCDF files.",
)
@click.option(
    "--variable",
    "variables",
    multiple=True,
    help="Variable to include (repeatable). Default: e3sm-aerosol ACE I/O set.",
)
@click.option(
    "--all-vars",
    is_flag=True,
    help="Include all data variables with scalar or (time, lat, lon) dims.",
)
@click.option("--start-date", help="Optional subset start, e.g. 1945-01-01.")
@click.option("--end-date", help="Optional subset stop, e.g. 1980-12-31.")
@click.option(
    "--time-chunk",
    default=124,
    show_default=True,
    help="Zarr time chunk size (124 ≈ one month; use 1 for ERA5-identical chunks).",
)
@click.option(
    "--batch-months",
    default=3,
    show_default=True,
    help="Months per read/concat/write batch (lower = less RAM).",
)
@click.option(
    "--workers",
    default=16,
    show_default=True,
    help="Parallel netCDF readers per batch (I/O threads).",
)
@click.option(
    "--legacy-monthly-append",
    is_flag=True,
    help="Use slow one-file-at-a-time append path (for benchmarking only).",
)
@click.option(
    "--consolidate/--no-consolidate",
    default=True,
    show_default=True,
    help="Consolidate Zarr metadata after writing.",
)
def main(
    input_directory: str,
    output_zarr: str,
    file_pattern: str,
    variables: tuple[str, ...],
    all_vars: bool,
    start_date: str | None,
    end_date: str | None,
    time_chunk: int,
    batch_months: int,
    workers: int,
    legacy_monthly_append: bool,
    consolidate: bool,
) -> None:
    """Write monthly netCDF files under INPUT_DIRECTORY to OUTPUT_ZARR."""
    if os.path.exists(output_zarr):
        raise click.ClickException(
            f"Output path already exists: {output_zarr}. Remove it or pick a new path."
        )

    paths = sorted(glob(os.path.join(input_directory, file_pattern)))
    if not paths:
        raise click.ClickException(
            f"No files found for pattern {input_directory}/{file_pattern}"
        )
    paths = _filter_paths_by_date(paths, start_date, end_date)
    if not paths:
        raise click.ClickException("No files remain after date filtering.")

    with xr.open_dataset(
        paths[0],
        decode_times=coder,
        drop_variables=DROP_VARIABLES,
    ) as first_raw:
        if all_vars:
            # Exclude ccn.3bl from ACE zarrs (kept only in monthly NetCDF sources).
            variable_names = sorted(
                name
                for name in first_raw.data_vars
                if name != "ccn.3bl" and _allowed_dims(tuple(first_raw[name].dims))
            )
        elif variables:
            variable_names = list(variables)
        else:
            variable_names = list(DEFAULT_E3SM_AEROSOL_VARS)

    drop_variables = _drop_variables_for(paths[0], variable_names)
    drop_variables = sorted(set(drop_variables) | set(DROP_VARIABLES))

    t0 = time.time()
    if legacy_monthly_append:
        total_steps = _convert_legacy_monthly(
            paths,
            output_zarr,
            variable_names,
            start_date,
            end_date,
            time_chunk,
            consolidate,
            drop_variables,
        )
    else:
        total_steps = _convert_batched(
            paths,
            output_zarr,
            variable_names,
            start_date,
            end_date,
            time_chunk,
            batch_months,
            workers,
            consolidate,
        )

    if total_steps == 0:
        raise click.ClickException("No timesteps written; check date subset.")

    with xr.open_zarr(output_zarr, consolidated=consolidate) as written:
        assert written.sizes["time"] == total_steps
        assert set(variable_names) <= set(written.data_vars)

    elapsed = time.time() - t0
    print(
        f"Wrote {total_steps} timesteps to {output_zarr} "
        f"from {len(paths)} files in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
