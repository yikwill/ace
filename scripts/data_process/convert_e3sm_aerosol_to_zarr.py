#!/usr/bin/env python3
"""Convert one e3sm-aerosol PI or PD netCDF directory to an ACE Zarr store."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).resolve().parent
CONVERTER = SCRIPT_DIR / "convert_monthly_netcdf_to_zarr.py"

# label, input dir name, output zarr name, use --all-vars (clim stores are aerosol-only)
DATASETS: dict[str, tuple[str, str, str, bool]] = {
    "pi": (
        "PI",
        "e3sm-aerosol-PI-1945-1980",
        "e3sm-aerosol-PI-1945-1980.zarr",
        False,
    ),
    "pd": (
        "PD",
        "e3sm-aerosol-PD-1945-1980",
        "e3sm-aerosol-PD-1945-1980.zarr",
        False,
    ),
    "pi-clim": (
        "PI-clim",
        "e3sm-aerosol-PI-monthly-clim-6hourly-1945-1980",
        "e3sm-aerosol-PI-aerosol-clim-6hourly-1945-1980.zarr",
        True,
    ),
    "pd-clim": (
        "PD-clim",
        "e3sm-aerosol-PD-monthly-clim-6hourly-1945-1980",
        "e3sm-aerosol-PD-aerosol-clim-6hourly-1945-1980.zarr",
        True,
    ),
}


def _compare_zarr_format(reference: Path, candidate: Path) -> None:
    """Raise AssertionError when candidate store layout differs from reference."""
    import zarr

    ref = zarr.open_group(str(reference), mode="r")
    cand = zarr.open_group(str(candidate), mode="r")
    ref_keys = sorted(ref.array_keys())
    cand_keys = sorted(cand.array_keys())
    assert cand_keys == ref_keys, (
        f"variable mismatch: only_in_ref={set(ref_keys)-set(cand_keys)} "
        f"only_in_cand={set(cand_keys)-set(ref_keys)}"
    )

    for name in ref_keys:
        ref_arr = ref[name]
        cand_arr = cand[name]
        assert (
            ref_arr.shape[1:] == cand_arr.shape[1:]
        ), f"{name}: spatial shape {cand_arr.shape} != {ref_arr.shape}"
        assert (
            ref_arr.dtype == cand_arr.dtype
        ), f"{name}: dtype {cand_arr.dtype} != {ref_arr.dtype}"
        assert (
            ref_arr.chunks[1:] == cand_arr.chunks[1:]
        ), f"{name}: spatial chunks {cand_arr.chunks} != {ref_arr.chunks}"
        if ref_arr.ndim == 3:
            assert (
                ref_arr.chunks[0] == cand_arr.chunks[0]
            ), f"{name}: time chunk {cand_arr.chunks[0]} != {ref_arr.chunks[0]}"

    ref_arr = ref[ref_keys[0]]
    time_chunk = ref_arr.chunks[0] if ref_arr.ndim == 3 else "n/a"
    print(
        f"Format OK: {len(cand_keys)} variables, "
        f"lat/lon {ref_arr.shape[-2:]}, time chunk {time_chunk}"
    )


@click.command()
@click.argument(
    "dataset",
    type=click.Choice(sorted(DATASETS), case_sensitive=False),
)
@click.option(
    "--data-root",
    default="/global/cfs/projectdirs/e3sm/yikwill/datasets",
    show_default=True,
    help="Root directory containing netCDF input folders and final zarr output.",
)
@click.option(
    "--scratch-root",
    default=None,
    help="Write zarr here first (default: $PSCRATCH/e3sm-aerosol-zarr).",
)
@click.option(
    "--copy/--no-copy",
    default=True,
    show_default=True,
    help="Copy finished zarr from scratch to data-root.",
)
@click.option("--start-date", help="Optional subset start, e.g. 1945-01-01.")
@click.option("--end-date", help="Optional subset stop, e.g. 1945-03-31.")
@click.option("--batch-months", default=3, show_default=True)
@click.option("--time-chunk", default=124, show_default=True)
@click.option("--workers", default=16, show_default=True)
@click.option(
    "--verify-against",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="After conversion, assert zarr layout matches this reference store.",
)
def main(
    dataset: str,
    data_root: str,
    scratch_root: str | None,
    copy: bool,
    start_date: str | None,
    end_date: str | None,
    batch_months: int,
    time_chunk: int,
    workers: int,
    verify_against: Path | None,
) -> None:
    label, input_name, output_name, all_vars = DATASETS[dataset.lower()]
    data_root_path = Path(data_root)
    scratch_root_path = Path(
        scratch_root
        or os.path.join(os.environ.get("PSCRATCH", "."), "e3sm-aerosol-zarr")
    )
    input_dir = data_root_path / input_name
    scratch_zarr = scratch_root_path / output_name
    final_zarr = data_root_path / output_name

    if not input_dir.is_dir():
        raise click.ClickException(f"Input directory not found: {input_dir}")

    print(f"=== {label}: {input_dir} -> {scratch_zarr}", end="")
    if copy:
        print(f" (then {final_zarr})", end="")
    print(" ===")

    scratch_root_path.mkdir(parents=True, exist_ok=True)
    if scratch_zarr.exists():
        shutil.rmtree(scratch_zarr)

    cmd = [
        sys.executable,
        "-u",
        str(CONVERTER),
        str(input_dir),
        str(scratch_zarr),
        "--batch-months",
        str(batch_months),
        "--time-chunk",
        str(time_chunk),
        "--workers",
        str(workers),
    ]
    if all_vars:
        cmd.append("--all-vars")
    if start_date:
        cmd.extend(["--start-date", start_date])
    if end_date:
        cmd.extend(["--end-date", end_date])

    subprocess.run(cmd, check=True)

    if verify_against is not None:
        _compare_zarr_format(verify_against, scratch_zarr)

    if copy:
        if final_zarr.exists():
            shutil.rmtree(final_zarr)
        final_zarr.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying {scratch_zarr} -> {final_zarr}")
        shutil.copytree(scratch_zarr, final_zarr)


if __name__ == "__main__":
    main()
