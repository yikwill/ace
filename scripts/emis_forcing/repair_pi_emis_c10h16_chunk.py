#!/usr/bin/env python3
"""Repair one corrupt zarr chunk in PI emis forcing store.

``emis_C10H16_sfc`` time chunk 69 (indices 8556:8680, 1950-11-11 .. 1950-12-11)
has a bad zstd payload. The store is a repeating seasonal climatology tiled
yearly, so the correct values are identical to the same calendar window in any
other year. This script copies slice 7096:7220 (1949) into 8556:8680 (1950).

Backs up the corrupt chunk file before overwriting. Use --dry-run to inspect only.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

DEFAULT_ZARR = Path(
    "/global/cfs/projectdirs/e3sm/yikwill/datasets/"
    "e3sm-aerosol-PI-emis-forcing-6hourly-1945-1980.zarr"
)
VAR = "emis_C10H16_sfc"
SRC_SLICE = slice(7096, 7220)
DST_SLICE = slice(8556, 8680)
CHUNK_IDX = 69


def _read_slice(zarr_path: Path, slc: slice) -> np.ndarray:
    ds = xr.open_zarr(zarr_path, consolidated=True)
    try:
        return ds[VAR].isel(time=slc).values
    finally:
        ds.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"zarr: {args.zarr}")
    print(f"var:  {VAR}")
    print(f"dst:  time[{DST_SLICE.start}:{DST_SLICE.stop}] (chunk {CHUNK_IDX})")
    print(f"src:  time[{SRC_SLICE.start}:{SRC_SLICE.stop}]")

    try:
        _read_slice(args.zarr, DST_SLICE)
        print("Destination slice already readable; nothing to repair.")
        return 0
    except RuntimeError as exc:
        print(f"Destination unreadable (expected): {exc}")

    src = _read_slice(args.zarr, SRC_SLICE)
    print(f"Source OK: shape={src.shape}, mean={src.mean():.6g}")

    if args.dry_run:
        print("Dry run; no writes.")
        return 0

    chunk_data = args.zarr / VAR / "c" / str(CHUNK_IDX) / "0" / "0"
    if not chunk_data.exists():
        print(f"Chunk data not found: {chunk_data}", file=sys.stderr)
        return 1

    bak = chunk_data.with_suffix(
        chunk_data.suffix + f".corrupt-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    shutil.copy2(chunk_data, bak)
    print(f"Backed up corrupt chunk to {bak}")

    root = zarr.open_group(str(args.zarr), mode="r+")
    root[VAR][DST_SLICE] = src

    repaired = _read_slice(args.zarr, DST_SLICE)
    if not np.array_equal(repaired, src):
        print("ERROR: post-repair data does not match source", file=sys.stderr)
        return 1

    print("Repair OK; destination slice matches source.")
    print("Re-run merge_emis_into_stats.py to refresh emis_C10H16_sfc stats.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
