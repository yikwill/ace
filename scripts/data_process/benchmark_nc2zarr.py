#!/usr/bin/env python3
"""Benchmark legacy vs batched nc→zarr conversion on a month subset."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "convert_monthly_netcdf_to_zarr.py"
INPUT = Path("/global/cfs/projectdirs/e3sm/yikwill/datasets/e3sm-aerosol-PI-1945-1980")
OUT_ROOT = Path("/pscratch/sd/y/yikwill/nc2zarr-bench")


def run(label: str, extra_args: list[str], out: Path) -> float:
    if out.exists():
        shutil.rmtree(out)
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(INPUT),
        str(out),
        "--start-date",
        "1945-01-01",
        "--end-date",
        "1947-12-31",
        *extra_args,
    ]
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"{label}: {elapsed:.1f}s")
    return elapsed


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    legacy = run(
        "legacy (432 appends, time-chunk=1)",
        ["--legacy-monthly-append", "--time-chunk", "1"],
        OUT_ROOT / "legacy.zarr",
    )
    batched = run(
        "batched (2 writes, time-chunk=124)",
        ["--batch-months", "12", "--time-chunk", "124", "--workers", "16"],
        OUT_ROOT / "batched.zarr",
    )
    speedup = legacy / batched
    print(f"\nSpeedup: {speedup:.1f}x ({legacy:.1f}s -> {batched:.1f}s on 36 months)")


if __name__ == "__main__":
    main()
