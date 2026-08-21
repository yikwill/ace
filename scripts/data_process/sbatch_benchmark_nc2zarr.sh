#!/bin/bash
#SBATCH --account e3sm
#SBATCH --qos regular
#SBATCH --constraint cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 01:00:00
#SBATCH --job-name nc2zarr-bench
#SBATCH --output /global/homes/y/yikwill/llnl-research/slurm-out/nc2zarr-bench-%j.out

set -euo pipefail
source ~/.bashrc
conda activate fme

SCRIPT=/global/homes/y/yikwill/llnl-research/ace-exp-e3sm-aerosol/scripts/data_process/convert_monthly_netcdf_to_zarr.py
INPUT=/global/cfs/projectdirs/e3sm/yikwill/datasets/e3sm-aerosol-PI-1945-1980
OUT=/pscratch/sd/y/yikwill/nc2zarr-bench
mkdir -p "$OUT"

echo "=== legacy 24mo (chunk=1, monthly append) ==="
rm -rf "$OUT/legacy24-cpu.zarr"
/usr/bin/time -f 'legacy elapsed %e s' python -u "$SCRIPT" "$INPUT" "$OUT/legacy24-cpu.zarr" \
  --start-date 1945-01-01 --end-date 1946-12-31 \
  --legacy-monthly-append --time-chunk 1

echo "=== batched 24mo (chunk=124, 3-month batches) ==="
rm -rf "$OUT/batch24-cpu.zarr"
/usr/bin/time -f 'batched elapsed %e s' python -u "$SCRIPT" "$INPUT" "$OUT/batch24-cpu.zarr" \
  --start-date 1945-01-01 --end-date 1946-12-31 \
  --batch-months 3 --time-chunk 124 --workers 32

python3 - <<'PY'
import xarray as xr
legacy = xr.open_zarr("/pscratch/sd/y/yikwill/nc2zarr-bench/legacy24-cpu.zarr")
batch = xr.open_zarr("/pscratch/sd/y/yikwill/nc2zarr-bench/batch24-cpu.zarr")
assert legacy.sizes["time"] == batch.sizes["time"]
assert legacy.time.equals(batch.time)
for v in ["T_0", "SOLIN", "PRECT", "cdnc"]:
    diff = float((legacy[v] - batch[v]).abs().max())
    assert diff == 0.0, (v, diff)
print("VALIDATION OK:", legacy.sizes["time"], "steps")
PY
