#!/bin/bash
# 4-node parallel flux-only radiation stats (interactive). Launch via:
#   salloc -A m1266 -C cpu -q interactive -N 4 -t 04:00:00 --immediate=1800 \
#     bash /path/to/run-radiation-stats-fluxes-4node.sh
set -euo pipefail

export PATH="/pscratch/sd/y/yikwill/mamba/envs/fme-s2unet/bin:${PATH}"
python -m pip install --quiet dask distributed

SCRIPT_DIR="/global/homes/y/yikwill/llnl-research/ace-exp-s2resnet/scripts/data_process"
CHUNK_ROOT="/pscratch/sd/y/yikwill/datasets/fme-radiation-data-stats-flux-chunks"
FINAL_ROOT="/pscratch/sd/y/yikwill/datasets/fme-radiation-data-stats-fluxes"
EXISTING_ROOT="/pscratch/sd/y/yikwill/datasets/fme-radiation-data-stats"
LOG_ROOT="/global/homes/y/yikwill/llnl-research/slurm-out"

mkdir -p "${CHUNK_ROOT}" "${FINAL_ROOT}" "${LOG_ROOT}"
cd "${SCRIPT_DIR}"

# Write per-quartile configs (same time splits as the full-stats 4-node run).
"${PATH%%:*}/python3" - <<'PY'
import copy
import pathlib
import yaml

base = yaml.safe_load(
    pathlib.Path("configs/shield-som-increasing-co2-radiation-stats-fluxes.yaml").read_text()
)
base["stats"].pop("time_chunk_size", None)

ranges = [
    ("2031-01-01T06:00:00", "2048-07-02T00:00:00"),
    ("2048-07-02T00:00:00", "2065-12-31T18:00:00"),
    ("2065-12-31T18:00:00", "2083-07-02T12:00:00"),
    ("2083-07-02T12:00:00", None),
]
chunk_root = "/pscratch/sd/y/yikwill/datasets/fme-radiation-data-stats-flux-chunks"
for i, (start, end) in enumerate(ranges):
    cfg = copy.deepcopy(base)
    cfg["stats"]["output_directory"] = f"{chunk_root}/chunk{i}"
    cfg["stats"]["start_date"] = start
    cfg["stats"]["end_date"] = end
    path = pathlib.Path(
        f"configs/shield-som-increasing-co2-radiation-stats-flux-chunk{i}.yaml"
    )
    path.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"wrote {path} {start} -> {end}")
PY

echo "=== starting 4 parallel flux chunk stats at $(date) ==="
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=128 \
    bash -lc '
  export PATH="/pscratch/sd/y/yikwill/mamba/envs/fme-s2unet/bin:${PATH}"
  export DASK_NUM_WORKERS=64
  i="${SLURM_PROCID}"
  cfg="/global/homes/y/yikwill/llnl-research/ace-exp-s2resnet/scripts/data_process/configs/shield-som-increasing-co2-radiation-stats-flux-chunk${i}.yaml"
  echo "flux chunk ${i} on $(hostname) at $(date)"
  python -u /global/homes/y/yikwill/llnl-research/ace-exp-s2resnet/scripts/data_process/get_stats.py "${cfg}" 0 --force \
    > "'"${LOG_ROOT}"'/radiation-stats-flux-chunk${i}.log" 2>&1
'

echo "=== combining flux chunk stats at $(date) ==="
# combine_stats skips if the destination already exists; clear sidecar first.
rm -rf "${FINAL_ROOT}/increasing-CO2"
mkdir -p "${FINAL_ROOT}/increasing-CO2"
python -u - <<PY
from combine_stats import combine_stats

chunk_root = "${CHUNK_ROOT}"
final_root = "${FINAL_ROOT}"
roots = [f"{chunk_root}/chunk{i}/increasing-CO2/" for i in range(4)]
combine_stats(
    stats_roots=roots,
    output_directory=final_root,
    subdirectory="increasing-CO2",
    history="combined from 4 time quartiles via run-radiation-stats-fluxes-4node.sh",
)
print("combined flux stats at", final_root + "/increasing-CO2")
PY

echo "=== merging flux vars into existing stats at $(date) ==="
python -u - <<'PY'
import shutil
from pathlib import Path

import xarray as xr

existing = Path("/pscratch/sd/y/yikwill/datasets/fme-radiation-data-stats/increasing-CO2")
flux = Path("/pscratch/sd/y/yikwill/datasets/fme-radiation-data-stats-fluxes/increasing-CO2")
backup = existing.parent / "increasing-CO2-pre-flux-merge-backup"
filenames = (
    "centering.nc",
    "scaling-full-field.nc",
    "scaling-residual.nc",
    "time-mean.nc",
)

if backup.exists():
    shutil.rmtree(backup)
backup.mkdir(parents=True)
for name in filenames:
    shutil.copy2(existing / name, backup / name)
print(f"backed up existing stats to {backup}")

for name in filenames:
    base = xr.open_dataset(existing / name).load()
    add = xr.open_dataset(flux / name).load()
    overlap = set(base.data_vars) & set(add.data_vars)
    if overlap:
        raise SystemExit(f"{name}: refusing to overwrite existing vars {sorted(overlap)}")
    merged = xr.merge([base, add])
    # Preserve base history; note the flux merge.
    hist = str(base.attrs.get("history", ""))
    merged.attrs.update(base.attrs)
    merged.attrs["history"] = (
        hist + "; merged flux vars from fme-radiation-data-stats-fluxes"
    ).lstrip("; ")
    out = existing / name
    tmp = existing / f".{name}.tmp"
    merged.to_netcdf(tmp)
    tmp.replace(out)
    print(f"merged {name}: {len(base.data_vars)} + {len(add.data_vars)} -> {len(merged.data_vars)}")
PY

ls -la "${EXISTING_ROOT}/increasing-CO2/"
echo "=== finished at $(date) ==="
