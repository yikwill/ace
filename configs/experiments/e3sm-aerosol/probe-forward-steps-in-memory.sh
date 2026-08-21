#!/bin/bash
# Probe max forward_steps_in_memory on 1x A100-40GB.
# Usage (inside GPU allocation):
#   CONFIG_FILE=config-infer-PI-1951-1980.yaml ./probe-forward-steps-in-memory.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-config-infer-PI-1951-1980.yaml}
FME_CHECKPOINT_PATH=${FME_CHECKPOINT_PATH:-/pscratch/sd/y/yikwill/fme-output/55145243/training_checkpoints/best_inference_ckpt.tar}
PROBE_OUT=${PROBE_OUT:-${PSCRATCH}/fme-output/forward-steps-in-memory-probe-$(date +%Y%m%d-%H%M%S)}
STEPS_LIST=${STEPS_LIST:-"40 80 160 320 480 640 800 1000 1200 1600 2000 2500 3000"}

source ~/.bashrc
conda activate "${CONDA_ENV:-fme}"
cd "${SCRIPT_DIR}/../../.."

mkdir -p "${PROBE_OUT}"
RESULTS="${PROBE_OUT}/results.tsv"
echo -e "forward_steps_in_memory\tstatus\tpeak_gpu_mib\tduration_s" | tee "${RESULTS}"

for N in ${STEPS_LIST}; do
  RUN_DIR="${PROBE_OUT}/steps_${N}"
  mkdir -p "${RUN_DIR}"
  LOG="${RUN_DIR}/probe.log"
  echo "=== probing forward_steps_in_memory=${N} ===" | tee -a "${RESULTS}"

  START=$(date +%s)
  set +e
  srun -u --gpus-per-node=1 --cpus-per-task=32 --gpu-bind=none \
    python -u - <<PY 2>&1 | tee "${LOG}"
import gc
import os
import sys
import tempfile
import uuid

import dacite
import torch
import yaml

from fme.ace.inference.evaluator import InferenceEvaluatorConfig, run_evaluator_from_config
from fme.core.cli import prepare_config, prepare_directory

config_path = "${SCRIPT_DIR}/${CONFIG_FILE}"
with open(config_path) as f:
    data = yaml.safe_load(f)

data["experiment_dir"] = "${RUN_DIR}"
data["checkpoint_path"] = "${FME_CHECKPOINT_PATH}"
data["n_forward_steps"] = ${N}
data["forward_steps_in_memory"] = ${N}
data["logging"]["log_to_wandb"] = False
data["logging"]["log_to_file"] = False
data["data_writer"] = {
    "save_prediction_files": True,
    "save_monthly_files": False,
    "names": ["FSUTOA", "FLUT"],
}
data["aggregator"] = {
    "log_histograms": False,
    "log_global_mean_time_series": False,
    "log_global_mean_norm_time_series": False,
    "log_zonal_mean_images": False,
    "log_nino34_index": False,
    "log_ipo_index": False,
    "log_step_means": [],
}

tmp = os.path.join(tempfile.gettempdir(), f"fme-probe-{uuid.uuid4().hex}.yaml")
with open(tmp, "w") as f:
    yaml.dump(data, f)

config_data = prepare_config(tmp, override=None)
config = dacite.from_dict(
    InferenceEvaluatorConfig, config_data, config=dacite.Config(strict=True)
)
prepare_directory(config.experiment_dir, config_data)

torch.cuda.reset_peak_memory_stats()
try:
    with torch.no_grad():
        run_evaluator_from_config(config)
except Exception as e:
    print(f"PROBE_FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"PROBE_PEAK_GPU_MIB={peak:.1f}")
    gc.collect()
    torch.cuda.empty_cache()
PY
  STATUS=$?
  set -e
  END=$(date +%s)
  DUR=$((END - START))

  PEAK=$(grep -oP 'PROBE_PEAK_GPU_MIB=\K[0-9.]+' "${LOG}" | tail -1)
  PEAK=${PEAK:-NA}
  if [ ${STATUS} -eq 0 ]; then
    echo -e "${N}\tok\t${PEAK}\t${DUR}" | tee -a "${RESULTS}"
  else
    echo -e "${N}\tfail\t${PEAK}\t${DUR}" | tee -a "${RESULTS}"
    if grep -qiE 'out of memory|CUDA error|PROBE_FAILED' "${LOG}"; then
      echo "OOM or failure at ${N}; stopping probe." | tee -a "${RESULTS}"
      break
    fi
  fi
done

echo "Probe complete. Results: ${RESULTS}"
