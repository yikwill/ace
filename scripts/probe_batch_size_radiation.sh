#!/bin/bash
# Probe max global batch sizes (powers of 2) for 4-node radiation training.
# Launch via:
#   salloc -A e3sm -C gpu -q interactive -N 4 --gpus-per-node=4 -t 04:00:00 \
#     --immediate=1800 bash /path/to/probe_batch_size_radiation.sh
set -euo pipefail

PERLMUTTER_DIR="/global/homes/y/yikwill/llnl-research/perlmutter"
ACE_LIVE="/global/homes/y/yikwill/llnl-research/ace-exp-s2resnet"
CONFIG_FILE="configs/experiments/s2resnet/config-train-radiation-increasing-co2.yaml"

export ACE_ROOT="${ACE_LIVE}"
export CONFIG_FILE
export MODE=debug
export WANDB_MODE=disabled
export FME_DEBUG_MAX_EPOCHS=1
export FME_DEBUG_INFER_STEPS=20
export FME_DEBUG_INFER_STARTS=16
export FME_DEBUG_TRAIN_STOP=2031-03-01
export FME_DEBUG_VAL_STOP=2031-04-01
export TRAIN_NODES="${SLURM_JOB_NUM_NODES:-4}"
export TRAIN_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"

# shellcheck source=/dev/null
source "${PERLMUTTER_DIR}/env.sh"
# shellcheck source=/dev/null
source "${PERLMUTTER_DIR}/stage.sh"

export CONFIG_DIR
PROBE_ROOT="${PSCRATCH}/fme-output-debug/batch-probe-$$"
mkdir -p "${PROBE_ROOT}"
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)"
export MASTER_PORT="${TORCH_MASTER_PORT}"

WORLD=$((TRAIN_NODES * TRAIN_GPUS_PER_NODE))
RESULT_FILE="${PROBE_ROOT}/batch_probe_results.txt"
echo "world_size=${WORLD}" | tee "${RESULT_FILE}"

MAX_OK=0
for bs in 16 32 64 128 256 512 1024 2048; do
    if (( bs % WORLD != 0 )); then
        echo "skip bs=${bs} (not divisible by world_size=${WORLD})" | tee -a "${RESULT_FILE}"
        continue
    fi
    export FME_OUTPUT_DIR="${PROBE_ROOT}/bs${bs}"
    mkdir -p "${FME_OUTPUT_DIR}"
    echo "=== trying global batch_size=${bs} at $(date) ===" | tee -a "${RESULT_FILE}"
    python3 - <<PY
import yaml
from pathlib import Path
p = Path("${CONFIG_DIR}/train-config.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["max_epochs"] = 1
cfg["save_checkpoint"] = False
cfg["train_loader"]["batch_size"] = ${bs}
cfg["validation"]["loader"]["batch_size"] = ${bs}
cfg["logging"]["log_to_wandb"] = False
cfg.pop("inference", None)
p.write_text(yaml.dump(cfg, sort_keys=False))
PY
  set +e
  "${CONFIG_DIR}/train.sh" >> "${FME_OUTPUT_DIR}/train.log" 2>&1
  ec=$?
  set -e
  if [[ ${ec} -eq 0 ]]; then
    echo "OK batch_size=${bs}" | tee -a "${RESULT_FILE}"
    MAX_OK=${bs}
  else
    echo "FAIL batch_size=${bs} exit=${ec}" | tee -a "${RESULT_FILE}"
    break
  fi
done

echo "max_ok_batch_size=${MAX_OK}" | tee -a "${RESULT_FILE}"
cat "${RESULT_FILE}"
