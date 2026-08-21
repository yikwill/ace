#!/bin/bash
# Probe max validation batch size with train batch fixed at TRAIN_BS.
# Launch via:
#   salloc -A e3sm -C gpu -q interactive -N 4 --gpus-per-node=4 -t 04:00:00 \
#     --immediate=1800 TRAIN_BS=256 bash /path/to/probe_val_batch_size_radiation.sh
set -euo pipefail

PERLMUTTER_DIR="/global/homes/y/yikwill/llnl-research/perlmutter"
ACE_LIVE="/global/homes/y/yikwill/llnl-research/ace-exp-s2resnet"
CONFIG_FILE="configs/experiments/s2resnet/config-train-radiation-increasing-co2.yaml"
TRAIN_BS="${TRAIN_BS:-256}"

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
PROBE_ROOT="${PSCRATCH}/fme-output-debug/val-batch-probe-$$"
mkdir -p "${PROBE_ROOT}"
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)"
export MASTER_PORT="${TORCH_MASTER_PORT}"

WORLD=$((TRAIN_NODES * TRAIN_GPUS_PER_NODE))
RESULT_FILE="${PROBE_ROOT}/val_batch_probe_results.txt"
echo "world_size=${WORLD} train_batch_size=${TRAIN_BS}" | tee "${RESULT_FILE}"

MAX_OK=1024
VAL_BS_LIST="${VAL_BS_LIST:-2048 4096 8192 16384}"
for val_bs in ${VAL_BS_LIST}; do
    if (( val_bs % WORLD != 0 )); then
        echo "skip val_bs=${val_bs} (not divisible by world_size=${WORLD})" | tee -a "${RESULT_FILE}"
        continue
    fi
    export FME_OUTPUT_DIR="${PROBE_ROOT}/train${TRAIN_BS}-val${val_bs}"
    mkdir -p "${FME_OUTPUT_DIR}"
    echo "=== train_bs=${TRAIN_BS} val_bs=${val_bs} at $(date) ===" | tee -a "${RESULT_FILE}"
    "${FME_VENV}/bin/python" - <<PY
import yaml
from pathlib import Path
p = Path("${CONFIG_DIR}/train-config.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["max_epochs"] = 1
cfg["save_checkpoint"] = False
cfg["train_loader"]["batch_size"] = ${TRAIN_BS}
cfg["validation"]["loader"]["batch_size"] = ${val_bs}
cfg["logging"]["log_to_wandb"] = False
cfg.pop("inference", None)
p.write_text(yaml.dump(cfg, sort_keys=False))
PY
    set +e
    "${CONFIG_DIR}/train.sh" >> "${FME_OUTPUT_DIR}/train.log" 2>&1
    ec=$?
    set -e
    if [[ ${ec} -eq 0 ]]; then
        echo "OK val_bs=${val_bs}" | tee -a "${RESULT_FILE}"
        MAX_OK=${val_bs}
    else
        echo "FAIL val_bs=${val_bs} exit=${ec}" | tee -a "${RESULT_FILE}"
        if grep -q "OutOfMemoryError\|out of memory" "${FME_OUTPUT_DIR}/train.log"; then
            echo "  (CUDA OOM)" | tee -a "${RESULT_FILE}"
        fi
        break
    fi
done

echo "max_ok_val_batch_size=${MAX_OK} (train_bs=${TRAIN_BS})" | tee -a "${RESULT_FILE}"
cat "${RESULT_FILE}"
