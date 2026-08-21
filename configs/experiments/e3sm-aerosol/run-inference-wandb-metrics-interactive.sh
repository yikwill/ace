#!/bin/bash
# Interactive WandB-metrics inference (no 96GB netCDF output).
# Usage:
#   ./run-inference-wandb-metrics-interactive.sh
#   TRAIN_JOB_ID=55328228 CONFIG_FILE=config-infer-PD-1951-1980-wandb-pd-only.yaml ./run-inference-wandb-metrics-interactive.sh
#
# Subset smoke test (~20-30 min on interactive GPU):
#   SMOKE=1 WANDB_MODE=disabled ./run-inference-wandb-metrics-interactive.sh
#
# Full 30-year run — request 4h interactive GPU (see WANDB_INFER_COMMANDS.md):
#   salloc --account e3sm --qos interactive --constraint gpu --nodes 1 \
#     --gpus-per-node 1 --cpus-per-task 128 --ntasks-per-node 1 --time 03:30:00 \
#     bash -lc 'cd .../e3sm-aerosol && TRAIN_JOB_ID=55145243 CONFIG_FILE=config-infer-PI-1951-1980-wandb-pi-pd.yaml ./run-inference-wandb-metrics-interactive.sh'

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

CONFIG_FILE=${CONFIG_FILE:-config-infer-PI-1951-1980-wandb-pi-pd.yaml}
CONFIG_SLUG=${CONFIG_FILE#config-infer-}
CONFIG_SLUG=${CONFIG_SLUG%.yaml}

export TRAIN_JOB_ID=${TRAIN_JOB_ID:-55145243}
export FME_CHECKPOINT_PATH=${FME_CHECKPOINT_PATH:-/pscratch/sd/y/yikwill/fme-output/${TRAIN_JOB_ID}/training_checkpoints/best_inference_ckpt.tar}

if [ ! -f "${FME_CHECKPOINT_PATH}" ]; then
  echo "Checkpoint not found: ${FME_CHECKPOINT_PATH}"
  exit 1
fi

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-E3SM-SFNO-wandb-infer}
export WANDB_NAME=${WANDB_NAME:-"$(date +%Y%m%d)-PM-E3SM-SFNO-${CONFIG_SLUG}"}
export WANDB_JOB_TYPE=inference
export FME_RESUME_KEY=${FME_RESUME_KEY:-wandb-infer-${CONFIG_SLUG}-$(date +%Y%m%d)}

UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p "${CONFIG_DIR}"
cp "${SCRIPT_DIR}/${CONFIG_FILE}" "${CONFIG_DIR}/inference-config.yaml"

export FME_OUTPUT_DIR=${FME_OUTPUT_DIR:-${PSCRATCH}/fme-wandb-infer/${FME_RESUME_KEY}}
mkdir -p "${FME_OUTPUT_DIR}"

export WANDB_NOTES="WandB-metrics inference, checkpoint: ${FME_CHECKPOINT_PATH}, scratch: ${FME_OUTPUT_DIR}"
if [ -f "${HOME}/.config/wandb/api" ]; then
  set +x
  export WANDB_API_KEY=$(cat ~/.config/wandb/api)
  set -x
fi

INFERENCE_CONFIG=${CONFIG_DIR}/inference-config.yaml
sed -i "s|FME_OUTPUT_DIR|${FME_OUTPUT_DIR}|" "${INFERENCE_CONFIG}"
sed -i "s|FME_CHECKPOINT_PATH|${FME_CHECKPOINT_PATH}|" "${INFERENCE_CONFIG}"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "SMOKE=1: subset validation to 1979-1980, n_forward_steps=600, disable val snapshot"
  python - <<'PY' "${INFERENCE_CONFIG}"
import sys
import yaml
path = sys.argv[1]
with open(path) as f:
    cfg = yaml.safe_load(f)
cfg["n_forward_steps"] = 600
for branch in cfg["validation"]["loader"]["dataset"]["merge"]:
    branch.setdefault("subset", {})
    branch["subset"]["start_time"] = "1979-01-01"
    branch["subset"]["stop_time"] = "1980-12-31"
cfg["validation"].setdefault("aggregator", {})
cfg["validation"]["aggregator"]["snapshot"] = {"enabled": False}
with open(path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PY
  export N_FORWARD_STEPS=600
  export WANDB_MODE=${WANDB_MODE:-disabled}
fi

cp -r "${CONFIG_DIR}" "${FME_OUTPUT_DIR}/job_config"

source ~/.bashrc
conda activate "${CONDA_ENV:-fme}"
cd "${REPO_ROOT}"
python -m fme.ace.validate_config --config_type evaluator "${INFERENCE_CONFIG}"

OVERRIDE_ARGS=()
if [ -n "${N_FORWARD_STEPS:-}" ]; then
  OVERRIDE_ARGS+=(--override "n_forward_steps=${N_FORWARD_STEPS}")
fi
if [ -n "${FORWARD_STEPS_IN_MEMORY:-}" ]; then
  OVERRIDE_ARGS+=(--override "forward_steps_in_memory=${FORWARD_STEPS_IN_MEMORY}")
fi
if [ "${WANDB_MODE:-}" = "disabled" ]; then
  OVERRIDE_ARGS+=(--override "logging.log_to_wandb=false")
fi

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "FME_CHECKPOINT_PATH=${FME_CHECKPOINT_PATH}"
echo "FME_OUTPUT_DIR=${FME_OUTPUT_DIR}"
echo "N_FORWARD_STEPS=${N_FORWARD_STEPS:-43799 (full)}"

run_evaluator() {
  python -u -m fme.ace.evaluator "${INFERENCE_CONFIG}" "${OVERRIDE_ARGS[@]}"
}

# Inside an salloc shell on a compute node, run directly (avoid nested srun CPU bind errors).
if [[ "$(hostname)" == nid* ]]; then
  run_evaluator
else
  srun -u --gpus-per-node=1 --cpus-per-task=32 --gpu-bind=none run_evaluator
fi
