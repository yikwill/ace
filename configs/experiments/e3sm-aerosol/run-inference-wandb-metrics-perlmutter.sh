#!/bin/bash
# Submit one WandB-metrics 30-yr inference batch job (no 96GB netCDF).
#
#   TRAIN_JOB_ID=55145243 CONFIG_FILE=config-infer-PI-1951-1980-wandb-pi-pd.yaml \
#     ./run-inference-wandb-metrics-perlmutter.sh

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
export FME_RESUME_KEY=${FME_RESUME_KEY:-wandb-infer-${CONFIG_SLUG}-$(date +%Y%m%d)}

UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p "${CONFIG_DIR}"
cp "${SCRIPT_DIR}/${CONFIG_FILE}" "${CONFIG_DIR}/inference-config.yaml"
cp "${SCRIPT_DIR}/sbatch-scripts/sbatch-inference-wandb.sh" "${CONFIG_DIR}/sbatch-inference-wandb.sh"

source ~/.bashrc
conda activate "${CONDA_ENV:-fme}"
cd "${REPO_ROOT}"
python -m fme.ace.validate_config --config_type evaluator "${CONFIG_DIR}/inference-config.yaml"

JOB_NAME="wandb-${CONFIG_SLUG}"
sbatch --export=ALL -J "${JOB_NAME}" "${CONFIG_DIR}/sbatch-inference-wandb.sh"
