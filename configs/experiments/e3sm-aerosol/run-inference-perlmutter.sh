#!/bin/bash

set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

CONFIG_FILE=${CONFIG_FILE:-config-infer-PI-1951-1980.yaml}
CONFIG_SLUG=${CONFIG_FILE#config-infer-}
CONFIG_SLUG=${CONFIG_SLUG%.yaml}

export WANDB_RUN_GROUP=E3SM-SFNO

export TRAIN_JOB_ID=${TRAIN_JOB_ID:-55145243}
export FME_CHECKPOINT_DIR=${PSCRATCH}/fme-output/${TRAIN_JOB_ID}/training_checkpoints
export CKPT_NAME=${CKPT_NAME:-best_inference_ckpt.tar}
export FME_CHECKPOINT_PATH=${FME_CHECKPOINT_DIR}/${CKPT_NAME}

export COMMIT=${COMMIT:-$(git -C "${REPO_ROOT}" rev-parse --short HEAD)}

export WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-infer-${CONFIG_SLUG}-$(basename "${CKPT_NAME}" .tar)"
export FME_RESUME_KEY=${FME_RESUME_KEY:-infer-${CONFIG_SLUG}-$(basename "${CKPT_NAME}" .tar)-$(date +%Y%m%d)}

if [ ! -f "${FME_CHECKPOINT_PATH}" ]; then
  echo "Checkpoint not found: ${FME_CHECKPOINT_PATH}"
  exit 1
fi

# user should not need to modify below

UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
cp "${SCRIPT_DIR}/${CONFIG_FILE}" $CONFIG_DIR/inference-config.yaml
cp "${SCRIPT_DIR}/run-inference-perlmutter.sh" $CONFIG_DIR/run-inference-perlmutter.sh
cp "${SCRIPT_DIR}/sbatch-scripts/sbatch-inference.sh" $CONFIG_DIR/sbatch-inference.sh
cp "${SCRIPT_DIR}/make-venv.sh" $CONFIG_DIR/make-venv.sh

export FME_VENV=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)
conda activate $FME_VENV
cd "${REPO_ROOT}"
python -m fme.ace.validate_config --config_type evaluator $CONFIG_DIR/inference-config.yaml

# sbatch -t 00:30:00 -q debug "${SCRIPT_DIR}/sbatch-scripts/sbatch-inference.sh"
sbatch --export=ALL -J infer-e3sm-aerosol-${CONFIG_SLUG} "${SCRIPT_DIR}/sbatch-scripts/sbatch-inference.sh"
