#!/bin/bash

set -x

CONFIG_FILE=${CONFIG_FILE:-config-infer-PI-1951-1980.yaml}
CONFIG_SLUG=${CONFIG_FILE#config-infer-}
CONFIG_SLUG=${CONFIG_SLUG%.yaml}

export FME_CHECKPOINT_PATH=${FME_CHECKPOINT_PATH:-/pscratch/sd/y/yikwill/fme-output/55145243/training_checkpoints/best_inference_ckpt.tar}

export WANDB_NAME=${WANDB_NAME:-"$(date +%Y%m%d)-PM-E3SM-SFNO-infer-${CONFIG_SLUG}"}
export WANDB_RUN_GROUP=E3SM-SFNO
export FME_RESUME_KEY=${FME_RESUME_KEY:-SMOKE-INFER-$(date +%Y%m%d)-e3sm-aerosol-${CONFIG_SLUG}}

UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
cp "$CONFIG_FILE" $CONFIG_DIR/inference-config.yaml

set -xe

# Default: CFS project storage (override with FME_OUTPUT_DIR or FME_RESUME_KEY on scratch)
if [ -z "${FME_OUTPUT_DIR}" ]; then
  if [ -z "${RESUME_JOB_ID}" ]; then
    export FME_OUTPUT_DIR=/global/cfs/projectdirs/e3sm/yikwill/inference-output/${FME_RESUME_KEY}
  else
    export FME_OUTPUT_DIR=/global/cfs/projectdirs/e3sm/yikwill/inference-output/${RESUME_JOB_ID}
  fi
fi
mkdir -p $FME_OUTPUT_DIR

export WANDB_JOB_TYPE=inference
export WANDB_NOTES="PM inference, checkpoint: $FME_CHECKPOINT_PATH, results: $FME_OUTPUT_DIR"
set +x
export WANDB_API_KEY=$(cat ~/.config/wandb/api)
set -x

INFERENCE_CONFIG=${CONFIG_DIR}/inference-config.yaml
sed -i "s|FME_OUTPUT_DIR|${FME_OUTPUT_DIR}|" ${INFERENCE_CONFIG}
sed -i "s|FME_CHECKPOINT_PATH|${FME_CHECKPOINT_PATH}|" ${INFERENCE_CONFIG}
cp -r $CONFIG_DIR $FME_OUTPUT_DIR/job_config

source ~/.bashrc
conda activate "${CONDA_ENV:-fme}"
cd "$(dirname "$0")/../../.."
python -m fme.ace.validate_config --config_type evaluator ${INFERENCE_CONFIG}

OVERRIDE_ARGS=()
if [ -n "${N_FORWARD_STEPS}" ]; then
  OVERRIDE_ARGS+=(--override "n_forward_steps=${N_FORWARD_STEPS}")
fi
if [ -n "${FORWARD_STEPS_IN_MEMORY}" ]; then
  OVERRIDE_ARGS+=(--override "forward_steps_in_memory=${FORWARD_STEPS_IN_MEMORY}")
fi

srun -u --gpus-per-node=1 --cpus-per-task=32 --gpu-bind=none \
  python -u -m fme.ace.evaluator ${INFERENCE_CONFIG} "${OVERRIDE_ARGS[@]}"
