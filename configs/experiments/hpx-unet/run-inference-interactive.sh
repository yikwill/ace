#!/bin/bash
#
# Run evaluator inference on an interactive GPU allocation.
#
# 1) Request a node (Perlmutter example):
#    salloc -A m1266 -q interactive -C gpu --nodes=1 --ntasks=1 \
#      --gpus-per-node=1 --cpus-per-task=32 -t 04:00:00
# 2) From this directory:
#    ./run-inference-interactive.sh
#
# Optional env overrides:
#   INFERENCE_CONFIG=config-inference-long-46year.yaml
#   CKPT_NAME=best_ckpt.tar
#   FME_RESUME_KEY=my-eval-run   # stable output dir under $PSCRATCH/fme-output/
#   FME_OUTPUT_DIR=/path/to/dir  # if set, ignores FME_RESUME_KEY

set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export INFERENCE_CONFIG=${INFERENCE_CONFIG:-config-inference-prescribed-q0-pressfc.yaml}
if [[ "${INFERENCE_CONFIG}" = /* ]]; then
  INFERENCE_CONFIG_PATH="${INFERENCE_CONFIG}"
else
  INFERENCE_CONFIG_PATH="${SCRIPT_DIR}/${INFERENCE_CONFIG}"
fi
INFERENCE_CONFIG_NAME=$(basename "${INFERENCE_CONFIG_PATH}" .yaml)
INFERENCE_CONFIG_NAME=${INFERENCE_CONFIG_NAME#config-inference}
INFERENCE_CONFIG_NAME=${INFERENCE_CONFIG_NAME#-}
INFERENCE_CONFIG_NAME=${INFERENCE_CONFIG_NAME:-default}

export WANDB_RUN_GROUP=2025-05-06-AIMIP-HPXUNET

# from job 53035680: /pscratch/sd/y/yikwill/fme-conda-envs/28d5b5dce
export COMMIT=28d5b5dce

export TRAIN_JOB_ID=53035680
export FME_CHECKPOINT_DIR=${PSCRATCH}/fme-output/${TRAIN_JOB_ID}/training_checkpoints

export CKPT_NAME=${CKPT_NAME:-best_ckpt.tar}
export FME_CHECKPOINT_PATH=${FME_CHECKPOINT_DIR}/${CKPT_NAME}

export FME_TRAIN_DIR=/pscratch/sd/e/elynnwu/fme-dataset
export FME_STATS_DIR=/pscratch/sd/y/yikwill/datasets/ace/2025-10-04-healpix-era5-dataset

export WANDB_NAME=PM-AIMIP-HPXUNET-eval-${TRAIN_JOB_ID}-${INFERENCE_CONFIG_NAME}-$(basename "${CKPT_NAME}" .tar)
export WANDB_NOTES="interactive eval train job ${TRAIN_JOB_ID}, config ${INFERENCE_CONFIG_NAME}, checkpoint ${CKPT_NAME}"

if [ ! -f "${INFERENCE_CONFIG_PATH}" ]; then
  echo "Inference config not found: ${INFERENCE_CONFIG_PATH}"
  exit 1
fi

if [ ! -f "${FME_CHECKPOINT_PATH}" ]; then
  echo "Checkpoint not found: ${FME_CHECKPOINT_PATH}"
  exit 1
fi

# user should not need to modify below

UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
cp "${INFERENCE_CONFIG_PATH}" $CONFIG_DIR/config-inference.yaml
cp "${SCRIPT_DIR}/run-inference-interactive.sh" $CONFIG_DIR/run-inference-interactive.sh
cp "${SCRIPT_DIR}/sbatch-scripts/sbatch-inference.sh" $CONFIG_DIR/sbatch-inference.sh
cp "${SCRIPT_DIR}/make-venv.sh" $CONFIG_DIR/make-venv.sh

export FME_VENV=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)
conda activate $FME_VENV
python -m fme.ace.validate_config --config_type evaluator $CONFIG_DIR/config-inference.yaml

set -xe

if [ -z "${FME_OUTPUT_DIR}" ]; then
  export FME_RESUME_KEY=${FME_RESUME_KEY:-hpx-unet-eval-${TRAIN_JOB_ID}-${INFERENCE_CONFIG_NAME}-$(basename "${CKPT_NAME}" .tar)}
  export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${FME_RESUME_KEY}
fi
mkdir -p $FME_OUTPUT_DIR

echo "FME_OUTPUT_DIR=${FME_OUTPUT_DIR}"
echo "FME_CHECKPOINT_PATH=${FME_CHECKPOINT_PATH}"

srun -u $CONFIG_DIR/sbatch-inference.sh
