#!/bin/bash

set -x

# wandb config (set WANDB_NAME per checkpoint below)
export WANDB_RUN_GROUP=2025-05-06-AIMIP-HPXUNET

# from job 53035680: /pscratch/sd/y/yikwill/fme-conda-envs/28d5b5dce (see slurm-out/slurm-53035680.out)
export COMMIT=28d5b5dce

# training job that produced checkpoints
export TRAIN_JOB_ID=53035680
export FME_CHECKPOINT_DIR=${PSCRATCH}/fme-output/${TRAIN_JOB_ID}/training_checkpoints

# checkpoint to evaluate: best_inference_ckpt.tar | best_ckpt.tar | ckpt.tar
export CKPT_NAME=${CKPT_NAME:-best_ckpt.tar}
export FME_CHECKPOINT_PATH=${FME_CHECKPOINT_DIR}/${CKPT_NAME}

# same data paths as run-train-perlmutter.sh (inference loader uses FME_TRAIN_DIR)
export FME_TRAIN_DIR=/pscratch/sd/e/elynnwu/fme-dataset
export FME_STATS_DIR=/pscratch/sd/y/yikwill/datasets/ace/2025-10-04-healpix-era5-dataset

export WANDB_NAME=PM-AIMIP-HPXUNET-eval-${TRAIN_JOB_ID}-$(basename "${CKPT_NAME}" .tar)
export WANDB_NOTES="eval train job ${TRAIN_JOB_ID}, checkpoint ${CKPT_NAME}"

if [ ! -f "${FME_CHECKPOINT_PATH}" ]; then
  echo "Checkpoint not found: ${FME_CHECKPOINT_PATH}"
  exit 1
fi

# user should not need to modify below

UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
cp config-inference-prescribed-q0.yaml $CONFIG_DIR/config-inference.yaml
cp run-inference-perlmutter.sh $CONFIG_DIR/run-inference-perlmutter.sh
cp sbatch-scripts/sbatch-inference.sh $CONFIG_DIR/sbatch-inference.sh
cp make-venv.sh $CONFIG_DIR/make-venv.sh

export FME_VENV=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)
conda activate $FME_VENV
python -m fme.ace.validate_config --config_type evaluator $CONFIG_DIR/config-inference.yaml

# submit one job for CKPT_NAME; for all three checkpoints:
#   for c in best_inference_ckpt.tar best_ckpt.tar ckpt.tar; do CKPT_NAME=$c ./run-inference-perlmutter.sh; done
# sbatch -t 00:30:00 -q debug sbatch-scripts/sbatch-inference.sh
sbatch sbatch-scripts/sbatch-inference.sh
