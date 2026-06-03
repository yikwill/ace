#!/bin/bash

set -x

# wandb config
export WANDB_NAME=PM-AIMIP-HPXUNET-DLESYM-RECIPE-train
export WANDB_RUN_GROUP=2025-05-14-AIMIP-HPXUNET-DLESYM-RECIPE

export COMMIT=$(git rev-parse --short HEAD)

# directories for input data (training, validation, inference, stats)
export FME_TRAIN_DIR=/pscratch/sd/e/elynnwu/fme-dataset
export FME_VALID_DIR=/pscratch/sd/e/elynnwu/fme-dataset
export FME_STATS_DIR=/pscratch/sd/y/yikwill/datasets/ace/2025-10-04-healpix-era5-dataset

CONFIG_FILE=config-train-dlesym-recipe.yaml

# if resuming a failed job, provide its slurm job ID below and uncomment;
# note that information entered above should be consistent with that of
# the failed job
# export RESUME_JOB_ID=12345678

# user should not need to modify below

# copy config to staging area so that local changes between job submission
# and job start will not effect the run
UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
if [ -z "${RESUME_JOB_ID}" ]; then
  cp $CONFIG_FILE $CONFIG_DIR/train-config.yaml
else
  cp ${PSCRATCH}/fme-output/${RESUME_JOB_ID}/job_config/train-config.yaml $CONFIG_DIR/train-config.yaml
fi
cp run-train-perlmutter.sh $CONFIG_DIR/run-train-perlmutter.sh  # copy for reproducibility/tracking
cp sbatch-scripts/requeueable-train.sh $CONFIG_DIR/requeueable-train.sh
cp make-venv.sh $CONFIG_DIR/make-venv.sh
cp upload-to-beaker.sh $CONFIG_DIR/upload-to-beaker.sh

export FME_VENV=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)
conda activate $FME_VENV
python -m fme.ace.validate_config --config_type train $CONFIG_DIR/train-config.yaml
# sbatch -t 00:30:00 -q debug sbatch-scripts/sbatch-train.sh  # use this for debugging config/submission
sbatch sbatch-scripts/sbatch-train.sh
