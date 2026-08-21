#!/bin/bash

set -x

CONFIG_FILE=config-train-PD-1945-1980-aerosol-clim-forcing.yaml
CONFIG_SLUG=${CONFIG_FILE#config-train-}
CONFIG_SLUG=${CONFIG_SLUG%.yaml}

# wandb config
export WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-train-${CONFIG_SLUG}"
export WANDB_RUN_GROUP=E3SM-SFNO

export COMMIT=$(git rev-parse --short HEAD)

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
sbatch -J train-e3sm-aerosol-${CONFIG_SLUG} sbatch-scripts/sbatch-train.sh
