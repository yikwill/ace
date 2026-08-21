#!/bin/bash

set -x

CONFIG_FILE=config-train-PD-1945-1980-aerosol-clim-forcing.yaml
CONFIG_SLUG=${CONFIG_FILE#config-train-}
CONFIG_SLUG=${CONFIG_SLUG%.yaml}

# wandb config
export WANDB_NAME="SMOKE-TEST-$(date +%Y%m%d)-PM-E3SM-SFNO-train-${CONFIG_SLUG}"
export WANDB_RUN_GROUP=E3SM-SFNO
# stable run key for interactive restarts; change if you want fresh run
export FME_RESUME_KEY=${FME_RESUME_KEY:-SMOKE-TEST-$(date +%Y%m%d)-e3sm-aerosol-${CONFIG_SLUG}}

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
cp $CONFIG_FILE $CONFIG_DIR/train-config.yaml
cp run-train-perlmutter.sh $CONFIG_DIR/run-train-perlmutter.sh  # copy for reproducibility/tracking
cp sbatch-scripts/requeueable-train.sh $CONFIG_DIR/requeueable-train.sh
cp make-venv.sh $CONFIG_DIR/make-venv.sh
cp upload-to-beaker.sh $CONFIG_DIR/upload-to-beaker.sh

python -m fme.ace.validate_config --config_type train $CONFIG_DIR/train-config.yaml

set -xe

# directory for saving output from training/inference job
# interactive runs should reuse same directory so checkpoints are found on restart
if [ -z "${RESUME_JOB_ID}" ]; then
  export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${FME_RESUME_KEY}
else
  export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${RESUME_JOB_ID}
fi
mkdir -p $FME_OUTPUT_DIR

# env variables
export WANDB_JOB_TYPE=training
export WANDB_NOTES="PM: $FME_IMAGE, results: $FME_OUTPUT_DIR"
set +x  # don't print API key to logs
export WANDB_API_KEY=$(cat ~/.config/wandb/api)
set -x

TRAIN_CONFIG=${CONFIG_DIR}/train-config.yaml

# replace placeholders in config with actual values
sed -i "s|FME_OUTPUT_DIR|${FME_OUTPUT_DIR}|" ${TRAIN_CONFIG}

cp -r $CONFIG_DIR $FME_OUTPUT_DIR/job_config

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST:-}" 2>/dev/null | head -1)
if [ -z "${MASTER_ADDR}" ]; then
  export MASTER_ADDR=$(hostname)
fi
export MASTER_PORT=29507

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

srun -u $CONFIG_DIR/requeueable-train.sh
