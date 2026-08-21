#!/bin/bash -l

#SBATCH -A e3sm
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -J train-e3sm-aerosol
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 48:00:00
#SBATCH --output=joblogs/%j.out
#SBATCH --signal=USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/global/homes/y/yikwill/llnl-research/slurm-out/slurm-%j.out
#SBATCH --mail-user=yikwill@uw.edu
#SBATCH --mail-type=ALL

# for pre-emptible jobs, update -q to preempt

set -xe

# directory for saving output from training/inference job
if [ -z "${RESUME_JOB_ID}" ]; then
  export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${SLURM_JOB_ID}
else
  export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${RESUME_JOB_ID}
fi
mkdir -p $FME_OUTPUT_DIR

module load python
conda activate $FME_VENV

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

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29507

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
# run the requeueable job
srun -u $CONFIG_DIR/requeueable-train.sh

sleep 120
