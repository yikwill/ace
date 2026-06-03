#!/bin/bash -l

#SBATCH -A e3sm
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -J infer-hpx-unet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 04:00:00
#SBATCH --output=/global/homes/y/yikwill/llnl-research/slurm-out/slurm-%j.out

set -xe

# batch jobs: new dir per Slurm job ID; interactive: launcher sets FME_OUTPUT_DIR first
export FME_OUTPUT_DIR=${FME_OUTPUT_DIR:-${PSCRATCH}/fme-output/${SLURM_JOB_ID}}
mkdir -p $FME_OUTPUT_DIR

module load python
conda activate $FME_VENV

export WANDB_JOB_TYPE=inference
set +x
export WANDB_API_KEY=$(cat ~/.config/wandb/api)
set -x

INFERENCE_CONFIG=$CONFIG_DIR/config-inference.yaml

sed -i "s|FME_OUTPUT_DIR|${FME_OUTPUT_DIR}|" ${INFERENCE_CONFIG}
sed -i "s|FME_CHECKPOINT_PATH|${FME_CHECKPOINT_PATH}|" ${INFERENCE_CONFIG}
sed -i "s|FME_TRAIN_DIR|${FME_TRAIN_DIR}|" ${INFERENCE_CONFIG}
sed -i "s|FME_STATS_DIR|${FME_STATS_DIR}|" ${INFERENCE_CONFIG}

cp -r $CONFIG_DIR $FME_OUTPUT_DIR/job_config

export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29507}

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES} SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE}"

torchrun --nnodes ${SLURM_JOB_NUM_NODES:-1} \
  --nproc_per_node ${SLURM_GPUS_PER_NODE:-1} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m fme.ace.evaluator $INFERENCE_CONFIG
