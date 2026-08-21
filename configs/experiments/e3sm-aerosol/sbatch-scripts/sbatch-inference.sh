#!/bin/bash -l

#SBATCH -A e3sm
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -J infer-e3sm-aerosol
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 4:00:00
#SBATCH --output=/global/homes/y/yikwill/llnl-research/slurm-out/slurm-%j.out

set -xe

export FME_OUTPUT_DIR=${FME_OUTPUT_DIR:-/global/cfs/projectdirs/e3sm/yikwill/inference-output/${FME_RESUME_KEY}-${SLURM_JOB_ID}}
mkdir -p $FME_OUTPUT_DIR

module load python
conda activate $FME_VENV

export WANDB_JOB_TYPE=inference
export WANDB_NOTES="PM inference, checkpoint: $FME_CHECKPOINT_PATH, results: $FME_OUTPUT_DIR"
set +x
export WANDB_API_KEY=$(cat ~/.config/wandb/api)
set -x

INFERENCE_CONFIG=$CONFIG_DIR/inference-config.yaml

sed -i "s|FME_OUTPUT_DIR|${FME_OUTPUT_DIR}|" ${INFERENCE_CONFIG}
sed -i "s|FME_CHECKPOINT_PATH|${FME_CHECKPOINT_PATH}|" ${INFERENCE_CONFIG}

cp -r $CONFIG_DIR $FME_OUTPUT_DIR/job_config

echo "FME_OUTPUT_DIR=${FME_OUTPUT_DIR}"
echo "FME_CHECKPOINT_PATH=${FME_CHECKPOINT_PATH}"

python -u -m fme.ace.evaluator ${INFERENCE_CONFIG}
