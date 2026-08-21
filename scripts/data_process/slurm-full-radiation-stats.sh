#!/bin/bash
#SBATCH -A e3sm
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -J radiation-stats-full
#SBATCH -o /global/homes/y/yikwill/llnl-research/slurm-out/radiation-stats-full-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yikwill@uw.edu

set -euo pipefail

export PATH="/pscratch/sd/y/yikwill/mamba/envs/fme-s2unet/bin:${PATH}"
cd /global/homes/y/yikwill/llnl-research/ace-exp-s2resnet/scripts/data_process

echo "Starting full radiation stats at $(date)"
python -u get_stats.py configs/shield-som-increasing-co2-radiation-stats-full.yaml 0 --force
echo "Finished full radiation stats at $(date)"
