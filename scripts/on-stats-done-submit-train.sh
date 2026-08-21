#!/bin/bash
# Run after full stats finish: patch config paths and submit preempt train.
set -euo pipefail

STATS_DIR="/pscratch/sd/y/yikwill/datasets/fme-radiation-data-stats/increasing-CO2"
CONFIG="/global/homes/y/yikwill/llnl-research/ace-exp-s2resnet/configs/experiments/s2resnet/config-train-radiation-increasing-co2.yaml"
LLNL="/global/homes/y/yikwill/llnl-research"

for f in centering.nc scaling-full-field.nc scaling-residual.nc; do
  test -f "${STATS_DIR}/${f}" || { echo "Missing ${STATS_DIR}/${f}"; exit 1; }
done

sed -i 's|fme-radiation-data-stats-sample|fme-radiation-data-stats|g' "${CONFIG}"

cd "${LLNL}"
MODE=preempt WALLTIME=24:00:00 TRAIN_NODES=4 \
  ACE_ROOT="${LLNL}/ace-exp-s2resnet" \
  CONFIG_FILE=configs/experiments/s2resnet/config-train-radiation-increasing-co2.yaml \
  ./perlmutter/submit_train.sh 2>&1 | tee "${LLNL}/slurm-out/on-stats-done-submit.log"
