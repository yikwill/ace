#!/bin/bash
# Per-node launcher for multi-node Polaris training (invoked by mpiexec --ppn 1).
#
# Uses static torchrun rendezvous (node-rank from PALS/PMI) instead of dynamic
# c10d rdzv, matching the ocean Polaris pattern in ai2cm/ace exp/e3sm.
# PALS launches this process on each allocated node — no SSH between nodes.
set -euo pipefail

: "${CONFIG_DIR:?CONFIG_DIR must be set}"
: "${FME_OUTPUT_DIR:?FME_OUTPUT_DIR must be set}"
: "${ACE_ROOT:?ACE_ROOT must be set}"
: "${FME_VENV:?FME_VENV must be set}"
: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:?MASTER_PORT must be set}"
: "${NNODES:?NNODES must be set}"

NPROC_PER_NODE="${TRAIN_GPUS_PER_NODE:-4}"
# With mpiexec --ppn 1, PMI_RANK / PALS_RANKID is the node index.
NODE_RANK="${PMI_RANK:-${PALS_RANKID:-0}}"

export PYTHONPATH="${ACE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_CONFIG="${CONFIG_DIR}/train-config.yaml"

echo "node-train: host=$(hostname) NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT} NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE}"

cd "$ACE_ROOT"

exec "$FME_VENV/bin/python" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    -m fme.ace.train "$TRAIN_CONFIG"
