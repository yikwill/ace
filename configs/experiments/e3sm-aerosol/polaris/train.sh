#!/bin/bash
# Shared training entrypoint — used by interactive, debug, and prod PBS jobs.
#
# Single-node: torchrun on the current host.
# Multi-node: mpiexec (PALS) launches one node-train.sh per node; each starts
# torchrun with static rendezvous. Do not pass a custom --hostfile — that can
# make Hydra fall back to SSH between nodes (Permission denied without keys).
# Pattern follows ai2cm/ace ocean/polaris pbs-train.sh + requeueable-train.sh.
set -euo pipefail

: "${CONFIG_DIR:?CONFIG_DIR must be set}"
: "${FME_OUTPUT_DIR:?FME_OUTPUT_DIR must be set}"

# Cap BLAS/OpenMP threads so DataLoader worker forks do not exhaust process slots.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Reduce CUDA allocator fragmentation on 40GB A100s.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_CONFIG="${CONFIG_DIR}/train-config.yaml"

# Replace experiment_dir placeholder (idempotent if already absolute).
if grep -q 'FME_OUTPUT_DIR' "$TRAIN_CONFIG"; then
    sed -i "s|FME_OUTPUT_DIR|${FME_OUTPUT_DIR}|g" "$TRAIN_CONFIG"
fi

mkdir -p "$FME_OUTPUT_DIR"
cp -r "$CONFIG_DIR" "$FME_OUTPUT_DIR/job_config"

export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-training}"
export WANDB_NOTES="${WANDB_NOTES:-Polaris train, results: $FME_OUTPUT_DIR}"
if [[ "${WANDB_MODE:-}" != "disabled" && -f "${HOME}/.config/wandb/api" ]]; then
    set +x
    export WANDB_API_KEY
    WANDB_API_KEY="$(cat "${HOME}/.config/wandb/api")"
    set -x
fi

NPROC_PER_NODE="${TRAIN_GPUS_PER_NODE:-4}"
export MASTER_PORT="${TORCH_MASTER_PORT:-29507}"

if [[ -n "${PBS_NODEFILE:-}" && -s "${PBS_NODEFILE}" ]]; then
    # Preserve PBS order (do not sort): first line is the master for static rdzv.
    NNODES="$(wc -l < "$PBS_NODEFILE" | tr -d ' ')"
    export MASTER_ADDR
    MASTER_ADDR="$(head -n 1 "$PBS_NODEFILE")"
else
    NNODES="${TRAIN_NODES:-1}"
    export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
fi
export NNODES

NTOTRANKS=$((NNODES * NPROC_PER_NODE))

echo "MODE=${MODE:-?} NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE NTOTRANKS=$NTOTRANKS"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "FME_OUTPUT_DIR=$FME_OUTPUT_DIR"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"

cd "$ACE_ROOT"

if [[ "${FME_TRAIN_DRY_RUN:-}" == "1" ]]; then
    echo "DRY RUN: checking GPU + imports (no training)"
    "$CONFIG_DIR/ace-python.sh" - <<'PY'
import torch
import fme
print(f"fme ok, torch={torch.__version__}, gpus={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
    exit 0
fi

if [[ "$NNODES" -eq 1 ]]; then
    exec "$CONFIG_DIR/ace-python.sh" \
        -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --rdzv-id="${PBS_JOBID:-$$}" \
        -m fme.ace.train "$TRAIN_CONFIG"
fi

# PALS mpiexec: one process per node. Default hostfile is $PBS_NODEFILE.
# Explicit --env so remote ranks see ACE paths (login-node exports are not enough).
# Must use Cray PALS — pbs-train.pbs prepends $FME_VENV/bin, which shadows PATH
# with conda MPICH Hydra mpiexec (rejects --cpu-bind / cannot launch across nodes).
NODE_LAUNCH="${CONFIG_DIR}/node-train.sh"
if [[ ! -x "$NODE_LAUNCH" ]]; then
    chmod +x "$NODE_LAUNCH" 2>/dev/null || true
fi
MPIEXEC="${MPIEXEC:-/opt/cray/pals/default/bin/mpiexec}"
if [[ ! -x "$MPIEXEC" ]]; then
    echo "ERROR: PALS mpiexec not found at $MPIEXEC" >&2
    exit 1
fi

exec "$MPIEXEC" -n "$NNODES" --ppn 1 \
    --cpu-bind none \
    --env CONFIG_DIR="$CONFIG_DIR" \
    --env FME_OUTPUT_DIR="$FME_OUTPUT_DIR" \
    --env ACE_ROOT="$ACE_ROOT" \
    --env FME_VENV="$FME_VENV" \
    --env MASTER_ADDR="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    --env NNODES="$NNODES" \
    --env TRAIN_GPUS_PER_NODE="$NPROC_PER_NODE" \
    --env OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    --env MODE="${MODE:-}" \
    --env WANDB_MODE="${WANDB_MODE:-}" \
    --env WANDB_NAME="${WANDB_NAME:-}" \
    --env WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}" \
    --env WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-}" \
    --env PYTHONPATH="${ACE_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
    "$NODE_LAUNCH"
