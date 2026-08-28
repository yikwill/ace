#!/bin/bash
# Submit a 1-node PBS job to print per-variable training loss for one batch.
#
# Uses the debug queue by default (1 node, 30 min walltime) for fast turnaround.
# Do not override PBS_QUEUE unless debug is unavailable; capacity has a
# per-project concurrency limit and preemptable can wait much longer.
#
# Usage:
#   CONFIG_FILE=config-train-PI-PD-1945-1980-aerosol-prognostic-emis-reduced.yaml \
#     ./polaris/diagnose-per-channel-loss.sh
#
#   CONFIG_FILE=... COMPARE_CONFIG=config-train-PI-PD-1945-1980-aerosol-clim-forcing.yaml \
#     ./polaris/diagnose-per-channel-loss.sh
#
# If debug slots are full (per-user running-job limit), wait for a slot or end
# other debug jobs before resubmitting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXPERIMENT_DIR"

export MODE="${MODE:-debug}"
export TRAIN_NODES=1
export TRAIN_GPUS_PER_NODE=1
DIAG_QUEUE="${PBS_QUEUE:-}"
DIAG_BATCH_SIZE="${DIAG_BATCH_SIZE:-1}"

# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

# Keep debug as the default queue; only honor an explicit PBS_QUEUE override.
if [[ -n "$DIAG_QUEUE" ]]; then
    export PBS_QUEUE="$DIAG_QUEUE"
fi

CONFIG_FILE="${CONFIG_FILE:?Set CONFIG_FILE=config-train-....yaml}"
COMPARE_CONFIG="${COMPARE_CONFIG:-}"

if [[ ! -f "$EXPERIMENT_DIR/$CONFIG_FILE" ]]; then
    echo "Config not found: $EXPERIMENT_DIR/$CONFIG_FILE" >&2
    exit 1
fi

DIAG_UUID="$(uuidgen)"
export CONFIG_DIR="${FME_CONFIG_ROOT}/diagnose-loss-${DIAG_UUID}"
mkdir -p "$CONFIG_DIR"

STAGED_CONFIG="$CONFIG_DIR/train-config.yaml"
"$SCRIPT_DIR/ace-python.sh" "$SCRIPT_DIR/patch-config.py" \
    "$EXPERIMENT_DIR/$CONFIG_FILE" "$STAGED_CONFIG"

cp "$SCRIPT_DIR/env.sh" "$CONFIG_DIR/env.sh"
cp "$SCRIPT_DIR/ace-python.sh" "$CONFIG_DIR/ace-python.sh"
echo "$ACE_ROOT" > "$CONFIG_DIR/ace_root.txt"
echo "$FME_VENV" > "$CONFIG_DIR/fme_venv.txt"

COMPARE_CLI=""
if [[ -n "$COMPARE_CONFIG" ]]; then
    if [[ ! -f "$EXPERIMENT_DIR/$COMPARE_CONFIG" ]]; then
        echo "Compare config not found: $EXPERIMENT_DIR/$COMPARE_CONFIG" >&2
        exit 1
    fi
    STAGED_COMPARE="$CONFIG_DIR/compare-config.yaml"
    "$SCRIPT_DIR/ace-python.sh" "$SCRIPT_DIR/patch-config.py" \
        "$EXPERIMENT_DIR/$COMPARE_CONFIG" "$STAGED_COMPARE"
    COMPARE_CLI="--compare ${STAGED_COMPARE}"
fi

JOB_ENV="$CONFIG_DIR/job.env"
cat > "$JOB_ENV" <<EOF
export CONFIG_DIR="$CONFIG_DIR"
export ACE_ROOT="$ACE_ROOT"
export FME_VENV="$FME_VENV"
export FME_DATA_ROOT="$FME_DATA_ROOT"
export FME_OUTPUT_ROOT="$FME_OUTPUT_ROOT"
export FME_CONFIG_ROOT="$FME_CONFIG_ROOT"
export PBS_LOG_ROOT="$PBS_LOG_ROOT"
export MODE="$MODE"
export TRAIN_NODES=1
export TRAIN_GPUS_PER_NODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
EOF

SHORT_SLUG="${CONFIG_FILE#config-train-}"
SHORT_SLUG="${SHORT_SLUG%.yaml}"
LOG_FILE="${PBS_LOG_ROOT}/diagnose-loss-${SHORT_SLUG}-${DIAG_UUID:0:8}.out"
mkdir -p "$PBS_LOG_ROOT"
PBS_QUEUE="${PBS_QUEUE:-debug}"

PBS_SCRIPT="$CONFIG_DIR/diagnose-loss.pbs"
cat > "$PBS_SCRIPT" <<EOF
#!/bin/bash -l
#PBS -N ace-loss-diag
#PBS -A ${PBS_ACCOUNT}
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q ${PBS_QUEUE}
#PBS -j oe
#PBS -o ${LOG_FILE}

set -euo pipefail
source "${JOB_ENV}"
export PATH="\${FME_VENV}/bin:\$PATH"
export PYTHONPATH="\${ACE_ROOT}:\${PYTHONPATH:-}"

echo "=== per-channel loss diagnostic \$(date) on \$(hostname) ==="
echo "CONFIG=${STAGED_CONFIG}"
nvidia-smi -L || true

"\$CONFIG_DIR/ace-python.sh" "\$ACE_ROOT/scripts/diagnose_per_channel_loss.py" \\
    --batch-size ${DIAG_BATCH_SIZE} \\
    "${STAGED_CONFIG}" ${COMPARE_CLI}

echo "=== finished \$(date) ==="
EOF

chmod +x "$PBS_SCRIPT"
JOB_ID="$(qsub "$PBS_SCRIPT")"
echo "Submitted $JOB_ID"
echo "Log: $LOG_FILE"
echo "Tail with: tail -f $LOG_FILE"
