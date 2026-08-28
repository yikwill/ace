#!/bin/bash
# Submit ACE production (or batch-debug) training on Polaris via PBS.
#
# Recommended workflow (batch debug → prod; no interactive required):
#   CONFIG_FILE=config-train-....yaml MODE=debug TRAIN_NODES=2 ./polaris/run-train.sh
#   CONFIG_FILE=config-train-....yaml MODE=prod TRAIN_NODES=4 ./polaris/run-train.sh
#
# One shared conda env; ACE code comes from the worktree via ACE_ROOT/PYTHONPATH.
# Edit configs in this experiment directory, then re-run — staging picks up changes.
#
# Examples:
#   CONFIG_FILE=config-train-PI-PD-1945-1980-aerosol-clim-forcing.yaml MODE=prod ./polaris/run-train.sh
#   CONFIG_FILE=config-train-....yaml MODE=debug ./polaris/run-train.sh          # unattended batch debug
#   CONFIG_FILE=config-train-....yaml ./polaris/run-train.sh --verify-only
#
# Different experiment worktree:
#   ACE_ROOT=/path/to/ace-exp-other ./polaris/run-train.sh
#
# Optional interactive/hold scripts exist but are not the default path (same
# queue wait as batch; see interactive-session.sh / interactive-hold.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXPERIMENT_DIR"

# Batch submission defaults to prod; use MODE=debug for unattended smoke tests.
export MODE="${MODE:-prod}"

# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

VERIFY_ONLY=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify-only) VERIFY_ONLY=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

CONFIG_FILE="${CONFIG_FILE:-config-train-PI-PD-1945-1980-aerosol-clim-forcing.yaml}"
export CONFIG_FILE

# shellcheck source=stage-train-config.sh
source "$SCRIPT_DIR/stage-train-config.sh"

if $VERIFY_ONLY; then
    echo "Running one-step zarr/train verification..."
    "$SCRIPT_DIR/ace-python.sh" "$ACE_ROOT/scripts/data_process/verify_zarr_training.py" \
        "$CONFIG_DIR/train-config.yaml"
    echo "verify-only OK"
    exit 0
fi

if $DRY_RUN; then
    export FME_TRAIN_DRY_RUN=1
    export FME_DEBUG_MAX_EPOCHS=1
fi

PBS_TEMPLATE="$SCRIPT_DIR/pbs-train.pbs"
PBS_SCRIPT="${CONFIG_DIR}/pbs-train.pbs"
sed \
    -e "s/NUMNODES/${TRAIN_NODES}/g" \
    -e "s/WALLTIME/${PBS_WALLTIME}/g" \
    -e "s/QUEUE/${PBS_QUEUE}/g" \
    -e "s/E3SMinput/${PBS_ACCOUNT}/g" \
    -e "s|@CONFIG_DIR@|${CONFIG_DIR}|g" \
    "$PBS_TEMPLATE" > "$PBS_SCRIPT"

# Mail directives: keep when PBS_MAIL is set, otherwise strip placeholder lines
# so qsub is not given an empty -M.
if [[ -n "${PBS_MAIL:-}" && "${PBS_MAIL_EVENTS:-n}" != "n" ]]; then
    sed -i \
        -e "s|@PBS_MAIL@|${PBS_MAIL}|g" \
        -e "s|@PBS_MAIL_EVENTS@|${PBS_MAIL_EVENTS}|g" \
        "$PBS_SCRIPT"
else
    sed -i -e '/@PBS_MAIL@/d' -e '/@PBS_MAIL_EVENTS@/d' "$PBS_SCRIPT"
fi

JOB_NAME="train-${CONFIG_SLUG}-${MODE}"
STAGE_UUID="$(basename "$CONFIG_DIR")"
mkdir -p "$PBS_LOG_ROOT"
# Unique pre-submit path (PBS has no Slurm-style %j). Symlink pbs-<jobid>.out after qsub.
PBS_LOG_FILE="${PBS_LOG_ROOT}/${JOB_NAME}-${STAGE_UUID}.out"
touch "$PBS_LOG_FILE"

ENV_FILE="${CONFIG_DIR}/job.env"
cat > "$ENV_FILE" <<EOF
export CONFIG_DIR='${CONFIG_DIR}'
export ACE_ROOT='${ACE_ROOT}'
export MODE='${MODE}'
export TRAIN_NODES='${TRAIN_NODES}'
export TRAIN_GPUS_PER_NODE='${TRAIN_GPUS_PER_NODE}'
export FME_VENV='${FME_VENV}'
export FME_OUTPUT_ROOT='${FME_OUTPUT_ROOT}'
export FME_LOG_ROOT='${FME_LOG_ROOT}'
export PBS_LOG_ROOT='${PBS_LOG_ROOT}'
export PBS_LOG_FILE='${PBS_LOG_FILE}'
export TORCH_MASTER_PORT='${TORCH_MASTER_PORT}'
export WANDB_NAME='${WANDB_NAME}'
export WANDB_RUN_GROUP='${WANDB_RUN_GROUP}'
export WANDB_MODE='${WANDB_MODE:-}'
export FME_TRAIN_DRY_RUN='${FME_TRAIN_DRY_RUN:-}'
export RESUME_JOB_ID='${RESUME_JOB_ID:-}'
EOF

# Patch staged PBS script placeholders (NUMNODES/etc. already applied above).
sed -i \
    -e "s|@PBS_LOG_FILE@|${PBS_LOG_FILE}|g" \
    "$PBS_SCRIPT"

# Pointer in staging dir for anyone still looking at Eagle config UUID.
printf 'PBS stdout: %s\n' "$PBS_LOG_FILE" > "${CONFIG_DIR}/pbs.log"

QSUB_ARGS=(-N "$JOB_NAME" -o "$PBS_LOG_FILE" -e "$PBS_LOG_FILE")
if [[ -n "${PBS_MAIL:-}" && "${PBS_MAIL_EVENTS:-n}" != "n" ]]; then
    QSUB_ARGS+=(-M "$PBS_MAIL" -m "$PBS_MAIL_EVENTS")
    echo "Submitting $JOB_NAME (queue=$PBS_QUEUE nodes=$TRAIN_NODES gpus/node=$TRAIN_GPUS_PER_NODE mail=$PBS_MAIL events=$PBS_MAIL_EVENTS)"
else
    echo "Submitting $JOB_NAME (queue=$PBS_QUEUE nodes=$TRAIN_NODES gpus/node=$TRAIN_GPUS_PER_NODE)"
fi
JOB_ID=$(qsub "${QSUB_ARGS[@]}" "$PBS_SCRIPT")
SHORT_ID="${JOB_ID%%.*}"
ln -sfn "$(basename "$PBS_LOG_FILE")" "${PBS_LOG_ROOT}/pbs-${SHORT_ID}.out"

echo "Submitted $JOB_ID"
echo "  ace root:       $ACE_ROOT"
echo "  python env:     $FME_VENV"
echo "  config staging: $CONFIG_DIR"
echo "  logs:           ${PBS_LOG_ROOT}/pbs-${SHORT_ID}.out"
echo "  logs (full):    $PBS_LOG_FILE"
echo "  output:         ${FME_OUTPUT_ROOT}/${JOB_ID}/"
echo ""
echo "Monitor:  qstat -u \$USER"
echo "Logs:     tail -f ${PBS_LOG_ROOT}/pbs-${SHORT_ID}.out"
