#!/bin/bash
# OPTIONAL: interactive GPU session (qsub -I). Not the default workflow.
#
# Polaris has no fast interactive queue — same wait as batch debug/debug-scaling.
# Prefer: MODE=debug ./polaris/run-train.sh → MODE=prod ./polaris/run-train.sh
#
# Use this only when you already want a live shell on a node after the queue wait.
# Agents / no-TTY: use interactive-hold.sh instead (also optional).
#
# Docs: https://docs.alcf.anl.gov/polaris/running-jobs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export TRAIN_NODES="${TRAIN_NODES:-1}"
export MODE=interactive

# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

WALLTIME="${PBS_WALLTIME}"
NODES="${TRAIN_NODES}"
QUEUE="${PBS_QUEUE}"

# qsub -I requires a real TTY. Prefer batch debug for smokes (same queues).
if ! tty -s; then
    echo "No TTY detected — qsub -I cannot attach from this shell." >&2
    echo "Preferred: MODE=debug TRAIN_NODES=${NODES} ./polaris/run-train.sh" >&2
    echo "Optional hold+SSH: TRAIN_NODES=${NODES} ./polaris/interactive-hold.sh" >&2
    exit 1
fi

echo "Requesting optional interactive session (same queue wait as batch):"
echo "  queue=$QUEUE nodes=$NODES walltime=$WALLTIME account=$PBS_ACCOUNT"
echo "  filesystems=$PBS_FILESYSTEMS"
echo "  Prefer batch smokes: MODE=debug TRAIN_NODES=${NODES} ./polaris/run-train.sh"
echo ""
echo "After the shell starts on a compute node, run:"
echo "  cd $EXPERIMENT_DIR"
echo "  CONFIG_FILE=config-train-....yaml ./polaris/run-interactive-train.sh"
echo ""

exec qsub -I \
    -A "$PBS_ACCOUNT" \
    -q "$QUEUE" \
    -l "select=${NODES}:system=polaris" \
    -l "walltime=${WALLTIME}" \
    -l "filesystems=${PBS_FILESYSTEMS}" \
    -l place=scatter
