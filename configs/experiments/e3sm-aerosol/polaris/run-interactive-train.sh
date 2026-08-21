#!/bin/bash
# OPTIONAL: train on an existing interactive/hold allocation.
# Default workflow uses MODE=debug ./polaris/run-train.sh instead.
#
# Stage config, then launch train.sh with 1-epoch / WandB-off settings.
# --verify-only and --dry-run work on login nodes too.
#
# Examples:
#   CONFIG_FILE=config-train-PI-PD-1945-1980-aerosol-clim-forcing.yaml ./polaris/run-interactive-train.sh
#   CONFIG_FILE=config-train-....yaml ./polaris/run-interactive-train.sh --dry-run
#   CONFIG_FILE=config-train-....yaml ./polaris/run-interactive-train.sh --verify-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXPERIMENT_DIR"

export MODE=interactive

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
elif ! command -v nvidia-smi &>/dev/null || ! nvidia-smi -L &>/dev/null; then
    echo "No GPU visible on this host." >&2
    echo "Start an interactive session: ./polaris/interactive-session.sh" >&2
    echo "Or use --dry-run / --verify-only on a login node." >&2
    exit 1
fi

RUN_TAG="$(date +%Y%m%d-%H%M%S)-$(hostname -s)"
export FME_OUTPUT_DIR="${FME_OUTPUT_ROOT}/interactive-${RUN_TAG}"
mkdir -p "$FME_OUTPUT_DIR"

echo "Interactive train run"
echo "  ace root:       $ACE_ROOT"
echo "  python env:     $FME_VENV"
echo "  config staging: $CONFIG_DIR"
echo "  output:         $FME_OUTPUT_DIR"
echo ""

bash "$SCRIPT_DIR/train.sh"
