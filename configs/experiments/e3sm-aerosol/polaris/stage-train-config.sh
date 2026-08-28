#!/bin/bash
# Stage a train config for Polaris and validate it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

CONFIG_FILE="${CONFIG_FILE:?Set CONFIG_FILE=config-train-....yaml}"
CONFIG_SLUG="${CONFIG_FILE#config-train-}"
CONFIG_SLUG="${CONFIG_SLUG%.yaml}"

if [[ ! -f "$EXPERIMENT_DIR/$CONFIG_FILE" ]]; then
    echo "Config not found: $EXPERIMENT_DIR/$CONFIG_FILE" >&2
    exit 1
fi

UUID="${UUID:-$(uuidgen)}"
export CONFIG_DIR="${FME_CONFIG_ROOT}/${UUID}"
mkdir -p "$CONFIG_DIR"

STAGED_CONFIG="$CONFIG_DIR/train-config.yaml"
PATCH_ARGS=("$EXPERIMENT_DIR/$CONFIG_FILE" "$STAGED_CONFIG")
if [[ -n "${FME_DEBUG_MAX_EPOCHS:-}" ]]; then
    PATCH_ARGS+=(--debug-max-epochs "$FME_DEBUG_MAX_EPOCHS")
fi
"$SCRIPT_DIR/ace-python.sh" "$SCRIPT_DIR/patch-config.py" "${PATCH_ARGS[@]}"

cp "$SCRIPT_DIR/env.sh" "$CONFIG_DIR/env.sh"
cp "$SCRIPT_DIR/train.sh" "$CONFIG_DIR/train.sh"
cp "$SCRIPT_DIR/node-train.sh" "$CONFIG_DIR/node-train.sh"
cp "$SCRIPT_DIR/ace-python.sh" "$CONFIG_DIR/ace-python.sh"
cp "$SCRIPT_DIR/patch-config.py" "$CONFIG_DIR/patch-config.py"
cp "$SCRIPT_DIR/../$CONFIG_FILE" "$CONFIG_DIR/source-config.yaml" 2>/dev/null || true
chmod +x "$CONFIG_DIR/train.sh" "$CONFIG_DIR/node-train.sh" "$CONFIG_DIR/ace-python.sh"
echo "$ACE_ROOT" > "$CONFIG_DIR/ace_root.txt"
echo "$FME_VENV" > "$CONFIG_DIR/fme_venv.txt"

export WANDB_NAME="${WANDB_NAME:-$(date +%Y%m%d)-POLARIS-E3SM-train-${CONFIG_SLUG}-${MODE}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-E3SM-SFNO}"
export WANDB_JOB_TYPE=training

"$SCRIPT_DIR/ace-python.sh" -m fme.ace.validate_config --config_type train "$STAGED_CONFIG"
echo "Staged config: $STAGED_CONFIG"
