#!/bin/bash
# OPTIONAL: attach to an interactive-hold session (SSH to head node).
# Default workflow uses MODE=debug ./polaris/run-train.sh instead.
#
# Examples:
#   CONFIG_FILE=config-train-PI-PD-1945-1980-aerosol-clim-forcing.yaml \
#     ./polaris/interactive-attach.sh
#   ./polaris/interactive-attach.sh <session-id> nvidia-smi -L

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODE=interactive
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

SESSION_ROOT="${FME_CONFIG_ROOT}/interactive-sessions"
WAIT_SECONDS="${WAIT_SECONDS:-3600}"
POLL_SECONDS="${POLL_SECONDS:-15}"

SESSION_ID=""
REMOTE_CMD=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait-seconds)
            WAIT_SECONDS="$2"
            shift 2
            ;;
        --help|-h)
            sed -n '2,16p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -z "$SESSION_ID" && -d "${SESSION_ROOT}/$1" ]]; then
                SESSION_ID="$1"
                shift
            else
                REMOTE_CMD=("$@")
                break
            fi
            ;;
    esac
done

if [[ -z "$SESSION_ID" ]]; then
    if [[ -f "${SESSION_ROOT}/latest" ]]; then
        SESSION_ID="$(cat "${SESSION_ROOT}/latest")"
    else
        echo "No session id given and no ${SESSION_ROOT}/latest" >&2
        echo "Start one with: TRAIN_NODES=N ./polaris/interactive-hold.sh" >&2
        exit 1
    fi
fi

SESSION_DIR="${SESSION_ROOT}/${SESSION_ID}"
if [[ ! -d "$SESSION_DIR" ]]; then
    echo "Session not found: $SESSION_DIR" >&2
    exit 1
fi

# Prefer session.info (written on the compute node); fall back to submit.info.
_load_info() {
    if [[ -f "${SESSION_DIR}/session.info" ]]; then
        # shellcheck source=/dev/null
        source "${SESSION_DIR}/session.info"
    elif [[ -f "${SESSION_DIR}/submit.info" ]]; then
        # shellcheck source=/dev/null
        source "${SESSION_DIR}/submit.info"
    else
        echo "Missing session metadata under $SESSION_DIR" >&2
        exit 1
    fi
}

_load_info

echo "Waiting for hold session $SESSION_ID (job=${JOB_ID:-?}, timeout=${WAIT_SECONDS}s)..."
deadline=$((SECONDS + WAIT_SECONDS))
while true; do
    if [[ -f "${SESSION_DIR}/ready" && -f "${SESSION_DIR}/session.info" ]]; then
        break
    fi
    if [[ -n "${JOB_ID:-}" ]]; then
        state="$(qstat -f "$JOB_ID" 2>/dev/null | awk -F'= ' '/job_state/{print $2; exit}' || true)"
        case "$state" in
            F|E)
                echo "Hold job $JOB_ID ended before ready (state=$state)." >&2
                [[ -f "${SESSION_DIR}/hold.log" ]] && tail -40 "${SESSION_DIR}/hold.log" >&2
                exit 1
                ;;
            "")
                # Job may have left the queue without writing ready.
                if [[ ! -f "${SESSION_DIR}/ready" ]] && (( SECONDS > deadline - WAIT_SECONDS + 30 )); then
                    if ! qstat "$JOB_ID" &>/dev/null; then
                        echo "Hold job $JOB_ID no longer in queue and session not ready." >&2
                        [[ -f "${SESSION_DIR}/hold.log" ]] && tail -40 "${SESSION_DIR}/hold.log" >&2
                        exit 1
                    fi
                fi
                ;;
        esac
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for session $SESSION_ID to become ready." >&2
        exit 1
    fi
    sleep "$POLL_SECONDS"
done

# shellcheck source=/dev/null
source "${SESSION_DIR}/session.info"

: "${HEAD_HOST:?HEAD_HOST missing in session.info}"
: "${PBS_NODEFILE_COPY:?PBS_NODEFILE_COPY missing in session.info}"
: "${EXPERIMENT_DIR:?EXPERIMENT_DIR missing in session.info}"

if [[ ${#REMOTE_CMD[@]} -eq 0 ]]; then
    if [[ -z "${CONFIG_FILE:-}" ]]; then
        echo "Session $SESSION_ID ready on $HEAD_HOST" >&2
        echo "Pass a remote command, or set CONFIG_FILE=... to run interactive train:" >&2
        echo "  CONFIG_FILE=config-train-....yaml ./polaris/interactive-attach.sh $SESSION_ID" >&2
        echo "  ./polaris/interactive-attach.sh $SESSION_ID nvidia-smi -L" >&2
        exit 1
    fi
    REMOTE_CMD=(./polaris/run-interactive-train.sh)
fi

# Quote remote argv for a login shell on the compute node. Export PBS_NODEFILE /
# TRAIN_NODES / PBS_JOBID so multi-node train.sh sees the full allocation.
remote_exports=(
    "export PBS_NODEFILE='${PBS_NODEFILE_COPY}'"
    "export TRAIN_NODES='${TRAIN_NODES}'"
    "export PBS_JOBID='${JOB_ID}'"
    "export ACE_ROOT='${ACE_ROOT}'"
    "export MODE=interactive"
)
if [[ -n "${CONFIG_FILE:-}" ]]; then
    remote_exports+=("export CONFIG_FILE='${CONFIG_FILE}'")
fi

remote_cmd_str=""
for arg in "${REMOTE_CMD[@]}"; do
    remote_cmd_str+=" $(printf '%q' "$arg")"
done

remote_script="$(
    printf '%s\n' "${remote_exports[@]}"
    printf 'cd %q\n' "$EXPERIMENT_DIR"
    printf '%s\n' "$remote_cmd_str"
)"

echo "Attaching to $HEAD_HOST (session=$SESSION_ID job=$JOB_ID)"
echo "  remote: ${REMOTE_CMD[*]}"
echo ""

# BatchMode fails fast if SSH keys / home perms are wrong (ALCF: $HOME and
# ~/.ssh should be 700). See https://docs.alcf.anl.gov/polaris/known-issues/
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$HEAD_HOST" \
    "bash -l -c $(printf '%q' "$remote_script")"
