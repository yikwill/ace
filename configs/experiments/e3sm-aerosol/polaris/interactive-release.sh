#!/bin/bash
# OPTIONAL: release an interactive-hold session (qdel).
#
# Usage:
#   ./polaris/interactive-release.sh
#   ./polaris/interactive-release.sh <session-id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODE=interactive
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

SESSION_ROOT="${FME_CONFIG_ROOT}/interactive-sessions"
SESSION_ID="${1:-}"

if [[ -z "$SESSION_ID" ]]; then
    if [[ -f "${SESSION_ROOT}/latest" ]]; then
        SESSION_ID="$(cat "${SESSION_ROOT}/latest")"
    else
        echo "No session id given and no ${SESSION_ROOT}/latest" >&2
        exit 1
    fi
fi

SESSION_DIR="${SESSION_ROOT}/${SESSION_ID}"
if [[ ! -d "$SESSION_DIR" ]]; then
    echo "Session not found: $SESSION_DIR" >&2
    exit 1
fi

JOB_ID=""
if [[ -f "${SESSION_DIR}/session.info" ]]; then
    # shellcheck source=/dev/null
    source "${SESSION_DIR}/session.info"
elif [[ -f "${SESSION_DIR}/submit.info" ]]; then
    # shellcheck source=/dev/null
    source "${SESSION_DIR}/submit.info"
fi

if [[ -z "${JOB_ID:-}" ]]; then
    echo "No JOB_ID in session metadata for $SESSION_ID" >&2
    exit 1
fi

if qstat "$JOB_ID" &>/dev/null; then
    echo "Releasing session $SESSION_ID (qdel $JOB_ID)"
    qdel "$JOB_ID"
else
    echo "Job $JOB_ID already gone; marking session released."
fi

touch "${SESSION_DIR}/released"
date -u +%Y-%m-%dT%H:%M:%SZ > "${SESSION_DIR}/released"
echo "Released $SESSION_ID"
