#!/bin/bash
# Shared Polaris settings for e3sm-aerosol training.
# Source from run-train.sh or pbs-train.pbs.
#
# Code selection: one shared conda env (deps); ACE_ROOT worktree on PYTHONPATH (code).
# Switch experiments by running from a different worktree or setting ACE_ROOT.
#
# Hardware/queues: Polaris nodes are 4× A100 40GB.
# Docs: https://docs.alcf.anl.gov/polaris/ and https://docs.alcf.anl.gov/polaris/running-jobs/

export POLARIS_DIR="${POLARIS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$(cd "$POLARIS_DIR/.." && pwd)}"

# Worktree root — auto-detected from polaris/ location when unset.
if [[ -z "${ACE_ROOT:-}" ]]; then
    if ACE_ROOT="$(git -C "$POLARIS_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
        :
    else
        ACE_ROOT="$(cd "$POLARIS_DIR/../../../.." && pwd)"
    fi
fi
export ACE_ROOT

# ALCF project / filesystems
export PBS_ACCOUNT="${PBS_ACCOUNT:-E3SMinput}"
export PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:eagle}"

# Data and output on Eagle (not home).
export FME_DATA_ROOT="${FME_DATA_ROOT:-/eagle/E3SMinput/yikwill/datasets}"
export FME_OUTPUT_ROOT="${FME_OUTPUT_ROOT:-/eagle/E3SMinput/yikwill/fme-output}"
export FME_CONFIG_ROOT="${FME_CONFIG_ROOT:-/eagle/E3SMinput/yikwill/fme-config}"

# Job stdout/stderr under the research repo (Polaris analog of slurm-out/).
# Derived from ACE_ROOT parent so worktree switches keep a single log dir.
_LLNL_RESEARCH_ROOT="$(cd "${ACE_ROOT}/.." && pwd)"
export PBS_LOG_ROOT="${PBS_LOG_ROOT:-${_LLNL_RESEARCH_ROOT}/pbs-out}"
# Back-compat alias (older job.env / docs may still reference FME_LOG_ROOT).
export FME_LOG_ROOT="${FME_LOG_ROOT:-$PBS_LOG_ROOT}"
unset _LLNL_RESEARCH_ROOT

# Single shared conda env for all experiments (torch, xarray, etc.).
export FME_VENV="${FME_VENV:-/eagle/E3SMinput/yikwill/conda/envs/fme}"

# Training layout (override for prod scaling).
export TRAIN_NODES="${TRAIN_NODES:-1}"
export TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-4}"

# MODE=interactive|debug|prod
#   debug       — 1-epoch batch smoke (primary debug path)
#   prod        — full training via batch job
#   interactive — optional live node session / hold (same queues as debug; not default)
export MODE="${MODE:-interactive}"

# Queue selection by node count (Polaris PBS policy):
#   debug:          1–2 nodes, ≤1h  — default for 1–2 node debug/interactive
#   debug-scaling:  1–10 nodes, ≤1h — required for 3–10 node debug/interactive
#   capacity:       1–4 nodes, ≤168h — 4-node production (prod routing needs ≥10 nodes)
#   prod:           ≥10 nodes routing to small/medium/large
# Override with PBS_QUEUE=... if needed.
_pick_queue_for_nodes() {
    local nodes="$1" mode="$2"
    case "$mode" in
        interactive|debug)
            if [[ "$nodes" -gt 2 ]]; then
                echo "debug-scaling"
            else
                echo "debug"
            fi
            ;;
        prod)
            if [[ "$nodes" -lt 10 ]]; then
                echo "capacity"
            else
                echo "prod"
            fi
            ;;
        *)
            echo "debug"
            ;;
    esac
}

case "$MODE" in
    interactive|debug)
        # Always recompute from TRAIN_NODES unless explicitly forced. A leftover
        # PBS_QUEUE=debug in the shell would otherwise reject TRAIN_NODES>2.
        if [[ -z "${FORCE_PBS_QUEUE:-}" ]]; then
            export PBS_QUEUE="$(_pick_queue_for_nodes "$TRAIN_NODES" "$MODE")"
        else
            export PBS_QUEUE="$FORCE_PBS_QUEUE"
        fi
        # Same for walltime: a leftover PBS_WALLTIME=01:00:00 must not stick when
        # switching modes. Override with FORCE_PBS_WALLTIME=... if needed.
        if [[ -z "${FORCE_PBS_WALLTIME:-}" ]]; then
            export PBS_WALLTIME="01:00:00"
        else
            export PBS_WALLTIME="$FORCE_PBS_WALLTIME"
        fi
        export FME_DEBUG_MAX_EPOCHS="${FME_DEBUG_MAX_EPOCHS:-1}"
        export WANDB_MODE="${WANDB_MODE:-disabled}"
        ;;
    prod)
        if [[ -z "${FORCE_PBS_QUEUE:-}" ]]; then
            export PBS_QUEUE="$(_pick_queue_for_nodes "$TRAIN_NODES" prod)"
        else
            export PBS_QUEUE="$FORCE_PBS_QUEUE"
        fi
        # capacity allows up to 168h; prod/small max walltimes are shorter and node-min is 10.
        if [[ -z "${FORCE_PBS_WALLTIME:-}" ]]; then
            export PBS_WALLTIME="48:00:00"
        else
            export PBS_WALLTIME="$FORCE_PBS_WALLTIME"
        fi
        unset FME_DEBUG_MAX_EPOCHS
        ;;
    *)
        echo "Unknown MODE=$MODE (use interactive, debug, or prod)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

export TORCH_MASTER_PORT="${TORCH_MASTER_PORT:-29507}"

# PBS email notifications (qsub -M / -m). Events: a=abort, b=begin, e=end.
# Override with PBS_MAIL=... or disable with PBS_MAIL_EVENTS=n / PBS_MAIL=.
export PBS_MAIL="${PBS_MAIL:-yikwill@uw.edu}"
export PBS_MAIL_EVENTS="${PBS_MAIL_EVENTS:-abe}"

# Import fme from the submitting worktree (overrides editable install in FME_VENV).
export PYTHONPATH="${ACE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
