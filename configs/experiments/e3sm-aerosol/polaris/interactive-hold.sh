#!/bin/bash
# OPTIONAL: hold GPU node(s) for agent SSH attach. Not the default workflow.
#
# Same PBS queues/waits as batch. Prefer MODE=debug ./polaris/run-train.sh for smokes.
# Use only when you need a multi-turn live allocation without qsub -I (no TTY).
#
#   TRAIN_NODES=2 ./polaris/interactive-hold.sh
#   CONFIG_FILE=... ./polaris/interactive-attach.sh
#   ./polaris/interactive-release.sh
#
# Docs: https://docs.alcf.anl.gov/polaris/running-jobs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export TRAIN_NODES="${TRAIN_NODES:-1}"
export MODE=interactive

# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

SESSION_ROOT="${FME_CONFIG_ROOT}/interactive-sessions"
SESSION_ID="${SESSION_ID:-$(date +%Y%m%d-%H%M%S)-$(uuidgen | cut -c1-8)}"
SESSION_DIR="${SESSION_ROOT}/${SESSION_ID}"
mkdir -p "$SESSION_DIR"

PBS_SCRIPT="${SESSION_DIR}/hold.pbs"
cat > "$PBS_SCRIPT" <<EOF
#!/bin/bash -l
#PBS -N hold-interactive-${TRAIN_NODES}n
#PBS -A ${PBS_ACCOUNT}
#PBS -l select=${TRAIN_NODES}:system=polaris
#PBS -l place=scatter
#PBS -l walltime=${PBS_WALLTIME}
#PBS -l filesystems=${PBS_FILESYSTEMS}
#PBS -q ${PBS_QUEUE}
#PBS -j oe
#PBS -o ${SESSION_DIR}/hold.log
#PBS -e ${SESSION_DIR}/hold.log

set -euo pipefail

SESSION_DIR="${SESSION_DIR}"
EXPERIMENT_DIR="${EXPERIMENT_DIR}"
ACE_ROOT="${ACE_ROOT}"
TRAIN_NODES="${TRAIN_NODES}"

HEAD_HOST="\$(hostname -f)"
NODEFILE_COPY="\${SESSION_DIR}/nodefile"
sort -u "\${PBS_NODEFILE}" > "\${NODEFILE_COPY}"

# Shell-sourceable metadata for attach/release (written before ready so attach
# never sees a ready flag without JOB_ID / HEAD_HOST).
cat > "\${SESSION_DIR}/session.info" <<INFO
SESSION_ID=${SESSION_ID}
SESSION_DIR=\${SESSION_DIR}
JOB_ID=\${PBS_JOBID}
HEAD_HOST=\${HEAD_HOST}
PBS_NODEFILE_COPY=\${NODEFILE_COPY}
TRAIN_NODES=\${TRAIN_NODES}
EXPERIMENT_DIR=\${EXPERIMENT_DIR}
ACE_ROOT=\${ACE_ROOT}
INFO

touch "\${SESSION_DIR}/ready"
echo "hold ready job=\${PBS_JOBID} head=\${HEAD_HOST} nodes=\${TRAIN_NODES} \$(date)" | tee -a "\${SESSION_DIR}/hold.log"

# Keep the allocation until walltime; PBS kills the job when time expires.
sleep infinity
EOF

echo "Submitting interactive hold:"
echo "  session=$SESSION_ID"
echo "  queue=$PBS_QUEUE nodes=$TRAIN_NODES walltime=$PBS_WALLTIME account=$PBS_ACCOUNT"
echo "  session_dir=$SESSION_DIR"
echo ""

QSUB_ARGS=()
if [[ -n "${PBS_MAIL:-}" && "${PBS_MAIL_EVENTS:-n}" != "n" ]]; then
    QSUB_ARGS+=(-M "$PBS_MAIL" -m "$PBS_MAIL_EVENTS")
    echo "  mail=$PBS_MAIL events=$PBS_MAIL_EVENTS"
fi

JOB_ID="$(qsub "${QSUB_ARGS[@]}" "$PBS_SCRIPT")"
# Record job id from the login side immediately so release works before ready.
cat > "${SESSION_DIR}/submit.info" <<INFO
SESSION_ID=${SESSION_ID}
SESSION_DIR=${SESSION_DIR}
JOB_ID=${JOB_ID}
TRAIN_NODES=${TRAIN_NODES}
EXPERIMENT_DIR=${EXPERIMENT_DIR}
ACE_ROOT=${ACE_ROOT}
PBS_QUEUE=${PBS_QUEUE}
INFO
echo "${SESSION_ID}" > "${SESSION_ROOT}/latest"

echo "Submitted ${JOB_ID}"
echo "  Wait/attach:  CONFIG_FILE=config-train-....yaml ./polaris/interactive-attach.sh ${SESSION_ID}"
echo "  Release:      ./polaris/interactive-release.sh ${SESSION_ID}"
echo "  Hold log:     ${SESSION_DIR}/hold.log"
echo "  Monitor:      qstat -f ${JOB_ID}"
