#!/usr/bin/env bash
# Convert one e3sm-aerosol PI or PD netCDF directory to ACE Zarr.
set -euo pipefail

source "${HOME}/.bashrc"
conda activate "${CONDA_ENV:-fme}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-}"

if [[ -z "${DATASET}" ]]; then
    echo "Usage: $0 <pi|pd|pi-clim|pd-clim>" >&2
    exit 1
fi

export DATA_ROOT="${DATA_ROOT:-/global/cfs/projectdirs/e3sm/yikwill/datasets}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-${PSCRATCH}/e3sm-aerosol-zarr}"
export BATCH_MONTHS="${BATCH_MONTHS:-3}"
export TIME_CHUNK="${TIME_CHUNK:-124}"
export WORKERS="${WORKERS:-16}"

python -u "${SCRIPT_DIR}/convert_e3sm_aerosol_to_zarr.py" \
    "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --scratch-root "${SCRATCH_ROOT}" \
    --batch-months "${BATCH_MONTHS}" \
    --time-chunk "${TIME_CHUNK}" \
    --workers "${WORKERS}"
