#!/bin/bash
#SBATCH --account e3sm
#SBATCH --qos regular
#SBATCH --constraint cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --time 16:00:00
#SBATCH --job-name e3sm-aerosol-nc2zarr
#SBATCH --output /global/homes/y/yikwill/llnl-research/slurm-out/e3sm-aerosol-nc2zarr-%j.out

set -euo pipefail

DATASET="${1:-}"
if [[ -z "${DATASET}" ]]; then
    echo "Usage: sbatch $0 <pi|pd|pi-clim|pd-clim>" >&2
    exit 1
fi

source "${HOME}/.bashrc"
conda activate "${CONDA_ENV:-fme}"

export BATCH_MONTHS=24
export TIME_CHUNK=124
export WORKERS=64
export SCRATCH_ROOT="${PSCRATCH}/e3sm-aerosol-zarr"

SCRIPT_DIR="/global/homes/y/yikwill/llnl-research/ace-exp-e3sm-aerosol/scripts/data_process"
bash "${SCRIPT_DIR}/convert_e3sm_aerosol_pi_pd_to_zarr.sh" "${DATASET}"
