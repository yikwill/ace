#!/bin/bash
# Run Python from the shared venv, importing fme from ACE_ROOT via PYTHONPATH.
set -euo pipefail

: "${FME_VENV:?FME_VENV must be set}"
: "${ACE_ROOT:?ACE_ROOT must be set}"

# Cap BLAS/OpenMP on login-node staging/validate (default OpenBLAS=64 can
# exhaust process slots and hang/fail config validation).
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

export PYTHONPATH="${ACE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "$FME_VENV/bin/python" "$@"
