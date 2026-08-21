# exp/e3sm-aerosol composition

Reconstructable integration branch for the E3SM PI/PD aerosol experiment.
Modular `fix/` / `feature/` branches target `main`; everything under **Exp-only** stays on this branch until generalized.

## Base

- `origin/main` @ `05c5a050f168d64ebcfef50ca0a8317f17246d2f`

## Modular branches (independent — any order into main)

No inter-branch dependencies. Merge into `main` in any order; the reconstruct
commands below use a fixed order for empty `git diff` checks.

| Branch | Role |
|--------|------|
| `fix/optional-cluster-env-logging` | Tolerate missing Slurm/PBS cluster env vars in logging helpers |
| `fix/git-rev-parse-stderr` | Silence `git rev-parse` stderr when recording training history |
| `fix/dataloader-persistent-workers-train-only` | Persist DataLoader workers only for **train**; val/inference never persist (full train-only policy). **Supersedes** `fix/inference-persistent-workers` (do not merge the old 1-liner into this experiment) |
| `fix/trainer-host-memory-trim` | `malloc_trim` / host RSS trim after inference and checkpoint spikes (Perlmutter cgroup OOM) |
| `feature/get-stats-multifile-netcdf` | Multifile NetCDF + E3SMV3 dims in `get_stats` (HPX dims intentionally omitted here) |
| `feature/monthly-netcdf-to-zarr` | Monthly NetCDF → zarr converters, sbatch wrappers, benchmark, verify |
| `feature/emis-forcing-zarr` | Emis forcing zarr build/validate/merge under `scripts/emis_forcing/` |

### Deliberately omitted from this experiment

- `feature/spherical-unet` / `feature/spherical-resnet` (and related irfft fixes)
- Old `fix/inference-persistent-workers` — superseded by `fix/dataloader-persistent-workers-train-only`
- Atmosphere variable-alias / HPX-only changes (left out of modularization; not required for lat/lon E3SM aerosol)

## Exp-only (do not PR to main as-is)

- `configs/experiments/e3sm-aerosol/**` — train/infer YAMLs (CFS dataset paths, including untracked prognostic configs), polaris helpers, make-venv, run/sbatch wrappers
- `configs/examples/perlmutter-conda/**` — personal Perlmutter conda launch examples (where customized)
- `scripts/data_process/configs/e3sm-aerosol-stats-*.yaml` only (no HPX ERA5 stats yaml)
- `scripts/diagnose_per_channel_loss.py`
- This file (`COMPOSITION.md`)

Launch/submit for day-to-day training lives in `llnl-research/perlmutter/`
(`submit_train.sh` / `submit_inference.sh`), not per-experiment copies in this repo.

## Reconstruct

From a clean clone / worktree of the fork (or with `origin` = `ai2cm/ace` and
`yikwill-ace-fork` = your fork):

```bash
git fetch origin main
git fetch yikwill-ace-fork \
  fix/optional-cluster-env-logging \
  fix/git-rev-parse-stderr \
  fix/dataloader-persistent-workers-train-only \
  fix/trainer-host-memory-trim \
  feature/get-stats-multifile-netcdf \
  feature/monthly-netcdf-to-zarr \
  feature/emis-forcing-zarr \
  exp/e3sm-aerosol

git checkout -B exp/e3sm-aerosol origin/main
git merge --no-ff yikwill-ace-fork/fix/optional-cluster-env-logging
git merge --no-ff yikwill-ace-fork/fix/git-rev-parse-stderr
git merge --no-ff yikwill-ace-fork/fix/dataloader-persistent-workers-train-only
git merge --no-ff yikwill-ace-fork/fix/trainer-host-memory-trim
git merge --no-ff yikwill-ace-fork/feature/get-stats-multifile-netcdf
git merge --no-ff yikwill-ace-fork/feature/monthly-netcdf-to-zarr
git merge --no-ff yikwill-ace-fork/feature/emis-forcing-zarr
# Then cherry-pick the exp-only tip commit(s) from yikwill-ace-fork/exp/e3sm-aerosol
# (configs + stats yamls + diagnose script + this COMPOSITION.md).
```

After modular merges, the remaining tip commits on `exp/e3sm-aerosol` that are
not on the seven modular branches are the exp-only layer.
