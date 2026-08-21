# exp/s2resnet composition

Reconstructable integration branch for the SphericalResNet radiation experiment.
Modular `fix/` / `feature/` branches target `main`; everything under **Exp-only** stays on this branch until generalized.

## Base

- `origin/main` @ `05c5a050f168d64ebcfef50ca0a8317f17246d2f`

## Modular branches

| Branch | Role | Depends on |
|--------|------|------------|
| `fix/inference-persistent-workers` | Inference DataLoader: `persistent_workers` only when `num_data_workers > 0` | — |
| `fix/irfft-autograd` | Functional DC/Nyquist imag clearing in `fme/fft.py` (multi-step autograd) | — |
| `feature/spherical-unet` | `SphericalUNet` / `NoiseConditionedSphericalUNet` + ACE registry + tests (includes optional DISCO `theta_cutoff`) | — |
| `feature/spherical-resnet` | `SphericalResNet` / `NoiseConditionedSphericalResNet` + ACE registry + tests | **`feature/spherical-unet`** (shared spherical helpers) |
| `feature/get-stats-time-chunked` | Time-chunked `get_stats` (`time_chunk_size`, `include_variables`, `max_time_samples`, `--force`, `DASK_NUM_WORKERS`) | — |

### Main merge order

1. Fixes (`fix/inference-persistent-workers`, `fix/irfft-autograd`) — any order.
2. `feature/spherical-unet`
3. `feature/spherical-resnet` (must follow spherical-unet)
4. `feature/get-stats-time-chunked` — any time relative to the above.

## Exp-only (do not PR to main as-is)

- `configs/experiments/s2resnet/config-train-radiation-increasing-co2.yaml`
- `configs/experiments/s2resnet/config-train-radiation-bsprobe-val128-base.yaml`
- `configs/experiments/s2resnet/config-train-radiation-bsprobe-val128.yaml`
- `configs/experiments/s2resnet/config-train-radiation-bsprobe-val256.yaml`
- `configs/experiments/s2resnet/config-train-radiation-bsprobe-val512.yaml`
- `configs/experiments/s2resnet/config-train-radiation-bsprobe-val1024.yaml`
- `scripts/data_process/configs/shield-som-increasing-co2-radiation-stats-*.yaml` (benchmark, sample, full, fluxes, chunk0–3, flux-chunk0–3)
- `scripts/data_process/run-radiation-stats-4node.sh`
- `scripts/data_process/run-radiation-stats-fluxes-4node.sh`
- `scripts/data_process/slurm-full-radiation-stats.sh`
- `scripts/on-stats-done-submit-train.sh`
- `scripts/probe_batch_size_radiation.sh`
- `scripts/probe_val_batch_size_radiation.sh`
- This file (`COMPOSITION.md`)

Launch/submit lives in `llnl-research/perlmutter/`, not per-experiment scripts in this repo.

## Reconstruct

From a clean clone / worktree of the fork (or with `origin` = `ai2cm/ace` and `yikwill-ace-fork` = your fork):

```bash
git fetch origin main
git fetch yikwill-ace-fork \
  fix/inference-persistent-workers \
  fix/irfft-autograd \
  feature/spherical-unet \
  feature/spherical-resnet \
  feature/get-stats-time-chunked

git checkout -B exp/s2resnet origin/main
git merge --no-ff yikwill-ace-fork/fix/inference-persistent-workers
git merge --no-ff yikwill-ace-fork/fix/irfft-autograd
git merge --no-ff yikwill-ace-fork/feature/spherical-unet
git merge --no-ff yikwill-ace-fork/feature/spherical-resnet
git merge --no-ff yikwill-ace-fork/feature/get-stats-time-chunked
# Then replay exp-only tip commits from yikwill-ace-fork/exp/s2resnet
# (configs + radiation stats scripts + this COMPOSITION.md), or cherry-pick those commits.
```

After modular merges, the remaining tip commits on `exp/s2resnet` that are not on the modular branches are the exp-only layer.
