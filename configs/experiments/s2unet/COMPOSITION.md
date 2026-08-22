# exp/s2unet composition

Reconstructable integration branch for the spherical U-Net ERA5 experiment.
Modular `fix/` / `feature/` branches target `main`; everything under **Exp-only** stays on this branch until generalized.

## Base

- `origin/main` @ `05c5a050f168d64ebcfef50ca0a8317f17246d2f`

## Modular branches (independent — any order into main)

No inter-branch dependencies. Merge into `main` in any order.

| Branch | Role |
|--------|------|
| `fix/inference-persistent-workers` | Inference DataLoader: `persistent_workers` only when `num_data_workers > 0` |
| `fix/irfft-autograd` | Functional DC/Nyquist imag clearing in `fme/fft.py` (multi-step autograd) |
| `feature/spherical-unet` | `SphericalUNet` / `NoiseConditionedSphericalUNet` model + ACE registry + tests (optional DISCO `theta_cutoff`; opt-in `unet_layout: classic`) |
| `feature/ar-input-noise` | Training-time AR input noise: `TrainStepperConfig.ar_input_noise_sigma` scales prognostic state noise by `|true Δ|` after each train step |
| `feature/residual-std-scale` | Opt-in `scale_residual_by_residual_std`: prognostic residual add uses `σ_res/σ_full` so the network predicts in residual units |
| `feature/spatial-mean-normalization` | Spatial means in `NormalizationConfig` (e.g. `time-mean.nc`) plus optional `scalar_means_path` / `scalar_means_names` overrides for static fields |

## Exp-only (do not PR to main as-is)

- `configs/experiments/s2unet/config-train-era5.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction-ar-noise.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction-res-scaled.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction-res-scaled-classic.yaml`
- `configs/experiments/s2unet/config-train-era5-classic-time-mean-centering.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction-classic-time-mean-centering.yaml`
- `configs/experiments/s2unet/config-train-era5-sfno-baseline.yaml`
- `fme/core/distributed/torch_distributed.py`: global `broadcast_buffers=False` for DDP (DISCO/SHT buffer workaround). Needs a narrower design before any `fix/` PR.
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
  feature/ar-input-noise \
  feature/residual-std-scale \
  feature/spatial-mean-normalization

git checkout -B exp/s2unet origin/main
git merge --no-ff yikwill-ace-fork/fix/inference-persistent-workers
git merge --no-ff yikwill-ace-fork/fix/irfft-autograd
git merge --no-ff yikwill-ace-fork/feature/spherical-unet
git merge --no-ff yikwill-ace-fork/feature/ar-input-noise
git merge --no-ff yikwill-ace-fork/feature/residual-std-scale
git merge --no-ff yikwill-ace-fork/feature/spatial-mean-normalization
# Then replay exp-only tip commits from yikwill-ace-fork/exp/s2unet
# (configs + DDP hack + this COMPOSITION.md), or cherry-pick those commits.
```

After modular merges, the remaining tip commits on `exp/s2unet` that are not on the modular branches are the exp-only layer.
