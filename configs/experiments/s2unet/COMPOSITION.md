# exp/s2unet composition

Reconstructable integration branch for the spherical U-Net ERA5 experiment.
Modular `fix/` / `feature/` branches target `main`; everything under **Exp-only** stays on this branch until generalized.

## Base

- `origin/main` @ `e57469c1ddf1ed4b701bf81f3aba4d72137a8385`

## Modular branches

| Branch | Role | Depends on |
|--------|------|------------|
| `fix/inference-persistent-workers` | Inference DataLoader: `persistent_workers` only when `num_data_workers > 0` | — |
| `fix/irfft-autograd` | Functional DC/Nyquist imag clearing in `fme/fft.py` (multi-step autograd) | — |
| `feature/spherical-unet` | `SphericalUNet` / `NoiseConditionedSphericalUNet` model + ACE registry + tests (optional DISCO `theta_cutoff`; opt-in `unet_layout: classic`) | — |
| `feature/ar-input-noise` | Training-time AR input noise: `TrainStepperConfig.ar_input_noise_sigma` scales prognostic state noise by `|true Δ|` after each train step | — |
| `feature/residual-lowpass` | Opt-in `residual_lowpass`: spherical-harmonic truncation on the residual add (`x + LP(dx)` or `LP(x) + dx`). `lmax` keeps degrees `l < lmax` | — |
| `feature/residual-std-scale` | Load legacy `scale_residual_by_residual_std` as `residual_prediction.normalized` (the scale itself is on main, and runs before the low-pass) | **`feature/residual-lowpass`** |
| `feature/full-field-prognostics` | Load legacy `full_field_prognostic_names` as the complement of `residual_prediction.names`. Those names stay out of the residual add and cannot be low-passed. Loss scaling follows main | **`feature/residual-std-scale`** |
| `feature/normalization-stat-overrides` | Spatial stats in `NormalizationConfig` plus `means_overrides` / `stds_overrides` lists (`{path, names}`) so selected fields can mix a time-mean map with residual std | — |
| `fix/evaluator-sst-perturbation` | Evaluator `get_inference_data` forwards stepper ocean field names so loader SST perturbations can apply | — |

### Merge into `main`

Any order among independent leaves. Stacks (parent then child, or stack tip after the parent is on `main`):

1. `feature/residual-lowpass` then `feature/residual-std-scale` then `feature/full-field-prognostics`

### Reconstruct merges

Any order among independent leaves; for stacks merge the tip only:

- `feature/full-field-prognostics` (contains `feature/residual-lowpass` and `feature/residual-std-scale`)
- `feature/normalization-stat-overrides`

## Exp-only (do not PR to main as-is)

- `configs/experiments/s2unet/config-train-era5.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction-ar-noise.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction-res-scaled.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction-res-scaled-classic.yaml`
- `configs/experiments/s2unet/config-train-era5-classic-time-mean-centering.yaml`
- `configs/experiments/s2unet/config-infer-era5-1980-2025.yaml`
- `configs/experiments/s2unet/config-infer-era5-1980-2020-enso.yaml`
- `configs/experiments/s2unet/config-infer-era5-1996-1997.yaml`
- `configs/experiments/s2unet/config-infer-era5-1996-1997-sst-p2k.yaml`
- `configs/experiments/s2unet/config-infer-era5-1996-1997-sst-p4k.yaml`
- `configs/experiments/s2unet/config-train-era5-classic-pressfc-time-mean.yaml`
- `configs/experiments/s2unet/config-train-era5-classic-pressfc-time-mean-residual-std.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction-classic-time-mean-centering.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction-classic-pressfc-full-field.yaml`
- `configs/experiments/s2unet/config-train-era5-residual-prediction-classic-pressfc-full-field-residual-std.yaml`
- `configs/experiments/s2unet/config-train-era5-sfno-baseline.yaml`
- `configs/experiments/s2unet/config-train-s2unet-era5-residual-pressfc-lowpass64.yaml`
- `configs/experiments/s2unet/config-train-s2unet-era5-residual-pressfc-skip-lowpass32.yaml`
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
  feature/full-field-prognostics \
  feature/normalization-stat-overrides \
  feature/residual-lowpass \
  fix/evaluator-sst-perturbation

git checkout -B exp/s2unet origin/main
git merge --no-ff yikwill-ace-fork/fix/inference-persistent-workers
git merge --no-ff yikwill-ace-fork/fix/irfft-autograd
git merge --no-ff yikwill-ace-fork/feature/spherical-unet
git merge --no-ff yikwill-ace-fork/feature/ar-input-noise
git merge --no-ff yikwill-ace-fork/feature/full-field-prognostics
git merge --no-ff yikwill-ace-fork/feature/normalization-stat-overrides
git merge --no-ff yikwill-ace-fork/fix/evaluator-sst-perturbation
# Then replay exp-only tip commits from yikwill-ace-fork/exp/s2unet
# (configs + DDP hack + this COMPOSITION.md), or cherry-pick those commits.
```

After modular merges, the remaining tip commits on `exp/s2unet` that are not on the modular branches are the exp-only layer.
