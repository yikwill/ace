# WandB-metrics 30-year inference (interactive)

Four configs log **training-style one-step validation** (`val/*`) plus **30-year rollout metrics** (`inference/*`) to WandB. No 96 GB prediction netCDF.

| Config | Model checkpoint | Climate |
|--------|------------------|---------|
| `config-infer-PI-1951-1980-wandb-pi-pd.yaml` | `55145243` | PI |
| `config-infer-PD-1951-1980-wandb-pi-pd.yaml` | `55145243` | PD |
| `config-infer-PI-1951-1980-wandb-pd-only.yaml` | `55328228` | PI |
| `config-infer-PD-1951-1980-wandb-pd-only.yaml` | `55328228` | PD |

Launcher: `run-inference-wandb-metrics-interactive.sh`

Scratch output: `$PSCRATCH/fme-wandb-infer/` (~200 MB diagnostics only)

---

## Smoke test (~50–55 min)

Proves validation + inference on interactive GPU without WandB or full period.

**Verified Jul 9:** validation completed (364 batches, ~25 min); inference started (600 steps, 2 windows). Needs **`--time 01:00:00`** (45 min too tight).

```bash
cd /global/homes/y/yikwill/llnl-research/ace-exp-e3sm-aerosol/configs/experiments/e3sm-aerosol

salloc --account e3sm --qos interactive --constraint gpu --nodes 1 \
  --gpus-per-node 1 --cpus-per-task 128 --ntasks-per-node 1 --time 01:00:00

# inside allocation:
source ~/.bashrc && conda activate fme
export SMOKE=1
export TRAIN_JOB_ID=55145243
export CONFIG_FILE=config-infer-PI-1951-1980-wandb-pi-pd.yaml
./run-inference-wandb-metrics-interactive.sh
```

`SMOKE=1` sets validation to 1979–1980, `n_forward_steps=600`, disables WandB.

**Measured smoke (PI pi_pd, Jul 9):**
- Validation 364 batches: **25 min**
- Inference: started window 2/2 before 45 min wall killed job; use **1 h** allocation

---

## Full runs — time estimates

From completed 30-yr jobs **with** netCDF + subset smoke **without** netCDF:

| Job | Steps | Wall (with netCDF) | Notes |
|-----|-------|-------------------|--------|
| PI pi_pd (`55641072`) | 43799 | 2h 28m | fsim=300 |
| PI pd_only (`55644032`) | 43799 | 1h 41m | fsim=300 |
| PD pi_pd (`55369304`) | 43799 | 1h 43m | fsim=240 |
| PD pd_only (`55686351`) | 43799 | 2h 58m | fsim=240 |

**Extra cost vs old inference:**
- One-step validation (1976–1980, batch_size 8): **~60–70 min** (~912 batches; smoke measured ~4 s/batch avg)
- No netCDF write: **~15 min saved** vs prior runs
- WandB + histogram aggregator: small overhead

**Estimated full wall per job (interactive):**

| Config | Estimate |
|--------|----------|
| PI + pi_pd | **3h 15m – 3h 45m** |
| PI + pd_only | **2h 30m – 3h 00m** |
| PD + pi_pd | **2h 30m – 3h 00m** |
| PD + pd_only | **3h 30m – 4h 00m** |

Request **`--time 04:00:00`** per job to be safe.

---

## Full run commands

### 1. Allocate (once per job)

```bash
cd /global/homes/y/yikwill/llnl-research/ace-exp-e3sm-aerosol/configs/experiments/e3sm-aerosol

salloc --account e3sm --qos interactive --constraint gpu --nodes 1 \
  --gpus-per-node 1 --cpus-per-task 128 --ntasks-per-node 1 --time 04:00:00
```

### 2. Run inside allocation

**PI+PD model — PI climate:**

```bash
source ~/.bashrc && conda activate fme
export TRAIN_JOB_ID=55145243
export CONFIG_FILE=config-infer-PI-1951-1980-wandb-pi-pd.yaml
export WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-infer-PI-wandb-pi-pd"
./run-inference-wandb-metrics-interactive.sh
```

**PI+PD model — PD climate:**

```bash
export TRAIN_JOB_ID=55145243
export CONFIG_FILE=config-infer-PD-1951-1980-wandb-pi-pd.yaml
export WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-infer-PD-wandb-pi-pd"
./run-inference-wandb-metrics-interactive.sh
```

**PD-only model — PI climate:**

```bash
export TRAIN_JOB_ID=55328228
export FME_CHECKPOINT_PATH=/pscratch/sd/y/yikwill/fme-output/55328228/training_checkpoints/best_inference_ckpt.tar
export CONFIG_FILE=config-infer-PI-1951-1980-wandb-pd-only.yaml
export WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-infer-PI-wandb-pd-only"
./run-inference-wandb-metrics-interactive.sh
```

**PD-only model — PD climate:**

```bash
export TRAIN_JOB_ID=55328228
export FME_CHECKPOINT_PATH=/pscratch/sd/y/yikwill/fme-output/55328228/training_checkpoints/best_inference_ckpt.tar
export CONFIG_FILE=config-infer-PD-1951-1980-wandb-pd-only.yaml
export WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-infer-PD-wandb-pd-only"
./run-inference-wandb-metrics-interactive.sh
```

Re-use same `salloc` for back-to-back jobs if total wall fits in 4 h × N allocations, or release and re-allocate per job.

---

## Batch (`sbatch`) — recommended for full 30-yr runs

5 h wall, `regular` QoS, scratch output. Submit from login node:

```bash
cd /global/homes/y/yikwill/llnl-research/ace-exp-e3sm-aerosol/configs/experiments/e3sm-aerosol
```

**PI+PD — PI:**
```bash
TRAIN_JOB_ID=55145243 CONFIG_FILE=config-infer-PI-1951-1980-wandb-pi-pd.yaml \
  WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-infer-PI-wandb-pi-pd" \
  ./run-inference-wandb-metrics-perlmutter.sh
```

**PI+PD — PD:**
```bash
TRAIN_JOB_ID=55145243 CONFIG_FILE=config-infer-PD-1951-1980-wandb-pi-pd.yaml \
  WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-infer-PD-wandb-pi-pd" \
  ./run-inference-wandb-metrics-perlmutter.sh
```

**PD-only — PI:**
```bash
TRAIN_JOB_ID=55328228 CONFIG_FILE=config-infer-PI-1951-1980-wandb-pd-only.yaml \
  WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-infer-PI-wandb-pd-only" \
  ./run-inference-wandb-metrics-perlmutter.sh
```

**PD-only — PD:**
```bash
TRAIN_JOB_ID=55328228 CONFIG_FILE=config-infer-PD-1951-1980-wandb-pd-only.yaml \
  WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-infer-PD-wandb-pd-only" \
  ./run-inference-wandb-metrics-perlmutter.sh
```

Monitor: `squeue -u $USER` · logs: `~/llnl-research/slurm-out/wandb-infer-<jobid>.out`

Submit all four at once:
```bash
for spec in \
  "55145243 config-infer-PI-1951-1980-wandb-pi-pd.yaml infer-PI-wandb-pi-pd" \
  "55145243 config-infer-PD-1951-1980-wandb-pi-pd.yaml infer-PD-wandb-pi-pd" \
  "55328228 config-infer-PI-1951-1980-wandb-pd-only.yaml infer-PI-wandb-pd-only" \
  "55328228 config-infer-PD-1951-1980-wandb-pd-only.yaml infer-PD-wandb-pd-only"
do
  set -- $spec
  TRAIN_JOB_ID=$1 CONFIG_FILE=$2 \
    WANDB_NAME="$(date +%Y%m%d)-PM-E3SM-SFNO-$3" \
    ./run-inference-wandb-metrics-perlmutter.sh
done
```

---

## WandB metrics logged

**Validation (climate-matched 1976–1980, before rollout):**
- `val/mean/weighted_rmse/{var}`, `val/mean/weighted_bias/{var}`
- `val/mean_map/image-error/{var}` (bias maps)
- `val/mean_norm/weighted_rmse/{var}`, `val/power_spectrum/...`, `val/mean/loss`

**Inference (30-yr rollout):**
- `inference/time_mean/bias_map/{var}`, `inference/time_mean/rmse/{var}`
- `inference/mean_step_20/...`, `inference/histogram/...`, `inference/annual/...`

---

## Notes

- **validation batch_size=8** required (batch_size 32 OOMs after inference init loads stepper on GPU).
- Script detects compute node (`nid*`) and runs Python directly (no nested `srun`).
- Do not tee multiple runs to one log file; use separate logs per job.
