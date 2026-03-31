# Training Methods and Process

This document describes how models are trained for the wildfire suppression environment (`WildfireEnv`) and how the benchmark pipeline is executed end to end.

It reflects the current hardened setup, including:
- cleaned-data default (`data/static/v2`)
- one-factor-at-a-time pilot sweeps
- explicit constant learning-rate schedule labeling
- fail-fast non-finite metric checks during checkpoint/final eval

---

## 1) Environment and Agent

`WildfireEnv` is a single-agent tactical wildfire suppression environment.

- Grid: `25 x 25`
- Horizon: `150` steps
- Agent role: one tactical firefighting controller
- Actions:
  - movement: `MOVE_N`, `MOVE_S`, `MOVE_E`, `MOVE_W`
  - suppression: `DEPLOY_HELICOPTER`, `DEPLOY_CREW`
- Objective: protect critical assets while containing spread under limited suppression budgets/cooldowns

The learned algorithms trained in this benchmark are:
- `PPO`
- `A2C`
- `DQN`

Baselines used for evaluation:
- `greedy`
- `random`

---

## 2) Data Contract and Input Sources

All benchmark runs use seeded split artifacts and strict split validation:

- train: `scenario_parameter_records_seeded_train.json`
- val: `scenario_parameter_records_seeded_val.json`
- holdout: `scenario_parameter_records_seeded_holdout.json`

### Current default data variant

The stage scripts default to cleaned data:

- `DATA_VARIANT=v2`
- `DATASET_BASE=data/static/${DATA_VARIANT}`

So, by default:
- `TRAIN_DATASET=data/static/v2/scenario_parameter_records_seeded_train.json`
- `VAL_DATASET=data/static/v2/scenario_parameter_records_seeded_val.json`
- `HOLDOUT_DATASET=data/static/v2/scenario_parameter_records_seeded_holdout.json`

You can override either by:
1) setting `DATA_VARIANT=v1`, or
2) setting explicit dataset paths (`TRAIN_DATASET=...`, etc.).

---

## 3) Canonical Training Protocol

Unless explicitly running an ablation, canonical settings are:

- Algorithms: `ppo`, `a2c`, `dqn`
- Seeds (final): `11,22,33,44,55`
- Budget: `200,000` steps per algorithm per seed
- Checkpoint cadence: every `20,000` steps
- Checkpoint eval episodes: `20`
- Final eval episodes: `100`
- Model selection rule: maximize `val.asset_survival_rate`, tie-break by `val.mean_return`
- Reporting model artifact: `best_model.zip` (not `last_model.zip`)

Learning-rate schedule policy:
- canonical runs use `constant` schedule mode
- schedule mode is recorded in run config (`learning_rate_schedule`)

---

## 4) Stage-by-Stage Pipeline

The benchmark process is organized into 5 stages under `scripts/stages/`.

## Stage 1/5: Karpathy one-record overfit checks

Script:
- `scripts/stages/01_karpathy_overfit.sh`

What runs:
- Each algorithm (`ppo`, `a2c`, `dqn`) trains on one-record train/val/holdout datasets built from the configured split files.

Design choice:
- A fast sanity test to verify the optimization path can overfit trivial data and produce expected learning-direction signals.
- Helps catch broken reward/action/wiring before expensive multi-seed runs.

Outputs:
- Karpathy artifacts under `outputs/benchmark/karpathy/<algo>/seed_<seed>/`
- Printed first-vs-last checkpoint train metrics (return and asset survival)

---

## Stage 2/5: Smoke training + reproducibility canary

Script:
- `scripts/stages/02_smoke_train_and_repro.sh`

What runs:
- Smoke training for each algorithm with short timesteps (default `20,000`).
- Optional reproducibility rerun with identical config and seed.
- Canary comparison script checks:
  - `checkpoint_metrics.json`
  - `best_checkpoint.json`

Design choice:
- Confirms basic pipeline viability and checks deterministic consistency before larger sweeps/final runs.

Outputs:
- Smoke artifacts: `outputs/benchmark/smoke/<algo>/seed_<seed>/...`
- Canary candidate artifacts: `outputs/benchmark/repro_canary/smoke/<algo>/seed_<seed>/...`
- Repro pass/fail printed by `scripts/canary.py`

---

## Stage 3/5: Smoke evaluation

Script:
- `scripts/stages/03_smoke_eval.sh`

What runs:
- Evaluates smoke `best_model.zip` checkpoints for `ppo`, `a2c`, `dqn` plus `greedy` and `random`.
- Uses the same dataset paths explicitly passed from stage context (train/val/holdout).

Design choice:
- Verifies load + score path, split handling, and metric outputs before pilot/final compute is spent.

Outputs:
- `outputs/benchmark/smoke/eval_smoke.json`

---

## Stage 4/5: Validation-only pilot sweeps

Script:
- `scripts/stages/04_pilot_sweep.sh`
- Core sweep logic in `scripts/stages/_common.sh` (`pilot_sweep_algo`)

What runs:
- Hyperparameter pilot runs at reduced budget (`PILOT_TIMESTEPS`, default `40,000`) with one seed (`PILOT_SEED`, default `11`).
- One-factor-at-a-time style sweep families:

`PPO`
- learning rate variations with anchor `n_steps` and `ent_coef`
- `n_steps` variations with anchor learning rate and entropy
- entropy coefficient variations with anchor learning rate and `n_steps`

`A2C`
- learning rate variations with anchor `n_steps` and `ent_coef`
- `n_steps` variations with anchor learning rate and entropy
- entropy coefficient variations with anchor learning rate and `n_steps`

`DQN`
- learning rate variations with anchor exploration/target update/buffer
- exploration parameter variations with anchor learning rate/target update/buffer
- target update interval variations with anchor learning rate/exploration/buffer
- replay buffer size variations with anchor learning rate/exploration/target update

Pilot selection:
- Rank by validation checkpoint metrics:
  1) `asset_survival_rate`
  2) `mean_return`
  3) `containment_success_rate`

Design choice:
- Limits confounding and makes pilot conclusions easier to justify than multi-knob changes.

Outputs:
- Per-config pilot artifacts under `outputs/benchmark/pilot_sweeps/<algo>/<config_id>/pilot/<algo>/seed_<seed>/`
- Winner manifest:
  - `outputs/benchmark/pilot_sweeps/<algo>/pilot_winner.json`
  - includes selected hyperparameters and generated `cli_flags` for final training reuse

---

## Stage 5/5: Full 5-seed benchmark training

Script:
- `scripts/stages/05_final_train.sh`

What runs:
- For each algorithm and each final seed:
  - train at full canonical budget (`200,000`)
  - checkpoint every `20,000`
  - select best checkpoint on validation rule
  - run final evaluation from selected checkpoint

Hyperparameter source:
- If `USE_PILOT_WINNERS=1` and winner JSON exists, stage loads `selected.cli_flags` from:
  - `outputs/benchmark/pilot_sweeps/<algo>/pilot_winner.json`
- Otherwise falls back to algorithm defaults.

Design choice:
- Keeps final runs protocol-fixed while allowing pilot-informed hyperparameters without manual hand-editing.

Outputs:
- `outputs/benchmark/final/<algo>/seed_<seed>/`
  - `config.json`
  - `checkpoint_metrics.json`
  - `best_checkpoint.json`
  - `best_model.zip`
  - `last_model.zip`
  - `final_eval_best.json`

---

## 5) Hardening Controls in the Training Runner

Implemented in `src/models/train_rl_agent.py`:

- strict split record loading (`train`, `val`, `holdout`)
- config serialization per run (`config.json`)
- explicit learning-rate schedule field (`learning_rate_schedule`, canonical `constant`)
- checkpoint-by-checkpoint evaluation loop
- fail-fast checks for malformed or non-finite metrics (`NaN`/`inf`) at checkpoint and final eval time
- best-checkpoint selection guard requiring a valid `val` split entry

This is intended to fail early on numeric instability or malformed metric artifacts.

---

## 6) Running the Pipeline

All stages (full pipeline):
```bash
./scripts/run_benchmark_train.sh
```

Individual stages:
```bash
./scripts/stages/01_karpathy_overfit.sh
./scripts/stages/02_smoke_train_and_repro.sh
./scripts/stages/03_smoke_eval.sh
./scripts/stages/04_pilot_sweep.sh
./scripts/stages/05_final_train.sh
```

Evaluate trained models:
```bash
./scripts/run_benchmark_eval.sh
```

### Useful overrides

Run on original data variant:
```bash
DATA_VARIANT=v1 ./scripts/run_benchmark_train.sh
```

Force explicit dataset paths:
```bash
TRAIN_DATASET=data/static/v2/scenario_parameter_records_seeded_train.json \
VAL_DATASET=data/static/v2/scenario_parameter_records_seeded_val.json \
HOLDOUT_DATASET=data/static/v2/scenario_parameter_records_seeded_holdout.json \
./scripts/run_benchmark_train.sh
```

Change artifact location:
```bash
ARTIFACT_ROOT=outputs/benchmark_v2 ./scripts/run_benchmark_train.sh
```

---

## 7) Artifact Layout

Default root:
- `outputs/benchmark`

Run layout:
- `<artifact_root>/<run_label>/<algo>/seed_<seed>/`

Primary run labels:
- `smoke`
- `pilot`
- `final`
- `karpathy` (sanity-only)

For reporting, use:
- `best_model.zip` + `best_checkpoint.json` + `final_eval_best.json`
