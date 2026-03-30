# Training and Benchmark Plan

This document defines the concrete plan for implementing, validating, tuning, and benchmarking learned agents for `WildfireEnv`.

It is narrower than `docs/planning/impl-plan.md`: this file focuses specifically on training infrastructure, metric definitions, verification steps, and the execution order needed to produce paper-ready results.

---

## 1) Goal

Benchmark standard RL methods against simple baselines on the frozen wildfire asset-protection task.

Required benchmark methods:

- `random`
- `greedy`
- `DQN`
- `A2C`
- `PPO`

Primary paper question:

> Under the same frozen environment, split datasets, and training budget, which method best protects critical assets while containing fire under limited suppression resources?

---

## 2) Frozen Protocol

Unless explicitly labeled as an ablation, use the following benchmark protocol.

- Grid size: `25 x 25`
- Episode horizon: `150`
- Train dataset: `data/static/scenario_parameter_records_seeded_train.json`
- Validation dataset: `data/static/scenario_parameter_records_seeded_val.json`
- Holdout dataset: `data/static/scenario_parameter_records_seeded_holdout.json`
- Training seeds: `11, 22, 33, 44, 55`
- Evaluation cadence during training: every `20,000` env steps
- Checkpoint evaluation episodes per split: `20`
- Final evaluation episodes per seed for train/val: `100`

Training budget per algorithm per seed:

- `PPO`: `200,000` env steps
- `A2C`: `200,000` env steps
- `DQN`: `200,000` env steps

The final paper may report a longer PPO run separately only if all compared methods receive the same additional budget or the extra run is clearly labeled as an ablation.

### 2.1 Split semantics are two-dimensional

Canonical benchmarking has two distinct notions of split, and both must be enforced explicitly in code.

1. Temporal data split from the seeded scenario-record files:
   - train records: `scenario_parameter_records_seeded_train.json`
   - validation records: `scenario_parameter_records_seeded_val.json`
   - holdout records: `scenario_parameter_records_seeded_holdout.json`
2. Scenario-family split inside `WildfireEnv`:
   - in-distribution families: `TRAIN_FAMILIES`
   - held-out OOD families: `HELD_OUT_FAMILIES`

Canonical meaning of each split:

- Train: train records + `TRAIN_FAMILIES`
- Validation: validation records + `TRAIN_FAMILIES`
- Family holdout: validation or expanded holdout records + `HELD_OUT_FAMILIES`
- Temporal holdout: holdout records + explicit family list, reported separately from train/val

Implementation rule:

- The training/evaluation runner must pass `scenario_families` explicitly.
- Canonical runs must not rely on the environment default of `TRAIN_FAMILIES` when split semantics matter.

### 2.2 Current holdout limitation

The current seeded temporal holdout dataset contains only one unique record.

Consequences:

- it is not a credible benchmark split for model selection or checkpoint-time monitoring
- repeated rollouts across seeds do not produce a meaningful `std_across_seeds` estimate for holdout
- it may still be used as a final, clearly labeled stress-test diagnostic

Paper-ready rule:

- Do not present the current single-record temporal holdout as a full held-out benchmark.
- Before making strong final holdout claims, expand `scenario_parameter_records_seeded_holdout.json` beyond one record.

---

## 3) Implementation Scope

### 3.1 Unified training runner

Extend `src/models/train_rl_agent.py` into a unified runner with `--algo {ppo,a2c,dqn}`.

Add a benchmark-safe preset in code:

- `--benchmark-preset canonical`

The canonical preset should fill in the frozen benchmark defaults unless the user explicitly overrides them for an ablation run.

Canonical preset values:

- train dataset: `data/static/scenario_parameter_records_seeded_train.json`
- validation dataset: `data/static/scenario_parameter_records_seeded_val.json`
- holdout dataset: `data/static/scenario_parameter_records_seeded_holdout.json`
- train/validation families: `TRAIN_FAMILIES`
- family-holdout families: `HELD_OUT_FAMILIES`
- checkpoint cadence: `20,000` env steps
- checkpoint evaluation episodes: `20`
- checkpoint-visible splits: `train`, `val`, and optional family holdout only
- final evaluation episodes for train/val: `100`
- benchmark-mode env creation enabled

Holdout visibility rule for the canonical preset:

- Do not surface temporal holdout metrics during checkpoint evaluation or hyperparameter sweeps.
- Temporal holdout is final-reporting-only until the holdout dataset is expanded beyond one record.

Requirements:

1. Shared dataset loading and benchmark-mode env construction.
2. Shared run config serialization.
3. Shared checkpoint evaluation path.
4. Per-algorithm model construction.
5. Per-algorithm output path naming.

Run artifact directory naming:

- `artifacts/benchmark/<run_label>/<algo>/seed_<seed>/`

Required `run_label` values:

- `smoke`
- `pilot`
- `final`

Purpose:

- prevent smoke tests and pilot sweeps from overwriting canonical final benchmark artifacts

Canonical per-seed artifacts:

- `config.json`
- `checkpoint_metrics.json`
- `best_checkpoint.json`
- `best_model.zip`
- `last_model.zip`
- `final_eval_best.json`

Optional convenience exports outside the artifact directory are allowed, but they are not the canonical benchmark outputs.

Artifact semantics:

- `best_model.zip` is the paper-facing artifact for that seed
- `last_model.zip` is retained for debugging and reproducibility only
- `best_checkpoint.json` records the selected training step and selection metric values

### 3.1.1 Frozen checkpoint metric schema

`checkpoint_metrics.json` should be a JSON array. Each element should have this structure:

```json
{
  "algo": "ppo",
  "seed": 11,
  "train_steps": 20000,
  "selected_for_best": false,
  "splits": {
    "train": {
      "mean_return": 0.0,
      "asset_survival_rate": 0.0,
      "containment_success_rate": 0.0,
      "mean_burned_area_fraction": 0.0,
      "mean_time_to_containment": null,
      "mean_resource_efficiency": 0.0,
      "wasted_deployment_rate": 0.0,
      "episodes": 20
    },
    "val": {
      "mean_return": 0.0,
      "asset_survival_rate": 0.0,
      "containment_success_rate": 0.0,
      "mean_burned_area_fraction": 0.0,
      "mean_time_to_containment": null,
      "mean_resource_efficiency": 0.0,
      "wasted_deployment_rate": 0.0,
      "episodes": 20
    }
  }
}
```

If a family-holdout evaluation is enabled during development, it should appear under a distinct key such as:

```json
{
  "splits": {
    "family_holdout": {
      "mean_return": 0.0,
      "asset_survival_rate": 0.0,
      "containment_success_rate": 0.0,
      "mean_burned_area_fraction": 0.0,
      "mean_time_to_containment": null,
      "mean_resource_efficiency": 0.0,
      "wasted_deployment_rate": 0.0,
      "episodes": 20
    }
  }
}
```

The training runner should update `selected_for_best` only after the full checkpoint comparison is complete.

Temporal holdout results, if produced, should be written only in final evaluation artifacts and clearly labeled as single-record diagnostic results when that constraint still applies.

### 3.2 Environment construction by algorithm

Use the same benchmark environment contract for all methods.

- `PPO`: vectorized envs
- `A2C`: vectorized envs
- `DQN`: single env by default

Rationale:

- `WildfireEnv` already exposes a flat observation and discrete action space compatible with all three methods.
- The main algorithm-specific difference is that `DQN` should not reuse the parallel vectorized setup intended for `PPO` and `A2C`.

Implementation requirement for benchmark metrics:

- the environment `info` payload or evaluator-accessible state must expose `assets_lost`, `step`, successful deployment counts, wasted deployment counts, and total deployment attempt counts
- these counters are required for `mean_time_to_containment`, `mean_resource_efficiency`, and `wasted_deployment_rate`
- zero-denominator cases for deployment-based metrics must be handled explicitly as defined in Section 4

### 3.3 Unified evaluation runner

Extend `src/models/evaluate_agents.py` so it can evaluate:

- `ppo`
- `a2c`
- `dqn`
- `greedy`
- `random`

The rollout loop can remain shared because all learned methods expose `model.predict(...)`.

Add a matching benchmark-safe preset in code for evaluation so canonical runs do not rely on ad hoc CLI arguments.

The evaluation preset should:

- use the canonical split dataset paths by default
- default to `100` episodes for train/val
- use the benchmark metric schema defined in this file
- evaluate the chosen artifact explicitly, such as `best_model.zip` or `last_model.zip`

Canonical evaluation outputs should distinguish:

- `train`
- `val`
- optional `family_holdout`
- optional `temporal_holdout_diagnostic`

---

## 4) Metric Definitions

These definitions must be used consistently in code, plots, tables, and paper text.

Core metrics:

1. `mean_return`
   - Mean episodic return over evaluation episodes.
2. `asset_survival_rate`
   - Fraction of episodes with `assets_lost == 0`.
3. `containment_success_rate`
   - Fraction of episodes that terminate by extinguishing the fire before truncation.
4. `mean_burned_area_fraction`
   - Mean final burned-area fraction.
   - Per episode: `(burned + burning + asset_burned cells) / 625`.
5. `std_across_seeds`
   - Standard deviation of the seed-level metric means, not pooled episode variance.

Secondary metrics:

1. `mean_time_to_containment`
   - Mean step of containment over only the episodes where containment occurs.
   - Report `null` or `NA` when no episode in the slice is contained.
2. `mean_resource_efficiency`
   - Mean of `successful_deployments / total_deployments`.
   - A successful deployment changes at least one cell state through suppression/firebreak action.
   - If `total_deployments == 0`, report `0.0`.
3. `wasted_deployment_rate`
   - Mean of `wasted_deployments / total_deployments_attempted`.
   - If `total_deployments_attempted == 0`, report `0.0`.
4. `heldout_performance_drop`
   - Difference between validation or holdout performance and train performance for the same metric.
   - Use `asset_survival_rate` as the primary reported generalization drop.
5. `mean_normalized_burn_ratio`
   - `final_burned_area_with_policy / final_burned_area_under_non_intervention_baseline_on_same_seed_same_record`.
   - Evaluation-only diagnostic.

Definition of `non_intervention_baseline`:

- This is not a trained policy.
- This is not a second task.
- It is a deterministic evaluation-only baseline that never deploys suppression.
- Under the frozen 6-action benchmark, implement it as a deterministic movement-only policy because `WAIT` is not part of the action space.
- Its purpose is to estimate the damage level when the agent does not intervene with helicopter or crew actions.

Reporting rule:

- Final paper tables should report seed-level `mean +- std` for the core metrics.
- Secondary metrics can appear in a second table, appendix, or supplemental JSON.
- `mean_normalized_burn_ratio` should remain a secondary diagnostic, not a headline benchmark metric.

Metrics not to use as the only model-selection criterion:

- pooled episode variance
- raw final burned cell count without normalization
- reward alone without asset survival context

---

## 5) Training-Time Logging

Checkpoint evaluation is required to make the training process auditable.

At each checkpoint, record per split:

- `mean_return`
- `asset_survival_rate`
- `containment_success_rate`
- `mean_burned_area_fraction`
- `mean_time_to_containment`
- `mean_resource_efficiency`
- `wasted_deployment_rate`

Checkpoint-visible splits for canonical runs:

- `train`
- `val`
- optional `family_holdout`

Do not log or inspect temporal holdout metrics during checkpoint evaluation, pilot tuning, or model selection.

The checkpoint metric file is the source of truth for best-checkpoint selection. Do not infer best-checkpoint status later from console logs.

Checkpoint plots to generate from the logs:

- validation `asset_survival_rate` vs env steps
- validation `mean_return` vs env steps
- train/val gap for `asset_survival_rate`
- train/family-holdout gap for `asset_survival_rate` after final training when family holdout is enabled

Model selection rule:

- Select the best checkpoint by highest validation `asset_survival_rate`.
- Use validation `mean_return` as tie-breaker.
- Do not select by holdout performance.
- Save both `best_model.zip` and `last_model.zip` for every seed.
- Use `best_model.zip` for the final paper-facing evaluation unless an ablation explicitly studies last-checkpoint behavior.

---

## 6) Hyperparameter Strategy

Do not choose final hyperparameters by naked-eye inspection alone.

Use a small, fixed validation sweep.

### 6.1 Tuning policy

1. Use a small coarse sweep on the validation split.
2. Tune only a few high-impact knobs per algorithm.
3. Freeze the chosen configuration before the full 5-seed benchmark.
4. Do not change hyperparameters after viewing family-holdout or temporal-holdout results.

### 6.2 Sweep budget

Recommended tuning budget per algorithm:

- short pilot runs only
- `1` seed per candidate config
- smaller training budget than final runs

The goal is to eliminate obviously bad settings, not to over-optimize.

### 6.3 Parameters to tune

`PPO`

- learning rate
- `n_steps`
- entropy coefficient

`A2C`

- learning rate
- `n_steps`
- entropy coefficient

`DQN`

- learning rate
- exploration fraction / final epsilon
- target update interval
- replay buffer size

### 6.4 Selection rule

Choose the config with the best validation `asset_survival_rate` at the end of the pilot budget.

Tie-breakers:

1. validation `mean_return`
2. validation `containment_success_rate`
3. lower instability across repeated quick checks if needed

---

## 7) Verification Ladder Before Full Training

Run the following checks in order. Do not launch full 5-seed benchmarks until each prior stage passes.

### 7.1 Environment and data contract check

Verify that:

- train/val/holdout seeded files load in benchmark mode
- split mismatches fail fast
- explicit `scenario_families` are passed for canonical train/val/family-holdout runs
- reset/step terminate correctly
- observations remain length `636`
- actions remain `Discrete(6)`

### 7.2 Algorithm structure smoke test

For each of `PPO`, `A2C`, `DQN`:

1. instantiate the model against the benchmark env
2. run a very short training job
3. save and reload the model
4. run a short deterministic evaluation rollout
5. verify that both `best_model.zip` and `last_model.zip` are emitted when checkpointing is enabled

This verifies:

- the chosen env wrapper is compatible with the algorithm
- the model serialization path works
- `evaluate_agents.py` can load and score the model

### 7.3 Checkpoint logging test

Run one short training job with checkpoint evaluation enabled and verify that:

- checkpoints fire at the expected step interval
- per-split metrics are written to JSON
- the best-checkpoint selection rule behaves as expected
- no fallback heuristic contaminates learned-agent evaluation
- the benchmark preset produces the frozen protocol values without needing manual CLI reconstruction
- temporal holdout metrics do not appear in checkpoint artifacts for canonical runs

### 7.4 Reward sanity pilot

Run `PPO` for `20,000` steps on one seed and confirm:

- asset-loss penalties appear in the return trace
- returns are not numerically unstable
- the agent learns something better than random on train at minimum

If reward coefficients must change, do it once here and refreeze.

### 7.5 Per-algorithm pilot benchmark

For each of `DQN`, `A2C`, `PPO`, run a short pilot with the candidate hyperparameters and compare against:

- `random`
- `greedy`

Only proceed to full benchmark if:

- the learned method beats `random` on validation `asset_survival_rate`, or
- the run is stable enough to justify a final budget run

---

## 8) Benchmark Execution Order

1. Implement unified train/eval support for `ppo`, `a2c`, `dqn`.
2. Add benchmark-safe presets for training and evaluation.
3. Fix evaluator metric definitions to match this document.
4. Add checkpoint evaluation, config serialization, and canonical artifact writing.
5. Run smoke tests for all methods.
6. Run one-seed pilot tuning sweeps.
7. Freeze one config per algorithm.
8. Run full 5-seed training for `DQN`, `A2C`, `PPO`.
9. Evaluate all learned methods plus `greedy`, `random`, and the non-intervention baseline on train/val and optional family holdout.
10. Run temporal holdout evaluation only as a final, separately labeled diagnostic until the holdout dataset is expanded.
11. Aggregate seed-level means and standard deviations.
12. Produce final plots and paper tables.

---

## 9) Minimum Paper-Ready Outputs

The benchmark is ready for reporting when the following artifacts exist.

Per learned algorithm:

- `best_model.zip` and `last_model.zip` for each seed
- checkpoint metric logs
- final split-wise evaluation JSON for the best checkpoint
- selected hyperparameter config

Aggregate benchmark outputs:

- table of train/val and optional family-holdout `mean +- std` across 5 seeds
- learning curve for at least `PPO`
- holdout comparison figure across methods only when the chosen holdout benchmark is credible for that claim
- explicit note of the model-selection rule and tuning budget
- explicit note that the non-intervention baseline is an evaluation-only secondary diagnostic
- explicit note that single-record temporal holdout results, if shown, are stress-test diagnostics rather than full benchmark evidence

---

## 10) Non-Goals

This plan does not include:

- recurrent policies
- hidden-regime-shift training
- large-scale automated hyperparameter optimization
- modifying the reward during final benchmarking
- using holdout performance for tuning
