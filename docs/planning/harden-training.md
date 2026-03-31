# Training Hardening Plan

This document defines the execution order for turning the existing baseline run into paper-defensible benchmark evidence.

One full run already exists and should be treated as a baseline sanity pass, not final evidence.

---

## Execution Sequence

1. Plot and analyze the existing baseline run in a notebook.
2. Harden the training and evaluation pipeline.
3. Re-run on the original data variant with hardened controls.
4. Reproduce on cleaned data and compare changes.

---

## Step 0: Baseline Run (Completed)

Purpose: preserve and interpret the initial run as a reference point.

Current baseline configuration:

- Data variant: original seeded datasets under `data/static/v1/`
- Algorithms: `PPO`, `A2C`, `DQN`
- Final budget: `200,000` steps per algorithm per seed
- Seed protocol: canonical 5 seeds (`11,22,33,44,55`) for final runs
- Checkpoint cadence: every `20,000` steps
- Checkpoint model selection: highest `val.asset_survival_rate`, tie-breaker `val.mean_return`
- Final artifact for reporting: `best_model.zip` (not `last_model.zip`)

Interpretation rules:

- Treat this run as a stability and feasibility baseline only.
- Do not use Step 0 as final paper evidence.
- Preserve all Step 0 artifacts for regression comparison.

---

## Step 1: Plot and Analyze Baseline Results (Done)

Purpose: quantify what happened in Step 0 before deciding how aggressively to rerun.

Checklist:

- Unpack and inspect the completed run artifacts (for example `training_0.zip`).
- Build a notebook summary of checkpoint/final metrics per algorithm and seed.
- Plot trajectories for key validation metrics (`asset_survival_rate`, `mean_return`, burned-area metrics).
- Confirm selection behavior (`best_checkpoint.json` and `best_model.zip` follow the stated rule).
- Record instability signals (high variance, collapse windows, missing/malformed artifacts).

Output:

- A short Step 0 baseline analysis notebook and a written go/no-go note for hardening depth.

---

## Step 2: Harden Training and Evaluation

Purpose: reduce debugging risk and increase defensibility before reruns.

### 2.1 One-factor-at-a-time pilot sweeps

For each algorithm:

1. Run small pilot sweeps where only one hyperparameter changes at a time.
2. Keep all other hyperparameters fixed to defaults or current anchor values.
3. Use validation-only selection for pilot ranking.

Recommended order:

- `PPO`: learning rate -> `n_steps` -> entropy coefficient
- `A2C`: learning rate -> `n_steps` -> entropy coefficient
- `DQN`: learning rate -> exploration params -> target update interval -> replay buffer size

### 2.2 Learning-rate schedule explicitness

- Keep constant learning rate in pilot/debug unless schedule testing is an explicit ablation.
- Record in config/logs when no scheduler is active.
- If scheduler is tested, label it clearly and isolate it from canonical results.

### 2.3 NaN and instability guardrails

Add fail-fast checks during training/eval:

- Abort on any non-finite metric/loss (`NaN`, `inf`, `-inf`).
- Abort when checkpoint artifacts are malformed or missing required keys.
- Flag severe instability patterns for manual review.

---

## Step 3: Reproduce on Cleaned Data and Compare

Purpose: quantify how data cleaning changes results under the same frozen training protocol.

Checklist:

- Run data audit notebook: `notebooks/data_outlier_audit.ipynb`.
- Inspect outliers, invalid ranges, missing fields, split consistency, and duplicates.
- If cleaning is justified, build cleaned artifacts under `data/static/v2/`.
- Regenerate all split artifacts consistently from the cleaned source (no overwrite of `v1`).
- Re-run frozen benchmark protocol on `v2` and compare against `v1`.
- Label cleaned-data runs explicitly in metadata/run labels.

Comparison rule:

- Keep budget, seeds, selection rule, and evaluation protocol identical between `v1` and `v2`.

---

## Acceptance Criteria

| Area | Criterion | Threshold / Rule | Action if Failed |
|---|---|---|---|
| Reproducibility canary | Same-config rerun consistency | `checkpoint_metrics.json` and `best_checkpoint.json` match within tolerance | Block final runs; inspect nondeterminism sources |
| Data contract | Required fields + split integrity | No missing required fields; no split mismatches | Rebuild/repair datasets before rerun |
| Numeric validity | Core env-feature ranges | `base_spread_prob` and `wind_strength` remain in valid bounds | Fix pipeline mapping or cleaning rules |
| Training stability | Non-finite values | Zero `NaN`/`inf` in training/eval metrics | Stop run immediately; debug before continue |
| Pilot sanity | Learned policy beats weak baseline trend | At least one stable pilot config improves over random on val asset survival | Revisit hyperparameters/reward/debug before final |
| Selection robustness | Top config repeatability | Top-1 config remains top or near-top across quick repeats | Expand pilot repeats or reduce search noise |
| Protocol fairness | Final-run comparability | Same fixed step budget across all learned algorithms | Reject run as non-canonical |
| Leakage control | Holdout usage discipline | No holdout-based tuning/model selection | Re-run selection using train/val only |
| Artifact completeness | Paper-traceable outputs | Config, checkpoint logs, best checkpoint, best/last model, final eval all present | Re-run missing seeds or repair pipeline |

---

## Reporting Notes

When writing paper-facing results:

- explicitly state fixed-budget protocol and seed count
- explicitly state model-selection rule
- explicitly state data variant used (`v1` original or `v2` cleaned)
- clearly separate pilot diagnostics from final benchmark evidence
