# Training Hardening Plan

This document defines a staged hardening workflow before launching paper-facing benchmark runs.

It assumes one full run has already been executed and should now be treated as a baseline sanity pass, not the final benchmark evidence.

---

## Step 0: Dummy Run

Purpose: establish a baseline run with the current unclean pipeline and confirm the end-to-end system is operational.

Current baseline configuration:

- Data variant: original seeded datasets under `data/static/`
- Algorithms: `PPO`, `A2C`, `DQN`
- Final budget: `200,000` steps per algorithm per seed
- Seed protocol: canonical 5 seeds (`11,22,33,44,55`) for final runs
- Checkpoint cadence: every `20,000` steps
- Checkpoint model selection: highest `val.asset_survival_rate`, tie-breaker `val.mean_return`
- Final artifact for reporting: `best_model.zip` (not `last_model.zip`)

Step 0 interpretation rules:

- Treat this run as a reference point for stability and feasibility.
- Do not treat it as final paper evidence until Step 1-3 hardening is complete.
- Preserve all Step 0 artifacts for regression comparison.

---

## Step 1: Pilot Hardening

Purpose: reduce debugging risk and improve interpretability before expensive reruns.

### 1.1 One-factor-at-a-time pilot sweeps

For each algorithm:

1. Run small pilot sweeps where only one hyperparameter changes at a time.
2. Keep all other hyperparameters fixed to algorithm defaults or current pilot anchor values.
3. Use validation-only selection for pilot ranking.

Recommended order:

- `PPO`: learning rate -> `n_steps` -> entropy coefficient
- `A2C`: learning rate -> `n_steps` -> entropy coefficient
- `DQN`: learning rate -> exploration params -> target update interval -> replay buffer size

Why: if performance changes unexpectedly, attribution is immediate and debugging is faster.

### 1.2 Learning-rate schedule explicitness

- Keep constant learning rate in pilot/debug unless a schedule is explicitly tested as an ablation.
- Record in run config/logs that no scheduler is active.
- If a scheduler is tested later, it must be clearly labeled and isolated from canonical results.

### 1.3 Early stopping policy

- Allow optional early stopping only in `pilot` or debug runs for faster iteration.
- Do not use early stopping in canonical final benchmark runs.
- Final benchmark remains fixed-budget for fairness (`200,000` per algorithm per seed).

### 1.4 NaN and instability guardrails

Add fail-fast checks during training/eval:

- Abort if any non-finite metric/loss appears (`NaN`, `inf`, `-inf`).
- Abort if checkpoint artifacts are malformed or missing expected keys.
- Flag severe instability patterns (for example prolonged metric collapse) for manual review.

---

## Step 2: Re-run and Data Audit Decision

Purpose: re-run after Step 1 hardening and decide whether dataset cleaning is required.

### 2.1 Re-run hardened pilot

- Re-run smoke + pilot with hardening controls enabled.
- Compare against Step 0 baseline on key validation metrics and stability.

### 2.2 Data outlier and null audit

- Run the audit notebook: `notebooks/data_outlier_audit.ipynb`.
- Inspect outliers, invalid ranges, missing fields, split consistency, and duplicates.
- Decide if cleaning is required based on documented findings.

### 2.3 If cleaning is required

Use strict versioning and isolation:

- Save cleaned variants under `data/static/clean/`.
- Regenerate all split artifacts consistently from the cleaned source.
- Keep old and cleaned artifacts side-by-side (no overwrite).
- Save model weights and run artifacts before rerunning on cleaned data.
- Label cleaned-data runs explicitly (for example run label suffix or metadata flag).

Reproducibility note:

- Do not mix pilot selection on one data variant with final reporting on another without clear disclosure.

---

## Step 3: Defensibility and Final Freeze

Purpose: increase confidence in model health and lock benchmark settings before full reporting runs.

### 3.1 Training-health diagnostics

Add periodic diagnostics per algorithm where available:

- gradient norm trends
- value loss / TD loss trends
- entropy (for policy-gradient methods)
- checkpoint metric trajectories for train vs val

### 3.2 Instability-aware pilot confirmation

- For each algorithm, take top-2 pilot candidates.
- Run 2 quick repeats per candidate.
- Use aggregate validation ranking with instability-aware tie-breaking.

### 3.3 Freeze and execute final benchmark

- Freeze one hyperparameter config per algorithm.
- Freeze data variant (`original` or specific `clean` version).
- Run canonical fixed-budget benchmark over 5 seeds.
- Report only frozen-protocol results as primary evidence.

### Step 3 Acceptance Criteria

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

When writing paper-facing results after hardening:

- explicitly state fixed-budget protocol and seed count
- explicitly state model-selection rule
- explicitly state data variant used (`original` or `clean/*`)
- clearly separate pilot diagnostics from final benchmark evidence
