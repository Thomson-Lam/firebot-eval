# Environment Spec: Wildfire Simulator + XGBoost Calibration

This document is a concrete implementation guide for how the wildfire RL environment should work and how XGBoost interfaces with it.

It is aligned with `impl-plan.md` and intended to remove ambiguity before coding and experiments.

---

## 1) What the Agent Represents

The agent is a single tactical controller operating on a grid map.

- It decides movement and suppression actions each step.
- It has limited suppression resources (helicopter and crew budgets).
- Its mission is not to minimize all fire everywhere; its mission is to protect critical assets under budget.

Think of it as a simplified incident-response decision unit.

---

## 2) Core Environment Definition

## 2.1 Grid and Episode Constants (canonical)

- Grid size: `25 x 25`
- Episode horizon: `150` steps
- Per-episode budgets:
  - `heli_left = 8`
  - `crew_left = 20`
- Cooldowns:
  - helicopter cooldown: `5` steps
  - crew cooldown: `2` steps

## 2.2 Cell encoding

- `0`: unburned
- `1`: burning
- `2`: burned
- `3`: suppressed (firebreak or suppressed burn)
- `4`: critical asset (unburned)
- `5`: critical asset damaged/burning (internal bookkeeping can be separate, but report this event explicitly)

---

## 3) Observation (State)

At each step, the policy receives:

1. Fire grid (`25x25`, encoded cells above)
2. Agent position `(row, col)`
3. Remaining resources: `heli_left`, `crew_left`
4. Cooldowns: `heli_cd`, `crew_cd`
5. Severity one-hot: `[low, medium, high]`
6. Wind bias vector: `(wx, wy)`

This state lets the policy reason about:

- where the fire is,
- what must be protected,
- what resources are still available,
- how spread is likely to move spatially.

---

## 4) Action Semantics (Hard Rules)

Action IDs:

- `0`: `MOVE_N`
- `1`: `MOVE_S`
- `2`: `MOVE_E`
- `3`: `MOVE_W`
- `4`: `DEPLOY_HELICOPTER`
- `5`: `DEPLOY_CREW`

Rules:

- Movement changes position by one cell if in bounds; otherwise no movement.
- Deployment actions act at the **current agent cell**.
- Helicopter footprint: **3x3** neighborhood centered at agent.
- Crew footprint: **1x1** current cell only.
- Both can target burning or non-burning cells.
- Burning cells in footprint become `suppressed`.
- Unburned cells in footprint become `suppressed` firebreaks.

Budget/cooldown gating:

- Helicopter requires `heli_left > 0` and `heli_cd == 0`.
- Crew requires `crew_left > 0` and `crew_cd == 0`.
- Successful helicopter use: `heli_left -= 1`, `heli_cd = 5`.
- Successful crew use: `crew_left -= 1`, `crew_cd = 2`.
- Cooldowns decrement by 1 each step down to 0.

Wasted action definition:

An action is wasted if either:

1. Deployment attempted while blocked by cooldown/budget, or
2. Deployment causes zero state change in its footprint.

---

## 5) Fire Dynamics and Transition

Each step after action execution:

1. Apply suppression effects.
2. Spread fire stochastically from burning cells to neighbors.
3. Apply burn progression/burnout rules.
4. Update asset-loss counters when fire reaches asset cells.

Spread probability is episode-parameterized:

- baseline from `base_spread_prob`
- adjusted by wind bias `(wx, wy)` relative to neighbor direction
- optional local modifiers if additional heterogeneity is enabled

Episode termination:

- success if no burning cells remain, or
- horizon reached at step 150.

---

## 6) Reward Function (Single-Objective)

Per-step reward:

```text
r_t =
  - 75.0 * asset_cells_lost_t
  - 0.4  * new_burned_cells_t
  + 3.0  * burning_cells_suppressed_t
  - 1.5  * heli_used_t
  - 0.5  * crew_used_t
  - 1.0  * wasted_action_t
```

Terminal shaping:

- `+100` if fire extinguished and no asset loss.
- `+40` if episode ends with all assets intact.

Interpretation:

- Asset protection is dominant objective.
- Burn suppression and burn growth provide dense learning signal.
- Resource costs prevent degenerate spam strategies.

---

## 7) XGBoost Interface: What It Does and Does Not Do

XGBoost is used to calibrate episode conditions from cached real-data snapshots.

- It does **not** choose actions.
- It does **not** replace the RL policy.
- It does **not** make real-time tactical decisions.

It outputs simulator parameters at episode reset.

## 7.1 Snapshot input features (for XGBoost)

Required canonical features:

- weather: `wind_speed_km_h`, `wind_direction_deg`, `temperature_c`, `relative_humidity_pct`, `precipitation_mm`
- danger: `fwi`, `isi`, `bui`
- incident: `area_hectares`, `latitude`, `longitude`, `province`

Optional useful features (if ingestion/training pipeline is extended):

- `frp_mw` (FIRMS)
- `cffdrs_station_distance_km`
- `dmc`, `dc`, `ffmc`
- temporal deltas from snapshot history

## 7.2 XGBoost output contract

For each snapshot record:

1. `spread_intensity` in `[0,1]`
2. `spread_rate_1h_m` (logging + interpretability)
3. `wind_dir_deg` (pass-through from snapshot)
4. `wind_strength` in `[0,1]` (normalized from wind speed)
5. `severity_bucket` from `spread_intensity`

Deterministic env mapping:

- `base_spread_prob = 0.04 + 0.18 * spread_intensity`
- severity:
  - low: `<0.33`
  - medium: `0.33-0.66`
  - high: `>0.66`
- wind vector:
  - `wx = wind_strength * cos(wind_dir_deg)`
  - `wy = wind_strength * sin(wind_dir_deg)`

Episode rule:

- Sample one parameter record at reset.
- Keep it fixed for the full episode in canonical runs.

---

## 8) Mandatory Snapshot Pipeline for Reproducibility

Training/evaluation must never depend on live API calls.

Required workflow:

1. Collect and normalize ingestion data.
2. Write versioned snapshot cache file(s).
3. Build XGBoost features from snapshots.
4. Produce env-parameter records.
5. Train/evaluate RL only from cached records + seeded RNG.

Fail-fast rule:

- If required fields are missing in benchmark mode, error out.
- Do not silently inject hidden defaults during benchmark runs.

---

## 9) Training Process for the Agent

## 9.1 Algorithms

- DQN
- A2C
- PPO
- Baselines: greedy, random

## 9.2 Fixed protocol

- Train steps: `200,000` per algorithm per seed
- Seeds: `11, 22, 33, 44, 55`
- Eval cadence: every `20,000` steps
- Eval episodes/checkpoint: `20`
- Final eval episodes per seed: `100`

## 9.3 Scenario families

Train families:

- ignition in `{center, edge, multi_cluster}`
- severity in `{low, medium, high}`
- asset layout `A`

Held-out families:

- ignition `corner` x all severities x layout `A`
- ignition in `{center, edge, multi_cluster}` x severity `medium` x layout `B`

## 9.4 Training loop (conceptual)

```text
for seed in [11,22,33,44,55]:
  set global RNG seed
  for algorithm in [DQN, A2C, PPO]:
    init agent + env factory
    for step in training_steps:
      collect transitions
      update policy/value per algorithm
      if step % eval_interval == 0:
        evaluate on fixed eval set (no fallback)
        log metrics
    run final 100-episode evaluation
aggregate results across seeds
compare against greedy and random baselines
```

---

## 10) What to Report

Primary metrics:

1. mean episodic return
2. asset survival rate
3. containment success rate
4. final burned area
5. variance across seeds

Secondary metrics:

- time to containment
- resource efficiency
- wasted deployment rate
- held-out performance drop

Interpretability checks:

- which assets are protected first,
- deployment timing under low vs high severity,
- behavior under held-out corner ignition scenarios.

---

## 11) Minimal Sanity Checklist Before Full Runs

1. Reward sanity pass (20k PPO steps, 1 seed).
2. Confirm non-zero asset-loss and suppression events in logs.
3. Confirm budgets/cooldowns are enforced.
4. Confirm no live API access during train/eval.
5. Confirm held-out scenario IDs are excluded from training.
