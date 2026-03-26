# Implementation Plan: Wildfire RL Benchmark (Canonical)

This file is the single source of truth for implementation and evaluation.

Project direction:

**Empirical comparison of standard RL algorithms on an enhanced custom wildfire simulator with one objective: protect critical assets under limited suppression budget.**

---

## 1) Canonical Problem and Claim

### Problem

Given a spreading wildfire on a grid and limited suppression resources, what tactical policy best protects critical assets?

### Single objective

**Maximize critical asset survival under fixed per-episode suppression budget.**

### Claim boundaries

- Do not claim operational readiness.
- Do not claim superiority over real emergency protocols.
- Treat real data ingestion and XGBoost as simulator calibration support only.

### Canonical claim text

"We design an enhanced wildfire tactical suppression benchmark with protected assets, finite suppression budget, and heterogeneous spread, then compare RL and heuristic baselines under fixed scenario families and held-out tests."

---

## 2) Frozen Experimental Protocol (No Ranges)

- Grid size: **25 x 25**
- Episode horizon: **150 steps**
- Training budget per algorithm per seed: **200,000 env steps**
- Evaluation cadence during training: **every 20,000 steps**
- Evaluation episodes per checkpoint: **20**
- Final evaluation episodes per seed: **100**
- Number of seeds: **5** (`11, 22, 33, 44, 55`)

Any deviation must be explicitly labeled as an ablation and reported separately.

---

## 3) System Overview

```mermaid
flowchart TD
    A[Feasible Data Sources] --> B[Ingestion and Normalization]
    B --> C[Mandatory Snapshot Cache]
    C --> D[XGBoost Spread Calibration]
    D --> E[Scenario Parameters]
    E --> F[Enhanced RL Environment]
    F --> G[DQN A2C PPO]
    F --> H[Greedy Random]
    G --> I[Benchmark Harness]
    H --> I
    I --> J[Metrics Tables Plots]

    K[Scenario Family Generator] --> F
    L[Asset Layout Generator] --> F
```

---

## 4) Environment Specification (Canonical)

## 4.1 State representation

Observation at step `t` contains:

1. `fire_grid` (`25x25`) with cell types:
   - `0` unburned
   - `1` burning
   - `2` burned
   - `3` suppressed
   - `4` critical asset (unburned)
   - `5` critical asset (burning/burned marker in internal bookkeeping)
2. agent position `(row, col)`
3. remaining helicopter budget `heli_left`
4. remaining crew budget `crew_left`
5. helicopter cooldown `heli_cd`
6. crew cooldown `crew_cd`
7. severity bucket (`low`, `medium`, `high`) encoded one-hot
8. wind bias vector `(wx, wy)` if wind-bias mode enabled

## 4.2 Action set and exact semantics

Actions:

- `0`: `MOVE_N`
- `1`: `MOVE_S`
- `2`: `MOVE_E`
- `3`: `MOVE_W`
- `4`: `DEPLOY_HELICOPTER`
- `5`: `DEPLOY_CREW`

Hard definitions:

- Movement actions move the agent by one cell if in bounds; otherwise no movement.
- `DEPLOY_HELICOPTER` acts at the **agent's current cell** and affects a **3x3 neighborhood** (radius 1 Chebyshev).
- `DEPLOY_CREW` acts at the **agent's current cell only**.
- Both deployment actions can be invoked on burning and non-burning cells.
- Suppression effect:
  - burning cells in affected area become `suppressed` immediately,
  - unburned cells in affected area become `suppressed` firebreak cells.
- Budget/cooldown constraints:
  - helicopter: requires `heli_left > 0` and `heli_cd == 0`; then `heli_left -= 1`, set `heli_cd = 5`
  - crew: requires `crew_left > 0` and `crew_cd == 0`; then `crew_left -= 1`, set `crew_cd = 2`
- A deployment action is marked **wasted** if it changes zero cells (no state transition in targeted footprint) or if attempted while blocked by cooldown/budget.

Per-episode budgets:

- `heli_left` initial value: **8**
- `crew_left` initial value: **20**

## 4.3 Fire dynamics

- Fire spreads stochastically from burning cells to neighbors.
- Baseline spread probability is scenario-dependent from XGBoost calibration.
- Heterogeneity mode for canonical runs: **wind bias enabled**.
- Wind bias increases ignition probability downwind and decreases upwind.

---

## 5) Reward Function (Single-Objective Aligned)

Reward at step `t`:

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
- `+40` if episode ends with all assets intact (even if fire not fully extinguished).

### Reward sanity check (required)

Before full benchmark runs, run a short pilot (`20,000` steps, PPO, 1 seed) and verify:

- asset-loss events appear in training signal,
- suppression terms are not drowned out,
- return scale is stable (no exploding positive/negative episodes).

If unstable, adjust only `asset` and `burn` coefficients once, then freeze.

---

## 6) Scenario Families and Held-Out Split (Frozen)

## 6.1 Scenario factors

- Ignition layout: `center`, `edge`, `corner`, `multi_cluster`
- Severity: `low`, `medium`, `high`
- Asset layout type: `A` (single critical cluster), `B` (two smaller critical clusters)

## 6.2 Training families (frozen)

- Train on ignition in `{center, edge, multi_cluster}` x severity in `{low, medium, high}` x asset layout `A`.
- Total train families: **9**.

## 6.3 Held-out test families (frozen)

- OOD ignition test: `corner` x `{low, medium, high}` x layout `A` (3 families).
- OOD asset test: `{center, edge, multi_cluster}` x `medium` x layout `B` (3 families).
- Total held-out families: **6**.

Report both in-distribution and held-out performance.

---

## 7) Algorithms to Benchmark

Required methods:

- **DQN** (value-based discrete baseline)
- **A2C** (lightweight on-policy baseline)
- **PPO** (strong policy-gradient baseline)
- **Greedy heuristic** (non-RL baseline)
- **Random** (sanity floor)

Do not include recurrent baseline unless hidden regime shifts are explicitly added and tested.

---

## 8) Benchmark Harness and Logging (Required Infrastructure)

The harness is mandatory and must be built before full training runs.

Requirements:

1. Unified runner for all algorithms.
2. Fixed config serialization to file per run.
3. Metrics written to CSV/JSON per checkpoint and final summary.
4. Distinct evaluation mode with fallback heuristics disabled for RL methods.
5. Seed-aware aggregation scripts for mean/std and confidence intervals.

Core metrics:

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

---

## 9) XGBoost Interface to Environment (Refined)

XGBoost is a calibration layer between ingested wildfire/weather snapshots and simulator episode parameters. It is not a control policy.

## 9.1 Input feature contract with availability status

Canonical feature groups:

1. **Weather (supported now)**
   - `wind_speed_km_h`
   - `wind_direction_deg`
   - `temperature_c`
   - `relative_humidity_pct`
   - `precipitation_mm`

2. **Fire danger indices (supported now)**
   - `fwi`, `isi`, `bui`

> TODO: check caveats to ensure data quality for pipeline! 

1. **Incident context (supported now, with caveats)**
   - `area_hectares` (often missing for FIRMS hotspots)
   - `latitude`, `longitude`
   - `province` (categorical coarse location)

4. **Useful optional features (partially supported or easy to add)**
   - `frp_mw` from FIRMS hotspot intensity (partially available)
   - `cffdrs_station_distance_km` (derive from nearest-station lookup)
   - `dmc`, `dc`, `ffmc` (already available from CFFDRS response)
   - simple temporal deltas from snapshots (e.g., 6h wind or RH change)

5. **Do not include as canonical unless truly ingested**
   - synthetic `slope_pct`
   - synthetic `rh_trend_24h`

### Required ingestion/training code updates for optional features

- Extend snapshot schema to persist `frp_mw`, station distance, and additional CFFDRS indices.
- Update XGBoost feature builder to include encoded `province` and missingness flags (e.g., `has_area`, `has_frp`).
- Remove hidden defaults during benchmark feature generation; fail fast if required canonical features are missing.

## 9.2 Output contract for simulator (refined)

For each snapshot record, produce:

1. `spread_intensity` in `[0, 1]` (primary scalar for fire-growth pressure)
2. `spread_rate_1h_m` (interpretable spread scale for logging and sanity checks)
3. `wind_dir_deg` (pass-through from weather input, not predicted)
4. `wind_strength` in `[0, 1]` (normalized from observed wind speed)
5. `severity_bucket` (`low`, `medium`, `high`) derived deterministically from `spread_intensity`

Deterministic mapping to env parameters:

- `base_spread_prob = 0.04 + 0.18 * spread_intensity`
- severity bucket thresholds:
  - low: `< 0.33`
  - medium: `0.33-0.66`
  - high: `> 0.66`
- wind bias vector:
  - `wx = wind_strength * cos(wind_dir_deg)`
  - `wy = wind_strength * sin(wind_dir_deg)`

Episode sampling rule:

- At reset, sample one snapshot-derived parameter record for the episode.
- Parameters remain fixed for the full episode in canonical runs.

## 9.3 Why this interface is chosen

- Keeps RL benchmark focused on tactical decision-making rather than end-to-end forecasting claims.
- Uses real ingested signals where available while preserving deterministic simulator reproducibility.
- Avoids modeling wind direction with XGBoost when it is already directly observed.

---

## 10) Data Pipeline Plan and Feasibility

## 10.1 Feasible now (already in codebase)

- NASA FIRMS
- CWFIS active fires
- Open-Meteo
- CFFDRS station data

## 10.2 Mandatory snapshot pipeline

```mermaid
flowchart TD
    A[NASA FIRMS] --> N[Normalize and Validate]
    B[CWFIS Active Fires] --> N
    C[Open-Meteo] --> N
    D[CFFDRS] --> N
    N --> S[Versioned Snapshot Cache]
    S --> X[XGBoost Feature Builder]
    X --> P[Env Parameter Records]
    P --> E[RL Scenario Generator]
```

Snapshot requirements:

- versioned file naming (date + schema version)
- schema validation at load
- no silent fallback defaults during benchmark runs (fail fast)

## 10.3 High-risk overclaimed sources (future work)

```mermaid
flowchart TD
    A[CIFFC reports]
    B[BC ArcGIS full ingestion]
    C[AB ArcGIS full ingestion]
    D[ECCC Datamart full integration]
    E[Historical spread label corpus]
    F[Production ETL scheduler]

    A --> Z[Future Work]
    B --> Z
    C --> Z
    D --> Z
    E --> Z
    F --> Z
```

---

## 11) Build Order (Updated)

1. Freeze objective, protocol numbers, held-out split.
2. Implement assets, budgets, cooldown semantics.
3. Implement wind-bias heterogeneity.
4. Implement mandatory benchmark harness/log schema/eval mode.
5. Implement scenario generator with frozen train/test families.
6. Implement snapshot cache loader and XGBoost-to-env parameter mapping.
7. Run reward sanity pass and freeze coefficients.
8. Run full multi-seed benchmarks for DQN/A2C/PPO + greedy/random.
9. Aggregate plots/tables and write limitations.

---

## 12) Out of Scope

- multi-agent coordination
- full GIS terrain integration
- historical replay validation
- complex dispatch logistics
- continuous-action routing

These remain future extensions.
