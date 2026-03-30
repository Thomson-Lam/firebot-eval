# Fire Environment Specification

This document describes the RL environment in `src/models/fire_env.py` and its training/evaluation usage in `src/models/train_rl_agent.py` and `src/models/evaluate_agents.py`.

It remains code-first for implemented behavior, but it also records the benchmark-alignment requirements that training and evaluation code must satisfy before full canonical runs are launched.

The concrete benchmark execution plan lives in `docs/planning/train-plan.md`.

---

## 1) Environment Definition

`WildfireEnv` is a single-agent, discrete-action tactical wildfire suppression environment.

- Agent count: `1`
- Grid size: `25 x 25` (`625` cells)
- Episode horizon: `150` steps
- Action space: `6` discrete actions (`MOVE_N`, `MOVE_S`, `MOVE_E`, `MOVE_W`, `DEPLOY_HELICOPTER`, `DEPLOY_CREW`)
- Fire process: one evolving fire field per episode (multiple burning cells can exist simultaneously)

Cell encodings:

- `0`: unburned
- `1`: burning
- `2`: burned
- `3`: suppressed
- `4`: critical asset
- `5`: critical asset burned/lost

Per-episode resource limits:

- helicopter budget: `8`, cooldown: `5`
- crew budget: `20`, cooldown: `2`

About "how many states": this environment has a very large combinatorial state space (not a small enumerable finite-state MDP in practice). The policy input vector shape is fixed at `636`.

---

## 2) How the Environment Gets and Uses Data from the Pipeline

Canonical benchmark mode consumes seeded split scenario records produced by the data pipeline:

- `data/static/scenario_parameter_records_seeded_train.json`
- `data/static/scenario_parameter_records_seeded_val.json`
- `data/static/scenario_parameter_records_seeded_holdout.json`

Loader and integration flow:

1. `load_scenario_parameter_records(...)` validates each record.
2. `create_benchmark_env(...)` creates `WildfireEnv` with strict benchmark settings.
3. `reset()` samples one cached record and maps it to `ScenarioConfig`.
4. The sampled record stays fixed for that episode.

Required benchmark fields:

- `record_id`, `split`
- `base_spread_prob`, `severity_bucket`
- `wind_direction` (8-direction string), `wind_strength`
- `ignition_seed`, `layout_seed`

How fields are used in env runtime:

- `base_spread_prob`: baseline spread probability
- `severity_bucket`: severity one-hot in observation
- `wind_direction` + `wind_strength`: wind-bias vector for spread
- `ignition_seed` / `layout_seed`: reproducible initialization RNGs for ignition and asset placement

Important boundary:

- ignition family and asset layout labels remain simulator-side controls
- seeded records do not store explicit ignition/layout labels
- seeds make simulator-side initialization reproducible

---

## 3) Implementation Details (Variable Updates and Core Functions)

Record loading and benchmark setup:

- `load_scenario_parameter_records`: schema/range/split validation
- `benchmark_env_kwargs`, `create_benchmark_env`: canonical benchmark env factory
- `scenario_from_parameter_record`: maps one record into `ScenarioConfig`

Episode construction and parameter sampling:

- `reset`: samples scenario family + parameter record, resets budgets/cooldowns/state, places assets, ignites fire
- `_sample_parameter_record`: seed-stable shuffled sampling over loaded records
- `_configure_initialization_rngs`: configures ignition/layout RNGs from record seeds

State transition internals:

- `_execute_action`: movement/suppression effects + immediate action reward terms
- `_spread_fire`: wind-biased stochastic spread + asset-loss accounting + burnout
- `_ignite`: ignition pattern initialization (`center`, `edge`, `corner`, `multi_cluster`)
- `_place_assets`: layout initialization (`A`, `B`)

Observation assembly:

- `_get_obs`: builds the flat `636`-length policy input vector

---

## 4) Observations (What the Agent Sees)

The policy receives a single flat vector with shape `636`:

1. flattened grid: `25 * 25 = 625`
2. agent position (normalized row, col): `2`
3. resources/cooldowns (normalized): `4`
   - `heli_left`, `crew_left`, `heli_cd`, `crew_cd`
4. severity one-hot: `3`
5. wind bias vector: `2` (`wx`, `wy`)

Total: `625 + 2 + 4 + 3 + 2 = 636`

The environment returns `obs, reward, terminated, truncated, info` at each step.

---

## 5) Fire Dynamics and Transition 

Per-step order:

1. decrement cooldowns
2. execute action
3. apply resource-use penalties (`-1.5` heli, `-0.5` crew when used)
4. advance spread via `_spread_fire`
5. apply asset-loss and burn-growth penalties
6. evaluate termination/truncation and terminal bonuses

Spread rule:

- base spread: `base_spread_prob`
- wind adjustment: `base + 0.15 * wind_dot`
- clipped spread probability: `[0.01, 0.95]`
- neighborhood: 4-connected (`N/S/E/W`)
- burning cells have `0.05` burnout chance each step

Wind handling:

- `wind_direction` is discrete 8-direction (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`)
- each direction maps to a unit/directional vector, scaled by `wind_strength`

Termination:

- `terminated=True` when no burning cells remain
- `truncated=True` when step count reaches `150`

---

## 6) The Reward Function 

Reward is assembled from multiple terms in `step()` and `_execute_action()`.

Main terms:

- `-75.0 * asset_cells_lost`
- `-0.4 * new_burned` where `new_burned = max(0, burning_now - prev_burning)`
- helicopter use cost: `-1.5`
- crew use cost: `-0.5`
- wasted/blocked deployment penalty: `-1.0`
- suppression bonuses:
  - helicopter: `+3.0` per suppressed affected cell in `3x3`
  - crew on burning cell: `+3.0`
  - crew firebreak on unburned cell: `+2.0`

Terminal shaping:

- `+100` if fire is extinguished and assets lost is `0`
- `+40` if episode ends (terminated or truncated) with assets lost still `0`

Optimization intent:

- primary objective is minimizing asset damage/loss
- other terms provide dense tactical shaping for learning stability

---

## 7) Training Process

Current training script:

- `src/models/train_rl_agent.py`
- currently implemented learned method: `PPO` (Stable-Baselines3)

Current implemented flow:

1. load seeded train split dataset
2. create vectorized benchmark envs (`n_envs`)
3. train PPO for configured timesteps
4. save model to `src/models/tactical_ppo_agent.zip`
5. run quick evaluation on train and optional val/holdout datasets

Benchmark-aligned target flow:

1. use a unified runner with `--algo {ppo,a2c,dqn}`
2. keep the same benchmark-mode dataset path and split validation for all methods
3. use vectorized envs for `PPO` and `A2C`
4. use a single benchmark env for `DQN` by default
5. write checkpoint metrics every fixed training interval
6. save per-run config and per-checkpoint metrics to disk
7. choose the best checkpoint by validation `asset_survival_rate`
8. run final split-wise evaluation on train/val/holdout after training completes

Current benchmark evaluation script:

- `src/models/evaluate_agents.py`
- evaluates agents across splits (`train`, `val`, `holdout`)
- currently implemented evaluated agents: `ppo`, `greedy`, `random`
- can output JSON summary via `--output`

Benchmark-aligned target support:

- `ppo`
- `a2c`
- `dqn`
- `greedy`
- `random`

Transparency outputs from current code:

- training console output: timesteps, env count, dataset path/count, quick split metrics
- model artifact: `tactical_ppo_agent.zip`
- evaluation console summary per split/agent
- optional evaluation JSON with aggregate metrics

Required benchmark transparency outputs:

- serialized run config per seed
- checkpoint metrics on train/val/holdout
- best-checkpoint selection record
- final evaluation JSON aggregated by seed, then across seeds

Recommended transparency plots (from saved eval JSON/logs):

- split-wise mean return (`train` vs `val` vs `holdout`)
- split-wise asset survival and containment rates
- final burned area distribution by split/agent
- seed variability/error bars for key metrics

---

## 8) Reporting Metrics

Primary optimization target: **Minimize assets damaged/lost**

Frozen benchmark metric definitions:

- mean episodic return
- asset survival rate
- containment success rate
- mean burned-area fraction: `(burned + burning + asset_burned) / 625`
- mean time to containment, conditioned on successful containment only
- mean resource efficiency: `successful_deployments / total_deployments`
- standard deviation across seeds for each reported metric
- wasted deployment rate
- mean normalized burn ratio (optional in evaluator)

Important alignment notes:

- pooled episode variance is not a substitute for seed-level aggregation
- raw burned-cell count can still be logged, but the benchmark-facing metric should be the normalized burned-area fraction
- holdout performance is for final reporting only and must not be used for tuning
- the current temporal holdout dataset has only one unique seeded record, so it is a final diagnostic only until expanded

Report these diagnostics during training checkpoints:

- train/val gap for each metric
- optional train/family-holdout gap for each metric
- per-seed summary tables
- baseline comparisons (`greedy`, `random`) against learned methods

## 9) Environment-Side Requirements For Benchmark Alignment

The environment and evaluator together must expose enough information to compute the benchmark metrics exactly.

Environment-side counters or `info` fields required for clean evaluation:

- `assets_lost`
- `step`
- `heli_left`
- `crew_left`
- count of successful helicopter deployments
- count of successful crew deployments
- count of wasted deployment attempts
- count of total deployment attempts

Operational metric rules:

- `mean_resource_efficiency = successful_deployments / total_deployments`
- if `total_deployments == 0`, report `0.0`
- `wasted_deployment_rate = wasted_deployments / total_deployment_attempts`
- if `total_deployment_attempts == 0`, report `0.0`

Evaluator-side aggregation requirements:

- aggregate per seed first, then summarize across seeds
- compute `time_to_containment` only on contained episodes
- compute normalized burn ratio against the same scenario record and evaluation seed under the deterministic non-intervention baseline defined in `docs/planning/train-plan.md`
- do not surface temporal holdout metrics during checkpoint evaluation in canonical runs
- pass `scenario_families` explicitly for canonical train, validation, and family-holdout runs rather than relying on the environment default

Verification requirement before full runs:

- all of the above metrics must appear in a short smoke-test evaluation artifact before any full 5-seed training sweep is launched
