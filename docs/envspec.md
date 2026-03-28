# Fire Environment Specification (Current Implementation)

This document describes the currently implemented RL environment in `src/models/fire_env.py` and its training/evaluation usage in `src/models/train_rl_agent.py` and `src/models/evaluate_agents.py`.

It is intentionally code-first and only documents behavior that exists in the current codebase.

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

## 7) Training Process (Current Code)

Current training script:

- `src/models/train_rl_agent.py`
- algorithm currently implemented in this script: `PPO` (Stable-Baselines3)

Canonical training flow:

1. load seeded train split dataset
2. create vectorized benchmark envs (`n_envs`)
3. train PPO for configured timesteps
4. save model to `src/models/tactical_ppo_agent.zip`
5. run quick evaluation on train and optional val/holdout datasets

Current benchmark evaluation script:

- `src/models/evaluate_agents.py`
- evaluates agents across splits (`train`, `val`, `holdout`)
- supported evaluated agents: `ppo`, `greedy`, `random`
- can output JSON summary via `--output`

Transparency outputs from current code:

- training console output: timesteps, env count, dataset path/count, quick split metrics
- model artifact: `tactical_ppo_agent.zip`
- evaluation console summary per split/agent
- optional evaluation JSON with aggregate metrics

Recommended transparency plots (from saved eval JSON/logs):

- split-wise mean return (`train` vs `val` vs `holdout`)
- split-wise asset survival and containment rates
- final burned area distribution by split/agent
- seed variability/error bars for key metrics

---

## 8) Reporting Metrics 

Primary optimization target/what the agent is trained to do: **Minimize assets damaged/lost**

Additional reported metrics (already computed or directly derivable from current eval):

- mean episodic return
- standard deviation / variance across episodes
- asset survival rate
- containment success rate
- mean final burned area
- mean time to containment
- mean resource efficiency
- mean normalized burn ratio (optional in evaluator)

Report these diagnostics during training:

- train/val/holdout gap for each metric
- per-seed summary tables
- baseline comparisons (`greedy`, `random`) against PPO

