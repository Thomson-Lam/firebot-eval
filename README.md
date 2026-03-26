# FireGrid

Empirical RL benchmark for wildfire tactical suppression. Compares DQN, A2C, PPO, and heuristic baselines on a 25x25 grid environment with critical assets and finite suppression budgets.

## Setup

```bash
uv sync
```

### Pre-commit hooks (optional)

Install [lefthook](https://github.com/evilmartians/lefthook) for local lint/format checks on commit:

```bash
# pick one
brew install lefthook
npm i -g lefthook

# then wire it up
lefthook install
```

## Usage

```bash
# Train PPO agent (200k steps)
uv run python -m src.models.train_rl_agent

# Quick test (10k steps)
uv run python -m src.models.train_rl_agent --timesteps 10000

# Train XGBoost spread model
uv run python -m src.models.spread_model
```

## Data Pipeline 

Build the static dataset at `src/ingestion/static_dataset.py`. The script:

- collects candidate fire records once
- enriches them with weather and CFFDRS fields
- writes frozen `snapshot_records.json`
- computes offline environment variables and write `scenario_parameter_records.json`

Usage:

```bash
uv run python -m src.ingestion.static_dataset --target-count 100
```

Optional usage with a precollected historical fire record file:

```bash
uv run python -m src.ingestion.static_dataset --fire-records path/to/fire_records.json --target-count 100
```

The cached scenario parameter file can then be consumed by `FireEnv` and PPO training.
