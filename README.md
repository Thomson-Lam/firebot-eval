# FireGrid

Empirical RL benchmark for wildfire tactical suppression. Compares DQN, A2C, PPO, and heuristic baselines on a 25x25 grid environment with critical assets and finite suppression budgets.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) 

1. clone the repo
2. in the project root, run: `uv venv && source .venv/bin/activate && uv sync`

### Pre-commit hooks 

Pre-commit hooks were used for the project for linting and checks. Install [lefthook](https://github.com/evilmartians/lefthook) for local lint/format checks on commit:

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
```

## Data Pipeline 

We ingest data from the following sources:

- CWFIS: the primary source of wildfire incidents 
- FIRMS: supplements CWFIS data with hotspot sources 
- CFFDRS: fire danger levels and dryness context

Refer to `docs/data-pipeline.md` for exact fields and data we ingest.

We build the static dataset at `src/ingestion/static_dataset.py`. The script:

- collects candidate fire records once
- enriches them with weather and CFFDRS fields
- writes frozen and normalized `snapshot_records.json` of snapshot records from the data pipeline inside `data/static`
- computes offline environment variables and write `scenario_parameter_records.json` in `data/static`. The environment variables written are:
    - `base_spread_prob`
    - `severity_bucket`
    - `wind_dir_deg`
    - `wind_strength`
- With the following extra fields stored:
    - `spread_rate_1h_m`
    - `spread_score`
    - `dryness_score`
    - `rh_factor`
    - `rain_factor`
    - `temp_factor`
    - `wind_factor`
    - `size_factor`
    - `record_quality_flag`

> NOTE: the stored extra fields are for checking whether the data pipeline is computing the primary metrics correctly, and checking why a record got a high/low spread setting. These variables' influence and effect have been collapsed into `based_spread_prob`, `severity_bucket` and `wind_strength`. They are not included because we want to reduce the amount of confounding variables and keep the initial environment design as simple as possible; and to reduce chances of data leakage and models overfitting.

For future improvements, consider using `dryness_score` to influence base burnout probability, `rain_factor` to damp spread for the whole episode, and `size_factor` if we think and agree that the incident size should affect the spread dynamics. However, these are arbitrary rates computed based on heuristics and introduce diminishing returns with a limited realistic environment. For simplicity, these will not be included.

Check `docs/data-pipeline.md` for how these variables are computed.

### How data is collected 

```
CWFIS incident records -> optional FIRMS hotspot supplementation -> candidate fire records.
```

Fire candidates are deduplicated by `fire_id` and coarse latitude/longitude and province, then sorted so CWFIS records with measured `area_hectares` are preferred. For each selected fire, `static_dataset.py` fetches weather from Open-Meteo and matches the nearest CFFDRS station by both distance and snapshot date. Then a normalized snapshot record is built and the environment parameters are computed from the snapshot record:

```
fire record -> snapshot record (`data/static/snapshot_records.json`) -> scenario (environment) parameter record (`data/static/scenario_parameter_records.json`).
```

For more details, check `docs/data-pipeline.md` 

### Usage from project root

Default usage without FIRMS data and only CWFIS data:

```bash
uv run python -m src.ingestion.static_dataset --target-count 100
```

With FIRMS:

```bash
uv run python -m src.ingestion.static_dataset --target-count 100 --include-firms 
```

With a fixed CFFDRS year for a historical record file that already contains matching snapshot dates:

```bash
uv run python -m src.ingestion.static_dataset --fire-records path/to/fire_records.json --target-count 100 --cffdrs-year 2024
```

If you have your own fire records file:

```bash
uv run python -m src.ingestion.static_dataset --fire-records path/to/fire_records.json --target-count 100
```

Notes:

- Do not hard-code `--cffdrs-year 2025` for live builds. The currently available 2025 CFFDRS station file may contain no usable danger-index values, which produces zero records.
- For live builds, prefer omitting `--cffdrs-year` and using the builder only when the current season has populated CFFDRS observations.
- For reproducible historical builds, use `--fire-records` with a curated historical file and a CFFDRS year known to contain populated station observations.

After building the dataset, you can train by running:

```bash
uv run python -m src.models.train_rl_agent --scenario-dataset data/static/scenario_parameter_records.json
```

The cached scenario parameter file can then be consumed by `FireEnv` and PPO training.
