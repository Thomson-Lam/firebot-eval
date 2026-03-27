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

The canonical benchmark pipeline now uses the Alberta historical wildfire dataset in `data/static/` as its primary source.

Source roles:

- Alberta historical wildfire dataset: primary incident, weather, spread-rate, and assessment-time source
- CFFDRS: optional supplementary fire-danger enrichment
- CWFIS and FIRMS: retained in the repo for legacy/live experiments, not part of the canonical build path

Refer to `docs/data-pipeline.md` for exact fields and data we ingest.

We build the static dataset at `src/ingestion/static_dataset.py`. The script:

- loads historical incident rows from `data/static/fp-historical-wildfire-data-2006-2025.csv`
- normalizes them into snapshot records anchored at assessment time
- optionally enriches them with CFFDRS fields when `--cffdrs-year` is provided and usable
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
Alberta historical wildfire CSV -> normalized snapshot records -> scenario parameter records.
```

Each record is anchored at `ASSESSMENT_DATETIME`. The builder uses observed spread rate, assessment weather, incident size, fire type, and fuel type to compute benchmark environment variables. If `--cffdrs-year` is passed and a usable station file exists, the builder also joins supplementary CFFDRS danger indices by both distance and date.

```
fire record -> snapshot record (`data/static/snapshot_records.json`) -> scenario (environment) parameter record (`data/static/scenario_parameter_records.json`).
```

For more details, check `docs/data-pipeline.md` 

### Usage from project root

Default usage from the Alberta historical CSV:

```bash
uv run python -m src.ingestion.static_dataset --target-count 100
```

`--target-count 100` means up to `100` records per split (`train`, `val`, `holdout`).

With optional supplementary CFFDRS enrichment:

```bash
uv run python -m src.ingestion.static_dataset --target-count 100 --cffdrs-year 2025
```

With a custom raw Alberta CSV path:

```bash
uv run python -m src.ingestion.static_dataset --raw-alberta-csv path/to/fp-historical-wildfire-data.csv --target-count 100
```

If you have your own normalized historical fire records JSON:

```bash
uv run python -m src.ingestion.static_dataset --fire-records path/to/fire_records.json --target-count 100
```

Notes:

- The canonical build no longer uses FIRMS or Open-Meteo.
- CFFDRS is supplementary. If the requested year is sparse or unavailable, the builder still works without it.
- The raw Alberta CSV already contains the main weather and spread fields used for the benchmark.

After building the dataset, you can train by running:

```bash
uv run python -m src.models.train_rl_agent --scenario-dataset data/static/scenario_parameter_records.json
```

The cached scenario parameter file can then be consumed by `FireEnv` and PPO training.

The builder also writes year-based split files for the benchmark:

- `train`: `2006-2022`
- `val`: `2023`
- `holdout`: `2024-2025`

Recommended training command:

```bash
uv run python -m src.models.train_rl_agent --scenario-dataset data/static/scenario_parameter_records_train.json --val-dataset data/static/scenario_parameter_records_val.json --holdout-dataset data/static/scenario_parameter_records_holdout.json
```
