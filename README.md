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

## Usage: Data Pipeline, Training and Eval 

The data pipeline now uses the Alberta historical wildfire dataset in `data/static/` as its primary source.

Data sources:

- Alberta historical wildfire dataset: primary incident, weather, spread-rate, and assessment-time source
- CFFDRS: optional supplementary fire-danger enrichment
- CWFIS and FIRMS: retained in the repo for legacy/live experiments, not part of the canonical build path

Refer to `docs/data-pipeline.md` for exact fields and data we ingest.

We build the static dataset at `src/ingestion/static_dataset.py`. The script:

- loads historical incident rows from `data/static/fp-historical-wildfire-data-2006-2025.csv`
- normalizes them into snapshot records anchored at assessment time
- applies lightweight cleaning (strip blank strings and drop unusable rows with missing required assessment-time fields)
- optionally enriches with CFFDRS fields when `--cffdrs-year` is provided and usable
- writes frozen and normalized `snapshot_records.json` and split snapshot files in `data/static`
- computes offline environment variables and writes `scenario_parameter_records.json` plus split files in `data/static`. The environment variables written are:
    - `base_spread_prob`
    - `severity_bucket`
    - `wind_dir_deg`
    - `wind_strength`
- With the following extra fields stored:
    - `spread_rate_1h_m`
    - `spread_score`
    - `weather_score`
    - `cffdrs_dryness_score`
    - `rain_factor`
    - `size_factor`
    - `fire_type_factor`
    - `fuel_factor`
    - `observed_spread_rate_m_min`
    - `assessment_hectares`
    - `fire_type`
    - `fuel_type`
    - `record_quality_flag`

> NOTE: the stored extra fields are for checking whether the data pipeline is computing the primary metrics correctly, and checking why a record got a high/low spread setting. Their influence has already been collapsed into `base_spread_prob`, `severity_bucket`, and `wind_strength` for the canonical environment. They are not directly consumed by the current `FireEnv` dynamics to keep the initial benchmark simple and reduce overfitting/confounding risk.

For future improvements, consider using `cffdrs_dryness_score` to influence burnout probability, `rain_factor` to damp spread for the whole episode, and `size_factor` if we agree incident size should affect spread dynamics. For now, these remain audit fields rather than direct transition inputs.

Check `docs/data-pipeline.md` for how these variables are computed.


### How data is collected 

```
Alberta historical wildfire CSV -> normalized snapshot records -> scenario parameter records.
```

Each record is anchored at `ASSESSMENT_DATETIME`. The builder uses observed spread rate, assessment weather, incident size, fire type, and fuel type to compute benchmark environment variables. If `--cffdrs-year` is passed and a usable station file exists, the builder also joins supplementary CFFDRS danger indices by both distance and date.

```
fire record -> snapshot record (`data/static/snapshot_records.json`) -> scenario (environment) parameter record (`data/static/scenario_parameter_records.json`).
```

- CFFDRS is supplementary. If the requested year is sparse or unavailable, the builder still works without it.
- The raw Alberta CSV already contains the main weather and spread fields used for the benchmark.

For more details, check `docs/data-pipeline.md` 

### Usage from project root

We run this command run to ingest our dataset (with a large cap to avoid split truncation):

```bash
uv run python -m src.ingestion.static_dataset --target-count 50000 --raw-alberta-csv data/static/fp-historical-wildfire-data-2006-2025.csv
```

If CFFDRS for the selected year is sparse, the builder still runs and writes records without supplementary CFFDRS enrichment.

Optionally, test with a smaller target count:

```bash
uv run python -m src.ingestion.static_dataset --target-count 100 --cffdrs-year 2025 --raw-alberta-csv data/static/fp-historical-wildfire-data-2006-2025.csv
```

If you have your own normalized historical fire records JSON:

```bash
uv run python -m src.ingestion.static_dataset --fire-records path/to/fire_records.json --target-count 100
```

### Training 

After building the dataset, you can train by running:

```bash
uv run python -m src.models.train_rl_agent --scenario-dataset data/static/scenario_parameter_records_train.json --val-dataset data/static/scenario_parameter_records_val.json --holdout-dataset data/static/scenario_parameter_records_holdout.json
```

The scenario parameter file can then be consumed by `FireEnv` and PPO training.

The builder also writes year-based split files for the benchmark:

- `train`: `2006-2022`
- `val`: `2023`
- `holdout`: `2024-2025`

Training command:

```bash
uv run python -m src.models.train_rl_agent --scenario-dataset data/static/scenario_parameter_records_train.json --val-dataset data/static/scenario_parameter_records_val.json --holdout-dataset data/static/scenario_parameter_records_holdout.json
```

General split benchmark evaluation (PPO + baselines):

```bash
uv run python -m src.models.evaluate_agents --agents ppo,greedy,random --train-dataset data/static/scenario_parameter_records_train.json --val-dataset data/static/scenario_parameter_records_val.json --holdout-dataset data/static/scenario_parameter_records_holdout.json --episodes 20 --seeds 42,43,44
```

The dataset builder prints cleaning/drop summaries to stdout and uses progress bars when `tqdm` is available.
