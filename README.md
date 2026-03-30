# FireGrid

Empirical RL benchmark for wildfire tactical suppression. Compares DQN, A2C, PPO, and heuristic baselines on a 25x25 grid environment with critical assets and finite suppression budgets.

The physics informed environment and built environment records from the [Alberta Historical Wildfires Database](https://open.alberta.ca/opendata/wildfire-data).

## Project Tree

```text
firebot/ 
├── README.md 
├── pyproject.toml 
├── uv.lock 
├── ruff.toml 
├── lefthook.yml 
├── fp-historical-wildfire-data-dictionary-2006-2025.pdf # from the dataset download
├── data/ 
│   └── static/ 
│       ├── fp-historical-wildfire-data-2006-2025.csv # raw Alberta historical wildfire CSV 
│       ├── snapshot_records.json # full normalized snapshot records from raw CSV
│       ├── snapshot_records_train.json # train-year snapshot subset
│       ├── snapshot_records_val.json # validation-year snapshot subset
│       ├── snapshot_records_holdout.json # holdout-year snapshot subset
│       ├── scenario_parameter_records.json # full unseeded environment parameter records
│       ├── scenario_parameter_records_train.json # train split unseeded records
│       ├── scenario_parameter_records_val.json # validation split unseeded records
│       ├── scenario_parameter_records_holdout.json # holdout split unseeded records
│       ├── scenario_parameter_records_seeded.json # full seeded records with ignition/layout seeds
│       ├── scenario_parameter_records_seeded_train.json # train runtime records
│       ├── scenario_parameter_records_seeded_val.json # validation runtime records
│       └── scenario_parameter_records_seeded_holdout.json # temporal holdout runtime records
├── docs/ 
│   ├── data-pipeline.md 
│   ├── envspec.md 
│   └── planning/ 
│       ├── env-checklist.md 
│       ├── impl-plan.md 
│       └── train-plan.md 
├── src/ 
│   ├── __init__.py 
│   ├── ingestion/ 
│   │   ├── __init__.py 
│   │   ├── clean_historical.py # row cleaning and required-field checks
│   │   ├── cffdrs.py # CFFDRS station ingestion, not used 
│   │   ├── weather.py # legacy Open-Meteo weather fetch helpers, not used 
│   │   └── static_dataset.py # builds snapshot/scenario parameter records in data/static
│   └── models/ # environment, training, evaluation, and shared benchmark utilities
│       ├── __init__.py 
│       ├── fire_env.py # WildfireEnv implementation and benchmark env construction helpers
│       ├── benchmarking.py # shared benchmark presets, rollout metrics, and aggregation functions
│       ├── train_rl_agent.py # unified PPO/A2C/DQN trainer with checkpoint and final evaluation artifacts
│       └── evaluate_agents.py # classdef for PPO/A2C/DQN plus greedy/random baselines
├── scripts/ 
│   ├── run_benchmark_train.sh # bash script for smoke validation then full 5-seed benchmark training
│   ├── run_benchmark_train.ps1 # powershell equivalent 
│   ├── run_benchmark_eval.sh # bash script for post-training benchmark evaluation by seed
│   └── run_benchmark_eval.ps1 # powershell equivalent 
├── tests/ 
│   ├── conftest.py 
│   └── models/ # environment and benchmark metric contract tests
│       ├── test_fire_env_setup_contract.py # benchmark-mode env loading/split/schema contract tests
│       └── test_benchmarking_metrics.py # benchmark metric/preset/aggregation tests
├── outputs/ # generated training and evaluation artifacts (gitignored)
└── drd-archive/ # archived prototype code from the earlier DRD proposal
```

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
    - `wind_direction` (8-direction string)
    - `wind_strength`
    - `ignition_seed`
    - `layout_seed`
- writes seeded benchmark variants (`scenario_parameter_records_seeded.json` and `scenario_parameter_records_seeded_{train|val|holdout}.json`) for reproducible initialization; holdout seeded export is currently a single unique held-out record.
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

This command builds data from the CSV and generates initialization seeds for ignition and asset layout for the corresponding environment. CFFDRS was not used to reduce confounding variables and any bias introduced due to incomplete CFFDRS data ingested for some specific fires.

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

For controlled and reproducible benchmark training, use the script wrappers in `scripts/`.

Run from project root on macOS/Linux (bash):

```bash
./scripts/run_benchmark_train.sh
```

Run from project root on Windows (PowerShell):

```powershell
./scripts/run_benchmark_train.ps1
```

Script runs:

- Stage 1 (smoke): runs short validation training for `ppo`, `a2c`, `dqn` on one seed
- Stage 2 (smoke eval): loads smoke `best_model.zip` artifacts and runs evaluator sanity checks
- Stage 3 (formal): runs full canonical training for all three algorithms across 5 seeds (`11,22,33,44,55`)
- Uses artifact root `outputs/benchmark/` and keeps default trainer settings for env count, timesteps, and checkpoint cadence on formal runs

Training script environment overrides (optional):

- `ARTIFACT_ROOT` (default `outputs/benchmark`)
- `SMOKE_TIMESTEPS` (default `20000`, one canonical checkpoint interval)
- `SMOKE_SEED` (default `11`)
- `SMOKE_EVAL_EPISODES` (default `5`)
- `FINAL_SEEDS_CSV` (default `11,22,33,44,55`)
- `ALGO_ORDER_CSV` (default `ppo,a2c,dqn`)

After `run_benchmark_train` completes, run benchmark evaluation wrappers.

Run from project root on macOS/Linux (bash):

```bash
./scripts/run_benchmark_eval.sh
```

Run from project root on Windows (PowerShell):

```powershell
./scripts/run_benchmark_eval.ps1
```

These are the default values that can be overridden via env ars or editing the `ps1` and `.sh` scripts.

- `ARTIFACT_ROOT` (default `outputs/benchmark`)
- `RUN_LABEL` (default `final`)
- `EVAL_SEEDS_CSV` (default `11,22,33,44,55`)
- `EVAL_EPISODES` (default `100`)
- `AGENTS` (default `ppo,a2c,dqn,greedy,random`)
- `OUTPUT_DIR` (default `outputs/benchmark/<run_label>/eval`)
- `INCLUDE_FAMILY_HOLDOUT` (`0` or `1`, default `0`)
- `INCLUDE_TEMPORAL_HOLDOUT` (`0` or `1`, default `0`)
- `NO_NORMALIZED_BURN` (`0` or `1`, default `0`)

The seeded scenario parameter files are the benchmark inputs for `FireEnv` training and script-driven evaluation.

The builder also writes year-based split files for the benchmark:

- `train`: `2006-2022`
- `val`: `2023`
- `holdout`: `2024-2025`

The dataset builder prints cleaning/drop summaries to stdout and uses progress bars when `tqdm` is available.
