# Firebot: An Empirical Evaluation of RL Agents on Wildfire Environments  

This project is an empirical RL benchmark for wildfire tactical suppression. We compare DQN, A2C, PPO, and random and greedy baselines on a 25x25 grid environment with critical assets and finite suppression budgets.

The physics informed environment and built environment records from the [Alberta Historical Wildfires Database](https://open.alberta.ca/opendata/wildfire-data).

## Our Process  

We ingested data from the Alberta Historical Wildfires Database, then built environment snapshots with a seeded random initialization for starting positions of fires and firefighter agents. We performed our first training run on the dataset directly from the data pipeline with only schema validation, and evaluated results inside `notebooks/training_0_analysis.ipynb`; we then extensively did a data audit (`notebooks/data_audit.ipynb`), cleaned the data (`notebooks/clean_data.ipynb`) and re-ran the full training process again. For both runs, we carried out the same training process: 

1. Overfitting on a single batch to check model architecture 
2. Smoke training run + seed behavior checks 
3. Smoke test evaluation check 
4. hyperparameter sweep on the validation set 
5. Full 5-seed training (trained seeds 11, 22, 33, 44, 55) for each model with hyperparameters from 4

We then ran the notebooks for analysis inside `notebooks`, `notebooks/training_<NUMBER>_analysis.ipynb`.

We did not include the 24 total models trained for both training runs in this GitHub repo, but we included our analysis of the results and our process in notebooks. To see our process and the results, please refer to the setup below, and open the notebooks and the relevant plots inside `notebook` to review our results directly. We have also included instructions to run the full data ingestion pipeline and training pipeline for reproducing. For more details, please refer to `docs/` and the file tree below.

## Project Tree

```text
firebot/
├── README.md
├── pyproject.toml
├── uv.lock
├── ruff.toml
├── lefthook.yml
├── .env.example
├── .python-version
├── fp-historical-wildfire-data-dictionary-2006-2025.pdf # from the dataset download
├── .github/
│   └── workflows/
│       └── ci.yml
├── .ci-smoke/
│   ├── results.json
│   ├── scenario_parameter_records_seeded_train.json
│   ├── scenario_parameter_records_seeded_val.json
│   └── scenario_parameter_records_seeded_holdout.json
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
│   └── training.md
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
│       ├── train_rl_agent.py # unified PPO/A2C/DQN trainer with checkpoint and final evaluation outputs 
│       └── evaluate_agents.py # classdef for PPO/A2C/DQN plus greedy/random baselines
├── scripts/
│   ├── canary.py # deterministic smoke/repro canary checks for trained artifacts
│   ├── run_benchmark_train.sh # legacy all-in-one bash runner (staged scripts are canonical)
│   ├── run_benchmark_train.ps1 # powershell equivalent
│   ├── run_benchmark_eval.sh # bash script for post-training benchmark evaluation by seed
│   ├── run_benchmark_eval.ps1 # powershell equivalent
│   └── stages/
│       ├── _common.sh # shared defaults, validation, and helper functions for staged runs
│       ├── 01_karpathy_overfit.sh # one-record overfit checks
│       ├── 02_smoke_train_and_repro.sh # smoke training + reproducibility canary
│       ├── 03_smoke_eval.sh # smoke checkpoint artifact loading + sanity eval
│       ├── 04_pilot_sweep.sh # one-seed validation-only pilot hyperparameter sweep
│       └── 05_final_train.sh # canonical full 5-seed training with frozen protocol
├── tests/
│   ├── conftest.py # test environment configuration 
│   └── models/ # environment and benchmark metric contract tests
│       ├── test_fire_env_setup_contract.py # benchmark-mode env loading/split/schema contract tests
│       └── test_benchmarking_metrics.py # benchmark metric/preset/aggregation tests
├── notebooks/
│   ├── clean_data.ipynb # used for cleaning data 
│   ├── data_audit.ipynb # data analysis and checks on the data 
│   ├── training_0_analysis.ipynb # first training run analysis 
│   ├── training_1_analysis.ipynb # second training run analysis 
│   ├── final_results_table.png
│   ├── final_results_table_compact.png
│   └── training_val_plots.png
├── outputs/ # generated training and evaluation outputs (gitignored)
└── drd-archive/ # archived prototype code from the initial RL proposal before wildfire RL evaluation 
    ├── main.py
    └── src/
        ├── __init__.py
        ├── config.py
        ├── env.py 
        ├── evaluate.py
        ├── networks.py
        ├── ppo.py
        ├── train.py
        ├── utils.py
        └── viz.py
```

## Setup

This project requires the [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager to run. After installing, please do the following: 

1. clone the repo
2. in the project root, run: `uv venv && source .venv/bin/activate && uv sync`

To run the notebooks or view them using the uv virtual environment:

```bash
source .venv/bin/activate # make sure the venv is active 
cd notebooks
jupyter notebook 
```

### For Development: Pre-Commit Hooks 

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
uv run python -m src.ingestion.static_dataset --target-count 50000
```

Or the following command for the full explicit paths:

```bash
uv run python -m src.ingestion.static_dataset --target-count 50000 --output-dir data/static/v1 --raw-alberta-csv data/static/raw/fp-historical-wildfire-data-2006-2025.csv
```

With default paths specified inside the data pipeline code: `data/static/raw/` for where the source dataset is; `data/static/v1` for where the initially compiled environment parameter records from the ingested data are

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

For controlled and reproducible benchmark training, run the staged bash scripts in order.

Run from project root on macOS/Linux (bash):

```bash
./scripts/stages/01_karpathy_overfit.sh
./scripts/stages/02_smoke_train_and_repro.sh
./scripts/stages/03_smoke_eval.sh
./scripts/stages/04_pilot_sweep.sh
./scripts/stages/05_final_train.sh
```

Run from project root on Windows (PowerShell):

```powershell
./scripts/run_benchmark_train.ps1
```

Staged bash flow:

- Stage 1: single-record overfit checks (`ppo`, `a2c`, `dqn`)
- Stage 2: smoke training + reproducibility check  
- Stage 3: smoke evaluation sanity check
- Stage 4: validation-only pilot sweeps and winner selection
- Stage 5: full canonical 5-seed training using frozen protocol values

Optional configurable fields that can be overridden for training:

- `ARTIFACT_ROOT` (default `outputs/benchmark`)
- `SMOKE_TIMESTEPS` (default `20000`, one canonical checkpoint interval)
- `SMOKE_SEED` (default `11`)
- `SMOKE_EVAL_EPISODES` (default `5`)
- `RUN_REPRO_CANARY` (default `1`)
- `REPRO_CANARY_TOL` (default `1e-9`)
- `KARPATHY_TIMESTEPS` (default `10000`)
- `KARPATHY_SEED` (default `11`)
- `KARPATHY_FAMILY` (default `center,medium,A`)
- `KARPATHY_CHECKPOINT_EVAL_EPISODES` (default `1`)
- `PILOT_TIMESTEPS` (default `40000`)
- `PILOT_SEED` (default `11`)
- `RUN_KARPATHY_CHECK` (default `1`)
- `RUN_PILOT_SWEEP` (default `1`)
- `USE_PILOT_WINNERS` (default `1`)
- `FINAL_SEEDS_CSV` (default `11,22,33,44,55`)
- `ALGO_ORDER_CSV` (default `ppo,a2c,dqn`)

After Stage 5 completes, run benchmark evaluation wrappers with the commands below.

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

## Using Notebooks

The notebook details data analysis. After running, move `outputs/` to `training-outputs/` in the project root and launch the notebooks via `jupyter notebook` in `notebooks/`, to be able to run `notebooks/training_*_analysis.ipynb`. Run `data_audit.ipynb` after running the data pipeline once, and run `clean_data.ipynb` to reproduce data cleaning after ingestion. The training scripts inside `stages/` use the cleaned data for training.
