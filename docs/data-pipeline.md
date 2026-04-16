# Data Pipeline

This document describes the current benchmark data pipeline for building `FireEnv` datasets.

We use the Alberta historical wildfire dataset stored under `data/static/` as the primary source for building `FireEnv` scenario records. While we originally included NASA FIRMS API and CFFDRS API for ingestion of data, we removed them to not introduce bias due to the fact that not all fires have the same consistent data across all 3 sources; some environments after being made would miss CFFDRS data because there are no stations nearby where the fire occurred to sample data from.

---

## 1) Overview

Our pipeline consists of two stages:

1. normalize historical wildfire incidents into frozen snapshot records
2. compute offline environment-variable records for `FireEnv`

When reproducing or re-running, you should use the seeded split parameter datasets (`scenario_parameter_records_seeded_{train|val|holdout}.json`), the outputs of the data pipeline.

---

## 2) Input Data Sources

### 2.1 Alberta historical wildfire dataset

Raw file path:

- `data/static/fp-historical-wildfire-data-2006-2025.csv`

This dataset is now the default primary input to `src/ingestion/static_dataset.py`.

The following fields are used directly by the pipeline:

- incident identity: `YEAR`, `FIRE_NUMBER`, `FIRE_NAME`
- location: `LATITUDE`, `LONGITUDE`
- timing: `FIRE_START_DATE`, `ASSESSMENT_DATETIME`, `DISCOVERED_DATE`, `REPORTED_DATE`, `DISPATCH_DATE`, `IA_ARRIVAL_AT_FIRE_DATE`, `FIRE_FIGHTING_START_DATE`
- fire state: `ASSESSMENT_HECTARES`, `CURRENT_SIZE`, `SIZE_CLASS`
- spread/weather: `FIRE_SPREAD_RATE`, `TEMPERATURE`, `RELATIVE_HUMIDITY`, `WIND_DIRECTION`, `WIND_SPEED`, `WEATHER_CONDITIONS_OVER_FIRE`
- fire context: `FIRE_TYPE`, `FUEL_TYPE`, `FIRE_POSITION_ON_SLOPE`, `FIRE_ORIGIN`
- optional response context: `INITIAL_ACTION_BY`, `IA_ACCESS`, `BUCKETING_ON_FIRE`, `DISTANCE_FROM_WATER_SOURCE`

### 2.2 `src/ingestion/cffdrs.py`

This module downloads annual CWFIS weather-station CSV data and parses:

- `fwi`
- `isi`
- `bui`
- `dc`
- `dmc`
- `ffmc`

We used this module for:

- supplementary enrichment only
- if `--cffdrs-year` is passed and usable observations exist, the builder joins the nearest station by distance, with date alignment to each fire snapshot (`max_date_offset_days=1`)
- the benchmark does not require CFFDRS availability to build records
- practical implication: one run fetches a single annual station file, so date-matched enrichment is usually concentrated in that selected year

We did not use CFFDRS for the environment design for the final benchmark evaluation presented.

### 2.3 `src/ingestion/weather.py`

This module fetches current-hour weather from Open-Meteo.

We did not use this for the environment design for the final benchmark evaluation presented. 

---

## 3) Build Flow of the Data Pipeline 

```text
Alberta historical wildfire CSV
-> normalized historical fire records
-> optional CFFDRS date-and-distance enrichment
-> snapshot_records.json
-> offline env-variable builder
-> scenario_parameter_records.json (unseeded build artifact)
-> scenario_parameter_records_seeded_{split}.json (benchmark runtime artifact)
-> FireEnv reset sampling
```

### 3.1 Cleaning and vetting specification

Data cleaning in the data pipeline is implemented in `src/ingestion/clean_historical.py`.

We did the following: 

- strip leading/trailing whitespace from all string fields
- convert blank strings to `null`
- drop rows missing any required core field:
  - `YEAR`
  - `FIRE_NUMBER`
  - `LATITUDE`
  - `LONGITUDE`
  - `ASSESSMENT_DATETIME`
  - `FIRE_SPREAD_RATE`
  - `TEMPERATURE`
  - `RELATIVE_HUMIDITY`
  - `WIND_DIRECTION`
  - `WIND_SPEED`
- drop rows where both size fields are missing:
  - `ASSESSMENT_HECTARES`
  - `CURRENT_SIZE`

Normalization-time vetting in `src/ingestion/static_dataset.py` additionally drops rows that fail parsing or mapping, such as invalid datetimes, non-numeric required values, unresolved wind direction values, or years outside the frozen split strategy.

### 3.2 Candidate Fire Selection and Truncation 

After normalization, candidate fires are selected in this order:

- deduplicate by `fire_id` (keep the first occurrence encountered for each fire id)
- rank remaining candidates by descending `(observed_spread_rate_m_min, assessment_hectares/area_hectares, year, fire_id)`
- apply `--target-count` as a per-split cap (`train`, `val`, `holdout`)

This means `--target-count 100` exports up to `100` records per split, not `100` total.

The data pipeline outputs to `stdout`: 

- total rows, kept rows, dropped rows
- top drop reasons (for example `missing_fire_spread_rate`, `normalization_failed`)
- per-year kept/total counts
- per-split built record counts

---

## 4) Snapshot Schema

The builder writes `data/static/snapshot_records.json`.

It also writes per-split files using the frozen year strategy:

- `train`: `2006-2022`
- `val`: `2023`
- `holdout`: `2024-2025`

Generated split files:

- `data/static/snapshot_records_train.json`
- `data/static/snapshot_records_val.json`
- `data/static/snapshot_records_holdout.json`
- `data/static/scenario_parameter_records_train.json`
- `data/static/scenario_parameter_records_val.json`
- `data/static/scenario_parameter_records_holdout.json`
- `data/static/scenario_parameter_records_seeded.json`
- `data/static/scenario_parameter_records_seeded_train.json`
- `data/static/scenario_parameter_records_seeded_val.json`
- `data/static/scenario_parameter_records_seeded_holdout.json`

Seeded parameter files include deterministic `ignition_seed` and `layout_seed` for reproducible environment initialization.
For the current benchmark setup, `scenario_parameter_records_seeded_holdout.json` is intentionally reduced to one unique held-out record.
In benchmark mode, `FireEnv` expects these seed fields to be present on all loaded records.

Each snapshot record represents one selected Alberta wildfire incident row after deduplication/ranking.

Core stored fields:

- identity: `record_id`, `fire_id`, `year`, `name`, `province`, `source`
- timing: `snapshot_date`, `snapshot_datetime`, `started_at`, `updated_at`
- location: `latitude`, `longitude`
- size: `area_hectares`, `assessment_hectares`, `current_size`, `size_class`
- observed spread/weather: `observed_spread_rate_m_min`, `temperature_c`, `relative_humidity_pct`, `wind_direction_deg`, `wind_speed_km_h`, `precipitation_mm`
- fire context: `fire_type`, `fuel_type`, `weather_conditions_over_fire`, `fire_position_on_slope`, `fire_origin`
- cause/admin metadata: `general_cause`, `activity_class`, `true_cause`
- response timing metadata: `detection_delay_h`, `report_delay_h`, `dispatch_delay_h`, `ia_travel_delay_h`
- optional supplementary enrichment: `fwi`, `isi`, `bui`, `dc`, `dmc`, `ffmc`, station metadata

Additional metadata currently written:

- split and lifecycle metadata: `split`, `status`, `record_quality_flag`, `snapshot_generated_at`
- CFFDRS alignment metadata: `cffdrs_station_distance_km`, `cffdrs_station_id`, `cffdrs_station_name`, `cffdrs_observation_date`, `cffdrs_date_offset_days`, `temporal_alignment_status`

Important notes:

- `snapshot_date` is anchored to `ASSESSMENT_DATETIME`
- `area_hectares` prefers `ASSESSMENT_HECTARES`, with `CURRENT_SIZE` as fallback
- `precipitation_mm` is estimated from `WEATHER_CONDITIONS_OVER_FIRE`
- CFFDRS fields may be `null` if supplementary enrichment is unavailable or no station/date match is found
- `temporal_alignment_status` is one of: `aligned`, `near_aligned`, `not_joined`

Top-level JSON payload shape for output files:

- `schema_version`
- `generated_at`
- `record_count`
- `records`

---

## 5) Environment-Variable Builder

The builder computes `data/static/scenario_parameter_records.json` from each snapshot record, then writes seeded benchmark variants in `data/static/scenario_parameter_records_seeded*.json`.

Environment fields:

- `base_spread_prob`
- `severity_bucket`
- `wind_direction` (8-direction string: `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`)
- `wind_strength`
- `ignition_seed`
- `layout_seed`

Stored fields:

- `spread_rate_1h_m`
- `spread_score`
- `weather_score`
- `cffdrs_dryness_score`
- `size_factor`
- `fire_type_factor`
- `fuel_factor`
- `rain_factor`
- `observed_spread_rate_m_min`
- `assessment_hectares`
- `fire_type`
- `fuel_type`
- `record_quality_flag`

### Builder logic

The current builder uses a blended physics-informed rule:

- dominant term: observed `fire_spread_rate`
- supporting terms: wind, temperature, relative humidity, estimated precipitation, assessment size
- optional supplementary term: CFFDRS dryness score from `ISI/FWI/BUI/FFMC`
- modifiers: `fire_type` and `fuel_type`

This is not a full Rothermel implementation. It is a benchmark-oriented, physics-informed calibration rule that keeps the simulator simple while grounding episode conditions in historical assessment data.

---

## 6) Mapping From Data to Environment Variables

| Stored env field | Source fields | Builder logic | Used by environment |
|---|---|---|---|
| `base_spread_prob` | `observed_spread_rate_m_min`, weather, size, optional CFFDRS dryness, `fire_type`, `fuel_type` | derived from blended `spread_score` | primary spread probability in `_spread_fire()` |
| `severity_bucket` | same fields as `base_spread_prob` | derived from `spread_score` thresholds | severity one-hot in observation |
| `wind_direction` | `wind_direction_deg` | mapped to 8-direction bins (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`) | converted to `(wx, wy)` wind bias |
| `wind_strength` | `wind_speed_km_h` | normalized and clipped from assessment wind speed | sets wind-bias magnitude |
| `ignition_seed` | `record_id`, `split` | deterministic stable hash | seeds ignition initialization RNG |
| `layout_seed` | `record_id`, `split` | deterministic stable hash | seeds asset-layout initialization RNG |
| `spread_rate_1h_m` | `observed_spread_rate_m_min` | direct conversion to `m/hour` for audit/logging | optional logging only |

Benchmark runtime file contract:

- canonical train/eval inputs are split-specific seeded files (`scenario_parameter_records_seeded_{split}.json`)
- benchmark loaders enforce split consistency from both filename hints and per-record `split` values
- mixed-split datasets are rejected in benchmark mode

Intermediates fields used only for auditing:

| Stored audit field | Source fields | Purpose |
|---|---|---|
| `spread_score` | spread + weather + size + optional CFFDRS + type/fuel modifiers | blended benchmark calibration score |
| `weather_score` | wind, temperature, RH | weather contribution summary |
| `cffdrs_dryness_score` | `ISI`, `FWI`, `BUI`, `FFMC` | supplementary dryness context |
| `size_factor` | `assessment_hectares` | weak size modifier |
| `fire_type_factor` | `fire_type` | fire-behavior modifier |
| `fuel_factor` | `fuel_type` | fuel-based modifier |
| `rain_factor` | `WEATHER_CONDITIONS_OVER_FIRE` -> `precipitation_mm` | precipitation damping |

---

## 7) Usage

Build the canonical dataset from the Alberta historical CSV:

```bash
uv run python -m src.ingestion.static_dataset --target-count 100
```

Here, `--target-count 100` means up to `100` records per split, not `100` total records overall.

For canonical benchmark builds, use a high cap to avoid truncating available records:

```bash
uv run python -m src.ingestion.static_dataset --target-count 50000 --raw-alberta-csv data/static/fp-historical-wildfire-data-2006-2025.csv
```

Build with optional supplementary CFFDRS enrichment:

```bash
uv run python -m src.ingestion.static_dataset --target-count 100 --cffdrs-year 2025
```

`--cffdrs-year` downloads one annual CFFDRS station file for that year and attempts snapshot-date alignment (within one day) for each candidate fire.

Canonical variant with CFFDRS enrichment:

```bash
uv run python -m src.ingestion.static_dataset --target-count 50000 --cffdrs-year 2025 --raw-alberta-csv data/static/fp-historical-wildfire-data-2006-2025.csv
```

Build from a pre-normalized historical JSON instead of the raw Alberta CSV:

```bash
uv run python -m src.ingestion.static_dataset --fire-records path/to/fire_records.json --target-count 100
```

Override the raw Alberta CSV path if needed:

```bash
uv run python -m src.ingestion.static_dataset --raw-alberta-csv path/to/fp-historical-wildfire-data.csv --target-count 100
```

Write outputs to a custom directory:

```bash
uv run python -m src.ingestion.static_dataset --output-dir path/to/output --target-count 100
```

To use the data for benchmark, you should point training/evaluation envs at seeded split files:

- `data/static/scenario_parameter_records_seeded_train.json`
- `data/static/scenario_parameter_records_seeded_val.json`
- `data/static/scenario_parameter_records_seeded_holdout.json`

This is already set as the default in the config for the environment definition code.

---

## 8) Practical Constraints and Considerations

- Alberta historical data is Alberta-only, so the benchmark is currently province-scoped rather than Canada-wide.
- CFFDRS annual station files may be sparse or unavailable for some years, so the data pipeline script treats them as optional.
- We did not ingest FIRMS or CWFIS or CFFDRS data for live-fire modules, and used a static dataset instead.

