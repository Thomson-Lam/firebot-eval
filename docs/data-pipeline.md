# Data Pipeline

This document describes the current benchmark data pipeline after the move away from live CWFIS-centered ingestion and XGBoost.

The canonical path now uses the Alberta historical wildfire dataset stored under `data/static/` as the primary source for building `FireEnv` scenario records.

---

## 1) Overview

The benchmark pipeline has two stages:

1. normalize historical wildfire incidents into frozen snapshot records
2. compute offline environment-variable records for `FireEnv`

Training and evaluation should then use only the cached parameter dataset plus seeded RNG.

Primary source hierarchy:

- primary: Alberta historical wildfire dataset
- supplementary: CFFDRS fire-danger indices when an annual station file is available and usable
- non-canonical: CWFIS live active fires and FIRMS hotspots

---

## 2) Input Data Sources

### 2.1 Alberta historical wildfire dataset

Raw file path:

- `data/static/fp-historical-wildfire-data-2006-2025.csv`

This dataset is now the default primary input to `src/ingestion/static_dataset.py`.

Important fields used directly by the pipeline:

- incident identity: `YEAR`, `FIRE_NUMBER`, `FIRE_NAME`
- location: `LATITUDE`, `LONGITUDE`
- timing: `FIRE_START_DATE`, `ASSESSMENT_DATETIME`, `DISCOVERED_DATE`, `REPORTED_DATE`, `DISPATCH_DATE`, `IA_ARRIVAL_AT_FIRE_DATE`, `FIRE_FIGHTING_START_DATE`
- fire state: `ASSESSMENT_HECTARES`, `CURRENT_SIZE`, `SIZE_CLASS`
- spread/weather: `FIRE_SPREAD_RATE`, `TEMPERATURE`, `RELATIVE_HUMIDITY`, `WIND_DIRECTION`, `WIND_SPEED`, `WEATHER_CONDITIONS_OVER_FIRE`
- fire context: `FIRE_TYPE`, `FUEL_TYPE`, `FIRE_POSITION_ON_SLOPE`, `FIRE_ORIGIN`
- optional response context: `INITIAL_ACTION_BY`, `IA_ACCESS`, `BUCKETING_ON_FIRE`, `DISTANCE_FROM_WATER_SOURCE`

Why this is now primary:

- historical instead of live-only
- provides assessment-time weather directly
- provides observed spread rate directly
- provides assessment-time size directly
- removes dependence on ad hoc live weather reconstruction for canonical builds

### 2.2 `src/ingestion/cffdrs.py`

This module downloads annual CWFIS weather-station CSV data and parses:

- `fwi`
- `isi`
- `bui`
- `dc`
- `dmc`
- `ffmc`

Current role:

- supplementary enrichment only
- if `--cffdrs-year` is passed and usable observations exist, the builder joins the nearest station by both distance and snapshot date
- the benchmark no longer depends on CFFDRS being available to build records

### 2.3 `src/ingestion/cwfis.py`

This module still downloads live active fires from CWFIS.

Current role:

- legacy / non-canonical
- useful for live experiments or future non-Alberta extensions
- not part of the canonical Alberta historical benchmark build

### 2.4 `src/ingestion/firms.py`

This module still fetches NASA FIRMS hotspots.

Current role:

- supplementary / non-canonical
- not used in the canonical Alberta historical benchmark build
- may still be useful for exploratory validation or future data discovery

### 2.5 `src/ingestion/weather.py`

This module fetches current-hour weather from Open-Meteo.

Current role:

- legacy / non-canonical for Alberta historical builds
- assessment-time weather now comes directly from the Alberta dataset

---

## 3) Canonical Build Flow

```text
Alberta historical wildfire CSV
-> normalized historical fire records
-> optional CFFDRS date-and-distance enrichment
-> snapshot_records.json
-> offline env-variable builder
-> scenario_parameter_records.json
-> FireEnv reset sampling
-> RL train/eval from cached records only
```

This path does not use FIRMS or Open-Meteo in the canonical benchmark build.

---

## 4) Snapshot Schema

The builder writes `data/static/snapshot_records.json`.

Each snapshot record represents one Alberta wildfire incident anchored at the initial assessment time.

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

Important notes:

- `snapshot_date` is anchored to `ASSESSMENT_DATETIME`
- `area_hectares` prefers `ASSESSMENT_HECTARES`, with `CURRENT_SIZE` as fallback
- `precipitation_mm` is estimated from `WEATHER_CONDITIONS_OVER_FIRE`
- CFFDRS fields may be `null` if supplementary enrichment is unavailable

---

## 5) Environment-Variable Builder

The builder computes `data/static/scenario_parameter_records.json` from each snapshot record.

Canonical env-facing fields:

- `base_spread_prob`
- `severity_bucket`
- `wind_dir_deg`
- `wind_strength`

Stored audit fields:

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
| `severity_bucket` | same fields as `base_spread_prob` | derived from `spread_score` thresholds | severity one-hot in observation and family matching |
| `wind_dir_deg` | `wind_direction_deg` | pass-through from Alberta assessment weather | converted to `(wx, wy)` wind bias |
| `wind_strength` | `wind_speed_km_h` | normalized and clipped from assessment wind speed | sets wind-bias magnitude |
| `spread_rate_1h_m` | `observed_spread_rate_m_min` | direct conversion to `m/hour` for audit/logging | optional logging only |

Audit-only intermediates:

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

Build with optional supplementary CFFDRS enrichment:

```bash
uv run python -m src.ingestion.static_dataset --target-count 100 --cffdrs-year 2025
```

Build from a pre-normalized historical JSON instead of the raw Alberta CSV:

```bash
uv run python -m src.ingestion.static_dataset --fire-records path/to/fire_records.json --target-count 100
```

Override the raw Alberta CSV path if needed:

```bash
uv run python -m src.ingestion.static_dataset --raw-alberta-csv path/to/fp-historical-wildfire-data.csv --target-count 100
```

Then train from the cached parameter file:

```bash
uv run python -m src.models.train_rl_agent --scenario-dataset data/static/scenario_parameter_records.json
```

---

## 8) Practical Constraints

- Alberta historical data is Alberta-only, so the canonical benchmark is currently province-scoped rather than Canada-wide.
- CFFDRS annual station files may be sparse or unavailable for some years; the builder treats them as optional.
- FIRMS and CWFIS remain available in the repo but are no longer part of the canonical benchmark build path.
- The benchmark still does not use terrain rasters, perimeter replay, or a full operational spread model.

That is acceptable for the current paper because the goal is a reproducible tactical RL benchmark, not an operational wildfire decision-support system.
