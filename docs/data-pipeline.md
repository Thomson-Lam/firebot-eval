# Data Pipeline

This document describes the current benchmark data pipeline after the redesign away from XGBoost and toward a one-time static dataset plus offline environment-variable builder.

---

## 1) Overview 

The benchmark pipeline has two stages:

1. one-time ingestion and normalization into frozen snapshot records
2. offline parameter building into cached environment-variable records for `FireEnv`

Training and evaluation should then use only the cached parameter dataset plus seeded RNG.

The current pipeline still fetches live source data during the one-time build step unless you provide precollected historical fire records via `--fire-records`, but benchmark runs themselves should not call live APIs.

---

## 2) Ingestion Modules

### 2.1 `src/ingestion/cffdrs.py`

This module downloads annual CWFIS weather-station CSV data and parses:

- `fwi`
- `isi`
- `bui`
- `dc`
- `dmc`
- `ffmc`
- station weather values: `temp_c`, `rh_pct`, `ws_km_h`, `precip_mm`

- `fetch_cffdrs_stations()` parses both fire-danger indices and station weather observations.
- `get_cffdrs_for_location()` returns nearest-station metadata plus `fwi`, `isi`, `bui`, `dc`, `dmc`, and `ffmc`.

### 2.2 `src/ingestion/cwfis.py`

This module downloads the CWFIS active fires CSV and normalizes each row into a fire-event dict.

Normalized fields:

- `fire_id`
- `province`
- `name`
- `status`
- `severity`
- `latitude`
- `longitude`
- `area_hectares`
- `started_at`
- `updated_at`
- `source`

- `severity` is derived from CWFIS status and is metadata only.
- canonical benchmark severity is now computed offline by the environment-variable builder, not taken directly from CWFIS.

### 2.3 `src/ingestion/firms.py`

This module fetches NASA FIRMS hotspot CSV data and normalizes each hotspot into a fire-event dict.

Normalized fields:

- `fire_id`
- `province`
- `name`
- `status`
- `severity`
- `latitude`
- `longitude`
- `area_hectares`
- `frp_mw`
- `confidence`
- `satellite`
- `started_at`
- `updated_at`
- `source`

- `province` is inferred from a rough BC/AB bounding-box rule.
- `area_hectares` is missing from FIRMS and may need imputation during snapshot building.
- `NASA_FIRMS_API_KEY` is required only if FIRMS is used.

### 2.4 `src/ingestion/weather.py`

This module fetches current-hour weather from Open-Meteo for a fire latitude/longitude.

Returns:

- `wind_speed_km_h`
- `wind_direction_deg`
- `temperature_c`
- `relative_humidity_pct`
- `precipitation_mm`
- `surface_pressure_hpa`
- `dew_point_c`
- `fetched_at`

- Open-Meteo requires no API key.
- these fields are used during one-time snapshot building, then cached.

### 2.5 `src/ingestion/static_dataset.py`

One-time builder script that converts source records into frozen benchmark artifacts.

Outputs:

- `snapshot_records.json`
- `scenario_parameter_records.json`

It also computes offline environment variables such as:

- `base_spread_prob`
- `severity_bucket`
- `wind_dir_deg`
- `wind_strength`
- audit fields like `spread_rate_1h_m`, `spread_score`, `dryness_score`, and `record_quality_flag`

---

## 3) Static Dataset Build Flow

```text
live fire sources or precollected historical fire records
-> normalized fire records
-> weather enrichment from Open-Meteo
-> nearest-station CFFDRS enrichment
-> snapshot_records.json
-> offline environment-variable builder
-> scenario_parameter_records.json
-> FireEnv reset sampling
-> RL train/eval from cached records only
```

To run the dataset builder: 

```bash
uv run python -m src.ingestion.static_dataset --target-count 100
```

Optional historical-record input:

```bash
uv run python -m src.ingestion.static_dataset --fire-records path/to/fire_records.json --target-count 100
```

---

## 4) Mapping From Data Pipeline to Environment Variables

The table below describes how ingested data fields map into the cached environment-variable record used by `FireEnv`.

| Stored env field | Source pipeline fields | Builder logic | Used by environment |
|---|---|---|---|
| `base_spread_prob` | `wind_speed_km_h`, `temperature_c`, `relative_humidity_pct`, `precipitation_mm`, `fwi`, `isi`, `bui`, `ffmc`, `area_hectares` | computed in `compute_environment_parameters()` from normalized dryness, RH, rain, wind, temperature, and size factors | primary spread probability in `_spread_fire()` |
| `severity_bucket` | same fields as `base_spread_prob` | derived from `spread_score` thresholds: low `<0.33`, medium `<0.66`, else high | severity one-hot in observation and family matching |
| `wind_dir_deg` | `wind_direction_deg` from Open-Meteo snapshot | pass-through from snapshot record | converted to `(wx, wy)` wind bias |
| `wind_strength` | `wind_speed_km_h` | normalized and clipped from wind speed | sets wind-bias magnitude |
| `spread_rate_1h_m` | same fields as `base_spread_prob` | audit/logging value derived from `spread_score` | optional logging only |
| `spread_score` | same fields as `base_spread_prob` | combined physics-informed intermediate score | audit/debug only |
| `dryness_score` | `isi`, `fwi`, `bui`, `ffmc` | weighted dryness subscore | audit/debug only |
| `rh_factor` | `relative_humidity_pct` | humidity damping factor | audit/debug only |
| `rain_factor` | `precipitation_mm` | precipitation damping factor | audit/debug only |
| `temp_factor` | `temperature_c` | mild heat multiplier | audit/debug only |
| `wind_factor` | `wind_speed_km_h` | wind multiplier | audit/debug only |
| `size_factor` | `area_hectares` | weak incident-size multiplier | audit/debug only |
| `record_quality_flag` | `area_hectares`, `frp_mw` | marks measured vs imputed area path | audit/debug only |

More detailed field provenance:

| Snapshot field | Source module | Notes |
|---|---|---|
| `wind_speed_km_h` | `src/ingestion/weather.py` | live during one-time build only |
| `wind_direction_deg` | `src/ingestion/weather.py` | live during one-time build only |
| `temperature_c` | `src/ingestion/weather.py` | live during one-time build only |
| `relative_humidity_pct` | `src/ingestion/weather.py` | live during one-time build only |
| `precipitation_mm` | `src/ingestion/weather.py` | live during one-time build only |
| `fwi` | `src/ingestion/cffdrs.py` | nearest-station lookup |
| `isi` | `src/ingestion/cffdrs.py` | nearest-station lookup |
| `bui` | `src/ingestion/cffdrs.py` | nearest-station lookup |
| `ffmc` | `src/ingestion/cffdrs.py` | nearest-station lookup, optional but used if available |
| `area_hectares` | `src/ingestion/cwfis.py` or imputed in `src/ingestion/static_dataset.py` | FIRMS path may infer area from `frp_mw` |
| `frp_mw` | `src/ingestion/firms.py` | optional metadata, used only for area imputation right now |

---

## 5) Where Fire Metadata Comes From

The current code supports two fire-incident inputs.

| Source module | Fields returned | Caveats |
|---|---|---|
| `src/ingestion/cwfis.py` | `fire_id`, `province`, `name`, `status`, `severity`, `latitude`, `longitude`, `area_hectares`, `started_at`, `updated_at`, `source` | best current source for measured incident area |
| `src/ingestion/firms.py` | `fire_id`, `province`, `name`, `status`, `severity`, `latitude`, `longitude`, `area_hectares`, `frp_mw`, `confidence`, `satellite`, `started_at`, `updated_at`, `source` | `area_hectares` missing; FIRMS is best treated as supplemental |

For benchmark quality, CWFIS should usually be the primary source and FIRMS should be supplemental unless you provide a historical record file with a cleaner schema.

---

## 6) Current Gaps and Constraints

The redesigned pipeline is much closer to the intended benchmark workflow, but some limits remain:

- source ingestion is still live during the one-time build unless `--fire-records` is used
- the available public feeds are current/recent feeds, not a curated historical spread-label dataset
- `area_hectares` may be imputed for FIRMS-derived records
- there is still no terrain, fuel-model, or perimeter-growth dataset in the canonical pipeline

This is acceptable for the current benchmark because the goal is to build realistic episode parameters for a fixed tactical RL environment, not an operational wildfire forecaster.

---

## 7) Practical Benchmark End State

The intended benchmark end state is:

```text
one-time ingestion run
-> normalized snapshot records
-> offline environment-variable builder
-> frozen scenario_parameter_records.json
-> train/eval using only cached records and seeded RNG
```

This removes runtime drift, removes hidden fallback contamination during benchmark runs, and makes the benchmark pipeline auditable.
