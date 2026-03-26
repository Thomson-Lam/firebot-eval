# Current Data Pipeline Findings

This document summarizes how the current ingestion and XGBoost spread pipeline works in code today.

It is intended to clarify the present behavior before the pipeline is redesigned into a static, snapshot-based workflow for reproducible paper experiments.

---

## 1) Executive Summary

The current pipeline is only partially static.

- Fire incident metadata comes from live CWFIS or live NASA FIRMS fetches.
- Weather inputs for spread prediction come from live Open-Meteo requests at prediction time.
- Fire danger indices come from live CFFDRS station downloads and nearest-station lookup at prediction time.
- Some XGBoost inputs are not ingested from real sources at all and are currently filled with fixed defaults.
- The XGBoost model is trained on synthetic data, not on archived real wildfire snapshots.

As a result, the current system is useful as a working prototype, but it is not yet a fully reproducible benchmark pipeline.

---

## 2) Current Ingestion Modules

### 2.1 `src/ingestion/cffdrs.py`

This module downloads annual CWFIS weather-station CSV data and parses:

- `fwi`
- `isi`
- `bui`
- `dc`
- `dmc`
- `ffmc`
- station weather values: `temp_c`, `rh_pct`, `ws_km_h`, `precip_mm`

Important implementation detail:

- `fetch_cffdrs_stations()` parses both the fire danger indices and the station weather observations.
- `get_cffdrs_for_location()` returns only nearest-station metadata plus `fwi`, `isi`, `bui`, `dc`, `dmc`, and `ffmc`.
- The parsed station weather values are not currently returned through the public nearest-station lookup interface.

### 2.2 `src/ingestion/cwfis.py`

This module downloads the CWFIS active fires CSV and normalizes each row into a fire-event dict.

Current normalized fields:

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

Important implementation detail:

- `severity` is derived from the CWFIS status/stage-of-control field.
- `source` is always recorded as `CWFIS_NRCAN`.

### 2.3 `src/ingestion/firms.py`

This module fetches NASA FIRMS hotspot CSV data and normalizes each hotspot into a fire-event dict.

Current normalized fields:

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

Important implementation details:

- `province` is inferred from a rough BC/AB bounding-box rule.
- `status` is set to `out_of_control` by assumption.
- `severity` is derived from `frp_mw`.
- `area_hectares` is always `None` because FIRMS does not provide incident area.
- `source` is always recorded as `NASA_FIRMS_VIIRS`.

### 2.4 `src/ingestion/weather.py`

This module fetches current-hour weather from Open-Meteo for a fire latitude/longitude.

Current returned fields:

- `wind_speed_km_h`
- `wind_direction_deg`
- `temperature_c`
- `relative_humidity_pct`
- `precipitation_mm`
- `surface_pressure_hpa`
- `dew_point_c`
- `fetched_at`

Important implementation detail:

- This is a live request made at prediction time, not a cached historical snapshot lookup.

---

## 3) Current XGBoost Model Behavior

The XGBoost model lives in `src/models/spread_model.py`.

### 3.1 Inputs used by the model

The model uses these 11 features:

- `wind_speed_km_h`
- `wind_u`
- `wind_v`
- `temperature_c`
- `relative_humidity_pct`
- `fwi`
- `isi`
- `bui`
- `area_hectares`
- `slope_pct`
- `rh_trend_24h`

### 3.2 Outputs produced by the model

The model predicts two spread-radius targets in meters:

- `spread_1h_m`
- `spread_3h_m`

So the high-level understanding that the model outputs fire spread radius at `+1h` and `+3h` horizons is correct.

### 3.3 Training data source

The current model is not trained from real ingestion snapshots.

- `generate_synthetic_dataset()` creates a synthetic training table.
- `train_spread_model()` trains two XGBoost regressors on that synthetic data.
- If saved models are missing, `_load_models()` trains them on the fly.

This is a prototype convenience path, not a reproducible paper-grade data pipeline.

---

## 4) Feature Provenance Table

The table below maps each current XGBoost input to where it actually comes from today.

| XGBoost input | Current source | How it is populated today | Notes |
|---|---|---|---|
| `wind_speed_km_h` | Open-Meteo | fetched by `get_fire_weather()` | live runtime fetch |
| `wind_u` | derived | computed from wind speed and wind direction | not directly ingested |
| `wind_v` | derived | computed from wind speed and wind direction | not directly ingested |
| `temperature_c` | Open-Meteo | fetched by `get_fire_weather()` | live runtime fetch |
| `relative_humidity_pct` | Open-Meteo | fetched by `get_fire_weather()` | live runtime fetch |
| `fwi` | CFFDRS | nearest-station lookup via `get_cffdrs_for_location()` | live runtime lookup |
| `isi` | CFFDRS | nearest-station lookup via `get_cffdrs_for_location()` | live runtime lookup |
| `bui` | CFFDRS | nearest-station lookup via `get_cffdrs_for_location()` | live runtime lookup |
| `area_hectares` | fire metadata | taken from `fire_data` when available | usually from CWFIS; FIRMS often has `None` |
| `slope_pct` | no real ingestion | fixed default `5.0` | placeholder only |
| `rh_trend_24h` | no real ingestion | fixed default `-8.0` | placeholder only |

---

## 5) Where Fire Metadata Comes From

The current code supports two live fire-incident sources.

| Source module | Fields returned | Caveats |
|---|---|---|
| `src/ingestion/cwfis.py` | `fire_id`, `province`, `name`, `status`, `severity`, `latitude`, `longitude`, `area_hectares`, `started_at`, `updated_at`, `source` | `severity` is derived from status |
| `src/ingestion/firms.py` | `fire_id`, `province`, `name`, `status`, `severity`, `latitude`, `longitude`, `area_hectares`, `frp_mw`, `confidence`, `satellite`, `started_at`, `updated_at`, `source` | `province`, `status`, and `severity` are derived; `area_hectares` is `None` |

This means the quality and completeness of the feature row depends on which fire source is used.

- CWFIS usually provides `area_hectares`.
- FIRMS provides `frp_mw`, but the current XGBoost feature vector does not use `frp_mw`.

---

## 6) Current Runtime Prediction Flow

The current prediction path is live and request-driven.

```text
fire record from CWFIS or FIRMS
-> select fire latitude/longitude (+ maybe area_hectares)
-> fetch live Open-Meteo weather for that location
-> fetch/download CFFDRS station data and do nearest-station lookup
-> build feature dict
-> derive wind_u and wind_v from wind speed + direction
-> fill any missing inputs with defaults
-> load XGBoost models from disk, or train them if absent
-> predict spread_1h_m and spread_3h_m
```

The default/fallback behavior currently includes:

- `area_hectares = 500.0` if missing
- `slope_pct = 5.0`
- `rh_trend_24h = -8.0`
- fallback weather and danger-index defaults if live lookups fail

This fallback behavior is convenient for demos, but it weakens reproducibility for benchmark experiments.

---

## 7) Current Model Training Flow

The current training path for the spread model is separate from the live ingestion path.

```text
synthetic feature generation in generate_synthetic_dataset()
-> synthetic labels spread_1h_m and spread_3h_m
-> train two XGBoost regressors
-> save spread_1h_model.joblib and spread_3h_model.joblib
-> runtime prediction later loads those saved models
```

This means the current ingestion modules are used mainly to support live inference-time feature assembly, not to build the XGBoost training dataset.

---

## 8) Corrections to the Initial Understanding

The following parts of the initial understanding are correct:

- `cffdrs.py` does fetch `FWI`, `ISI`, `BUI`, `DC`, `DMC`, and `FFMC`.
- `cwfis.py` does produce normalized fire metadata including `area_hectares`.
- `firms.py` does produce normalized hotspot metadata including `frp_mw`, `confidence`, and `satellite`.
- The XGBoost model does output spread radius in meters at `+1h` and `+3h` horizons.

The following parts are not fully correct in the current code:

- `wind_u` and `wind_v` do not come from an ingestion source; they are derived from wind speed and wind direction.
- `slope_pct` is not currently ingested from terrain/GIS data; it is a fixed default placeholder.
- `rh_trend_24h` is not currently built from historical weather snapshots; it is a fixed default placeholder.
- The XGBoost model is not currently trained on snapshot-derived real data from CWFIS, CFFDRS, FIRMS, and weather ingestion.
- The prediction pipeline still performs live runtime data access and default-based fallback behavior.

---

## 9) Practical Implication for the Planned Redesign

If the goal is a reproducible paper pipeline, the main gap is not just replacing live ingestion with one-time ingestion.

The redesign must also decide how to handle features that are currently synthetic or defaulted:

- `slope_pct`
- `rh_trend_24h`
- missing `area_hectares` for FIRMS hotspots
- model training data provenance

For a static benchmark pipeline, the intended end state should be:

```text
one-time ingestion run
-> normalized snapshot files
-> validated feature records
-> deterministic env/XGBoost parameter records
-> train/eval using only cached files and seeded RNG
```

That would remove runtime drift, remove silent fallback contamination, and make the paper pipeline auditable.

