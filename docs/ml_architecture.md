# FireGrid ML Architecture

## Overview

FireGrid uses a two-stage ML pipeline:

1. **XGBoost Spread Model** — Supervised learning. Answers: *"How far will this fire spread?"*
2. **PPO Tactical Agent** *(planned)* — Reinforcement learning. Answers: *"Where should we deploy assets to stop it?"*

---

## Stage 1: XGBoost Fire Spread Model

**File:** `backend/src/models/spread_model.py`
**Saved weights:** `backend/src/models/spread_1h_model.joblib`, `spread_3h_model.joblib`

### What It Predicts
Two separate `XGBRegressor` models trained in parallel:
- `spread_1h_m` — predicted fire spread **radius in metres** after 1 hour
- `spread_3h_m` — predicted spread radius after 3 hours

These radii feed the map's fire spread ring visualization.

### Training Data Strategy
Real historical spread data requires years of CWFIS archives. For this hackathon, we generate **6,000 physics-informed synthetic samples** using the **Rothermel fire spread formula** as a label generator:

```
spread_rate ∝ ISI × exp(0.05039 × wind_speed) × (1 - RH/120) × FFMC_factor × slope_factor
```

We add Gaussian noise so XGBoost learns a smooth, non-linear approximation rather than memorizing the formula.

### Feature Vector (11 inputs)

| Feature | Source | Why It Matters |
|---|---|---|
| `wind_speed_km_h` | Open-Meteo API | Primary spread driver (exponential effect) |
| `wind_u` | Derived: `speed × cos(dir_rad)` | Eastward wind vector — fixes cyclic discontinuity |
| `wind_v` | Derived: `speed × sin(dir_rad)` | Northward wind vector |
| `temperature_c` | Open-Meteo API | Fuel pre-heating, moisture evaporation |
| `relative_humidity_pct` | Open-Meteo API | Dampens spread — key moisture indicator |
| `fwi` | CWFIS/CFFDRS | Overall Canadian fire danger index |
| `isi` | CWFIS/CFFDRS | Initial Spread Index — rate of spread |
| `bui` | CWFIS/CFFDRS | Buildup Index — available fuel load |
| `area_hectares` | DynamoDB | Current fire size context |
| `slope_pct` | Synthetic (–20 to +45%) | Uphill fires accelerate via convective preheating |
| `rh_trend_24h` | Synthetic / ECCC | RH change over 24h — drying conditions amplify danger |

### Key Design Decisions

**Wind U/V decomposition** — Tree-based models cannot handle cyclical features. Feeding `wind_direction_deg = 359°` and `1°` as raw numbers makes them look 358 units apart, when they are physically 2° apart. Projecting onto Cartesian `(U, V)` vectors eliminates this discontinuity. This is a standard practice in numerical weather prediction (NWP) preprocessing.

**Slope factor** — The Rothermel (1972) model includes a slope intensification term. Fires moving uphill receive convective preheating from the rising column of hot gas ahead of the flame front, dramatically increasing spread rate. We apply `slope_factor = 1 + max(0, slope%)/20`, giving up to a 3.25× multiplier at 45% slope.

**RH trend (temporal context)** — A fire at 40% RH is more dangerous if RH was 80% yesterday and is dropping fast. Adding `rh_trend_24h` lets the model distinguish static vs. rapidly-drying conditions without building a full time-series architecture.

### Observed Feature Importances (v2 model)

```
isi                    ███████████████  0.301  ← CFFDRS Initial Spread Index
wind_speed_km_h        ██████████████   0.293  ← Wind magnitude
slope_pct              ███████          0.145  ← Terrain topography
relative_humidity_pct  ███              0.064  ← Moisture damper
wind_v                 ███              0.063  ← Northward component
wind_u                 ██               0.048  ← Eastward component
rh_trend_24h           █                0.037  ← Temporal drying context
temperature_c          █                0.026
area_hectares                           0.009
bui                                     0.009
fwi                                     0.005
```

### Evaluation Results

| Model | MAE | R² |
|---|---|---|
| 1-hour spread | 421 m | 0.977 |
| 3-hour spread | 1,323 m | 0.977 |

High R² expected on synthetic data — the model is learning the correct physical relationships between features and spread rate.

### API Integration

```
GET /api/v1/predictions/{fire_id}
```
1. Fetches fire record from DynamoDB (lat/lon, area)
2. Calls `get_fire_weather(lat, lon)` → Open-Meteo API (real-time)
3. Builds 11-feature vector (with CFFDRS fallback defaults)
4. Runs `model_1h.predict()` and `model_3h.predict()`
5. Returns `{spread_1h_m, spread_3h_m, features_used, model}`

---

## Stage 2: PPO Tactical Agent *(Planned — Phase 3b)*

**See:** `docs/ppo_plan.md`

The RL agent takes the XGBoost spread prediction as input and outputs tactical deployment coordinates for helicopters, ground crews, and dozers.

---

## Running the Model

```bash
# Train from scratch and run live demo predictions
uv run python -m src.models.spread_model

# Call via API (backend must be running)
Invoke-RestMethod http://localhost:8000/api/v1/predictions/BC-2026-001
```
