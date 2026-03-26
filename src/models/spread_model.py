"""
spread_model.py — XGBoost wildfire spread prediction model.

Predicts fire spread radius (metres) at +1h and +3h horizons.

TRAINING STRATEGY (Hackathon POC):
  Physics-informed synthetic data using a Rothermel-inspired formula.
  No historical fire spread archives needed.

FEATURES (11 inputs) — research-informed:
  wind_speed_km_h      — wind magnitude
  wind_u               — eastward wind vector (cos decomposition — fixes cyclical issue)
  wind_v               — northward wind vector (sin decomposition)
  temperature_c        — air temperature
  relative_humidity_pct
  fwi                  — Fire Weather Index (CFFDRS)
  isi                  — Initial Spread Index
  bui                  — Buildup Index
  area_hectares        — current fire size
  slope_pct            — terrain slope (negative=downhill, positive=uphill)
  rh_trend_24h         — change in RH over last 24h (temporal context)

NOTE on wind_u/wind_v: Tree models cannot handle cyclical features (359° and 1°
are physically adjacent but numerically 358 apart). Projecting onto U/V Cartesian
vectors eliminates this problem — the research paper confirmed this is the correct fix.

Run to train + test:
    uv run python -m src.models.spread_model
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
MODEL_1H_PATH = MODEL_DIR / "spread_1h_model.joblib"
MODEL_3H_PATH = MODEL_DIR / "spread_3h_model.joblib"

# Updated feature set — 11 features with wind U/V, slope, and RH trend
FEATURE_COLS = [
    "wind_speed_km_h",
    "wind_u",                  # eastward component: speed × cos(dir_rad)
    "wind_v",                  # northward component: speed × sin(dir_rad)
    "temperature_c",
    "relative_humidity_pct",
    "fwi",
    "isi",
    "bui",
    "area_hectares",
    "slope_pct",               # terrain slope (–20 to +45 %)
    "rh_trend_24h",            # RH change over last 24h (negative = drying out)
]


# ── Physics Formula ───────────────────────────────────────────────────────────

def _rothermel_spread_m_per_min(
    wind_speed: float,
    rh: float,
    isi: float,
    ffmc: float,
    slope: float = 0.0,
) -> float:
    """
    Rothermel-inspired fire spread rate (metres per minute).

    Improvements over original version:
    - wind_speed is the magnitude (|U|, |V| already resolved)
    - slope_factor: uphill fires accelerate due to convective preheating
    """
    ffmc_factor = max(0.1, (101 - ffmc) / 100)
    wind_factor = math.exp(0.05039 * wind_speed)
    rh_damping = max(0.01, 1 - (rh / 120))

    # Uphill acceleration (Rothermel): positive slope → faster spread
    # Downhill: small dampening. Flat: neutral (1.0)
    slope_factor = 1.0 + (max(0.0, slope) / 20.0)

    base = isi * ffmc_factor * wind_factor * rh_damping * slope_factor * 2.5
    return max(0.5, base)


# ── Synthetic Data Generator ──────────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Build a physics-informed synthetic training dataset.
    Ranges: BC/AB wildfire season (May–September) observations.

    Key improvements applied:
    1. wind_direction → (wind_u, wind_v) via trigonometric decomposition
    2. slope_pct feature added (terrain topography)
    3. rh_trend_24h feature added (temporal drying context)
    """
    rng = np.random.default_rng(seed)

    wind_speed   = rng.uniform(0, 60, n_samples)      # km/h
    wind_dir_deg = rng.uniform(0, 360, n_samples)      # degrees
    temperature  = rng.uniform(5, 42, n_samples)       # °C
    humidity     = rng.uniform(8, 85, n_samples)       # %
    fwi          = rng.uniform(0, 100, n_samples)
    isi          = rng.uniform(0, 40, n_samples)
    bui          = rng.uniform(0, 200, n_samples)
    area_ha      = rng.uniform(1, 25000, n_samples)

    # [FIX 1] Wind U/V decomposition — eliminates cyclic discontinuity
    wind_dir_rad = np.radians(wind_dir_deg)
    wind_u = wind_speed * np.cos(wind_dir_rad)   # eastward
    wind_v = wind_speed * np.sin(wind_dir_rad)   # northward

    # [FIX 2] Slope topography — uphill fires spread much faster
    slope_pct = rng.uniform(-20, 45, n_samples)  # –20 (downhill) to +45% (steep uphill)

    # [FIX 3] Temporal RH trend — drying conditions amplify danger
    # Biased negative (more often drying than wetting during fire season)
    rh_trend_24h = rng.normal(-5, 15, n_samples)  # % RH change per 24h

    # FFMC derived from humidity + temp
    ffmc = np.clip(101 - humidity * 0.7 + temperature * 0.4, 0, 101)

    # RH trend amplification: fast-drying conditions increase effective spread
    # (a fire in drop-40%-RH conditions is much more dangerous)
    rh_drying_factor = np.clip(1 + (-rh_trend_24h / 80), 0.8, 1.5)

    # Labels from Rothermel formula × drying amplification + noise
    spread_1h_m = np.array([
        _rothermel_spread_m_per_min(
            wind_speed[i], humidity[i], isi[i], ffmc[i], slope_pct[i]
        ) * 60 * rh_drying_factor[i]
        + rng.normal(0, 50)
        for i in range(n_samples)
    ])
    spread_3h_m = np.array([
        _rothermel_spread_m_per_min(
            wind_speed[i], humidity[i], isi[i], ffmc[i], slope_pct[i]
        ) * 180 * rh_drying_factor[i]
        + rng.normal(0, 200)
        for i in range(n_samples)
    ])

    spread_1h_m = np.clip(spread_1h_m, 50, 15000)
    spread_3h_m = np.clip(spread_3h_m, 100, 50000)

    return pd.DataFrame({
        "wind_speed_km_h":       wind_speed,
        "wind_u":                wind_u,
        "wind_v":                wind_v,
        "temperature_c":         temperature,
        "relative_humidity_pct": humidity,
        "fwi":                   fwi,
        "isi":                   isi,
        "bui":                   bui,
        "area_hectares":         area_ha,
        "slope_pct":             slope_pct,
        "rh_trend_24h":          rh_trend_24h,
        "spread_1h_m":           spread_1h_m,
        "spread_3h_m":           spread_3h_m,
    })


# ── Training ──────────────────────────────────────────────────────────────────

def train_spread_model(n_samples: int = 6000) -> tuple[XGBRegressor, XGBRegressor, dict]:
    print(f"Generating {n_samples} synthetic fire spread samples...")
    df = generate_synthetic_dataset(n_samples=n_samples)

    X = df[FEATURE_COLS]
    y_1h = df["spread_1h_m"]
    y_3h = df["spread_3h_m"]

    X_train, X_test, y1_train, y1_test, y3_train, y3_test = train_test_split(
        X, y_1h, y_3h, test_size=0.2, random_state=42
    )

    xgb_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    print("Training 1-hour spread model...")
    model_1h = XGBRegressor(**xgb_params)
    model_1h.fit(X_train, y1_train)

    print("Training 3-hour spread model...")
    model_3h = XGBRegressor(**xgb_params)
    model_3h.fit(X_train, y3_train)

    pred_1h = model_1h.predict(X_test)
    pred_3h = model_3h.predict(X_test)

    metrics = {
        "1h_mae_m": round(mean_absolute_error(y1_test, pred_1h), 1),
        "1h_r2":    round(r2_score(y1_test, pred_1h), 3),
        "3h_mae_m": round(mean_absolute_error(y3_test, pred_3h), 1),
        "3h_r2":    round(r2_score(y3_test, pred_3h), 3),
    }

    joblib.dump(model_1h, MODEL_1H_PATH)
    joblib.dump(model_3h, MODEL_3H_PATH)
    print(f"Models saved -> {MODEL_DIR}")

    return model_1h, model_3h, metrics


# ── Lazy Model Loading ────────────────────────────────────────────────────────

_model_1h: XGBRegressor | None = None
_model_3h: XGBRegressor | None = None


def _load_models() -> tuple[XGBRegressor, XGBRegressor]:
    global _model_1h, _model_3h
    if _model_1h is None or _model_3h is None:
        if MODEL_1H_PATH.exists() and MODEL_3H_PATH.exists():
            logger.info("Loading pre-trained spread models from disk...")
            _model_1h = joblib.load(MODEL_1H_PATH)
            _model_3h = joblib.load(MODEL_3H_PATH)
        else:
            logger.info("No saved models — training now...")
            _model_1h, _model_3h, _ = train_spread_model()
    return _model_1h, _model_3h


# ── Prediction API ────────────────────────────────────────────────────────────

def predict_spread_from_features(features: dict) -> dict:
    """
    Build feature row, run both XGBoost models, return predictions.
    Any missing feature is filled with a safe fire-season default.
    """
    model_1h, model_3h = _load_models()

    # Compute U/V from raw wind direction if provided
    wind_speed = features.get("wind_speed_km_h", 20.0)
    wind_dir   = features.get("wind_direction_deg", 180.0)
    wind_dir_rad = math.radians(wind_dir)

    defaults = {
        "wind_speed_km_h":       wind_speed,
        "wind_u":                wind_speed * math.cos(wind_dir_rad),
        "wind_v":                wind_speed * math.sin(wind_dir_rad),
        "temperature_c":         25.0,
        "relative_humidity_pct": 35.0,
        "fwi":                   25.0,
        "isi":                   10.0,
        "bui":                   60.0,
        "area_hectares":         500.0,
        "slope_pct":             5.0,    # mild uphill default
        "rh_trend_24h":          -8.0,   # slight drying — typical fire season
    }

    row = {col: features.get(col, defaults[col]) for col in FEATURE_COLS}
    df_row = pd.DataFrame([row])

    spread_1h = float(model_1h.predict(df_row)[0])
    spread_3h = float(model_3h.predict(df_row)[0])

    return {
        "spread_1h_m":   round(max(50, spread_1h)),
        "spread_3h_m":   round(max(100, spread_3h)),
        "features_used": row,
        "model":         "XGBoost-Rothermel-v2-WindUV-Slope-Trend",
    }


def predict_spread(fire_id: str, fire_data: dict | None = None) -> dict:
    """
    High-level call: fetch live weather → build features → predict.
    Called by GET /api/v1/predictions/{fire_id}.
    """
    from src.ingestion.weather import get_fire_weather

    lat  = fire_data.get("latitude",  49.9071) if fire_data else 49.9071
    lon  = fire_data.get("longitude", -119.496) if fire_data else -119.496
    area = fire_data.get("area_hectares", 500)   if fire_data else 500

    weather = get_fire_weather(lat, lon)

    if weather:
        wind_speed = weather.get("wind_speed_km_h", 20.0) or 20.0
        wind_dir   = weather.get("wind_direction_deg", 180.0) or 180.0
        wind_dir_rad = math.radians(wind_dir)
        features = {
            "wind_speed_km_h":       wind_speed,
            "wind_direction_deg":    wind_dir,    # kept for U/V calc in predict_spread_from_features
            "wind_u":                wind_speed * math.cos(wind_dir_rad),
            "wind_v":                wind_speed * math.sin(wind_dir_rad),
            "temperature_c":         weather.get("temperature_c", 25.0),
            "relative_humidity_pct": weather.get("relative_humidity_pct", 35.0),
            "fwi":                   25.0,        # CFFDRS fallback (overwritten below if available)
            "isi":                   10.0,
            "bui":                   60.0,
            "area_hectares":         float(area or 500),
            "slope_pct":             5.0,         # default mild uphill
            "rh_trend_24h":          -8.0,        # typical fire-season drying trend
        }

        # Enrich with real CFFDRS fire danger indices from nearest NRCan weather station
        try:
            from src.ingestion.cffdrs import get_cffdrs_for_location
            cffdrs = get_cffdrs_for_location(lat, lon)
            if cffdrs:
                if cffdrs.get("fwi") is not None:
                    features["fwi"] = cffdrs["fwi"]
                if cffdrs.get("isi") is not None:
                    features["isi"] = cffdrs["isi"]
                if cffdrs.get("bui") is not None:
                    features["bui"] = cffdrs["bui"]
                logger.info(
                    f"CFFDRS station '{cffdrs['source_station']}' "
                    f"({cffdrs['distance_km']} km away): "
                    f"FWI={cffdrs['fwi']}, ISI={cffdrs['isi']}, BUI={cffdrs['bui']}"
                )
        except Exception as e:
            logger.warning(f"CFFDRS lookup failed for {fire_id}: {e} — using fallback indices")
    else:
        logger.warning(f"No weather for {fire_id} — using defaults")
        features = {}

    prediction = predict_spread_from_features(features)
    prediction["fire_id"] = fire_id
    return prediction


# ── CLI: Train + Evaluate + Live Demo ───────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # suppress httpx INFO noise

    print("=" * 65)
    print("  FireGrid XGBoost Spread Model v2 — Wind U/V · Slope · Trend")
    print("=" * 65)

    model_1h, model_3h, metrics = train_spread_model(n_samples=6000)

    print("\nEvaluation (20% held-out test set):")
    print(f"  1h model:  MAE = {metrics['1h_mae_m']} m   R² = {metrics['1h_r2']}")
    print(f"  3h model:  MAE = {metrics['3h_mae_m']} m   R² = {metrics['3h_r2']}")

    print("\nLive predictions (real weather from Open-Meteo):\n")
    demo_fires = [
        {"fire_id": "BC-2026-001", "name": "Okanagan Ridge Fire",    "latitude": 49.9071, "longitude": -119.4960, "area_hectares": 12450},
        {"fire_id": "BC-2026-003", "name": "Fraser Valley Approach", "latitude": 49.3845, "longitude": -121.4483, "area_hectares": 250},
        {"fire_id": "AB-2026-001", "name": "Peace River Complex",    "latitude": 56.2370, "longitude": -117.2900, "area_hectares": 12500},
    ]

    for fire in demo_fires:
        result = predict_spread(fire["fire_id"], fire)
        wx = result.get("features_used", {})
        print(f"━━━ {fire['name']} ({fire['fire_id']}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  Wind:        {wx.get('wind_speed_km_h', '?'):.1f} km/h  "
              f"-> U={wx.get('wind_u', 0):.1f}, V={wx.get('wind_v', 0):.1f}")
        print(f"  Temp/RH:     {wx.get('temperature_c', '?')}°C / {wx.get('relative_humidity_pct', '?')}% RH")
        print(f"  Slope:       {wx.get('slope_pct', 5.0):.0f}%   RH trend: {wx.get('rh_trend_24h', -8.0):.0f}% per 24h")
        print(f"  Area:        {fire['area_hectares']:,} ha")
        print(f"  +1h spread:  {result['spread_1h_m']:,} m")
        print(f"  +3h spread:  {result['spread_3h_m']:,} m")
        print()

    print("Feature importances (1h model — sorted):")
    fi = dict(zip(FEATURE_COLS, model_1h.feature_importances_, strict=True))
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {feat:<28} {bar} ({imp:.3f})")
