"""
weather.py — Real-time fire weather ingestion via Open-Meteo API.

No API key required. Open-Meteo is a free, open-source weather API.

Given a fire's (latitude, longitude), this module returns the weather
variables needed as features for the XGBoost spread model:
  - wind_speed_km_h
  - wind_direction_deg
  - temperature_c
  - relative_humidity_pct
  - precipitation_mm
  - surface_pressure_hpa

Data source: https://api.open-meteo.com (current-hour forecast)
Docs: https://open-meteo.com/en/docs

Run to test:
    uv run python -m src.ingestion.weather
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Open-Meteo current-conditions endpoint (no key needed)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Variables we need for the ML feature vector
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
    "dew_point_2m",
]


def get_fire_weather(
    latitude: float,
    longitude: float,
    *,
    timeout: int = 10,
) -> Optional[dict]:
    """
    Fetch current weather conditions at a fire's coordinates.

    Args:
        latitude:  Fire latitude (e.g. 49.9071)
        longitude: Fire longitude (e.g. -119.496)
        timeout:   HTTP timeout in seconds

    Returns:
        Dict with weather features, or None on failure.

    Example output:
        {
          "latitude": 49.9071,
          "longitude": -119.496,
          "wind_speed_km_h": 28.4,
          "wind_direction_deg": 225,
          "temperature_c": 31.2,
          "relative_humidity_pct": 14,
          "precipitation_mm": 0.0,
          "surface_pressure_hpa": 1013.2,
          "dew_point_c": 8.4,
          "fetched_at": "2026-03-21T19:30:00+00:00"
        }
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
        "forecast_days": 1,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
    except httpx.TimeoutException:
        logger.error(f"Open-Meteo timed out for ({latitude}, {longitude})")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"Open-Meteo HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Open-Meteo request error: {e}")
        return None

    data = resp.json()
    current = data.get("current", {})

    if not current:
        logger.warning(f"Open-Meteo returned empty 'current' block for ({latitude}, {longitude})")
        return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "wind_speed_km_h": current.get("wind_speed_10m"),
        "wind_direction_deg": current.get("wind_direction_10m"),
        "temperature_c": current.get("temperature_2m"),
        "relative_humidity_pct": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation", 0.0),
        "surface_pressure_hpa": current.get("surface_pressure"),
        "dew_point_c": current.get("dew_point_2m"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def get_weather_for_fires(fires: list[dict]) -> dict[str, dict]:
    """
    Fetch weather for a list of fire dicts (each must have 'fire_id', 'latitude', 'longitude').

    Returns a dict keyed by fire_id → weather dict.
    Makes one HTTP call per fire. With 4 fires, that's 4 calls.
    Open-Meteo has no rate limit for reasonable usage.
    """
    results = {}
    for fire in fires:
        fire_id = fire.get("fire_id", "unknown")
        lat = fire.get("latitude")
        lon = fire.get("longitude")

        if lat is None or lon is None:
            logger.warning(f"Skipping {fire_id} — missing lat/lon")
            continue

        logger.info(f"Fetching weather for {fire_id} at ({lat}, {lon})")
        wx = get_fire_weather(lat, lon)
        if wx:
            results[fire_id] = wx
        else:
            logger.warning(f"No weather data returned for {fire_id}")

    return results


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    # Test fires
    test_fires = [
        {"fire_id": "BC-2026-001", "name": "Okanagan Ridge Fire",   "latitude": 49.9071,  "longitude": -119.496},
        {"fire_id": "BC-2026-002", "name": "Kamloops Plateau Fire", "latitude": 50.6745,  "longitude": -120.3273},
        {"fire_id": "BC-2026-003", "name": "Fraser Valley Approach","latitude": 49.3845,  "longitude": -121.4483},
        {"fire_id": "AB-2026-001", "name": "Peace River Complex",   "latitude": 56.2370,  "longitude": -117.2900},
    ]

    print("Fetching fire weather from Open-Meteo...\n")
    results = get_weather_for_fires(test_fires)

    for fire_id, wx in results.items():
        print(f"━━━ {fire_id} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  Wind:        {wx['wind_speed_km_h']} km/h @ {wx['wind_direction_deg']}°")
        print(f"  Temperature: {wx['temperature_c']} °C")
        print(f"  Humidity:    {wx['relative_humidity_pct']} %")
        print(f"  Precip:      {wx['precipitation_mm']} mm")
        print(f"  Dew Point:   {wx['dew_point_c']} °C")
        print()

    print(f"Weather data fetched for {len(results)}/{len(test_fires)} fires.")
