"""
firms.py — NASA FIRMS (Fire Information for Resource Management System) ingestion.

Pulls real VIIRS/NOAA satellite fire hotspot data over Western Canada (BC + AB)
and normalizes it into FireGrid FireEvent format.

API docs: https://firms.modaps.eosdis.nasa.gov/api/area/
Rate limit: 5000 transactions / 10 minutes — we use 1 call per request (safe).

Run this from backend/ to test:
    uv run python -m src.ingestion.firms
"""

import csv
import io
import logging
from datetime import UTC, datetime

import httpx

from src.core.config import settings

logger = logging.getLogger(__name__)

# ── Canada Bounding Box (BC + AB focus) ─────────────────────────────────────────
# Format: W,S,E,N  (longitude_min, latitude_min, longitude_max, latitude_max)
CANADA_WEST_BBOX = "-140,48,-110,62"

# FIRMS endpoint for CSV area data
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Use VIIRS NOAA-20 (most recent, best resolution ~375m)
FIRMS_SOURCE = "VIIRS_NOAA20_NRT"

# How many days back to fetch (1 = last 24h — minimizes data volume + rate hits)
DEFAULT_DAY_RANGE = 1


def _assign_province(lat: float, lon: float) -> str:
    """
    Rough bounding-box province assignment for BC and AB.
    Anything else gets tagged as 'OTHER'.
    """
    if -139.0 <= lon <= -114.0 and 48.3 <= lat <= 60.0:
        return "BC"
    elif -120.0 <= lon <= -110.0 and 49.0 <= lat <= 60.0:
        return "AB"
    return "OTHER"


def _frp_to_severity(frp: float) -> str:
    """
    Convert Fire Radiative Power (MW) to a FireGrid severity label.
    Thresholds based on FIRMS documentation and wildfire research.
    """
    if frp >= 500:
        return "extreme"
    elif frp >= 100:
        return "high"
    elif frp >= 20:
        return "moderate"
    return "low"


def _normalize_hotspot(row: dict, idx: int) -> dict | None:
    """
    Normalize a single FIRMS CSV row into a FireGrid FireEvent dict.
    Returns None if the row is missing critical fields.
    """
    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        frp = float(row.get("frp", 0) or 0)
        acq_date = row.get("acq_date", "")   # e.g. "2026-03-21"
        acq_time = row.get("acq_time", "0000")  # e.g. "2315"

        # Build ISO timestamp from acquisition date + time
        try:
            dt_str = f"{acq_date} {acq_time.zfill(4)}"
            detected_at = datetime.strptime(dt_str, "%Y-%m-%d %H%M").replace(tzinfo=UTC).isoformat()
        except ValueError:
            detected_at = datetime.now(UTC).isoformat()

        province = _assign_province(lat, lon)
        severity = _frp_to_severity(frp)

        # Build a deterministic fire_id from date + grid position
        # Round to 2 decimal places (~1km) to group nearby hotspots
        grid_lat = round(lat, 2)
        grid_lon = round(lon, 2)
        fire_id = f"FIRMS-{acq_date.replace('-', '')}-{abs(int(grid_lat * 100))}-{abs(int(grid_lon * 100))}"

        return {
            "fire_id": fire_id,
            "province": province,
            "name": f"Satellite Hotspot ({province}) #{idx + 1}",
            "status": "out_of_control",   # FIRMS detects active burning
            "severity": severity,
            "latitude": lat,
            "longitude": lon,
            "area_hectares": None,         # FIRMS doesn't provide area
            "frp_mw": frp,                 # Fire Radiative Power in megawatts
            "confidence": row.get("confidence", "n"),
            "satellite": row.get("satellite", "N20"),
            "started_at": detected_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "source": "NASA_FIRMS_VIIRS",
        }
    except (ValueError, KeyError) as e:
        logger.warning(f"Skipping malformed FIRMS row: {e}")
        return None


def fetch_firms_hotspots(
    day_range: int = DEFAULT_DAY_RANGE,
    bbox: str = CANADA_WEST_BBOX,
    min_confidence: str = "n",   # "l"=low, "n"=nominal, "h"=high — filter low quality
) -> list[dict]:
    """
    Fetch active fire hotspots from NASA FIRMS VIIRS over Western Canada.

    Args:
        day_range: Number of days to look back (1-10). Default 1 = last 24h.
        bbox: Bounding box "W,S,E,N". Default covers BC + AB.
        min_confidence: Minimum confidence level to include ('n' = nominal or higher).

    Returns:
        List of normalized FireEvent dicts.

    Rate limit: 1 API call. FIRMS allows 5000 calls/10min — this is very safe.
    """
    api_key = settings.NASA_FIRMS_API_KEY
    if not api_key or api_key in ("dummy_key", ""):
        logger.error("NASA_FIRMS_API_KEY is not set. Cannot fetch real fire data.")
        return []

    url = f"{FIRMS_BASE_URL}/{api_key}/{FIRMS_SOURCE}/{bbox}/{day_range}"
    logger.info(f"Fetching FIRMS data: {url}")

    try:
        # Single request — well within rate limits
        with httpx.Client(timeout=15) as client:
            resp = client.get(url)
        resp.raise_for_status()
    except httpx.TimeoutException:
        logger.error("FIRMS API timed out.")
        return []
    except httpx.HTTPStatusError as e:
        logger.error(f"FIRMS API HTTP error: {e.response.status_code} — {e.response.text[:200]}")
        return []
    except httpx.RequestError as e:
        logger.error(f"FIRMS API request failed: {e}")
        return []

    # Parse CSV response
    reader = csv.DictReader(io.StringIO(resp.text))
    hotspots = []
    for idx, row in enumerate(reader):
        # Filter to nominal/high confidence only to reduce noise
        confidence = row.get("confidence", "n").lower()
        if min_confidence == "n" and confidence == "l":
            continue
        if min_confidence == "h" and confidence != "h":
            continue

        normalized = _normalize_hotspot(row, idx)
        if normalized:
            hotspots.append(normalized)

    logger.info(f"FIRMS: fetched {len(hotspots)} hotspots over bbox {bbox}")
    return hotspots


def get_firms_fires() -> list[dict]:
    """
    Public interface: fetch and return all VIIRS hotspots over Western Canada.
    Used by the fires API endpoint when USE_DUMMY_DATA=False.
    """
    return fetch_firms_hotspots(day_range=DEFAULT_DAY_RANGE)


# ── Manual test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    print("Fetching NASA FIRMS hotspots over Western Canada...")
    fires = get_firms_fires()
    print(f"\nGot {len(fires)} hotspots.\n")
    if fires:
        print("Sample hotspot:")
        print(json.dumps(fires[0], indent=2))
    else:
        print("No hotspots found — might be no active fires right now, or check your API key.")
