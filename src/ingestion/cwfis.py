"""
cwfis.py — Canadian Wildland Fire Information System (CWFIS) ingestion.

Downloads the official active fire list from the NRCan open data portal.
No API key required — this is a public government dataset.

Data source: https://cwfis.cfs.nrcan.gc.ca/downloads/activefires/
Updated daily around 13:00 UTC by Natural Resources Canada.

Run from backend/ to test:
    uv run python -m src.ingestion.cwfis
"""

import csv
import io
import logging
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)

# ── CWFIS Open Data URLs ──────────────────────────────────────────────────────
# This CSV is updated daily by NRCan. No auth required.
CWFIS_ACTIVEFIRES_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/activefires/activefires.csv"

# Only ingest BC and AB for our scope
TARGET_PROVINCES = {"BC", "AB"}


def _severity_from_status(status: str) -> str:
    """Map CWFIS fire status codes to FireGrid severity labels."""
    s = status.upper().strip()
    if "OUT OF CONTROL" in s or s == "OC":
        return "extreme"
    elif "BEING HELD" in s or s == "BH":
        return "high"
    elif "UNDER CONTROL" in s or s == "UC":
        return "moderate"
    return "low"


def _normalize_cwfis_row(row: dict) -> dict | None:
    """
    Normalize a single CWFIS CSV row into a FireGrid FireEvent dict.

    CWFIS CSV columns (as of 2024):
        agency, firename, lat, lon, startdate, hectares, status, stage_of_control
    Returns None if critical fields are missing.
    """
    try:
        # Province comes from the agency code (e.g. "BC", "AB", "ON")
        agency = row.get("agency", "").strip().upper()
        province = agency[:2] if len(agency) >= 2 else "OTHER"

        # Filter to just BC + AB
        if province not in TARGET_PROVINCES:
            return None

        lat = float(row.get("lat") or row.get("latitude", 0))
        lon = float(row.get("lon") or row.get("longitude", 0))

        if lat == 0 and lon == 0:
            return None

        fire_name = row.get("firename", "").strip() or f"CWFIS Fire ({province})"
        fire_number = row.get("firenumber", "").strip() or row.get("firename", "UNK")
        status = row.get("stage_of_control", row.get("status", "Unknown")).strip()
        hectares_raw = row.get("hectares", row.get("area", "0")) or "0"
        hectares = float(hectares_raw) if hectares_raw else 0.0
        start_date = row.get("startdate", row.get("discovered", "")).strip()

        # Build ISO timestamp from start date
        try:
            started_at = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC).isoformat()
        except (ValueError, TypeError):
            started_at = datetime.now(UTC).isoformat()

        # Build a stable fire_id using province + fire number
        safe_num = fire_number.replace(" ", "_").replace("/", "-")
        fire_id = f"CWFIS-{province}-{safe_num}"

        return {
            "fire_id": fire_id,
            "province": province,
            "name": fire_name,
            "status": status.lower().replace(" ", "_"),
            "severity": _severity_from_status(status),
            "latitude": lat,
            "longitude": lon,
            "area_hectares": hectares,
            "started_at": started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "source": "CWFIS_NRCAN",
        }
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Skipping malformed CWFIS row: {e} | row={row}")
        return None


def fetch_cwfis_activefires() -> list[dict]:
    """
    Download and parse the CWFIS active fires CSV.
    Filters to BC + AB only.

    Returns:
        List of normalized FireEvent dicts.

    No rate limit — this is a static daily file. One HTTP GET.
    """
    logger.info(f"Fetching CWFIS active fires from {CWFIS_ACTIVEFIRES_URL}")

    try:
        with httpx.Client(timeout=20) as client:
            resp = client.get(CWFIS_ACTIVEFIRES_URL)
        resp.raise_for_status()
    except httpx.TimeoutException:
        logger.error("CWFIS download timed out.")
        return []
    except httpx.HTTPStatusError as e:
        logger.error(f"CWFIS HTTP error: {e.response.status_code}")
        return []
    except httpx.RequestError as e:
        logger.error(f"CWFIS request failed: {e}")
        return []

    # CWFIS CSV sometimes has a BOM or extra header lines — strip them
    text = resp.text.lstrip("\ufeff")  # strip BOM if present
    reader = csv.DictReader(io.StringIO(text))

    fires = []
    for row in reader:
        normalized = _normalize_cwfis_row(row)
        if normalized:
            fires.append(normalized)

    logger.info(f"CWFIS: found {len(fires)} active fires in BC + AB")
    return fires


def get_cwfis_fires() -> list[dict]:
    """
    Public interface: fetch and return active fires from CWFIS over BC + AB.
    Called by the API endpoint when real data is requested.
    """
    return fetch_cwfis_activefires()


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    print("Fetching CWFIS active fires (NRCan)...")
    fires = get_cwfis_fires()
    print(f"\nGot {len(fires)} fires in BC + AB.\n")
    if fires:
        print("Sample fire:")
        print(json.dumps(fires[0], indent=2))
    else:
        print("No active fires right now (or off-season). CSV was empty for BC/AB.")
