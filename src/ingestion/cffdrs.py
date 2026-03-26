"""
cffdrs.py — Canadian Forest Fire Danger Rating System (CFFDRS) index ingestion.

Downloads daily fire weather indices from the CWFIS open data portal.
No API key required — these are public NRCan datasets.

Fetches:
  - FWI (Fire Weather Index)      — overall fire danger
  - ISI (Initial Spread Index)    — rate of fire spread
  - BUI (Buildup Index)           — available fuel
  - DC  (Drought Code)            — deep organic layer dryness
  - DMC (Duff Moisture Code)      — upper organic layer dryness
  - FFMC (Fine Fuel Moisture Code)— fine surface fuel dryness

Data source: https://cwfis.cfs.nrcan.gc.ca/downloads/fwi_obs/
File format: CSV with weather station observations

Run to test:
    uv run python -m src.ingestion.cffdrs
"""

import csv
import io
import logging
import math
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)

# ── CWFIS CFFDRS Observation Files ────────────────────────────────────────────
# Today's FWI station observations (updates ~13:00 UTC daily)
# The URL uses YYYY-MM-DD format for the date
CFFDRS_BASE_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/fwi_obs/cwfis_allstn{year}.csv"

# Provinces we care about
TARGET_PROVINCES = {"BC", "AB"}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in km between two (lat, lon) pairs."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_float(val: str) -> float | None:
    """Safely parse a float from a CSV string, returning None if invalid."""
    try:
        f = float(val)
        return None if f < -900 else f  # CWFIS uses -999 as missing value sentinel
    except (ValueError, TypeError):
        return None


def fetch_cffdrs_stations(year: int | None = None) -> list[dict]:
    """
    Download the full CWFIS annual FWI observation CSV and parse it.

    Returns a list of station dicts with lat, lon, and all CFFDRS indices.
    Filters to BC + AB stations only.
    """
    if year is None:
        year = datetime.now(UTC).year

    url = CFFDRS_BASE_URL.format(year=year)
    logger.info(f"Fetching CFFDRS station data from {url}")

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url)
        resp.raise_for_status()
    except httpx.TimeoutException:
        logger.error("CFFDRS download timed out.")
        return []
    except httpx.HTTPStatusError as e:
        logger.error(f"CFFDRS HTTP {e.response.status_code}")
        # Try prior year as fallback (may not have current year yet)
        if e.response.status_code == 404 and year == datetime.now(UTC).year:
            logger.info("Trying prior year as fallback...")
            return fetch_cffdrs_stations(year - 1)
        return []
    except httpx.RequestError as e:
        logger.error(f"CFFDRS request failed: {e}")
        return []

    text = resp.text.lstrip("\ufeff")  # strip BOM
    reader = csv.DictReader(io.StringIO(text))

    stations = []
    for row in reader:
        # Province filter
        prov = row.get("prov", row.get("PROV", "")).strip().upper()
        if prov not in TARGET_PROVINCES:
            continue

        lat = _parse_float(row.get("lat", row.get("LAT", "")))
        lon = _parse_float(row.get("lon", row.get("LON", "")))
        if lat is None or lon is None:
            continue

        station = {
            "station_id": row.get("id", row.get("ID", "")).strip(),
            "station_name": row.get("name", row.get("NAME", "")).strip(),
            "province": prov,
            "latitude": lat,
            "longitude": lon,
            "date": row.get("date", row.get("DATE", "")).strip(),
            # Core CFFDRS indices
            "fwi":  _parse_float(row.get("fwi",  row.get("FWI",  ""))),
            "isi":  _parse_float(row.get("isi",  row.get("ISI",  ""))),
            "bui":  _parse_float(row.get("bui",  row.get("BUI",  ""))),
            "dc":   _parse_float(row.get("dc",   row.get("DC",   ""))),
            "dmc":  _parse_float(row.get("dmc",  row.get("DMC",  ""))),
            "ffmc": _parse_float(row.get("ffmc", row.get("FFMC", ""))),
            # Observed weather at station
            "temp_c":     _parse_float(row.get("temp", row.get("TEMP",  ""))),
            "rh_pct":     _parse_float(row.get("rh",   row.get("RH",    ""))),
            "ws_km_h":    _parse_float(row.get("ws",   row.get("WS",    ""))),
            "precip_mm":  _parse_float(row.get("prec", row.get("PREC",  ""))),
        }
        stations.append(station)

    logger.info(f"CFFDRS: loaded {len(stations)} BC/AB stations")
    return stations


def get_cffdrs_for_location(
    latitude: float,
    longitude: float,
    stations: list[dict] | None = None,
    max_radius_km: float = 200.0,
) -> dict | None:
    """
    Find the nearest CWFIS weather station and return its CFFDRS indices.

    Args:
        latitude:     Fire latitude
        longitude:    Fire longitude
        stations:     Pre-fetched station list (to avoid repeat downloads)
        max_radius_km: Maximum distance to consider a station valid

    Returns:
        Dict with FWI, ISI, BUI, DC, DMC, FFMC and station metadata,
        or None if no station found within max_radius_km.
    """
    if stations is None:
        stations = fetch_cffdrs_stations()

    if not stations:
        return None

    # Find nearest station with valid FWI data
    best = None
    best_dist = float("inf")

    for stn in stations:
        if stn.get("fwi") is None:
            continue  # skip stations with missing data
        dist = _haversine_km(latitude, longitude, stn["latitude"], stn["longitude"])
        if dist < best_dist:
            best_dist = dist
            best = stn

    if best is None or best_dist > max_radius_km:
        logger.warning(
            f"No CFFDRS station within {max_radius_km}km of ({latitude}, {longitude}). "
            f"Closest was {best_dist:.0f}km away."
        )
        return None

    return {
        "source_station": best["station_name"],
        "source_station_id": best["station_id"],
        "distance_km": round(best_dist, 1),
        "date": best["date"],
        "fwi":  best["fwi"],
        "isi":  best["isi"],
        "bui":  best["bui"],
        "dc":   best["dc"],
        "dmc":  best["dmc"],
        "ffmc": best["ffmc"],
    }


def get_cffdrs_for_fires(fires: list[dict]) -> dict[str, dict]:
    """
    Batch fetch CFFDRS indices for multiple fires.
    Downloads station data once, then does a nearest-station lookup per fire.

    Returns dict keyed by fire_id → CFFDRS indices dict.
    """
    # Download once, reuse for all fires
    stations = fetch_cffdrs_stations()

    if not stations:
        logger.warning("No CFFDRS station data available — skipping all fires")
        return {}

    results = {}
    for fire in fires:
        fire_id = fire.get("fire_id", "unknown")
        lat = fire.get("latitude")
        lon = fire.get("longitude")

        if lat is None or lon is None:
            continue

        cffdrs = get_cffdrs_for_location(lat, lon, stations=stations)
        if cffdrs:
            results[fire_id] = cffdrs

    return results


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_fires = [
        {"fire_id": "BC-2026-001", "name": "Okanagan Ridge Fire",   "latitude": 49.9071,  "longitude": -119.496},
        {"fire_id": "BC-2026-002", "name": "Kamloops Plateau Fire", "latitude": 50.6745,  "longitude": -120.3273},
        {"fire_id": "BC-2026-003", "name": "Fraser Valley Approach","latitude": 49.3845,  "longitude": -121.4483},
        {"fire_id": "AB-2026-001", "name": "Peace River Complex",   "latitude": 56.2370,  "longitude": -117.2900},
    ]

    print("Fetching CFFDRS fire danger indices from CWFIS/NRCan...\n")
    results = get_cffdrs_for_fires(test_fires)

    for fire_id, cffdrs in results.items():
        print(f"━━━ {fire_id} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  Nearest station: {cffdrs['source_station']} ({cffdrs['distance_km']} km away)")
        print(f"  Date: {cffdrs['date']}")
        print(f"  FWI:  {cffdrs['fwi']}  (Fire Weather Index — overall danger)")
        print(f"  ISI:  {cffdrs['isi']}  (Initial Spread Index — spread rate)")
        print(f"  BUI:  {cffdrs['bui']}  (Buildup Index — fuel availability)")
        print(f"  DC:   {cffdrs['dc']}   (Drought Code — deep dryness)")
        print(f"  DMC:  {cffdrs['dmc']}  (Duff Moisture Code)")
        print(f"  FFMC: {cffdrs['ffmc']} (Fine Fuel Moisture Code)")
        print()

    print(f"CFFDRS data fetched for {len(results)}/{len(test_fires)} fires.")

    if not results:
        print("\nNo results — CWFIS may not have today's data yet (updates ~13:00 UTC),")
        print("   or fire season hasn't started. Try again later or check manually:")
        print(f"   {CFFDRS_BASE_URL.format(year=datetime.now().year)}")
