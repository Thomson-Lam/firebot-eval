"""
static_dataset.py - Build frozen benchmark datasets from historical wildfire data.

Canonical path:
1. load Alberta historical wildfire incidents from `data/static/raw/`
2. normalize them into snapshot records
3. optionally enrich with CFFDRS fire-danger fields
4. compute environment-variable records for FireEnv

Run once, store the outputs, and train/evaluate only from the cached files.

Example:
    uv run python -m src.ingestion.static_dataset --target-count 100 --output-dir data/static/v1 --raw-alberta-csv data/static/raw/fp-historical-wildfire-data-2006-2025.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import blake2b
from pathlib import Path

from src.ingestion.clean_historical import clean_raw_historical_row_with_reason

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency

    def tqdm(iterable, **_kwargs):
        return iterable


logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/static/v1")
DEFAULT_ALBERTA_CSV = Path("data/static/raw/fp-historical-wildfire-data-2006-2025.csv")

WIND_DIR_TO_DEG = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}

FIRE_TYPE_FACTOR = {
    "ground": 0.8,
    "surface": 1.0,
    "crown": 1.18,
}

WIND_DIRECTIONS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass
class SnapshotBuildResult:
    snapshots: list[dict]
    parameter_records: list[dict]
    output_dir: Path


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _norm(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low), 0.0, 1.0)


def _clean_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_float(value: object) -> float | None:
    text = _clean_str(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_datetime(value: object) -> datetime | None:
    text = _clean_str(value)
    if text is None:
        return None
    try:
        if "T" in text:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_wind_direction(value: object) -> float | None:
    text = _clean_str(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return WIND_DIR_TO_DEG.get(text.upper())


def _wind_direction_8_from_deg(value: float) -> str:
    idx = int((value % 360.0) / 45.0 + 0.5) % 8
    return WIND_DIRECTIONS_8[idx]


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _with_initialization_seeds(record: dict) -> dict:
    seeded = dict(record)
    record_id = str(seeded.get("record_id") or "unknown")
    split = str(seeded.get("split") or "unknown")
    seeded["ignition_seed"] = _stable_seed(record_id, split, "ignition")
    seeded["layout_seed"] = _stable_seed(record_id, split, "layout")
    return seeded


def _single_unique_record(records: list[dict]) -> list[dict]:
    if not records:
        return []
    seen: set[str] = set()
    for record in records:
        record_id = str(record.get("record_id") or "")
        if not record_id or record_id in seen:
            continue
        seen.add(record_id)
        return [record]
    return []


def _estimate_precipitation_mm(condition: str | None) -> float:
    if condition is None:
        return 0.0
    normalized = condition.strip().lower()
    if normalized == "rain showers":
        return 2.0
    if normalized == "cb wet":
        return 1.0
    return 0.0


def _fuel_type_factor(fuel_type: str | None) -> float:
    if not fuel_type:
        return 1.0
    fuel = fuel_type.strip().upper()
    if (
        fuel.startswith("C-")
        or fuel.startswith("C")
        or fuel.startswith("S-")
        or fuel.startswith("S")
    ):
        return 1.12
    if fuel.startswith("M-") or fuel.startswith("M"):
        return 1.06
    if fuel.startswith("O-1B"):
        return 1.08
    if fuel.startswith("O-"):
        return 1.03
    if fuel.startswith("D-"):
        return 0.92
    return 1.0


def _canonical_record_id(fire: dict) -> str:
    fire_id = str(fire.get("fire_id", "unknown"))
    anchor = str(
        fire.get("snapshot_date") or fire.get("updated_at") or fire.get("started_at") or "unknown"
    )
    safe_time = anchor.replace(":", "").replace("-", "").replace("+", "_")
    return f"{fire_id}__{safe_time}"


def split_for_year(year: int | None) -> str | None:
    if year is None:
        return None
    if 2006 <= year <= 2022:
        return "train"
    if year == 2023:
        return "val"
    if 2024 <= year <= 2025:
        return "holdout"
    return None


def _dedupe_fires(fires: list[dict]) -> list[dict]:
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for fire in fires:
        fire_id = str(fire.get("fire_id", ""))
        if not fire_id or fire_id in seen_ids:
            continue
        seen_ids.add(fire_id)
        unique.append(fire)
    return unique


def _fire_priority(fire: dict) -> tuple[float, float, float, str]:
    spread = float(fire.get("observed_spread_rate_m_min") or 0.0)
    size = float(fire.get("assessment_hectares") or fire.get("area_hectares") or 0.0)
    year = float(fire.get("year") or 0.0)
    fire_id = str(fire.get("fire_id", ""))
    return (spread, size, year, fire_id)


def _load_fire_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    return [record for record in records if isinstance(record, dict)]


def _normalize_alberta_row(row: dict) -> dict | None:
    cleaned, _reason = clean_raw_historical_row_with_reason(row)
    if cleaned is None:
        return None

    year = _clean_str(cleaned.get("YEAR"))
    fire_number = _clean_str(cleaned.get("FIRE_NUMBER"))
    lat = _parse_float(cleaned.get("LATITUDE"))
    lon = _parse_float(cleaned.get("LONGITUDE"))
    assessment_dt = _parse_datetime(cleaned.get("ASSESSMENT_DATETIME"))
    assessment_hectares = _parse_float(cleaned.get("ASSESSMENT_HECTARES"))
    current_size = _parse_float(cleaned.get("CURRENT_SIZE"))
    spread_rate = _parse_float(cleaned.get("FIRE_SPREAD_RATE"))
    temp_c = _parse_float(cleaned.get("TEMPERATURE"))
    rh_pct = _parse_float(cleaned.get("RELATIVE_HUMIDITY"))
    wind_direction_deg = _parse_wind_direction(cleaned.get("WIND_DIRECTION"))
    wind_speed = _parse_float(cleaned.get("WIND_SPEED"))

    if not all([year, fire_number]) or lat is None or lon is None or assessment_dt is None:
        return None

    area_hectares = assessment_hectares if assessment_hectares not in (None, 0.0) else current_size
    if area_hectares is None or spread_rate is None or temp_c is None or rh_pct is None:
        return None
    if wind_direction_deg is None or wind_speed is None:
        return None

    started_at = _parse_datetime(cleaned.get("FIRE_START_DATE"))
    discovered_at = _parse_datetime(cleaned.get("DISCOVERED_DATE"))
    reported_at = _parse_datetime(cleaned.get("REPORTED_DATE"))
    dispatch_at = _parse_datetime(cleaned.get("DISPATCH_DATE"))
    arrival_at = _parse_datetime(cleaned.get("IA_ARRIVAL_AT_FIRE_DATE"))
    firefighting_start = _parse_datetime(cleaned.get("FIRE_FIGHTING_START_DATE"))

    fire_id = f"AB-{year}-{fire_number}"
    fire_name = _clean_str(cleaned.get("FIRE_NAME")) or fire_id
    fire_type = (_clean_str(cleaned.get("FIRE_TYPE")) or "Surface").strip()
    fuel_type = _clean_str(cleaned.get("FUEL_TYPE"))
    weather_over_fire = _clean_str(cleaned.get("WEATHER_CONDITIONS_OVER_FIRE"))
    year_int = int(year)
    split = split_for_year(year_int)
    if split is None:
        return None

    return {
        "record_id": fire_id,
        "fire_id": fire_id,
        "year": year_int,
        "split": split,
        "province": "AB",
        "name": fire_name,
        "source": "AB_HISTORICAL_WILDFIRE",
        "status": "historical",
        "snapshot_date": assessment_dt.date().isoformat(),
        "snapshot_datetime": _to_iso(assessment_dt),
        "started_at": _to_iso(started_at),
        "updated_at": _to_iso(assessment_dt),
        "latitude": lat,
        "longitude": lon,
        "area_hectares": float(area_hectares),
        "assessment_hectares": assessment_hectares,
        "current_size": current_size,
        "size_class": _clean_str(cleaned.get("SIZE_CLASS")),
        "observed_spread_rate_m_min": spread_rate,
        "temperature_c": temp_c,
        "relative_humidity_pct": rh_pct,
        "wind_direction_deg": wind_direction_deg,
        "wind_speed_km_h": wind_speed,
        "precipitation_mm": _estimate_precipitation_mm(weather_over_fire),
        "fire_type": fire_type.lower(),
        "fuel_type": fuel_type,
        "weather_conditions_over_fire": weather_over_fire,
        "fire_position_on_slope": _clean_str(cleaned.get("FIRE_POSITION_ON_SLOPE")),
        "fire_origin": _clean_str(cleaned.get("FIRE_ORIGIN")),
        "general_cause": _clean_str(cleaned.get("GENERAL_CAUSE")),
        "activity_class": _clean_str(cleaned.get("ACTIVITY_CLASS")),
        "true_cause": _clean_str(cleaned.get("TRUE_CAUSE")),
        "discovered_date": _to_iso(discovered_at),
        "reported_date": _to_iso(reported_at),
        "dispatch_date": _to_iso(dispatch_at),
        "ia_arrival_at_fire_date": _to_iso(arrival_at),
        "fire_fighting_start_date": _to_iso(firefighting_start),
        "discovered_size": _parse_float(cleaned.get("DISCOVERED_SIZE")),
        "fire_fighting_start_size": _parse_float(cleaned.get("FIRE_FIGHTING_START_SIZE")),
        "initial_action_by": _clean_str(cleaned.get("INITIAL_ACTION_BY")),
        "ia_access": _clean_str(cleaned.get("IA_ACCESS")),
        "bucketing_on_fire": _clean_str(cleaned.get("BUCKETING_ON_FIRE")),
        "distance_from_water_source": _parse_float(cleaned.get("DISTANCE_FROM_WATER_SOURCE")),
    }


def load_alberta_historical_fires(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        msg = f"Alberta historical wildfire CSV not found: {csv_path}"
        raise FileNotFoundError(msg)

    fires: list[dict] = []
    drop_reasons: Counter[str] = Counter()
    yearly_total: Counter[int] = Counter()
    yearly_kept: Counter[int] = Counter()
    raw_rows = 0
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(reader, desc="Cleaning historical rows", unit="row"):
            raw_rows += 1
            year_raw = _clean_str(row.get("YEAR"))
            if year_raw and year_raw.isdigit():
                yearly_total[int(year_raw)] += 1

            cleaned, reason = clean_raw_historical_row_with_reason(row)
            if cleaned is None:
                drop_reasons[reason or "cleaning_failed"] += 1
                continue

            normalized = _normalize_alberta_row(cleaned)
            if normalized is not None:
                fires.append(normalized)
                yearly_kept[int(normalized["year"])] += 1
            else:
                drop_reasons["normalization_failed"] += 1
    logger.info("Loaded %s Alberta historical wildfire incidents", len(fires))
    logger.info(
        "Historical input rows: %s | kept: %s | dropped: %s",
        raw_rows,
        len(fires),
        raw_rows - len(fires),
    )
    if drop_reasons:
        for reason, count in drop_reasons.most_common(10):
            logger.info("Dropped %s rows due to %s", count, reason)
    for year in sorted(yearly_total):
        logger.info(
            "Year %s: kept %s / %s",
            year,
            yearly_kept.get(year, 0),
            yearly_total[year],
        )
    return fires


def collect_candidate_fires(
    fire_records_path: Path | None = None,
    raw_alberta_csv: Path | None = None,
) -> list[dict]:
    """Collect and prioritize historical fire records for snapshot export."""
    if fire_records_path is not None:
        fires = _load_fire_records(fire_records_path)
    else:
        fires = load_alberta_historical_fires(raw_alberta_csv or DEFAULT_ALBERTA_CSV)

    unique = _dedupe_fires(fires)
    unique.sort(key=_fire_priority, reverse=True)
    return unique


def _hours_between(start: str | None, end: str | None) -> float | None:
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    return round((end_dt - start_dt).total_seconds() / 3600.0, 2)


def build_snapshot_record(fire: dict, *, stations: list[dict] | None = None) -> dict | None:
    """Convert one historical fire record into a snapshot record."""
    from src.ingestion.cffdrs import get_cffdrs_for_location

    lat = fire.get("latitude")
    lon = fire.get("longitude")
    if lat is None or lon is None:
        return None

    snapshot_date = fire.get("snapshot_date")
    snapshot_dt = _parse_datetime(fire.get("snapshot_datetime"))
    snapshot_day = snapshot_dt.date() if snapshot_dt is not None else None

    cffdrs = None
    if stations:
        cffdrs = get_cffdrs_for_location(
            float(lat),
            float(lon),
            stations=stations,
            target_date=snapshot_day,
            max_date_offset_days=1,
        )

    record = {
        "record_id": _canonical_record_id(fire),
        "fire_id": fire.get("fire_id"),
        "source": fire.get("source"),
        "province": fire.get("province", "AB"),
        "year": fire.get("year"),
        "split": fire.get("split"),
        "name": fire.get("name"),
        "status": fire.get("status"),
        "snapshot_date": snapshot_date,
        "snapshot_datetime": fire.get("snapshot_datetime"),
        "latitude": float(lat),
        "longitude": float(lon),
        "area_hectares": float(fire["area_hectares"]),
        "assessment_hectares": fire.get("assessment_hectares"),
        "current_size": fire.get("current_size"),
        "size_class": fire.get("size_class"),
        "started_at": fire.get("started_at"),
        "updated_at": fire.get("updated_at"),
        "wind_speed_km_h": float(fire["wind_speed_km_h"]),
        "wind_direction_deg": float(fire["wind_direction_deg"]),
        "temperature_c": float(fire["temperature_c"]),
        "relative_humidity_pct": float(fire["relative_humidity_pct"]),
        "precipitation_mm": float(fire.get("precipitation_mm") or 0.0),
        "observed_spread_rate_m_min": float(fire["observed_spread_rate_m_min"]),
        "fire_type": fire.get("fire_type"),
        "fuel_type": fire.get("fuel_type"),
        "weather_conditions_over_fire": fire.get("weather_conditions_over_fire"),
        "fire_position_on_slope": fire.get("fire_position_on_slope"),
        "fire_origin": fire.get("fire_origin"),
        "general_cause": fire.get("general_cause"),
        "activity_class": fire.get("activity_class"),
        "true_cause": fire.get("true_cause"),
        "discovered_date": fire.get("discovered_date"),
        "reported_date": fire.get("reported_date"),
        "dispatch_date": fire.get("dispatch_date"),
        "ia_arrival_at_fire_date": fire.get("ia_arrival_at_fire_date"),
        "fire_fighting_start_date": fire.get("fire_fighting_start_date"),
        "discovered_size": fire.get("discovered_size"),
        "fire_fighting_start_size": fire.get("fire_fighting_start_size"),
        "initial_action_by": fire.get("initial_action_by"),
        "ia_access": fire.get("ia_access"),
        "bucketing_on_fire": fire.get("bucketing_on_fire"),
        "distance_from_water_source": fire.get("distance_from_water_source"),
        "detection_delay_h": _hours_between(fire.get("started_at"), fire.get("discovered_date")),
        "report_delay_h": _hours_between(fire.get("discovered_date"), fire.get("reported_date")),
        "dispatch_delay_h": _hours_between(fire.get("reported_date"), fire.get("dispatch_date")),
        "ia_travel_delay_h": _hours_between(
            fire.get("dispatch_date"), fire.get("ia_arrival_at_fire_date")
        ),
        "record_quality_flag": "measured",
        "snapshot_generated_at": datetime.now(UTC).isoformat(),
    }

    if cffdrs is not None:
        record.update(
            {
                "fwi": cffdrs.get("fwi"),
                "isi": cffdrs.get("isi"),
                "bui": cffdrs.get("bui"),
                "dc": cffdrs.get("dc"),
                "dmc": cffdrs.get("dmc"),
                "ffmc": cffdrs.get("ffmc"),
                "cffdrs_station_distance_km": cffdrs.get("distance_km"),
                "cffdrs_station_id": cffdrs.get("source_station_id"),
                "cffdrs_station_name": cffdrs.get("source_station"),
                "cffdrs_observation_date": cffdrs.get("observation_date"),
                "cffdrs_date_offset_days": cffdrs.get("date_offset_days"),
                "temporal_alignment_status": "aligned"
                if cffdrs.get("date_offset_days", 0) == 0
                else "near_aligned",
            }
        )
    else:
        record.update(
            {
                "fwi": None,
                "isi": None,
                "bui": None,
                "dc": None,
                "dmc": None,
                "ffmc": None,
                "cffdrs_station_distance_km": None,
                "cffdrs_station_id": None,
                "cffdrs_station_name": None,
                "cffdrs_observation_date": None,
                "cffdrs_date_offset_days": None,
                "temporal_alignment_status": "not_joined",
            }
        )

    required_fields = (
        "wind_speed_km_h",
        "wind_direction_deg",
        "temperature_c",
        "relative_humidity_pct",
        "area_hectares",
        "observed_spread_rate_m_min",
    )
    if any(record.get(field) is None for field in required_fields):
        return None
    return record


def compute_environment_parameters(snapshot: dict) -> dict:
    """Map one snapshot record into deterministic FireEnv parameter fields."""
    observed_spread = float(snapshot["observed_spread_rate_m_min"])
    wind_speed = float(snapshot["wind_speed_km_h"])
    wind_direction_deg = float(snapshot["wind_direction_deg"])
    wind_direction = _wind_direction_8_from_deg(wind_direction_deg)
    temp_c = float(snapshot["temperature_c"])
    rh_pct = float(snapshot["relative_humidity_pct"])
    precip_mm = float(snapshot.get("precipitation_mm") or 0.0)
    area_hectares = float(snapshot["area_hectares"])
    fire_type = str(snapshot.get("fire_type") or "surface").lower()
    fuel_type = snapshot.get("fuel_type")

    spread_norm = _norm(observed_spread, 0.0, 25.0)
    wind_norm = _norm(wind_speed, 0.0, 40.0)
    temp_norm = _norm(temp_c, 0.0, 35.0)
    rh_norm = _norm(rh_pct, 10.0, 95.0)
    rain_norm = _norm(precip_mm, 0.0, 5.0)
    size_norm = _norm(area_hectares, 0.0, 2000.0)

    cffdrs_terms = [
        snapshot.get("isi"),
        snapshot.get("fwi"),
        snapshot.get("bui"),
        snapshot.get("ffmc"),
    ]
    cffdrs_present = any(value is not None for value in cffdrs_terms)
    cffdrs_dryness = 0.0
    if cffdrs_present:
        isi_norm = _norm(float(snapshot.get("isi") or 0.0), 0.0, 25.0)
        fwi_norm = _norm(float(snapshot.get("fwi") or 0.0), 0.0, 40.0)
        bui_norm = _norm(float(snapshot.get("bui") or 0.0), 0.0, 120.0)
        ffmc_norm = _norm(float(snapshot.get("ffmc") or 85.0), 70.0, 96.0)
        cffdrs_dryness = _clamp(
            0.4 * isi_norm + 0.25 * fwi_norm + 0.15 * bui_norm + 0.2 * ffmc_norm,
            0.0,
            1.0,
        )

    weather_score = _clamp(
        0.45 * wind_norm + 0.2 * temp_norm + 0.35 * (1.0 - rh_norm),
        0.0,
        1.0,
    )
    rain_factor = 1.0 - 0.5 * rain_norm
    size_factor = 0.95 + 0.15 * size_norm
    fire_type_factor = FIRE_TYPE_FACTOR.get(fire_type, 1.0)
    fuel_factor = _fuel_type_factor(fuel_type)

    spread_score = _clamp(
        (0.6 * spread_norm + 0.2 * weather_score + 0.1 * size_norm + 0.1 * cffdrs_dryness)
        * rain_factor
        * fire_type_factor
        * fuel_factor,
        0.0,
        1.0,
    )

    base_spread_prob = round(_clamp(0.04 + 0.18 * spread_score, 0.04, 0.22), 4)
    wind_strength = round(_clamp(0.1 + 0.5 * wind_norm, 0.1, 0.6), 4)
    spread_rate_1h_m = round(observed_spread * 60.0, 1)

    if spread_score < 0.33:
        severity_bucket = "low"
    elif spread_score < 0.66:
        severity_bucket = "medium"
    else:
        severity_bucket = "high"

    return {
        "record_id": snapshot["record_id"],
        "fire_id": snapshot.get("fire_id"),
        "source": snapshot.get("source"),
        "province": snapshot.get("province"),
        "year": snapshot.get("year"),
        "split": snapshot.get("split"),
        "base_spread_prob": base_spread_prob,
        "severity_bucket": severity_bucket,
        "wind_direction": wind_direction,
        "wind_strength": wind_strength,
        "spread_rate_1h_m": spread_rate_1h_m,
        "spread_score": round(spread_score, 4),
        "weather_score": round(weather_score, 4),
        "cffdrs_dryness_score": round(cffdrs_dryness, 4),
        "size_factor": round(size_factor, 4),
        "fire_type_factor": round(fire_type_factor, 4),
        "fuel_factor": round(fuel_factor, 4),
        "rain_factor": round(rain_factor, 4),
        "observed_spread_rate_m_min": observed_spread,
        "assessment_hectares": snapshot.get("assessment_hectares"),
        "fire_type": snapshot.get("fire_type"),
        "fuel_type": snapshot.get("fuel_type"),
        "record_quality_flag": snapshot.get("record_quality_flag", "measured"),
    }


def build_static_datasets(
    *,
    target_count: int = 100,
    output_dir: Path | None = None,
    cffdrs_year: int | None = None,
    fire_records_path: Path | None = None,
    raw_alberta_csv: Path | None = None,
) -> SnapshotBuildResult:
    """Run the one-time pipeline and write frozen benchmark artifacts."""
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stations: list[dict] | None = None
    if cffdrs_year is not None:
        from src.ingestion.cffdrs import fetch_cffdrs_stations

        stations = fetch_cffdrs_stations(year=cffdrs_year)
        if not stations:
            logger.warning(
                "CFFDRS station download failed for year %s; continuing without supplementary CFFDRS enrichment.",
                cffdrs_year,
            )

    candidates = collect_candidate_fires(
        fire_records_path=fire_records_path,
        raw_alberta_csv=raw_alberta_csv,
    )
    snapshots: list[dict] = []
    parameter_records: list[dict] = []
    split_counts = {"train": 0, "val": 0, "holdout": 0}

    for fire in tqdm(candidates, desc="Building snapshots", unit="record"):
        split_name = fire.get("split")
        if split_name not in split_counts:
            continue
        if split_counts[split_name] >= target_count:
            continue
        snapshot = build_snapshot_record(fire, stations=stations)
        if snapshot is None:
            continue
        params = compute_environment_parameters(snapshot)
        snapshots.append(snapshot)
        parameter_records.append(params)
        split_counts[split_name] += 1
        if all(count >= target_count for count in split_counts.values()):
            break

    snapshot_payload = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "record_count": len(snapshots),
        "records": snapshots,
    }
    params_payload = {
        "schema_version": 3,
        "generated_at": datetime.now(UTC).isoformat(),
        "record_count": len(parameter_records),
        "records": parameter_records,
    }
    seeded_parameter_records = [_with_initialization_seeds(record) for record in parameter_records]
    seeded_params_payload = {
        "schema_version": 3,
        "generated_at": datetime.now(UTC).isoformat(),
        "record_count": len(seeded_parameter_records),
        "records": seeded_parameter_records,
    }

    snapshot_path = output_dir / "snapshot_records.json"
    params_path = output_dir / "scenario_parameter_records.json"
    seeded_params_path = output_dir / "scenario_parameter_records_seeded.json"
    snapshot_path.write_text(json.dumps(snapshot_payload, indent=2))
    params_path.write_text(json.dumps(params_payload, indent=2))
    seeded_params_path.write_text(json.dumps(seeded_params_payload, indent=2))

    split_names = ("train", "val", "holdout")
    for split_name in split_names:
        split_snapshots = [record for record in snapshots if record.get("split") == split_name]
        split_params = [record for record in parameter_records if record.get("split") == split_name]
        split_seeded_params = [
            record for record in seeded_parameter_records if record.get("split") == split_name
        ]
        if split_name == "holdout":
            split_seeded_params = _single_unique_record(split_seeded_params)
        (output_dir / f"snapshot_records_{split_name}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "split": split_name,
                    "record_count": len(split_snapshots),
                    "records": split_snapshots,
                },
                indent=2,
            )
        )
        (output_dir / f"scenario_parameter_records_{split_name}.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "split": split_name,
                    "record_count": len(split_params),
                    "records": split_params,
                },
                indent=2,
            )
        )
        (output_dir / f"scenario_parameter_records_seeded_{split_name}.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "split": split_name,
                    "record_count": len(split_seeded_params),
                    "records": split_seeded_params,
                },
                indent=2,
            )
        )

    logger.info("Wrote %s snapshot records to %s", len(snapshots), snapshot_path)
    logger.info("Wrote %s scenario parameter records to %s", len(parameter_records), params_path)
    logger.info(
        "Wrote %s seeded scenario parameter records to %s",
        len(seeded_parameter_records),
        seeded_params_path,
    )
    for split_name in split_names:
        logger.info(
            "Split %s: %s records",
            split_name,
            sum(1 for record in parameter_records if record.get("split") == split_name),
        )
    if not parameter_records:
        logger.warning(
            "No scenario parameter records were built. Check whether the Alberta historical file has valid assessment fields or whether your optional CFFDRS join is too sparse."
        )
    return SnapshotBuildResult(
        snapshots=snapshots, parameter_records=parameter_records, output_dir=output_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen wildfire benchmark datasets")
    parser.add_argument(
        "--target-count", type=int, default=100, help="Target number of records to export per split"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for snapshot and parameter JSON files",
    )
    parser.add_argument(
        "--cffdrs-year",
        type=int,
        default=None,
        help="Optional CFFDRS observation year for supplementary danger-index enrichment",
    )
    parser.add_argument(
        "--fire-records",
        type=Path,
        default=None,
        help="Optional JSON file of normalized fire records to use instead of the Alberta historical CSV",
    )
    parser.add_argument(
        "--raw-alberta-csv",
        type=Path,
        default=DEFAULT_ALBERTA_CSV,
        help="Path to the raw Alberta historical wildfire CSV",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = build_static_datasets(
        target_count=args.target_count,
        output_dir=args.output_dir,
        cffdrs_year=args.cffdrs_year,
        fire_records_path=args.fire_records,
        raw_alberta_csv=args.raw_alberta_csv,
    )
    print(
        f"Built {len(result.parameter_records)} scenario parameter records in {result.output_dir}"
    )


if __name__ == "__main__":
    main()
