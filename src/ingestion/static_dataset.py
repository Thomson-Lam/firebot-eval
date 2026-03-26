"""
static_dataset.py - One-time snapshot export and offline environment parameter builder.

This module converts live ingestion sources into frozen benchmark artifacts:

1. normalized snapshot records
2. scenario parameter records for FireEnv episode setup

Run once, store the outputs, and train/evaluate only from the cached files.

Example:
    uv run python -m src.ingestion.static_dataset --target-count 100
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/static")


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


def _canonical_record_id(fire: dict) -> str:
    fire_id = str(fire.get("fire_id", "unknown"))
    updated_at = str(fire.get("updated_at", "unknown"))
    safe_time = updated_at.replace(":", "").replace("-", "").replace("+", "_")
    return f"{fire_id}__{safe_time}"


def _dedupe_fires(fires: list[dict]) -> list[dict]:
    seen_ids: set[str] = set()
    seen_cells: set[tuple[str, float, float]] = set()
    unique: list[dict] = []
    for fire in fires:
        fire_id = str(fire.get("fire_id", ""))
        if fire_id and fire_id in seen_ids:
            continue
        lat = fire.get("latitude")
        lon = fire.get("longitude")
        if lat is None or lon is None:
            continue
        cell_key = (
            str(fire.get("province", "OTHER")),
            round(float(lat), 2),
            round(float(lon), 2),
        )
        if cell_key in seen_cells:
            continue
        seen_ids.add(fire_id)
        seen_cells.add(cell_key)
        unique.append(fire)
    return unique


def _fire_priority(fire: dict) -> tuple[int, float, str]:
    has_area = 1 if fire.get("area_hectares") not in (None, 0, 0.0, "") else 0
    area = float(fire.get("area_hectares") or 0.0)
    source = str(fire.get("source", "zzz"))
    return (has_area, area, source)


def _load_fire_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    return [record for record in records if isinstance(record, dict)]


def collect_candidate_fires(
    target_count: int,
    include_firms: bool = False,
    fire_records_path: Path | None = None,
) -> list[dict]:
    """Collect and prioritize candidate fire records for snapshot export."""
    if fire_records_path is not None:
        fires = _load_fire_records(fire_records_path)
    else:
        from src.ingestion.cwfis import get_cwfis_fires

        fires = list(get_cwfis_fires())

    if include_firms and fire_records_path is None:
        try:
            from src.ingestion.firms import fetch_firms_hotspots

            fires.extend(fetch_firms_hotspots(day_range=7))
        except Exception as exc:
            logger.warning("Skipping FIRMS candidate collection: %s", exc)

    unique = _dedupe_fires(fires)
    unique.sort(key=_fire_priority, reverse=True)
    return unique[:target_count]


def build_snapshot_record(fire: dict, *, stations: list[dict]) -> dict | None:
    """Enrich one fire record into a normalized snapshot record."""
    from src.ingestion.cffdrs import get_cffdrs_for_location
    from src.ingestion.weather import get_fire_weather

    lat = fire.get("latitude")
    lon = fire.get("longitude")
    if lat is None or lon is None:
        return None

    weather = get_fire_weather(float(lat), float(lon))
    cffdrs = get_cffdrs_for_location(float(lat), float(lon), stations=stations)

    if not weather or not cffdrs:
        return None

    area_hectares = fire.get("area_hectares")
    quality_flag = "measured"
    if area_hectares in (None, ""):
        frp = fire.get("frp_mw")
        if frp is None:
            return None
        area_hectares = round(max(25.0, float(frp) * 2.5), 1)
        quality_flag = "area_imputed_from_frp"

    record = {
        "record_id": _canonical_record_id(fire),
        "fire_id": fire.get("fire_id"),
        "source": fire.get("source"),
        "province": fire.get("province"),
        "name": fire.get("name"),
        "status": fire.get("status"),
        "latitude": float(lat),
        "longitude": float(lon),
        "area_hectares": float(area_hectares),
        "started_at": fire.get("started_at"),
        "updated_at": fire.get("updated_at"),
        "wind_speed_km_h": weather.get("wind_speed_km_h"),
        "wind_direction_deg": weather.get("wind_direction_deg"),
        "temperature_c": weather.get("temperature_c"),
        "relative_humidity_pct": weather.get("relative_humidity_pct"),
        "precipitation_mm": weather.get("precipitation_mm"),
        "surface_pressure_hpa": weather.get("surface_pressure_hpa"),
        "dew_point_c": weather.get("dew_point_c"),
        "fwi": cffdrs.get("fwi"),
        "isi": cffdrs.get("isi"),
        "bui": cffdrs.get("bui"),
        "dc": cffdrs.get("dc"),
        "dmc": cffdrs.get("dmc"),
        "ffmc": cffdrs.get("ffmc"),
        "cffdrs_station_distance_km": cffdrs.get("distance_km"),
        "cffdrs_station_id": cffdrs.get("source_station_id"),
        "cffdrs_station_name": cffdrs.get("source_station"),
        "frp_mw": fire.get("frp_mw"),
        "record_quality_flag": quality_flag,
        "snapshot_generated_at": datetime.now(UTC).isoformat(),
    }

    required_fields = (
        "wind_speed_km_h",
        "wind_direction_deg",
        "temperature_c",
        "relative_humidity_pct",
        "precipitation_mm",
        "fwi",
        "isi",
        "bui",
        "area_hectares",
    )
    if any(record.get(field) is None for field in required_fields):
        return None
    return record


def compute_environment_parameters(snapshot: dict) -> dict:
    """Map one snapshot record into deterministic FireEnv parameter fields."""
    wind_speed = float(snapshot["wind_speed_km_h"])
    wind_dir_deg = float(snapshot["wind_direction_deg"])
    temp_c = float(snapshot["temperature_c"])
    rh_pct = float(snapshot["relative_humidity_pct"])
    precip_mm = float(snapshot["precipitation_mm"])
    fwi = float(snapshot["fwi"])
    isi = float(snapshot["isi"])
    bui = float(snapshot["bui"])
    area_hectares = float(snapshot["area_hectares"])
    ffmc = float(snapshot.get("ffmc") or 85.0)

    isi_norm = _norm(isi, 0.0, 25.0)
    fwi_norm = _norm(fwi, 0.0, 40.0)
    bui_norm = _norm(bui, 0.0, 120.0)
    ffmc_norm = _norm(ffmc, 70.0, 96.0)
    wind_norm = _norm(wind_speed, 0.0, 40.0)
    temp_norm = _norm(temp_c, 5.0, 35.0)
    rh_norm = _norm(rh_pct, 15.0, 90.0)
    rain_norm = _norm(precip_mm, 0.0, 10.0)
    area_norm = _norm(area_hectares, 0.0, 20000.0)

    dryness_score = _clamp(
        0.45 * isi_norm + 0.25 * fwi_norm + 0.15 * bui_norm + 0.15 * ffmc_norm,
        0.0,
        1.0,
    )
    rh_factor = 1.0 - 0.65 * rh_norm
    rain_factor = 1.0 - 0.55 * rain_norm
    temp_factor = 0.85 + 0.35 * temp_norm
    wind_factor = 0.9 + 0.85 * wind_norm
    size_factor = 0.95 + 0.1 * area_norm

    spread_score = _clamp(
        dryness_score * rh_factor * rain_factor * temp_factor * wind_factor * size_factor,
        0.0,
        1.0,
    )
    spread_rate_1h_m = round(150.0 + 2850.0 * spread_score, 1)
    base_spread_prob = round(_clamp(0.04 + 0.18 * spread_score, 0.04, 0.22), 4)
    wind_strength = round(_clamp(0.1 + 0.5 * wind_norm, 0.1, 0.6), 4)

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
        "base_spread_prob": base_spread_prob,
        "severity_bucket": severity_bucket,
        "wind_dir_deg": round(wind_dir_deg, 2),
        "wind_strength": wind_strength,
        "spread_rate_1h_m": spread_rate_1h_m,
        "spread_score": round(spread_score, 4),
        "dryness_score": round(dryness_score, 4),
        "rh_factor": round(rh_factor, 4),
        "rain_factor": round(rain_factor, 4),
        "temp_factor": round(temp_factor, 4),
        "wind_factor": round(wind_factor, 4),
        "size_factor": round(size_factor, 4),
        "record_quality_flag": snapshot.get("record_quality_flag", "measured"),
    }


def build_static_datasets(
    *,
    target_count: int = 100,
    output_dir: Path | None = None,
    include_firms: bool = False,
    cffdrs_year: int | None = None,
    fire_records_path: Path | None = None,
) -> SnapshotBuildResult:
    """Run the one-time pipeline and write frozen benchmark artifacts."""
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.ingestion.cffdrs import fetch_cffdrs_stations

    stations = fetch_cffdrs_stations(year=cffdrs_year)
    if not stations:
        msg = "CFFDRS station download failed; cannot build static dataset"
        raise RuntimeError(msg)

    candidates = collect_candidate_fires(
        target_count=target_count * 2,
        include_firms=include_firms,
        fire_records_path=fire_records_path,
    )
    snapshots: list[dict] = []
    parameter_records: list[dict] = []

    for fire in candidates:
        if len(snapshots) >= target_count:
            break
        snapshot = build_snapshot_record(fire, stations=stations)
        if snapshot is None:
            continue
        params = compute_environment_parameters(snapshot)
        snapshots.append(snapshot)
        parameter_records.append(params)

    snapshot_payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "record_count": len(snapshots),
        "records": snapshots,
    }
    params_payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "record_count": len(parameter_records),
        "records": parameter_records,
    }

    snapshot_path = output_dir / "snapshot_records.json"
    params_path = output_dir / "scenario_parameter_records.json"
    snapshot_path.write_text(json.dumps(snapshot_payload, indent=2))
    params_path.write_text(json.dumps(params_payload, indent=2))

    logger.info("Wrote %s snapshot records to %s", len(snapshots), snapshot_path)
    logger.info("Wrote %s scenario parameter records to %s", len(parameter_records), params_path)
    return SnapshotBuildResult(
        snapshots=snapshots, parameter_records=parameter_records, output_dir=output_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen wildfire benchmark datasets")
    parser.add_argument(
        "--target-count", type=int, default=100, help="Target number of records to export"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for snapshot and parameter JSON files",
    )
    parser.add_argument(
        "--include-firms",
        action="store_true",
        help="Include FIRMS hotspots as fallback candidates when available",
    )
    parser.add_argument(
        "--cffdrs-year",
        type=int,
        default=None,
        help="Override CFFDRS observation year for reproducible exports",
    )
    parser.add_argument(
        "--fire-records",
        type=Path,
        default=None,
        help="Optional JSON file of normalized fire records to use instead of live incident collection",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = build_static_datasets(
        target_count=args.target_count,
        output_dir=args.output_dir,
        include_firms=args.include_firms,
        cffdrs_year=args.cffdrs_year,
        fire_records_path=args.fire_records,
    )
    print(
        f"Built {len(result.parameter_records)} scenario parameter records in {result.output_dir}"
    )


if __name__ == "__main__":
    main()
