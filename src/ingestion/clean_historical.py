"""Utilities for lightweight cleaning of Alberta historical wildfire rows."""

from __future__ import annotations

REQUIRED_RAW_FIELDS = (
    "YEAR",
    "FIRE_NUMBER",
    "LATITUDE",
    "LONGITUDE",
    "ASSESSMENT_DATETIME",
    "FIRE_SPREAD_RATE",
    "TEMPERATURE",
    "RELATIVE_HUMIDITY",
    "WIND_DIRECTION",
    "WIND_SPEED",
)


def clean_raw_historical_row(row: dict) -> dict | None:
    """Trim strings and drop rows missing required canonical fields.

    This stays intentionally lightweight: strip blanks, normalize empty strings,
    and reject rows that lack core assessment-time fields needed by the builder.
    """
    cleaned: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, str):
            stripped = value.strip()
            cleaned[key] = stripped if stripped != "" else None
        else:
            cleaned[key] = value

    for field in REQUIRED_RAW_FIELDS:
        if cleaned.get(field) in (None, ""):
            return None

    area_fields_present = cleaned.get("ASSESSMENT_HECTARES") not in (None, "") or cleaned.get(
        "CURRENT_SIZE"
    ) not in (None, "")
    if not area_fields_present:
        return None

    return cleaned
