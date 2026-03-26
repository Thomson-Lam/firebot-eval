"""
dummy.py — Dummy data generators for all FireGrid data types.

These helpers keep the API usable when live data sources are unavailable.
"""

import random
from datetime import UTC, datetime, timedelta

# ── Seed for reproducible dummy data ───────────────────────────────────────────
random.seed(42)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _rand_bc_coord() -> tuple[float, float]:
    """Random coordinate inside British Columbia."""
    lat = random.uniform(49.0, 59.0)
    lon = random.uniform(-139.0, -114.0)
    return round(lat, 5), round(lon, 5)


def _rand_ab_coord() -> tuple[float, float]:
    """Random coordinate inside Alberta."""
    lat = random.uniform(49.0, 60.0)
    lon = random.uniform(-120.0, -110.0)
    return round(lat, 5), round(lon, 5)


def _rand_coord() -> tuple[float, float]:
    return random.choice([_rand_bc_coord, _rand_ab_coord])()


# ── Fire Incidents ───────────────────────────────────────────────────────────────

DUMMY_FIRE_INCIDENTS = [
    {
        "fire_id": "BC-2026-001",
        "province": "BC",
        "name": "Okanagan Ridge Fire",
        "status": "out_of_control",
        "severity": "extreme",
        "latitude": 49.9071,
        "longitude": -119.4960,
        "area_hectares": 4200.0,
        "started_at": (datetime.now(UTC) - timedelta(hours=18)).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "source": "dummy",
    },
    {
        "fire_id": "BC-2026-002",
        "province": "BC",
        "name": "Kamloops Plateau Fire",
        "status": "being_held",
        "severity": "high",
        "latitude": 50.6745,
        "longitude": -120.3273,
        "area_hectares": 800.0,
        "started_at": (datetime.now(UTC) - timedelta(hours=6)).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "source": "dummy",
    },
    {
        "fire_id": "BC-2026-003",
        "province": "BC",
        "name": "Fraser Valley Approach",
        "status": "being_held",
        "severity": "high",
        "latitude": 49.3845,
        "longitude": -121.4483,
        "area_hectares": 250.0,
        "started_at": (datetime.now(UTC) - timedelta(hours=8)).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "source": "dummy",
    },
    {
        "fire_id": "AB-2026-001",
        "province": "AB",
        "name": "Peace River Complex",
        "status": "out_of_control",
        "severity": "extreme",
        "latitude": 56.2370,
        "longitude": -117.2900,
        "area_hectares": 12500.0,
        "started_at": (datetime.now(UTC) - timedelta(hours=36)).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "source": "dummy",
    },
]


def get_dummy_fires() -> list[dict]:
    return DUMMY_FIRE_INCIDENTS


def get_dummy_fire_by_id(fire_id: str) -> dict | None:
    return next((f for f in DUMMY_FIRE_INCIDENTS if f["fire_id"] == fire_id), None)


# ── Burn Probability Grid ────────────────────────────────────────────────────────

def get_dummy_burn_probability(fire_id: str) -> dict:
    """
    Returns a 5x5 grid of burn probability cells around the fire origin.
    In production this will be replaced by XGBoost model inference.
    """
    fire = get_dummy_fire_by_id(fire_id)
    if not fire:
        return {}

    origin_lat = fire["latitude"]
    origin_lon = fire["longitude"]
    grid_step = 0.05  # ~5km cells

    cells = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Probability peaks at origin and decays outward (simulates wind pushing east)
            dist = abs(i) + abs(j - 1)  # offset east to simulate wind direction
            probability = max(0.0, round(0.95 - (dist * 0.18) + random.uniform(-0.05, 0.05), 3))
            cells.append({
                "latitude": round(origin_lat + i * grid_step, 5),
                "longitude": round(origin_lon + j * grid_step, 5),
                "burn_probability": probability,
                "cell_size_km": 5.0,
            })

    return {
        "fire_id": fire_id,
        "model": "dummy_v0",
        "horizon_hours": 24,
        "generated_at": datetime.now(UTC).isoformat(),
        "wind_speed_kmh": random.uniform(20, 60),
        "wind_direction_deg": random.uniform(220, 280),  # SW winds, pushing NE
        "cells": cells,
    }


# ── Asset Inventory ──────────────────────────────────────────────────────────────

DUMMY_ASSETS = [
    # Ground crews
    {"asset_id": "CREW-001", "type": "hotshot_crew", "size": 20, "status": "available", "latitude": 49.8880, "longitude": -119.4960, "province": "BC"},
    {"asset_id": "CREW-002", "type": "hotshot_crew", "size": 20, "status": "available", "latitude": 50.6745, "longitude": -120.1010, "province": "BC"},
    {"asset_id": "CREW-003", "type": "hotshot_crew", "size": 20, "status": "deployed", "latitude": 56.1200, "longitude": -117.3500, "province": "AB"},
    {"asset_id": "CREW-004", "type": "initial_attack_crew", "size": 6, "status": "available", "latitude": 49.9500, "longitude": -119.5000, "province": "BC"},
    # Heavy equipment
    {"asset_id": "DOZER-001", "type": "bulldozer", "size": 1, "status": "available", "latitude": 49.9100, "longitude": -119.5200, "province": "BC"},
    {"asset_id": "DOZER-002", "type": "bulldozer", "size": 1, "status": "available", "latitude": 56.2000, "longitude": -117.2500, "province": "AB"},
    # Aircraft
    {"asset_id": "AIR-001", "type": "water_bomber", "size": 1, "status": "available", "latitude": 49.4627, "longitude": -119.5720, "province": "BC"},
    {"asset_id": "AIR-002", "type": "water_bomber", "size": 1, "status": "deployed", "latitude": 56.2370, "longitude": -117.2800, "province": "AB"},
    {"asset_id": "AIR-003", "type": "helicopter", "size": 1, "status": "available", "latitude": 50.7000, "longitude": -120.3500, "province": "BC"},
]


def get_dummy_assets(province: str | None = None) -> list[dict]:
    if province:
        return [a for a in DUMMY_ASSETS if a["province"] == province]
    return DUMMY_ASSETS


def get_dummy_assets_summary() -> dict:
    available = [a for a in DUMMY_ASSETS if a["status"] == "available"]
    deployed = [a for a in DUMMY_ASSETS if a["status"] == "deployed"]
    return {
        "total": len(DUMMY_ASSETS),
        "available": len(available),
        "deployed": len(deployed),
        "by_type": {
            "hotshot_crew": len([a for a in DUMMY_ASSETS if a["type"] == "hotshot_crew"]),
            "initial_attack_crew": len([a for a in DUMMY_ASSETS if a["type"] == "initial_attack_crew"]),
            "bulldozer": len([a for a in DUMMY_ASSETS if a["type"] == "bulldozer"]),
            "water_bomber": len([a for a in DUMMY_ASSETS if a["type"] == "water_bomber"]),
            "helicopter": len([a for a in DUMMY_ASSETS if a["type"] == "helicopter"]),
        },
    }


# ── Choke Point Recommendations (Greedy MVP) ────────────────────────────────────

def get_dummy_choke_points(fire_id: str) -> dict:
    """
    MVP greedy heuristic: returns ranked deployment zones for the given fire.
    Each node is scored by simulated burn_probability × accessibility.
    In production this is replaced by the RL agent inference.
    """
    fire = get_dummy_fire_by_id(fire_id)
    if not fire:
        return {}

    lat = fire["latitude"]
    lon = fire["longitude"]

    recommendations = [
        {
            "choke_point_id": f"{fire_id}-CP-001",
            "latitude": round(lat + 0.12, 5),
            "longitude": round(lon + 0.08, 5),
            "priority_score": 0.94,
            "recommended_action": "selective_backburn",
            "recommended_assets": ["hotshot_crew", "bulldozer"],
            "estimated_crew_size": 20,
            "rationale": "Highest predicted burn probability in 24h window. Ridgeline break creates natural containment anchor.",
        },
        {
            "choke_point_id": f"{fire_id}-CP-002",
            "latitude": round(lat + 0.07, 5),
            "longitude": round(lon + 0.14, 5),
            "priority_score": 0.78,
            "recommended_action": "firebreak_construction",
            "recommended_assets": ["bulldozer"],
            "estimated_crew_size": 0,
            "rationale": "Secondary threat corridor. Dozer line along logging road will cut off eastern flank.",
        },
        {
            "choke_point_id": f"{fire_id}-CP-003",
            "latitude": round(lat - 0.05, 5),
            "longitude": round(lon + 0.10, 5),
            "priority_score": 0.61,
            "recommended_action": "aerial_retardant_drop",
            "recommended_assets": ["water_bomber"],
            "estimated_crew_size": 0,
            "rationale": "Dense fuel load in valley. Retardant drop will slow spread before ground crews can reach.",
        },
    ]

    return {
        "fire_id": fire_id,
        "model": "greedy_heuristic_v0",
        "generated_at": datetime.now(UTC).isoformat(),
        "total_choke_points": len(recommendations),
        "recommendations": recommendations,
    }
