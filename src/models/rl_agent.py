"""
rl_agent.py — PPO tactical agent inference interface.

Loads the trained PPO model and returns tactical deployment waypoints
for a given fire, converting grid coordinates back to real lat/lon.

Usage:
    from src.models.rl_agent import get_tactical_recommendations
    waypoints = get_tactical_recommendations("BC-2026-001", fire_data, spread_output)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "tactical_ppo_agent.zip"
GRID_SIZE = 25


def _grid_to_latlon(
    grid_pos: list[int],
    fire_center_lat: float,
    fire_center_lon: float,
    spread_radius_m: float,
) -> tuple[float, float]:
    """Convert a grid cell position to real-world lat/lon."""
    metres_per_deg_lat = 111_320
    metres_per_deg_lon = 111_320 * math.cos(math.radians(fire_center_lat))

    cell_size_m = (spread_radius_m * 2) / GRID_SIZE
    center_cell = GRID_SIZE // 2

    delta_row = grid_pos[0] - center_cell
    delta_col = grid_pos[1] - center_cell

    lat = fire_center_lat - (delta_row * cell_size_m / metres_per_deg_lat)
    lon = fire_center_lon + (delta_col * cell_size_m / metres_per_deg_lon)

    return round(lat, 5), round(lon, 5)


def _greedy_fallback(
    fire_lat: float,
    fire_lon: float,
    spread_1h_m: float,
    spread_3h_m: float,
) -> list[dict]:
    """
    Greedy heuristic recommendations if the PPO model isn't trained yet.
    Deploys along the fire perimeter at cardinal/intercardinal points.
    """
    directions = [
        ("N", -1, 0, "ground_crew", "Establish northern anchor line"),
        ("NE", -0.7, 0.7, "helicopter", "Pre-position tanker for NE flank"),
        ("E",  0, 1, "ground_crew", "Cut containment line on eastern flank"),
        ("S",  1, 0, "ground_crew", "Southern backfire opportunity"),
        ("W",  0, -1, "helicopter", "Air attack on advancing western head"),
    ]

    metres_per_deg_lat = 111_320
    metres_per_deg_lon = 111_320 * math.cos(math.radians(fire_lat))
    offset_m = spread_1h_m * 0.8

    waypoints = []
    for name, dlat_factor, dlon_factor, asset, rationale in directions:
        lat = fire_lat + (dlat_factor * offset_m / metres_per_deg_lat)
        lon = fire_lon + (dlon_factor * offset_m / metres_per_deg_lon)
        waypoints.append({
            "direction": name,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "asset_type": asset,
            "rationale": rationale,
            "score": round(0.9 - len(waypoints) * 0.1, 2),
            "source": "greedy_heuristic",
        })

    return waypoints


def get_tactical_recommendations(
    fire_id: str,
    fire_data: Optional[dict] = None,
    spread_output: Optional[dict] = None,
    n_inference_steps: int = 60,
) -> list[dict]:
    """
    Generate tactical deployment waypoints for a fire.

    If the PPO model is trained and saved, uses it for inference.
    Falls back to the greedy heuristic if model isn't available.
    """
    fire_lat  = float(fire_data.get("latitude",  49.9071)) if fire_data else 49.9071
    fire_lon  = float(fire_data.get("longitude", -119.496)) if fire_data else -119.496
    spread_1h = float(spread_output.get("spread_1h_m", 1200)) if spread_output else 1200
    spread_3h = float(spread_output.get("spread_3h_m", 3600)) if spread_output else 3600

    if not MODEL_PATH.exists():
        logger.info(f"PPO model not found at {MODEL_PATH} — using greedy heuristic")
        return _greedy_fallback(fire_lat, fire_lon, spread_1h, spread_3h)

    try:
        from stable_baselines3 import PPO
        from src.models.fire_env import WildfireEnv

        spread_rate_m_per_min = spread_1h / 60.0
        env = WildfireEnv(
            base_spread_rate_m_per_min=spread_rate_m_per_min,
        )
        model = PPO.load(str(MODEL_PATH), env=env)

        obs, _ = env.reset()
        waypoints = []
        deployment_actions = {4, 5}

        for _ in range(n_inference_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))

            if int(action) in deployment_actions:
                lat, lon = _grid_to_latlon(
                    env.agent_pos, fire_lat, fire_lon, spread_1h
                )
                asset = "helicopter" if int(action) == 4 else "ground_crew"
                waypoints.append({
                    "latitude":   lat,
                    "longitude":  lon,
                    "asset_type": asset,
                    "rationale":  f"PPO recommended {asset} deployment (step {env.step_count})",
                    "score":      round(float(reward), 2),
                    "source":     "ppo_agent",
                })

            if done or truncated:
                break

        # Deduplicate very close waypoints (within 200m)
        unique = []
        for wp in waypoints:
            too_close = any(
                abs(wp["latitude"] - u["latitude"]) < 0.002
                and abs(wp["longitude"] - u["longitude"]) < 0.002
                for u in unique
            )
            if not too_close:
                unique.append(wp)

        if not unique:
            logger.warning("PPO produced no deployments — falling back to heuristic")
            return _greedy_fallback(fire_lat, fire_lon, spread_1h, spread_3h)

        return unique[:8]

    except Exception as e:
        logger.error(f"PPO inference failed: {e} — using greedy fallback")
        return _greedy_fallback(fire_lat, fire_lon, spread_1h, spread_3h)
