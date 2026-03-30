"""
fire_env.py — Gymnasium wildfire simulation environment for RL benchmarking.

A 25x25 cellular automata grid where:
  0 = Unburned fuel
  1 = Actively burning
  2 = Burned / scorched
  3 = Suppressed (retardant or firebreak)
  4 = Critical asset (unburned)
  5 = Critical asset (burned -- lost)

The agent must protect critical assets under a finite suppression budget.
Scenarios vary by ignition pattern, severity, asset layout, and wind bias.

Canonical benchmark mode consumes precomputed offline scenario parameter records.
It does not fetch FIRMS/CWFIS/Open-Meteo/CFFDRS data at runtime.

Usage:
    from src.models.fire_env import WildfireEnv, ScenarioConfig
    env = WildfireEnv(
        scenario=ScenarioConfig(ignition="edge", severity="high"),
        benchmark_mode=False,
    )
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# Cell types
UNBURNED = 0
BURNING = 1
BURNED = 2
SUPPRESSED = 3
ASSET = 4
ASSET_BURNED = 5

# Actions
MOVE_N = 0
MOVE_S = 1
MOVE_E = 2
MOVE_W = 3
DEPLOY_HELICOPTER = 4  # suppresses 3x3 area
DEPLOY_CREW = 5  # creates 1-cell firebreak

GRID_SIZE = 25

# Legacy fallback only (dev/ablation): severity -> base spread probability.
LEGACY_SEVERITY_SPREAD_PROB = {
    "low": 0.04 + 0.18 * 0.17,  # spread_intensity ~ 0.17
    "medium": 0.04 + 0.18 * 0.50,  # spread_intensity ~ 0.50
    "high": 0.04 + 0.18 * 0.83,  # spread_intensity ~ 0.83
}

SEVERITY_INDEX = {"low": 0, "medium": 1, "high": 2}
VALID_SPLITS = ("train", "val", "holdout")
WIND_DIRECTIONS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_DIAG = 0.70710678118
WIND_VECTOR_BY_DIR = {
    "N": (0.0, -1.0),
    "NE": (_DIAG, -_DIAG),
    "E": (1.0, 0.0),
    "SE": (_DIAG, _DIAG),
    "S": (0.0, 1.0),
    "SW": (-_DIAG, _DIAG),
    "W": (-1.0, 0.0),
    "NW": (-_DIAG, -_DIAG),
}

PARAMETER_METADATA_FIELDS = (
    "record_id",
    "split",
    "fire_id",
    "year",
    "source",
    "province",
    "record_quality_flag",
    "ignition_seed",
    "layout_seed",
)

PARAMETER_AUDIT_FIELDS = (
    "spread_rate_1h_m",
    "spread_score",
    "weather_score",
    "cffdrs_dryness_score",
    "size_factor",
    "fire_type_factor",
    "fuel_factor",
    "rain_factor",
    "record_quality_flag",
)

# ── Scenario families ────────────────────────────────────────────────────────

IGNITION_TYPES = ("center", "edge", "corner", "multi_cluster")
SEVERITY_LEVELS = ("low", "medium", "high")
ASSET_LAYOUTS = ("A", "B")

# Frozen train/test split per impl-plan section 6
TRAIN_FAMILIES: list[tuple[str, str, str]] = [
    (ign, sev, "A") for ign in ("center", "edge", "multi_cluster") for sev in SEVERITY_LEVELS
]

HELD_OUT_FAMILIES: list[tuple[str, str, str]] = [
    # OOD ignition: corner
    *[("corner", sev, "A") for sev in SEVERITY_LEVELS],
    # OOD asset layout: layout B
    *[(ign, "medium", "B") for ign in ("center", "edge", "multi_cluster")],
]


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario family."""

    ignition: str = "center"
    severity: str = "medium"
    asset_layout: str = "A"
    wind_direction: str = "N"
    wind_strength: float = 0.3  # [0, 1]
    base_spread_prob: float | None = None
    record_id: str | None = None

    def __post_init__(self):
        assert self.ignition in IGNITION_TYPES, f"Unknown ignition: {self.ignition}"
        assert self.severity in SEVERITY_LEVELS, f"Unknown severity: {self.severity}"
        assert self.asset_layout in ASSET_LAYOUTS, f"Unknown asset layout: {self.asset_layout}"
        assert self.wind_direction in WIND_DIRECTIONS_8, (
            f"Unknown wind direction: {self.wind_direction}"
        )

    @property
    def spread_prob(self) -> float:
        if self.base_spread_prob is not None:
            return float(self.base_spread_prob)
        return LEGACY_SEVERITY_SPREAD_PROB[self.severity]

    @property
    def wind_bias(self) -> tuple[float, float]:
        """Wind bias vector (wx, wy) for directional spread."""
        wx, wy = WIND_VECTOR_BY_DIR[self.wind_direction]
        return (self.wind_strength * wx, self.wind_strength * wy)

    @property
    def severity_onehot(self) -> list[float]:
        vec = [0.0, 0.0, 0.0]
        vec[SEVERITY_INDEX[self.severity]] = 1.0
        return vec


def random_scenario(
    rng: np.random.Generator,
    families: list[tuple[str, str, str]] | None = None,
) -> ScenarioConfig:
    """Sample a random scenario for dev/ablation runs (default: train families)."""
    if families is None:
        families = TRAIN_FAMILIES
    ign, sev, layout = families[rng.integers(len(families))]
    return ScenarioConfig(
        ignition=ign,
        severity=sev,
        asset_layout=layout,
        wind_direction=WIND_DIRECTIONS_8[int(rng.integers(len(WIND_DIRECTIONS_8)))],
        wind_strength=float(rng.uniform(0.1, 0.6)),
    )


def _split_hint_from_path(path: Path) -> str | None:
    stem = path.stem.lower()
    for split in VALID_SPLITS:
        token = f"_{split}"
        if stem.endswith(token) or token in stem:
            return split
    return None


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def load_scenario_parameter_records(
    path: str | Path,
    *,
    benchmark_mode: bool = True,
    expected_split: str | None = None,
) -> list[dict]:
    """Load and validate precomputed scenario parameter records from a JSON file."""
    records_path = Path(path)
    payload = json.loads(records_path.read_text())
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        msg = f"Invalid scenario parameter dataset: {records_path}"
        raise ValueError(msg)

    valid_splits = set(VALID_SPLITS)
    valid_severities = set(SEVERITY_LEVELS)
    validated: list[dict] = []
    errors: list[str] = []

    if expected_split is not None:
        expected_split = expected_split.strip().lower()
        if expected_split not in valid_splits:
            msg = (
                f"Invalid expected_split '{expected_split}' for dataset {records_path}; "
                f"expected one of {sorted(valid_splits)}"
            )
            raise ValueError(msg)

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record[{idx}]: expected object, got {type(record).__name__}")
            continue

        missing = [
            field
            for field in (
                "record_id",
                "split",
                "base_spread_prob",
                "severity_bucket",
                "wind_direction",
                "wind_strength",
                *(("ignition_seed", "layout_seed") if benchmark_mode else ()),
            )
            if record.get(field) is None
        ]
        if missing:
            errors.append(f"record[{idx}]: missing required fields {missing}")
            continue

        record_id = str(record.get("record_id", "")).strip()
        if not record_id:
            errors.append(f"record[{idx}]: record_id must be non-empty")
            continue

        split = str(record.get("split", "")).strip().lower()
        if split not in valid_splits:
            errors.append(
                f"record[{idx}]: invalid split '{record.get('split')}' (expected one of {sorted(valid_splits)})"
            )
            continue

        severity = str(record.get("severity_bucket", "")).strip().lower()
        if severity not in valid_severities:
            errors.append(
                f"record[{idx}]: invalid severity_bucket "
                f"'{record.get('severity_bucket')}' (expected one of {sorted(valid_severities)})"
            )
            continue

        try:
            base_spread_prob = float(record["base_spread_prob"])
            wind_strength = float(record["wind_strength"])
        except (TypeError, ValueError) as exc:
            errors.append(f"record[{idx}]: numeric parse failed ({exc})")
            continue

        if not (math.isfinite(base_spread_prob) and math.isfinite(wind_strength)):
            errors.append(f"record[{idx}]: numeric fields must be finite floats")
            continue

        if not 0.0 <= base_spread_prob <= 1.0:
            errors.append(
                f"record[{idx}]: base_spread_prob {base_spread_prob} out of range [0.0, 1.0]"
            )
            continue

        wind_direction = str(record.get("wind_direction", "")).strip().upper()
        if wind_direction not in WIND_DIRECTIONS_8:
            errors.append(
                f"record[{idx}]: invalid wind_direction '{record.get('wind_direction')}' "
                f"(expected one of {list(WIND_DIRECTIONS_8)})"
            )
            continue

        if not 0.0 <= wind_strength <= 1.0:
            errors.append(f"record[{idx}]: wind_strength {wind_strength} out of range [0.0, 1.0]")
            continue

        seed_invalid = False
        for seed_key in ("ignition_seed", "layout_seed"):
            if record.get(seed_key) is None:
                continue
            try:
                seed_value = int(record[seed_key])
            except (TypeError, ValueError) as exc:
                errors.append(f"record[{idx}]: {seed_key} parse failed ({exc})")
                seed_invalid = True
                continue
            if seed_value < 0:
                errors.append(f"record[{idx}]: {seed_key} must be >= 0")
                seed_invalid = True
                continue

        if seed_invalid:
            continue

        normalized = dict(record)
        normalized["record_id"] = record_id
        normalized["split"] = split
        normalized["severity_bucket"] = severity
        normalized["base_spread_prob"] = base_spread_prob
        normalized["wind_direction"] = wind_direction
        normalized["wind_strength"] = wind_strength
        if normalized.get("ignition_seed") is not None:
            normalized["ignition_seed"] = int(normalized["ignition_seed"])
        if normalized.get("layout_seed") is not None:
            normalized["layout_seed"] = int(normalized["layout_seed"])
        validated.append(normalized)

    path_split_hint = _split_hint_from_path(records_path)
    if (
        path_split_hint is not None
        and expected_split is not None
        and path_split_hint != expected_split
    ):
        msg = (
            f"Split mismatch for {records_path}: expected_split='{expected_split}' "
            f"but filename suggests '{path_split_hint}'"
        )
        if benchmark_mode:
            raise ValueError(msg)
        logger.warning(msg)

    effective_expected_split = expected_split or path_split_hint
    if benchmark_mode and effective_expected_split is None and validated:
        record_splits = sorted({str(record["split"]) for record in validated})
        if len(record_splits) == 1:
            effective_expected_split = record_splits[0]
        else:
            msg = (
                f"Could not infer a single split for {records_path}; found mixed splits {record_splits}. "
                "Provide expected_split explicitly or use split-specific datasets."
            )
            raise ValueError(msg)

    if effective_expected_split is not None and validated:
        split_mismatch = [
            f"record[{idx}]: split '{record['split']}' != expected '{effective_expected_split}'"
            for idx, record in enumerate(validated)
            if str(record["split"]) != effective_expected_split
        ]
        if split_mismatch and benchmark_mode:
            sample = "\n  - " + "\n  - ".join(split_mismatch[:10])
            if len(split_mismatch) > 10:
                sample += f"\n  - ... and {len(split_mismatch) - 10} more"
            msg = (
                f"Split consistency check failed for {records_path}: "
                f"{len(split_mismatch)} record(s) do not match expected split "
                f"'{effective_expected_split}'.{sample}"
            )
            raise ValueError(msg)
        if split_mismatch and not benchmark_mode:
            logger.warning(
                "Scenario dataset %s has %s split-mismatched record(s); skipping them in dev mode.",
                records_path,
                len(split_mismatch),
            )
            for detail in split_mismatch[:10]:
                logger.warning("  %s", detail)
            if len(split_mismatch) > 10:
                logger.warning("  ... and %s more", len(split_mismatch) - 10)
            validated = [
                record for record in validated if str(record["split"]) == effective_expected_split
            ]

    if errors and benchmark_mode:
        sample = "\n  - " + "\n  - ".join(errors[:10])
        if len(errors) > 10:
            sample += f"\n  - ... and {len(errors) - 10} more"
        msg = (
            f"Invalid scenario parameter dataset at {records_path}: "
            f"{len(errors)} invalid record(s) found.{sample}"
        )
        raise ValueError(msg)

    if errors and not benchmark_mode:
        logger.warning(
            "Scenario dataset %s has %s invalid record(s); skipping them in dev mode.",
            records_path,
            len(errors),
        )
        for detail in errors[:10]:
            logger.warning("  %s", detail)
        if len(errors) > 10:
            logger.warning("  ... and %s more", len(errors) - 10)

    if benchmark_mode and not validated:
        msg = (
            f"Scenario dataset {records_path} has no usable records after validation. "
            "Benchmark mode requires a non-empty validated dataset."
        )
        raise ValueError(msg)

    return validated


def benchmark_env_kwargs(
    *,
    expected_split: str,
    scenario_parameter_records: list[dict] | None = None,
    dataset_path: str | Path | None = None,
) -> dict:
    """Build canonical benchmark env kwargs from validated frozen records."""
    if scenario_parameter_records is None:
        if dataset_path is None:
            msg = "Provide scenario_parameter_records or dataset_path for benchmark env creation"
            raise ValueError(msg)
        scenario_parameter_records = load_scenario_parameter_records(
            dataset_path,
            benchmark_mode=True,
            expected_split=expected_split,
        )

    return {
        "scenario_parameter_records": scenario_parameter_records,
        "benchmark_mode": True,
        "expected_split": expected_split,
        "randomize_scenario": True,
    }


def create_benchmark_env(
    *,
    expected_split: str,
    scenario_parameter_records: list[dict] | None = None,
    dataset_path: str | Path | None = None,
    **env_overrides,
) -> WildfireEnv:
    """Create a canonical benchmark env instance from frozen records."""
    kwargs = benchmark_env_kwargs(
        expected_split=expected_split,
        scenario_parameter_records=scenario_parameter_records,
        dataset_path=dataset_path,
    )
    kwargs.update(env_overrides)
    return WildfireEnv(**kwargs)


def scenario_from_parameter_record(
    record: dict,
    *,
    ignition: str,
    asset_layout: str,
) -> ScenarioConfig:
    """Build a ScenarioConfig from a cached parameter record."""
    severity = str(record["severity_bucket"]).lower()
    return ScenarioConfig(
        ignition=ignition,
        severity=severity,
        asset_layout=asset_layout,
        wind_direction=str(record["wind_direction"]),
        wind_strength=float(record["wind_strength"]),
        base_spread_prob=float(record["base_spread_prob"]),
        record_id=str(record["record_id"]),
    )


# ── Environment ──────────────────────────────────────────────────────────────


class WildfireEnv(gym.Env):
    """
    Single-agent wildfire tactical response environment.

    The agent controls one tactical commander unit on a 25x25 grid.
    It can move in 4 directions or deploy helicopter/ground crew at its position.

    Critical assets are placed on the grid and must be protected.
    Helicopter and ground crew deployments are limited by per-episode budgets
    and cooldowns. Wind bias creates directional fire spread.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        max_steps: int = 150,
        scenario: ScenarioConfig | None = None,
        n_assets: int = 3,
        heli_budget: int = 8,
        crew_budget: int = 20,
        heli_cooldown: int = 5,
        crew_cooldown: int = 2,
        randomize_scenario: bool = True,
        scenario_families: list[tuple[str, str, str]] | None = None,
        scenario_parameter_records: list[dict] | None = None,
        expected_split: str | None = None,
        benchmark_mode: bool = True,
        # Legacy compat -- ignored if scenario is provided
        base_spread_rate_m_per_min: float | None = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_assets = n_assets
        self.heli_budget_init = heli_budget
        self.crew_budget_init = crew_budget
        self.heli_cooldown_duration = heli_cooldown
        self.crew_cooldown_duration = crew_cooldown
        self.randomize_scenario = randomize_scenario
        self.scenario_families = scenario_families
        self.scenario_parameter_records = scenario_parameter_records or []
        self.expected_split = expected_split.lower() if expected_split is not None else None
        self.benchmark_mode = benchmark_mode
        self._active_parameter_record: dict | None = None
        self._active_record_id: str | None = None
        self._record_order: list[int] = []
        self._record_cursor: int = 0

        if self.benchmark_mode:
            if self.expected_split is not None and self.expected_split not in VALID_SPLITS:
                msg = (
                    f"Invalid expected_split '{self.expected_split}' for WildfireEnv; "
                    f"expected one of {list(VALID_SPLITS)}"
                )
                raise ValueError(msg)
            if not self.scenario_parameter_records:
                msg = (
                    "benchmark_mode=True requires non-empty scenario_parameter_records. "
                    "Load frozen scenario_parameter_records_*.json and pass them to WildfireEnv."
                )
                raise ValueError(msg)
            if scenario is not None:
                msg = (
                    "benchmark_mode=True does not accept fixed ScenarioConfig input. "
                    "Use scenario_parameter_records for record-driven resets."
                )
                raise ValueError(msg)
            if not self.randomize_scenario:
                msg = (
                    "benchmark_mode=True requires randomize_scenario=True for record-driven resets."
                )
                raise ValueError(msg)
            if base_spread_rate_m_per_min is not None:
                msg = (
                    "base_spread_rate_m_per_min is a legacy dev-mode path and cannot be used "
                    "with benchmark_mode=True."
                )
                raise ValueError(msg)
            record_splits = {
                str(record.get("split", "")).strip().lower()
                for record in self.scenario_parameter_records
                if isinstance(record, dict)
            }
            if not record_splits:
                msg = "benchmark_mode=True requires records with valid split fields"
                raise ValueError(msg)
            invalid_splits = [s for s in record_splits if s not in VALID_SPLITS]
            if invalid_splits:
                msg = (
                    f"benchmark_mode=True found invalid split values {sorted(invalid_splits)}; "
                    f"expected one of {list(VALID_SPLITS)}"
                )
                raise ValueError(msg)
            if self.expected_split is not None and any(
                s != self.expected_split for s in record_splits
            ):
                msg = (
                    f"benchmark_mode=True expected split '{self.expected_split}' but got record splits "
                    f"{sorted(record_splits)}"
                )
                raise ValueError(msg)
            if self.expected_split is None and len(record_splits) != 1:
                msg = (
                    "benchmark_mode=True requires a single split dataset when expected_split is not "
                    f"provided; got splits {sorted(record_splits)}"
                )
                raise ValueError(msg)
            if self.expected_split is None:
                self.expected_split = next(iter(record_splits))
            missing_init = [
                idx
                for idx, record in enumerate(self.scenario_parameter_records)
                if record.get("ignition_seed") is None or record.get("layout_seed") is None
            ]
            if missing_init:
                msg = (
                    "benchmark_mode=True requires initialization seeds on all records "
                    f"(missing ignition_seed/layout_seed in {len(missing_init)} record(s)). "
                    "Use scenario_parameter_records_seeded_*.json artifacts."
                )
                raise ValueError(msg)

        # Scenario (may be overridden each reset if randomize_scenario=True)
        if scenario is not None:
            self._scenario = scenario
            self.randomize_scenario = False
        elif base_spread_rate_m_per_min is not None:
            # Legacy: convert spread rate to a fixed scenario
            self._scenario = ScenarioConfig(severity="medium", wind_strength=0.0)
            self._scenario_spread_override = min(0.85, base_spread_rate_m_per_min / 250)
            self.randomize_scenario = False
        else:
            self._scenario = ScenarioConfig()

        self._scenario_spread_override: float | None = getattr(
            self, "_scenario_spread_override", None
        )

        # Observation: grid (grid_size^2) + agent_pos (2) + resources (4)
        #            + severity_onehot (3) + wind_bias (2)
        obs_size = grid_size * grid_size + 2 + 4 + 3 + 2
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # 6 discrete actions
        self.action_space = spaces.Discrete(6)

        # State (initialized in reset)
        self.grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.agent_pos: list[int] = [0, 0]
        self.step_count: int = 0
        self._prev_burning: int = 0
        self.heli_left: int = 0
        self.crew_left: int = 0
        self.heli_cd: int = 0
        self.crew_cd: int = 0
        self.assets_lost: int = 0
        self.initial_asset_count: int = 0
        self.successful_heli_deployments: int = 0
        self.successful_crew_deployments: int = 0
        self.wasted_deployment_attempts: int = 0
        self.total_deployment_attempts: int = 0
        self._ignition_seed_used: int | None = None
        self._layout_seed_used: int | None = None
        self._ignition_rng: np.random.Generator | None = None
        self._layout_rng: np.random.Generator | None = None

    @property
    def scenario(self) -> ScenarioConfig:
        return self._scenario

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.step_count = 0
        self.assets_lost = 0
        self.successful_heli_deployments = 0
        self.successful_crew_deployments = 0
        self.wasted_deployment_attempts = 0
        self.total_deployment_attempts = 0
        self._active_parameter_record = None
        self._active_record_id = self._scenario.record_id
        self._ignition_seed_used = None
        self._layout_seed_used = None
        self._ignition_rng = None
        self._layout_rng = None

        # Optionally sample a new scenario
        if self.randomize_scenario:
            # Ignition and asset layout remain simulator-side controls.
            # Cached records provide spread/weather conditions; optional seeds
            # can pin reproducible ignition/layout realizations.
            families = self.scenario_families or TRAIN_FAMILIES
            ign, _sev, layout = families[int(self.np_random.integers(len(families)))]
            if self.scenario_parameter_records:
                # Canonical path: consume precomputed offline parameters only.
                record = self._sample_parameter_record(reshuffle=seed is not None)
                self._active_parameter_record = record
                self._active_record_id = (
                    str(record.get("record_id")) if record.get("record_id") else None
                )
                self._scenario = scenario_from_parameter_record(
                    record,
                    ignition=ign,
                    asset_layout=layout,
                )
                self._configure_initialization_rngs(record=record, reset_seed=seed)
            else:
                if self.benchmark_mode:
                    msg = (
                        "benchmark_mode=True cannot reset without scenario_parameter_records. "
                        "Disable benchmark_mode only for explicit dev/ablation runs."
                    )
                    raise RuntimeError(msg)
                self._active_parameter_record = None
                self._active_record_id = None
                self._scenario = random_scenario(self.np_random, families)

        if self._ignition_rng is None:
            self._ignition_rng = self.np_random
        if self._layout_rng is None:
            self._layout_rng = self.np_random

        # Reset budgets and cooldowns
        self.heli_left = self.heli_budget_init
        self.crew_left = self.crew_budget_init
        self.heli_cd = 0
        self.crew_cd = 0

        # Place critical assets
        self._place_assets()

        # Ignite fire
        self._ignite()

        # Agent starts at top-left corner
        self.agent_pos = [0, 0]
        self._prev_burning = int(np.sum(self.grid == BURNING))

        return self._get_obs(), {
            "scenario": self._scenario,
            "record_id": self._active_record_id,
            "split": self._active_parameter_record.get("split")
            if self._active_parameter_record
            else None,
            "ignition_seed": self._ignition_seed_used,
            "layout_seed": self._layout_seed_used,
            "parameter_record_meta": self._parameter_metadata(),
            "parameter_audit": self._parameter_audit(),
            "parameter_record": self._active_parameter_record,
        }

    def step(self, action: int):
        self.step_count += 1
        reward = 0.0

        # Tick cooldowns
        if self.heli_cd > 0:
            self.heli_cd -= 1
        if self.crew_cd > 0:
            self.crew_cd -= 1

        # 1. Execute agent action
        action_reward, heli_used, crew_used = self._execute_action(action)
        reward += action_reward

        # Resource costs
        if heli_used:
            reward -= 1.5
        if crew_used:
            reward -= 0.5

        # 2. Advance fire spread
        asset_cells_lost = self._spread_fire()

        # 3. Reward components
        reward -= 75.0 * asset_cells_lost

        burning_now = int(np.sum(self.grid == BURNING))
        new_burned = max(0, burning_now - self._prev_burning)
        reward -= 0.4 * new_burned

        self._prev_burning = burning_now

        # 4. Check termination
        terminated = burning_now == 0
        truncated = self.step_count >= self.max_steps

        # Terminal shaping
        if terminated and self.assets_lost == 0:
            reward += 100.0
        elif (terminated or truncated) and self.assets_lost == 0:
            reward += 40.0

        info = {
            "burning_cells": burning_now,
            "step": self.step_count,
            "assets_lost": self.assets_lost,
            "assets_remaining": self.initial_asset_count - self.assets_lost,
            "successful_heli_deployments": self.successful_heli_deployments,
            "successful_crew_deployments": self.successful_crew_deployments,
            "successful_deployments": self.successful_heli_deployments
            + self.successful_crew_deployments,
            "wasted_deployment_attempts": self.wasted_deployment_attempts,
            "total_deployment_attempts": self.total_deployment_attempts,
            "heli_left": self.heli_left,
            "crew_left": self.crew_left,
            "scenario": self._scenario,
            "record_id": self._active_record_id,
            "split": self._active_parameter_record.get("split")
            if self._active_parameter_record
            else None,
            "ignition_seed": self._ignition_seed_used,
            "layout_seed": self._layout_seed_used,
            "parameter_record_meta": self._parameter_metadata(),
            "parameter_audit": self._parameter_audit(),
            "parameter_record": self._active_parameter_record,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _configure_initialization_rngs(
        self,
        *,
        record: dict,
        reset_seed: int | None,
    ) -> None:
        record_id = str(record.get("record_id") or "unknown")

        ignition_seed = record.get("ignition_seed")
        layout_seed = record.get("layout_seed")

        if self.benchmark_mode and (ignition_seed is None or layout_seed is None):
            msg = (
                "benchmark_mode=True requires per-record ignition_seed and layout_seed "
                "for reproducible initialization."
            )
            raise RuntimeError(msg)

        if ignition_seed is None and reset_seed is not None:
            ignition_seed = _stable_seed(record_id, reset_seed, "ignition")
        if layout_seed is None and reset_seed is not None:
            layout_seed = _stable_seed(record_id, reset_seed, "layout")

        self._ignition_seed_used = int(ignition_seed) if ignition_seed is not None else None
        self._layout_seed_used = int(layout_seed) if layout_seed is not None else None

        self._ignition_rng = (
            np.random.default_rng(self._ignition_seed_used)
            if self._ignition_seed_used is not None
            else self.np_random
        )
        self._layout_rng = (
            np.random.default_rng(self._layout_seed_used)
            if self._layout_seed_used is not None
            else self.np_random
        )

    def _parameter_metadata(self) -> dict:
        record = self._active_parameter_record
        if not record:
            return {}
        return {key: record.get(key) for key in PARAMETER_METADATA_FIELDS}

    def _parameter_audit(self) -> dict:
        record = self._active_parameter_record
        if not record:
            return {}
        return {key: record.get(key) for key in PARAMETER_AUDIT_FIELDS if key in record}

    def _sample_parameter_record(self, *, reshuffle: bool = False) -> dict:
        if not self.scenario_parameter_records:
            msg = "No scenario_parameter_records available for sampling"
            raise RuntimeError(msg)

        if reshuffle or not self._record_order or self._record_cursor >= len(self._record_order):
            self._record_order = [
                int(i) for i in self.np_random.permutation(len(self.scenario_parameter_records))
            ]
            self._record_cursor = 0

        index = self._record_order[self._record_cursor]
        self._record_cursor += 1
        return self.scenario_parameter_records[index]

    def _get_obs(self) -> np.ndarray:
        # Normalize grid: 6 cell types -> [0, 1]
        flat_grid = self.grid.flatten().astype(np.float32) / 5.0
        agent = np.array(self.agent_pos, dtype=np.float32) / self.grid_size
        resources = np.array(
            [
                self.heli_left / self.heli_budget_init,
                self.crew_left / self.crew_budget_init,
                self.heli_cd / self.heli_cooldown_duration
                if self.heli_cooldown_duration > 0
                else 0.0,
                self.crew_cd / self.crew_cooldown_duration
                if self.crew_cooldown_duration > 0
                else 0.0,
            ],
            dtype=np.float32,
        )
        severity = np.array(self._scenario.severity_onehot, dtype=np.float32)
        wx, wy = self._scenario.wind_bias
        wind = np.array([wx, wy], dtype=np.float32)
        return np.concatenate([flat_grid, agent, resources, severity, wind])

    def _ignite(self):
        """Set initial fire cells based on scenario ignition pattern."""
        rng = self._ignition_rng or self.np_random
        gs = self.grid_size
        cx, cy = gs // 2, gs // 2
        pattern = self._scenario.ignition

        if pattern == "center":
            seeds = [(cx, cy), (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
        elif pattern == "edge":
            # Fire starts along a random edge
            edge = int(rng.integers(4))
            if edge == 0:  # top
                seeds = [(0, cy - 1), (0, cy), (0, cy + 1)]
            elif edge == 1:  # bottom
                seeds = [(gs - 1, cy - 1), (gs - 1, cy), (gs - 1, cy + 1)]
            elif edge == 2:  # left
                seeds = [(cx - 1, 0), (cx, 0), (cx + 1, 0)]
            else:  # right
                seeds = [(cx - 1, gs - 1), (cx, gs - 1), (cx + 1, gs - 1)]
        elif pattern == "corner":
            corner = int(rng.integers(4))
            offsets = [(0, 0), (0, gs - 1), (gs - 1, 0), (gs - 1, gs - 1)]
            cr, cc = offsets[corner]
            seeds = [(cr, cc)]
            # Add neighbors that are in bounds
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if self._in_bounds(nr, nc):
                    seeds.append((nr, nc))
        elif pattern == "multi_cluster":
            # 2-3 small fire clusters scattered across the grid
            n_clusters = int(rng.integers(2, 4))
            seeds = []
            for _ in range(n_clusters):
                r = int(rng.integers(2, gs - 2))
                c = int(rng.integers(2, gs - 2))
                seeds.append((r, c))
                seeds.append((r + 1, c))
                seeds.append((r, c + 1))
        else:
            msg = f"Unknown ignition pattern: {pattern}"
            raise ValueError(msg)

        for r, c in seeds:
            if self._in_bounds(r, c) and self.grid[r, c] == UNBURNED:
                self.grid[r, c] = BURNING

    def _place_assets(self):
        """Place critical asset cells based on scenario asset layout."""
        rng = self._layout_rng or self.np_random
        gs = self.grid_size
        cx, cy = gs // 2, gs // 2
        min_dist = gs // 4

        if self._scenario.asset_layout == "A":
            # Layout A: single cluster of n_assets cells
            # Pick a cluster center away from grid center
            placed = 0
            cluster_r, cluster_c = 0, 0
            for _ in range(100):
                cluster_r = int(rng.integers(0, gs))
                cluster_c = int(rng.integers(0, gs))
                if abs(cluster_r - cx) + abs(cluster_c - cy) >= min_dist:
                    break

            # Place assets in a tight cluster around the chosen center
            candidates = [(cluster_r, cluster_c)]
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    candidates.append((cluster_r + dr, cluster_c + dc))
            rng.shuffle(candidates)

            for r, c in candidates:
                if placed >= self.n_assets:
                    break
                if self._in_bounds(r, c) and self.grid[r, c] == UNBURNED:
                    dist = abs(r - cx) + abs(c - cy)
                    if dist >= min_dist:
                        self.grid[r, c] = ASSET
                        placed += 1

            self.initial_asset_count = placed

        elif self._scenario.asset_layout == "B":
            # Layout B: two smaller clusters
            assets_per_cluster = max(1, self.n_assets // 2)
            placed = 0

            for _ in range(2):
                cluster_r, cluster_c = 0, 0
                for _ in range(100):
                    cluster_r = int(rng.integers(0, gs))
                    cluster_c = int(rng.integers(0, gs))
                    if abs(cluster_r - cx) + abs(cluster_c - cy) >= min_dist:
                        break

                candidates = [
                    (cluster_r + dr, cluster_c + dc) for dr in range(-1, 2) for dc in range(-1, 2)
                ]
                rng.shuffle(candidates)

                cluster_placed = 0
                for r, c in candidates:
                    if cluster_placed >= assets_per_cluster:
                        break
                    if self._in_bounds(r, c) and self.grid[r, c] == UNBURNED:
                        dist = abs(r - cx) + abs(c - cy)
                        if dist >= min_dist:
                            self.grid[r, c] = ASSET
                            cluster_placed += 1
                            placed += 1

            self.initial_asset_count = placed

    def _execute_action(self, action: int) -> tuple[float, bool, bool]:
        """Execute action, return (reward, heli_used, crew_used)."""
        r, c = self.agent_pos
        reward = 0.0
        heli_used = False
        crew_used = False

        if action == MOVE_N and r > 0:
            self.agent_pos[0] -= 1
        elif action == MOVE_S and r < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == MOVE_E and c < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == MOVE_W and c > 0:
            self.agent_pos[1] -= 1
        elif action == DEPLOY_HELICOPTER:
            self.total_deployment_attempts += 1
            if self.heli_left > 0 and self.heli_cd == 0:
                # Suppress 3x3 area around agent
                suppressed = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if self._in_bounds(nr, nc):
                            cell = self.grid[nr, nc]
                            if cell in (BURNING, UNBURNED):
                                self.grid[nr, nc] = SUPPRESSED
                                suppressed += 1
                self.heli_left -= 1
                self.heli_cd = self.heli_cooldown_duration
                heli_used = True
                if suppressed > 0:
                    self.successful_heli_deployments += 1
                    reward += suppressed * 3.0
                else:
                    self.wasted_deployment_attempts += 1
                    reward -= 1.0  # wasted
            else:
                self.wasted_deployment_attempts += 1
                reward -= 1.0  # blocked by budget or cooldown
        elif action == DEPLOY_CREW:
            self.total_deployment_attempts += 1
            if self.crew_left > 0 and self.crew_cd == 0:
                cell = self.grid[r, c]
                if cell == BURNING:
                    self.grid[r, c] = SUPPRESSED
                    self.successful_crew_deployments += 1
                    reward += 3.0
                elif cell == UNBURNED:
                    self.grid[r, c] = SUPPRESSED
                    self.successful_crew_deployments += 1
                    reward += 2.0  # firebreak
                else:
                    self.wasted_deployment_attempts += 1
                    reward -= 1.0  # wasted
                self.crew_left -= 1
                self.crew_cd = self.crew_cooldown_duration
                crew_used = True
            else:
                self.wasted_deployment_attempts += 1
                reward -= 1.0  # blocked by budget or cooldown

        return reward, heli_used, crew_used

    def _spread_fire(self) -> int:
        """Stochastic fire spread with wind bias. Returns asset cells lost."""
        base_prob = (
            self._scenario_spread_override
            if self._scenario_spread_override is not None
            else self._scenario.spread_prob
        )
        wx, wy = self._scenario.wind_bias

        new_burning = []
        asset_cells_lost = 0
        burning_cells = list(zip(*np.where(self.grid == BURNING), strict=True))

        for r, c in burning_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc):
                    continue

                # Wind bias: dot product of spread direction and wind vector
                # Positive = downwind (easier spread), negative = upwind
                wind_dot = dr * wy + dc * wx
                prob = min(0.95, max(0.01, base_prob + 0.15 * wind_dot))

                cell = self.grid[nr, nc]
                if cell == UNBURNED and self.np_random.random() < prob:
                    new_burning.append((nr, nc))
                elif cell == ASSET and self.np_random.random() < prob:
                    self.grid[nr, nc] = BURNING
                    asset_cells_lost += 1
                    self.assets_lost += 1

            # Burning cell may burn out
            if self.np_random.random() < 0.05:
                self.grid[r, c] = BURNED

        for r, c in new_burning:
            if self.grid[r, c] == UNBURNED:
                self.grid[r, c] = BURNING

        return asset_cells_lost
