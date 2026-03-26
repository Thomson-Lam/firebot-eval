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

Usage:
    from src.models.fire_env import WildfireEnv, ScenarioConfig
    env = WildfireEnv(scenario=ScenarioConfig(ignition="edge", severity="high"))
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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

# Severity -> base spread probability (from impl-plan section 9.2)
SEVERITY_SPREAD_PROB = {
    "low": 0.04 + 0.18 * 0.17,  # spread_intensity ~ 0.17
    "medium": 0.04 + 0.18 * 0.50,  # spread_intensity ~ 0.50
    "high": 0.04 + 0.18 * 0.83,  # spread_intensity ~ 0.83
}

SEVERITY_INDEX = {"low": 0, "medium": 1, "high": 2}

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
    wind_dir_deg: float = 0.0  # 0 = wind blowing north->south
    wind_strength: float = 0.3  # [0, 1]

    def __post_init__(self):
        assert self.ignition in IGNITION_TYPES, f"Unknown ignition: {self.ignition}"
        assert self.severity in SEVERITY_LEVELS, f"Unknown severity: {self.severity}"
        assert self.asset_layout in ASSET_LAYOUTS, f"Unknown asset layout: {self.asset_layout}"

    @property
    def spread_prob(self) -> float:
        return SEVERITY_SPREAD_PROB[self.severity]

    @property
    def wind_bias(self) -> tuple[float, float]:
        """Wind bias vector (wx, wy) for directional spread."""
        rad = math.radians(self.wind_dir_deg)
        return (
            self.wind_strength * math.cos(rad),
            self.wind_strength * math.sin(rad),
        )

    @property
    def severity_onehot(self) -> list[float]:
        vec = [0.0, 0.0, 0.0]
        vec[SEVERITY_INDEX[self.severity]] = 1.0
        return vec


def random_scenario(
    rng: np.random.Generator,
    families: list[tuple[str, str, str]] | None = None,
) -> ScenarioConfig:
    """Sample a random scenario from the given families (default: train)."""
    if families is None:
        families = TRAIN_FAMILIES
    ign, sev, layout = families[rng.integers(len(families))]
    return ScenarioConfig(
        ignition=ign,
        severity=sev,
        asset_layout=layout,
        wind_dir_deg=float(rng.uniform(0, 360)),
        wind_strength=float(rng.uniform(0.1, 0.6)),
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

    @property
    def scenario(self) -> ScenarioConfig:
        return self._scenario

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.step_count = 0
        self.assets_lost = 0

        # Optionally sample a new scenario
        if self.randomize_scenario:
            self._scenario = random_scenario(self.np_random, self.scenario_families)

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

        return self._get_obs(), {"scenario": self._scenario}

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
            "heli_left": self.heli_left,
            "crew_left": self.crew_left,
            "scenario": self._scenario,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

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
        gs = self.grid_size
        cx, cy = gs // 2, gs // 2
        pattern = self._scenario.ignition

        if pattern == "center":
            seeds = [(cx, cy), (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
        elif pattern == "edge":
            # Fire starts along a random edge
            edge = int(self.np_random.integers(4))
            if edge == 0:  # top
                seeds = [(0, cy - 1), (0, cy), (0, cy + 1)]
            elif edge == 1:  # bottom
                seeds = [(gs - 1, cy - 1), (gs - 1, cy), (gs - 1, cy + 1)]
            elif edge == 2:  # left
                seeds = [(cx - 1, 0), (cx, 0), (cx + 1, 0)]
            else:  # right
                seeds = [(cx - 1, gs - 1), (cx, gs - 1), (cx + 1, gs - 1)]
        elif pattern == "corner":
            corner = int(self.np_random.integers(4))
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
            n_clusters = int(self.np_random.integers(2, 4))
            seeds = []
            for _ in range(n_clusters):
                r = int(self.np_random.integers(2, gs - 2))
                c = int(self.np_random.integers(2, gs - 2))
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
        gs = self.grid_size
        cx, cy = gs // 2, gs // 2
        min_dist = gs // 4

        if self._scenario.asset_layout == "A":
            # Layout A: single cluster of n_assets cells
            # Pick a cluster center away from grid center
            placed = 0
            cluster_r, cluster_c = 0, 0
            for _ in range(100):
                cluster_r = int(self.np_random.integers(0, gs))
                cluster_c = int(self.np_random.integers(0, gs))
                if abs(cluster_r - cx) + abs(cluster_c - cy) >= min_dist:
                    break

            # Place assets in a tight cluster around the chosen center
            candidates = [(cluster_r, cluster_c)]
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    candidates.append((cluster_r + dr, cluster_c + dc))
            self.np_random.shuffle(candidates)

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
                    cluster_r = int(self.np_random.integers(0, gs))
                    cluster_c = int(self.np_random.integers(0, gs))
                    if abs(cluster_r - cx) + abs(cluster_c - cy) >= min_dist:
                        break

                candidates = [
                    (cluster_r + dr, cluster_c + dc) for dr in range(-1, 2) for dc in range(-1, 2)
                ]
                self.np_random.shuffle(candidates)

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
                    reward += suppressed * 3.0
                else:
                    reward -= 1.0  # wasted
            else:
                reward -= 1.0  # blocked by budget or cooldown
        elif action == DEPLOY_CREW:
            if self.crew_left > 0 and self.crew_cd == 0:
                cell = self.grid[r, c]
                if cell == BURNING:
                    self.grid[r, c] = SUPPRESSED
                    reward += 3.0
                elif cell == UNBURNED:
                    self.grid[r, c] = SUPPRESSED
                    reward += 2.0  # firebreak
                else:
                    reward -= 1.0  # wasted
                self.crew_left -= 1
                self.crew_cd = self.crew_cooldown_duration
                crew_used = True
            else:
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
