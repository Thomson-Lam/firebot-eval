"""
fire_env.py — Gymnasium wildfire simulation environment for RL benchmarking.

A 25×25 cellular automata grid where:
  0 = Unburned fuel
  1 = Actively burning
  2 = Burned / scorched
  3 = Suppressed (retardant or firebreak)
  4 = Critical asset (unburned)
  5 = Critical asset (burned — lost)

The agent must protect critical assets under a finite suppression budget.
Spread probability per timestep is driven by the XGBoost spread model.

Usage:
    from src.models.fire_env import WildfireEnv
    env = WildfireEnv()
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Cell types
UNBURNED      = 0
BURNING       = 1
BURNED        = 2
SUPPRESSED    = 3
ASSET         = 4
ASSET_BURNED  = 5

# Actions
MOVE_N            = 0
MOVE_S            = 1
MOVE_E            = 2
MOVE_W            = 3
DEPLOY_HELICOPTER = 4   # suppresses 3×3 area
DEPLOY_CREW       = 5   # creates 1-cell firebreak

GRID_SIZE = 25


class WildfireEnv(gym.Env):
    """
    Single-agent wildfire tactical response environment.

    The agent controls one tactical commander unit on a 25×25 grid.
    It can move in 4 directions or deploy helicopter/ground crew at its position.

    Critical assets are placed on the grid and must be protected.
    Helicopter and ground crew deployments are limited by per-episode budgets
    and cooldowns.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        max_steps: int = 150,
        base_spread_rate_m_per_min: float = 15.0,
        n_assets: int = 3,
        heli_budget: int = 8,
        crew_budget: int = 20,
        heli_cooldown: int = 5,
        crew_cooldown: int = 2,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_assets = n_assets
        self.heli_budget_init = heli_budget
        self.crew_budget_init = crew_budget
        self.heli_cooldown_duration = heli_cooldown
        self.crew_cooldown_duration = crew_cooldown

        # spread_rate from XGBoost (m/min) → per-step probability
        self.spread_prob = min(0.85, base_spread_rate_m_per_min / 250)

        # Observation: grid (grid_size^2) + agent_pos (2) + heli_left (1)
        #            + crew_left (1) + heli_cd (1) + crew_cd (1)
        obs_size = grid_size * grid_size + 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
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

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.step_count = 0
        self.assets_lost = 0

        # Reset budgets and cooldowns
        self.heli_left = self.heli_budget_init
        self.crew_left = self.crew_budget_init
        self.heli_cd = 0
        self.crew_cd = 0

        # Place critical assets away from center (fire start)
        self._place_assets()

        # Start fire at center
        cx, cy = self.grid_size // 2, self.grid_size // 2
        self.grid[cx, cy] = BURNING
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if self._in_bounds(nx, ny) and self.grid[nx, ny] == UNBURNED:
                self.grid[nx, ny] = BURNING

        # Agent starts at top-left corner
        self.agent_pos = [0, 0]
        self._prev_burning = int(np.sum(self.grid == BURNING))

        return self._get_obs(), {}

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
        # Asset loss penalty
        reward -= 75.0 * asset_cells_lost

        # New burned cells penalty
        burning_now = int(np.sum(self.grid == BURNING))
        new_burned = max(0, burning_now - self._prev_burning)
        reward -= 0.4 * new_burned

        # Suppression bonus (burning cells reduced by actions, not burnout)
        # Already accounted for in _execute_action

        self._prev_burning = burning_now

        # 4. Check termination
        terminated = burning_now == 0
        truncated = self.step_count >= self.max_steps

        # Terminal shaping
        if terminated and self.assets_lost == 0:
            reward += 100.0  # fire out, no asset loss
        elif (terminated or truncated) and self.assets_lost == 0:
            reward += 40.0   # episode ends with all assets intact

        info = {
            "burning_cells": burning_now,
            "step": self.step_count,
            "assets_lost": self.assets_lost,
            "assets_remaining": self.initial_asset_count - self.assets_lost,
            "heli_left": self.heli_left,
            "crew_left": self.crew_left,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _get_obs(self) -> np.ndarray:
        # Normalize grid: 6 cell types → [0, 1]
        flat_grid = self.grid.flatten().astype(np.float32) / 5.0
        agent = np.array(self.agent_pos, dtype=np.float32) / self.grid_size
        resources = np.array([
            self.heli_left / self.heli_budget_init,
            self.crew_left / self.crew_budget_init,
            self.heli_cd / self.heli_cooldown_duration if self.heli_cooldown_duration > 0 else 0.0,
            self.crew_cd / self.crew_cooldown_duration if self.crew_cooldown_duration > 0 else 0.0,
        ], dtype=np.float32)
        return np.concatenate([flat_grid, agent, resources])

    def _place_assets(self):
        """Place critical asset cells away from the fire center."""
        cx, cy = self.grid_size // 2, self.grid_size // 2
        min_dist = self.grid_size // 4  # at least this far from center
        placed = 0
        attempts = 0

        while placed < self.n_assets and attempts < 200:
            r = self.np_random.integers(0, self.grid_size)
            c = self.np_random.integers(0, self.grid_size)
            dist = abs(r - cx) + abs(c - cy)
            if dist >= min_dist and self.grid[r, c] == UNBURNED:
                self.grid[r, c] = ASSET
                placed += 1
            attempts += 1

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
                # Suppress 3×3 area around agent
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
                    reward -= 1.0  # wasted — nothing to suppress
            else:
                reward -= 1.0  # wasted — blocked by budget or cooldown
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
                    reward -= 1.0  # wasted — already burned/suppressed/asset
                self.crew_left -= 1
                self.crew_cd = self.crew_cooldown_duration
                crew_used = True
            else:
                reward -= 1.0  # wasted — blocked by budget or cooldown

        return reward, heli_used, crew_used

    def _spread_fire(self) -> int:
        """Stochastic fire spread. Returns number of asset cells lost this step."""
        new_burning = []
        asset_cells_lost = 0
        burning_cells = list(zip(*np.where(self.grid == BURNING), strict=True))

        for (r, c) in burning_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc):
                    continue
                cell = self.grid[nr, nc]
                if cell == UNBURNED and self.np_random.random() < self.spread_prob:
                    new_burning.append((nr, nc))
                elif cell == ASSET and self.np_random.random() < self.spread_prob:
                    # Asset catches fire — mark as lost
                    self.grid[nr, nc] = BURNING
                    asset_cells_lost += 1
                    self.assets_lost += 1

            # Burning cell may burn out
            if self.np_random.random() < 0.05:
                self.grid[r, c] = BURNED

        for (r, c) in new_burning:
            if self.grid[r, c] == UNBURNED:  # guard against double-set
                self.grid[r, c] = BURNING

        return asset_cells_lost
