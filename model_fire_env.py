"""
fire_env.py — Gymnasium wildfire simulation environment for PPO training.

A 50×50 cellular automata grid where:
  0 = Unburned fuel
  1 = Actively burning
  2 = Burned / scorched
  3 = Suppressed (retardant or firebreak)

The spread probability per timestep is driven by the XGBoost spread model,
linking the two ML components together. High ISI + low RH = fast-spreading fire.

Usage:
    from src.models.fire_env import WildfireEnv
    env = WildfireEnv(grid_size=50)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Fire grid cell values
UNBURNED  = 0
BURNING   = 1
BURNED    = 2
SUPPRESSED = 3

# Actions
MOVE_N    = 0
MOVE_S    = 1
MOVE_E    = 2
MOVE_W    = 3
DEPLOY_HELICOPTER = 4   # drops water: suppresses 3×3 area
DEPLOY_CREW       = 5   # creates 1-cell firebreak

GRID_SIZE = 50


class WildfireEnv(gym.Env):
    """
    Single-agent wildfire tactical response environment.

    The agent controls one tactical commander unit on a 50×50 grid.
    It can move in 4 directions or deploy helicopter/ground crew at its position.

    The fire starts at the grid center and spreads stochastically each step,
    with spread probability determined by weather conditions (via XGBoost rate).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        max_steps: int = 200,
        base_spread_rate_m_per_min: float = 15.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # spread_rate from XGBoost (m/min) → convert to per-step probability
        # grid cell ≈ 100m. spread_rate/100 = cells/min. /10 steps per min = prob per step.
        self.spread_prob = min(0.85, base_spread_rate_m_per_min / 250)

        # Flat observation: fire grid (grid_size^2) + agent position (2)
        self.observation_space = spaces.Box(
            low=0, high=3,
            shape=(grid_size * grid_size + 2,),
            dtype=np.float32,
        )

        # 6 discrete actions
        self.action_space = spaces.Discrete(6)

        self.grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.agent_pos: list[int] = [0, 0]
        self.step_count: int = 0
        self._prev_burning: int = 0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.step_count = 0

        # Start fire at center
        cx, cy = self.grid_size // 2, self.grid_size // 2
        self.grid[cx, cy] = BURNING
        # Small initial fire cluster (more realistic start)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cx+dx, cy+dy
            if self._in_bounds(nx, ny):
                self.grid[nx, ny] = BURNING

        # Agent starts at top-left corner (has to navigate to fire)
        self.agent_pos = [0, 0]
        self._prev_burning = int(np.sum(self.grid == BURNING))

        return self._get_obs(), {}

    def step(self, action: int):
        self.step_count += 1
        reward = 0.0

        # 1. Execute agent action
        reward += self._execute_action(action)

        # 2. Advance fire spread
        self._spread_fire()

        # 3. Calculate step reward
        burning_now = int(np.sum(self.grid == BURNING))
        delta = burning_now - self._prev_burning
        if delta > 0:
            reward -= delta * 0.5   # penalty for each new cell caught fire
        elif delta < 0:
            reward += abs(delta) * 1.0  # bonus for reducing burning cells
        self._prev_burning = burning_now

        # 4. Check termination
        terminated = burning_now == 0  # fire out — agent wins
        truncated = self.step_count >= self.max_steps

        # Bonus for extinguishing fire
        if terminated:
            reward += 100.0

        return self._get_obs(), reward, terminated, truncated, {
            "burning_cells": burning_now,
            "step": self.step_count,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _get_obs(self) -> np.ndarray:
        flat_grid = self.grid.flatten().astype(np.float32) / 3.0  # normalize 0-1
        agent = np.array(self.agent_pos, dtype=np.float32) / self.grid_size
        return np.concatenate([flat_grid, agent])

    def _execute_action(self, action: int) -> float:
        r, c = self.agent_pos
        reward = 0.0

        if action == MOVE_N and r > 0:
            self.agent_pos[0] -= 1
        elif action == MOVE_S and r < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == MOVE_E and c < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == MOVE_W and c > 0:
            self.agent_pos[1] -= 1
        elif action == DEPLOY_HELICOPTER:
            # Drop water: suppress 3×3 area around agent
            suppressed = 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if self._in_bounds(nr, nc) and self.grid[nr, nc] == BURNING:
                        self.grid[nr, nc] = SUPPRESSED
                        suppressed += 1
            if suppressed > 0:
                reward += suppressed * 3.0   # +3 per cell suppressed
            else:
                reward -= 2.0   # wasted deployment
        elif action == DEPLOY_CREW:
            # Create firebreak at current cell
            if self.grid[r, c] == BURNING:
                self.grid[r, c] = SUPPRESSED
                reward += 5.0   # direct suppression
            elif self.grid[r, c] == UNBURNED:
                # Check if adjacent to fire — good tactical positioning
                adjacent_fire = any(
                    self._in_bounds(r+dr, c+dc) and self.grid[r+dr, c+dc] == BURNING
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                )
                if adjacent_fire:
                    self.grid[r, c] = SUPPRESSED
                    reward += 2.0   # proactive firebreak
                else:
                    reward -= 1.0   # wasted — not near fire
            elif self.grid[r, c] == BURNED:
                reward -= 3.0   # wasted on already-burned land

        return reward

    def _spread_fire(self):
        """Stochastic fire spread: each burning cell has spread_prob chance per neighbour."""
        new_burning = []
        burning_cells = list(zip(*np.where(self.grid == BURNING)))

        for (r, c) in burning_cells:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if (self._in_bounds(nr, nc)
                        and self.grid[nr, nc] == UNBURNED
                        and self.np_random.random() < self.spread_prob):
                    new_burning.append((nr, nc))
            # Burning cell may burn out with small probability
            if self.np_random.random() < 0.05:
                self.grid[r, c] = BURNED

        for (r, c) in new_burning:
            self.grid[r, c] = BURNING
