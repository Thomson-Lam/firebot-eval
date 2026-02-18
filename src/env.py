from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import EnvConfig


class RegimeSwitchingGridWorld(gym.Env):
    """10x10 grid with regime-switching hazards.

    Regime A (Minefield): hazards deal heavy penalties, safe path preferred.
    Regime B (Time Pressure): hazards deactivated, step cost ramps up.

    The regime is NOT observable — the agent must infer it from experience.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        c = self.config

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # up, down, left, right

        self._rng = np.random.RandomState(c.seed)
        self.hazards = self._place_hazards()
        self.start = (0, 0)
        self.goal = (c.grid_size - 1, c.grid_size - 1)

        # Global step counter persists across episodes for regime switching
        self._global_step = 0
        self._episode_step = 0
        self._agent_pos = self.start
        self._prev_dist = self._manhattan(self.start)
        self._episode_regime: str | None = None  # for per-episode randomization

    @property
    def state_dim(self) -> int:
        return 4

    def _place_hazards(self) -> set[tuple[int, int]]:
        """Place hazards with a barrier forcing unavoidable hazard crossings.

        A horizontal barrier at the grid midpoint spans all columns. The agent
        must cross it to reach the goal, ensuring regime-dependent sub-reward
        patterns (safety penalty in A, free passage in B) on every episode.
        Remaining hazard budget is scattered along center/diagonal.
        """
        c = self.config
        hazards: set[tuple[int, int]] = set()

        mid = c.grid_size // 2
        for j in range(c.grid_size):
            cell = (mid, j)
            if cell != (0, 0) and cell != (c.grid_size - 1, c.grid_size - 1):
                hazards.add(cell)

        remaining = c.num_hazard_cells - len(hazards)
        if remaining > 0:
            candidates = []
            for i in range(c.grid_size):
                for j in range(c.grid_size):
                    if (i, j) in hazards or (i, j) == (0, 0) or (i, j) == (c.grid_size - 1, c.grid_size - 1):
                        continue
                    center_dist = abs(i - c.grid_size / 2) + abs(j - c.grid_size / 2)
                    diag_dist = abs(i - j)
                    if center_dist < c.grid_size * 0.4 or diag_dist < 2:
                        candidates.append((i, j))
            self._rng.shuffle(candidates)
            for cell in candidates[:remaining]:
                hazards.add(cell)

        return hazards

    def _manhattan(self, pos: tuple[int, int]) -> int:
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    @property
    def regime_switch_steps(self) -> list[int]:
        """Global steps where regime switches occur (for plotting)."""
        interval = self.config.regime_switch_interval
        return [i * interval for i in range(1, 20)]

    def get_regime(self) -> str:
        if self.config.randomize_regime_per_episode and self._episode_regime is not None:
            return self._episode_regime
        cycle = self._global_step // self.config.regime_switch_interval
        return "A" if cycle % 2 == 0 else "B"

    def _get_obs(self) -> np.ndarray:
        c = self.config
        x, y = self._agent_pos
        # Local hazard indicator: 1 if any adjacent cell (or current) is hazard
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]]
        hazard_nearby = float(any(n in self.hazards for n in neighbors))
        step_frac = self._episode_step / max(c.max_episode_steps, 1)
        return np.array([
            x / (c.grid_size - 1),
            y / (c.grid_size - 1),
            hazard_nearby,
            step_frac,
        ], dtype=np.float32)

    def _compute_sub_rewards(self) -> np.ndarray:
        c = self.config
        regime = self.get_regime()

        # R_progress: normalized distance reduction + goal bonus
        curr_dist = self._manhattan(self._agent_pos)
        r_progress = (self._prev_dist - curr_dist) / c.grid_size
        if self._agent_pos == self.goal:
            r_progress += c.goal_reward

        # R_safety: hazard contact penalty (regime-modulated)
        on_hazard = self._agent_pos in self.hazards
        if on_hazard:
            r_safety = c.hazard_penalty_full if regime == "A" else c.hazard_penalty_reduced
        else:
            r_safety = 0.0

        # R_efficiency: step cost (ramps in Regime B)
        if regime == "B":
            r_efficiency = c.step_cost_base + c.step_cost_ramp * self._episode_step
        else:
            r_efficiency = c.step_cost_base

        return np.array([r_progress, r_safety, r_efficiency], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._agent_pos = self.start
        self._prev_dist = self._manhattan(self.start)
        self._episode_step = 0

        if self.config.randomize_regime_per_episode:
            self._episode_regime = self._rng.choice(["A", "B"])

        obs = self._get_obs()
        info = {
            "sub_rewards": np.zeros(3, dtype=np.float32),
            "regime": self.get_regime(),
        }
        return obs, info

    def step(self, action: int):
        c = self.config
        x, y = self._agent_pos

        # Movement: 0=up, 1=down, 2=left, 3=right
        dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nx, ny = x + dx, y + dy
        if 0 <= nx < c.grid_size and 0 <= ny < c.grid_size:
            self._agent_pos = (nx, ny)

        self._episode_step += 1
        self._global_step += 1

        sub_rewards = self._compute_sub_rewards()
        self._prev_dist = self._manhattan(self._agent_pos)

        # Scalar reward is sum of sub-rewards (goal reward is included in r_progress)
        reward = float(sub_rewards.sum())

        terminated = self._agent_pos == self.goal

        truncated = self._episode_step >= c.max_episode_steps

        info = {
            "sub_rewards": sub_rewards,
            "regime": self.get_regime(),
            "on_hazard": self._agent_pos in self.hazards,
        }
        if terminated:
            info["goal_reached"] = True
            info["steps_to_goal"] = self._episode_step

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
