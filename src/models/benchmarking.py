"""Shared benchmark config and evaluation helpers for wildfire RL runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.models.fire_env import (
    ASSET_BURNED,
    BURNED,
    BURNING,
    DEPLOY_CREW,
    DEPLOY_HELICOPTER,
    HELD_OUT_FAMILIES,
    MOVE_E,
    MOVE_N,
    MOVE_S,
    MOVE_W,
    TRAIN_FAMILIES,
    WildfireEnv,
    create_benchmark_env,
    load_scenario_parameter_records,
)

CANONICAL_TRAIN_DATASET = Path("data/static/scenario_parameter_records_seeded_train.json")
CANONICAL_VAL_DATASET = Path("data/static/scenario_parameter_records_seeded_val.json")
CANONICAL_HOLDOUT_DATASET = Path("data/static/scenario_parameter_records_seeded_holdout.json")
CANONICAL_TRAINING_SEEDS = [11, 22, 33, 44, 55]
CANONICAL_CHECKPOINT_INTERVAL_STEPS = 20_000
CANONICAL_CHECKPOINT_EVAL_EPISODES = 20
CANONICAL_FINAL_EVAL_EPISODES = 100
CANONICAL_TIMESTEPS_BY_ALGO = {
    "ppo": 200_000,
    "a2c": 200_000,
    "dqn": 200_000,
}
RUN_LABELS = ("smoke", "pilot", "final", "karpathy")

ROLLOUT_AGENT_TYPES = ("ppo", "a2c", "dqn", "greedy", "random", "non_intervention")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for evaluating one split."""

    name: str
    expected_split: str
    dataset_path: Path
    scenario_families: list[tuple[str, str, str]]


def canonical_train_preset(algo: str) -> dict[str, Any]:
    """Return canonical training defaults for one algorithm."""
    algo_name = algo.lower()
    if algo_name not in CANONICAL_TIMESTEPS_BY_ALGO:
        msg = f"Unknown algorithm '{algo_name}' for canonical preset"
        raise ValueError(msg)

    return {
        "algo": algo_name,
        "total_timesteps": CANONICAL_TIMESTEPS_BY_ALGO[algo_name],
        "train_dataset": CANONICAL_TRAIN_DATASET,
        "val_dataset": CANONICAL_VAL_DATASET,
        "holdout_dataset": CANONICAL_HOLDOUT_DATASET,
        "train_families": list(TRAIN_FAMILIES),
        "val_families": list(TRAIN_FAMILIES),
        "family_holdout_families": list(HELD_OUT_FAMILIES),
        "checkpoint_interval_steps": CANONICAL_CHECKPOINT_INTERVAL_STEPS,
        "checkpoint_eval_episodes": CANONICAL_CHECKPOINT_EVAL_EPISODES,
        "final_eval_episodes": CANONICAL_FINAL_EVAL_EPISODES,
        "checkpoint_visible_splits": ["train", "val"],
        "benchmark_mode": True,
        "holdout_checkpoint_visibility": False,
    }


def canonical_eval_preset() -> dict[str, Any]:
    """Return canonical evaluation defaults."""
    return {
        "train_dataset": CANONICAL_TRAIN_DATASET,
        "val_dataset": CANONICAL_VAL_DATASET,
        "holdout_dataset": CANONICAL_HOLDOUT_DATASET,
        "episodes": CANONICAL_FINAL_EVAL_EPISODES,
        "seeds": list(CANONICAL_TRAINING_SEEDS),
        "train_families": list(TRAIN_FAMILIES),
        "val_families": list(TRAIN_FAMILIES),
        "family_holdout_families": list(HELD_OUT_FAMILIES),
        "temporal_holdout_families": list(TRAIN_FAMILIES),
    }


def load_records(path: Path, *, expected_split: str) -> list[dict]:
    """Load validated benchmark records for one split."""
    return load_scenario_parameter_records(path, benchmark_mode=True, expected_split=expected_split)


def build_default_splits(
    *,
    train_dataset: Path,
    val_dataset: Path,
    holdout_dataset: Path,
    include_family_holdout: bool,
    include_temporal_holdout: bool,
    train_families: list[tuple[str, str, str]],
    val_families: list[tuple[str, str, str]],
    family_holdout_families: list[tuple[str, str, str]],
    temporal_holdout_families: list[tuple[str, str, str]],
) -> list[SplitConfig]:
    """Build canonical split evaluation configs."""
    splits = [
        SplitConfig(
            name="train",
            expected_split="train",
            dataset_path=train_dataset,
            scenario_families=train_families,
        ),
        SplitConfig(
            name="val",
            expected_split="val",
            dataset_path=val_dataset,
            scenario_families=val_families,
        ),
    ]
    if include_family_holdout:
        splits.append(
            SplitConfig(
                name="family_holdout",
                expected_split="val",
                dataset_path=val_dataset,
                scenario_families=family_holdout_families,
            )
        )
    if include_temporal_holdout:
        splits.append(
            SplitConfig(
                name="temporal_holdout_diagnostic",
                expected_split="holdout",
                dataset_path=holdout_dataset,
                scenario_families=temporal_holdout_families,
            )
        )
    return splits


def _nearest_burning_cell(env: WildfireEnv) -> tuple[int, int] | None:
    burning_positions = np.argwhere(env.grid == BURNING)
    if burning_positions.size == 0:
        return None
    ar, ac = env.agent_pos
    dists = np.abs(burning_positions[:, 0] - ar) + np.abs(burning_positions[:, 1] - ac)
    index = int(np.argmin(dists))
    return int(burning_positions[index, 0]), int(burning_positions[index, 1])


def greedy_action(env: WildfireEnv) -> int:
    """Simple greedy baseline policy."""
    ar, ac = env.agent_pos

    if env.heli_left > 0 and env.heli_cd == 0:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = ar + dr, ac + dc
                if (
                    0 <= rr < env.grid_size
                    and 0 <= cc < env.grid_size
                    and env.grid[rr, cc] == BURNING
                ):
                    return DEPLOY_HELICOPTER

    if env.crew_left > 0 and env.crew_cd == 0 and env.grid[ar, ac] == BURNING:
        return DEPLOY_CREW

    target = _nearest_burning_cell(env)
    if target is None:
        return MOVE_N

    tr, tc = target
    if tr < ar:
        return MOVE_N
    if tr > ar:
        return MOVE_S
    if tc > ac:
        return MOVE_E
    if tc < ac:
        return MOVE_W

    if env.crew_left > 0 and env.crew_cd == 0:
        return DEPLOY_CREW
    return MOVE_N


def _select_action(agent_name: str, env: WildfireEnv, obs: np.ndarray, model) -> int:
    if agent_name == "random":
        return int(env.action_space.sample())
    if agent_name == "greedy":
        return greedy_action(env)
    if agent_name == "non_intervention":
        return MOVE_N
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def rollout_episode(
    env: WildfireEnv, *, agent_name: str, model, seed: int
) -> dict[str, float | int | None]:
    """Roll one deterministic benchmark episode and return scalar metrics."""
    if agent_name not in ROLLOUT_AGENT_TYPES:
        msg = f"Unsupported agent type '{agent_name}'"
        raise ValueError(msg)
    if agent_name in {"ppo", "a2c", "dqn"} and model is None:
        msg = f"Agent '{agent_name}' requires a loaded model"
        raise ValueError(msg)

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}

    for _ in range(env.max_steps):
        action = _select_action(agent_name, env, obs, model)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        if terminated or truncated:
            break

    final_burned_cells = int(
        np.sum((env.grid == BURNED) | (env.grid == BURNING) | (env.grid == ASSET_BURNED))
    )
    total_cells = float(env.grid_size * env.grid_size)

    total_deployments = int(info.get("total_deployment_attempts", 0))
    successful_deployments = int(info.get("successful_deployments", 0))
    wasted_deployments = int(info.get("wasted_deployment_attempts", 0))
    resource_efficiency = (
        float(successful_deployments / total_deployments) if total_deployments > 0 else 0.0
    )
    wasted_deployment_rate = (
        float(wasted_deployments / total_deployments) if total_deployments > 0 else 0.0
    )
    containment_success = bool(terminated and not truncated)

    return {
        "return": float(episode_return),
        "assets_lost": int(info.get("assets_lost", env.assets_lost)),
        "containment_success": 1.0 if containment_success else 0.0,
        "burned_area_fraction": float(final_burned_cells / total_cells),
        "time_to_containment": int(info.get("step", env.step_count))
        if containment_success
        else None,
        "resource_efficiency": resource_efficiency,
        "wasted_deployment_rate": wasted_deployment_rate,
        "final_burned_area_cells": final_burned_cells,
    }


def summarize_episodes(episode_metrics: list[dict[str, float | int | None]]) -> dict[str, Any]:
    """Summarize episode-level benchmark metrics for one split."""
    if not episode_metrics:
        raise ValueError("No episode metrics to summarize")

    returns = np.array([float(m["return"]) for m in episode_metrics], dtype=float)
    assets_lost = np.array([float(m["assets_lost"]) for m in episode_metrics], dtype=float)
    containment = np.array([float(m["containment_success"]) for m in episode_metrics], dtype=float)
    burned_fractions = np.array(
        [float(m["burned_area_fraction"]) for m in episode_metrics], dtype=float
    )
    resource_eff = np.array([float(m["resource_efficiency"]) for m in episode_metrics], dtype=float)
    wasted_rates = np.array(
        [float(m["wasted_deployment_rate"]) for m in episode_metrics], dtype=float
    )

    containment_steps = [
        float(m["time_to_containment"])
        for m in episode_metrics
        if m.get("time_to_containment") is not None
    ]
    mean_time_to_containment = (
        float(np.mean(np.array(containment_steps, dtype=float))) if containment_steps else None
    )

    summary: dict[str, Any] = {
        "episodes": len(episode_metrics),
        "mean_return": float(returns.mean()),
        "asset_survival_rate": float(np.mean(assets_lost == 0.0)),
        "containment_success_rate": float(containment.mean()),
        "mean_burned_area_fraction": float(burned_fractions.mean()),
        "mean_time_to_containment": mean_time_to_containment,
        "mean_resource_efficiency": float(resource_eff.mean()),
        "wasted_deployment_rate": float(wasted_rates.mean()),
    }

    if "normalized_burn_ratio" in episode_metrics[0]:
        normalized = np.array(
            [float(m["normalized_burn_ratio"]) for m in episode_metrics],
            dtype=float,
        )
        summary["mean_normalized_burn_ratio"] = float(normalized.mean())

    return summary


def _mean_and_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def aggregate_seed_summaries(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed summaries into mean/std_across_seeds values."""
    if not seed_summaries:
        raise ValueError("No seed summaries provided")

    metric_keys = [
        "mean_return",
        "asset_survival_rate",
        "containment_success_rate",
        "mean_burned_area_fraction",
        "mean_time_to_containment",
        "mean_resource_efficiency",
        "wasted_deployment_rate",
    ]
    if "mean_normalized_burn_ratio" in seed_summaries[0]:
        metric_keys.append("mean_normalized_burn_ratio")

    aggregate: dict[str, Any] = {
        "episodes_per_seed": int(seed_summaries[0]["episodes"]),
        "num_seeds": len(seed_summaries),
        "std_across_seeds": {},
    }
    for key in metric_keys:
        values = [float(summary[key]) for summary in seed_summaries if summary.get(key) is not None]
        mean_value, std_value = _mean_and_std(values)
        aggregate[key] = mean_value
        aggregate["std_across_seeds"][key] = std_value

    return aggregate


def evaluate_agent_on_split(
    *,
    agent_name: str,
    model,
    records: list[dict],
    expected_split: str,
    scenario_families: list[tuple[str, str, str]],
    seeds: list[int],
    episodes_per_seed: int,
    compute_normalized_burn_ratio: bool,
) -> dict[str, Any]:
    """Evaluate one agent on one split over one or more seeds."""
    seed_summaries: list[dict[str, Any]] = []

    for seed in seeds:
        env = create_benchmark_env(
            expected_split=expected_split,
            scenario_parameter_records=records,
            scenario_families=scenario_families,
        )
        baseline_env = None
        if compute_normalized_burn_ratio:
            baseline_env = create_benchmark_env(
                expected_split=expected_split,
                scenario_parameter_records=records,
                scenario_families=scenario_families,
            )
        try:
            episode_metrics = []
            for ep in range(episodes_per_seed):
                eval_seed = seed * 1_000_000 + ep
                metrics = rollout_episode(env, agent_name=agent_name, model=model, seed=eval_seed)
                if baseline_env is not None:
                    baseline_metrics = rollout_episode(
                        baseline_env,
                        agent_name="non_intervention",
                        model=None,
                        seed=eval_seed,
                    )
                    baseline_burned = int(baseline_metrics["final_burned_area_cells"])
                    metrics["normalized_burn_ratio"] = float(
                        float(metrics["final_burned_area_cells"]) / max(1, baseline_burned)
                    )
                episode_metrics.append(metrics)

            seed_summary = summarize_episodes(episode_metrics)
            seed_summary["seed"] = seed
            seed_summaries.append(seed_summary)
        finally:
            env.close()
            if baseline_env is not None:
                baseline_env.close()

    return {
        "seed_metrics": seed_summaries,
        "aggregate": aggregate_seed_summaries(seed_summaries),
    }


def load_model_for_algo(algo: str, model_path: Path):
    """Load a Stable-Baselines3 model for the given algorithm name."""
    algo_name = algo.lower()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    if algo_name == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(model_path))
    if algo_name == "a2c":
        from stable_baselines3 import A2C

        return A2C.load(str(model_path))
    if algo_name == "dqn":
        from stable_baselines3 import DQN

        return DQN.load(str(model_path))

    msg = f"Unsupported model algo '{algo_name}'"
    raise ValueError(msg)


def heldout_performance_drop(train_metric: float, heldout_metric: float) -> float:
    """Compute held-out drop for a metric (held-out - train)."""
    return float(heldout_metric - train_metric)
