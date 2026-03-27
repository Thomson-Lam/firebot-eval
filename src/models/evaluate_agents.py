"""General benchmark evaluation interface for RL agents on split datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.models.fire_env import (
    ASSET_BURNED,
    BURNED,
    BURNING,
    DEPLOY_CREW,
    DEPLOY_HELICOPTER,
    MOVE_E,
    MOVE_N,
    MOVE_S,
    MOVE_W,
    WildfireEnv,
    load_scenario_parameter_records,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency

    def tqdm(iterable, **_kwargs):
        return iterable


DEFAULT_TRAIN_DATASET = Path("data/static/scenario_parameter_records_train.json")
DEFAULT_VAL_DATASET = Path("data/static/scenario_parameter_records_val.json")
DEFAULT_HOLDOUT_DATASET = Path("data/static/scenario_parameter_records_holdout.json")
DEFAULT_PPO_MODEL = Path("src/models/tactical_ppo_agent.zip")


def _load_ppo_model(path: Path):
    from stable_baselines3 import PPO

    if not path.exists():
        raise FileNotFoundError(f"PPO model not found at {path}")
    return PPO.load(str(path))


def _nearest_burning_cell(env: WildfireEnv) -> tuple[int, int] | None:
    burning_positions = np.argwhere(env.grid == BURNING)
    if burning_positions.size == 0:
        return None
    ar, ac = env.agent_pos
    dists = np.abs(burning_positions[:, 0] - ar) + np.abs(burning_positions[:, 1] - ac)
    idx = int(np.argmin(dists))
    return int(burning_positions[idx, 0]), int(burning_positions[idx, 1])


def _greedy_action(env: WildfireEnv) -> int:
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
    return DEPLOY_CREW if env.crew_left > 0 and env.crew_cd == 0 else MOVE_N


def _run_episode(env: WildfireEnv, agent_name: str, model, seed: int) -> dict:
    obs, _info = env.reset(seed=seed)
    episode_return = 0.0
    terminated = False
    truncated = False
    info = {}

    for _ in range(env.max_steps):
        if agent_name == "random":
            action = int(env.action_space.sample())
        elif agent_name == "greedy":
            action = _greedy_action(env)
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        if terminated or truncated:
            break

    final_burned_area = int(
        np.sum((env.grid == BURNED) | (env.grid == BURNING) | (env.grid == ASSET_BURNED))
    )
    containment_success = 1 if terminated and not truncated else 0
    heli_used = env.heli_budget_init - info.get("heli_left", env.heli_left)
    crew_used = env.crew_budget_init - info.get("crew_left", env.crew_left)

    return {
        "return": episode_return,
        "assets_lost": int(info.get("assets_lost", env.assets_lost)),
        "containment_success": containment_success,
        "final_burned_area": final_burned_area,
        "time_to_containment": int(info.get("step", env.step_count)),
        "heli_used": int(heli_used),
        "crew_used": int(crew_used),
        "resource_efficiency": float(final_burned_area / max(1, heli_used + crew_used)),
    }


def _evaluate_agent_on_split(
    *,
    agent_name: str,
    records: list[dict],
    seeds: list[int],
    episodes_per_seed: int,
    model,
    compute_normalized_burn_ratio: bool,
) -> dict:
    episode_metrics = []

    for seed in seeds:
        env = WildfireEnv(scenario_parameter_records=records, randomize_scenario=True)
        baseline_env = WildfireEnv(scenario_parameter_records=records, randomize_scenario=True)
        iterator = tqdm(range(episodes_per_seed), desc=f"{agent_name} seed={seed}", unit="ep")
        for ep in iterator:
            eval_seed = seed * 10_000 + ep
            metrics = _run_episode(env, agent_name, model, seed=eval_seed)
            if compute_normalized_burn_ratio:
                # Use MOVE_N-only as deterministic no-action surrogate baseline.
                _obs, _ = baseline_env.reset(seed=eval_seed)
                for _ in range(baseline_env.max_steps):
                    _obs, _reward, done, trunc, _base_info = baseline_env.step(MOVE_N)
                    if done or trunc:
                        break
                baseline_burned = int(
                    np.sum(
                        (baseline_env.grid == BURNED)
                        | (baseline_env.grid == BURNING)
                        | (baseline_env.grid == ASSET_BURNED)
                    )
                )
                metrics["normalized_burn_ratio"] = float(
                    metrics["final_burned_area"] / max(1, baseline_burned)
                )
            episode_metrics.append(metrics)

    arr = {
        key: np.array([m[key] for m in episode_metrics], dtype=float) for key in episode_metrics[0]
    }
    summary = {
        "episodes": len(episode_metrics),
        "mean_return": float(arr["return"].mean()),
        "std_return": float(arr["return"].std()),
        "asset_survival_rate": float((arr["assets_lost"] == 0).mean()),
        "containment_success_rate": float(arr["containment_success"].mean()),
        "mean_final_burned_area": float(arr["final_burned_area"].mean()),
        "mean_time_to_containment": float(arr["time_to_containment"].mean()),
        "mean_resource_efficiency": float(arr["resource_efficiency"].mean()),
        "variance_across_episodes": float(arr["return"].var()),
    }
    if "normalized_burn_ratio" in arr:
        summary["mean_normalized_burn_ratio"] = float(arr["normalized_burn_ratio"].mean())
    return summary


def _load_split_records(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return load_scenario_parameter_records(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark agents on train/val/holdout splits"
    )
    parser.add_argument("--agents", type=str, default="ppo,greedy,random")
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--val-dataset", type=Path, default=DEFAULT_VAL_DATASET)
    parser.add_argument("--holdout-dataset", type=Path, default=DEFAULT_HOLDOUT_DATASET)
    parser.add_argument("--ppo-model", type=Path, default=DEFAULT_PPO_MODEL)
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per seed per split")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--no-normalized-burn", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    agents = [a.strip().lower() for a in args.agents.split(",") if a.strip()]

    split_records = {
        "train": _load_split_records(args.train_dataset),
        "val": _load_split_records(args.val_dataset),
        "holdout": _load_split_records(args.holdout_dataset),
    }

    results: dict[str, dict] = {}
    ppo_model = None
    if "ppo" in agents:
        ppo_model = _load_ppo_model(args.ppo_model)

    for agent_name in agents:
        results[agent_name] = {}
        for split_name, records in split_records.items():
            if not records:
                continue
            model = ppo_model if agent_name == "ppo" else None
            summary = _evaluate_agent_on_split(
                agent_name=agent_name,
                records=records,
                seeds=seeds,
                episodes_per_seed=args.episodes,
                model=model,
                compute_normalized_burn_ratio=not args.no_normalized_burn,
            )
            results[agent_name][split_name] = summary

    print("\nBenchmark Summary")
    print("=" * 72)
    for agent_name, split_summaries in results.items():
        for split_name, summary in split_summaries.items():
            print(
                f"{agent_name:>8} | {split_name:<7} | episodes={summary['episodes']:>4} "
                f"| return={summary['mean_return']:.1f} "
                f"| assets_survival={summary['asset_survival_rate']:.3f} "
                f"| containment={summary['containment_success_rate']:.3f} "
                f"| burned={summary['mean_final_burned_area']:.1f}"
            )

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
