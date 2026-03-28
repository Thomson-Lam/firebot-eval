"""
train_rl_agent.py — PPO tactical agent training script.

Trains a PPO agent on the WildfireEnv gymnasium environment (25×25 grid
with critical assets and finite suppression budgets).

Run:
    uv run python -m src.models.train_rl_agent
    uv run python -m src.models.train_rl_agent --timesteps 10000  # quick test
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_SAVE_PATH = Path(__file__).parent / "tactical_ppo_agent"
DEFAULT_SCENARIO_DATASET = Path("data/static/scenario_parameter_records_train.json")


def _resolve_dataset_path(path: str | None) -> str | None:
    if path:
        return path
    if DEFAULT_SCENARIO_DATASET.exists():
        return str(DEFAULT_SCENARIO_DATASET)
    return None


def _existing_path(path: str | None) -> str | None:
    if path and Path(path).exists():
        return path
    return None


def _evaluate_model(
    model,
    dataset_path: str,
    seed: int,
    episodes: int = 5,
    expected_split: str | None = None,
) -> tuple[float, float]:
    from src.models.fire_env import WildfireEnv, load_scenario_parameter_records

    records = load_scenario_parameter_records(
        dataset_path,
        benchmark_mode=True,
        expected_split=expected_split,
    )
    eval_env = WildfireEnv(
        scenario_parameter_records=records,
        benchmark_mode=True,
        expected_split=expected_split,
    )
    returns = []
    assets_lost_total = []
    for ep in range(episodes):
        obs, _ = eval_env.reset(seed=seed + ep + 100)
        ep_return = 0.0
        for _ in range(150):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(int(action))
            ep_return += reward
            if done or truncated:
                break
        returns.append(ep_return)
        assets_lost_total.append(info["assets_lost"])
    return sum(returns) / len(returns), sum(assets_lost_total) / len(assets_lost_total)


def train(
    total_timesteps: int = 200_000,
    spread_rate_m_per_min: float = 15.0,
    n_envs: int = 4,
    seed: int = 42,
    scenario_dataset_path: str | None = None,
    val_dataset_path: str | None = None,
    holdout_dataset_path: str | None = None,
    allow_legacy_dev_fallback: bool = False,
) -> None:
    """
    Train the PPO tactical agent.

    Args:
        total_timesteps:       Total env steps to train for.
        spread_rate_m_per_min: Legacy fixed spread rate used only in dev fallback mode.
        n_envs:                Parallel environments.
        seed:                  Random seed for reproducibility.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        from src.models.fire_env import WildfireEnv, load_scenario_parameter_records
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("   Run: uv sync")
        sys.exit(1)

    print("=" * 60)
    print("  FireGrid PPO Tactical Agent — Training")
    print("=" * 60)
    print(f"  Timesteps:    {total_timesteps:,}")
    print(f"  Environments: {n_envs} parallel")
    print("  Grid:         25×25 with critical assets")
    print("  Budgets:      heli=8, crew=20")
    print()

    scenario_dataset_path = _resolve_dataset_path(scenario_dataset_path)

    env_kwargs: dict = {}
    if scenario_dataset_path:
        records = load_scenario_parameter_records(
            scenario_dataset_path,
            benchmark_mode=True,
            expected_split="train",
        )
        env_kwargs["scenario_parameter_records"] = records
        env_kwargs["benchmark_mode"] = True
        env_kwargs["expected_split"] = "train"
        print("  Runtime data: frozen offline scenario records (no live ingestion)")
        print(f"  Scenario records: {len(records)} from {scenario_dataset_path}")
    else:
        if not allow_legacy_dev_fallback:
            msg = (
                "No training scenario dataset found. Canonical training requires precomputed "
                "scenario_parameter_records_train.json. To run non-canonical dev mode, pass "
                "--allow-legacy-dev-fallback with --spread-rate."
            )
            raise ValueError(msg)
        print(
            "  No scenario dataset found; running explicit legacy dev mode "
            "with --spread-rate fallback."
        )
        print(f"  Legacy spread rate: {spread_rate_m_per_min} m/min")
        env_kwargs["benchmark_mode"] = False
        env_kwargs["base_spread_rate_m_per_min"] = spread_rate_m_per_min
    vec_env = make_vec_env(
        WildfireEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        device="cpu",
    )

    print("Training PPO agent...\n")
    model.learn(total_timesteps=total_timesteps)

    model.save(str(MODEL_SAVE_PATH))
    print(f"\nPPO model saved -> {MODEL_SAVE_PATH}.zip")

    # Quick evaluation
    print("\nRunning quick evaluation (5 episodes)...")
    eval_targets = [("train", scenario_dataset_path)]
    if _existing_path(val_dataset_path):
        eval_targets.append(("val", val_dataset_path))
    if _existing_path(holdout_dataset_path):
        eval_targets.append(("holdout", holdout_dataset_path))

    for split_name, dataset_path in eval_targets:
        if not dataset_path:
            continue
        mean_return, mean_assets_lost = _evaluate_model(
            model,
            dataset_path,
            seed=seed,
            episodes=5,
            expected_split=split_name,
        )
        print(f"  [{split_name}] Mean return:      {mean_return:.1f}")
        print(f"  [{split_name}] Mean assets lost: {mean_assets_lost:.1f}")
    print(f"\nTraining complete. Model ready at {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO wildfire tactical agent")
    parser.add_argument(
        "--timesteps", type=int, default=200_000, help="Total training timesteps (default: 200000)"
    )
    parser.add_argument(
        "--spread-rate",
        type=float,
        default=15.0,
        help="Legacy dev-mode fixed spread rate in m/min (default: 15.0)",
    )
    parser.add_argument(
        "--allow-legacy-dev-fallback",
        action="store_true",
        help="Allow non-canonical fallback when no scenario dataset is available",
    )
    parser.add_argument(
        "--envs", type=int, default=4, help="Number of parallel environments (default: 4)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--scenario-dataset",
        type=str,
        default=None,
        help="Path to cached training scenario parameter JSON dataset",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default="data/static/scenario_parameter_records_val.json",
        help="Path to cached validation scenario parameter JSON dataset",
    )
    parser.add_argument(
        "--holdout-dataset",
        type=str,
        default="data/static/scenario_parameter_records_holdout.json",
        help="Path to cached holdout scenario parameter JSON dataset",
    )
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        spread_rate_m_per_min=args.spread_rate,
        n_envs=args.envs,
        seed=args.seed,
        scenario_dataset_path=args.scenario_dataset,
        val_dataset_path=args.val_dataset,
        holdout_dataset_path=args.holdout_dataset,
        allow_legacy_dev_fallback=args.allow_legacy_dev_fallback,
    )
