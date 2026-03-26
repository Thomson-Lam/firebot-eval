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


def train(
    total_timesteps: int = 200_000,
    spread_rate_m_per_min: float = 15.0,
    n_envs: int = 4,
    seed: int = 42,
    scenario_dataset_path: str | None = None,
) -> None:
    """
    Train the PPO tactical agent.

    Args:
        total_timesteps:       Total env steps to train for.
        spread_rate_m_per_min: Fire spread rate (from XGBoost) for training.
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
    print(f"  Spread rate:  {spread_rate_m_per_min} m/min")
    print(f"  Environments: {n_envs} parallel")
    print("  Grid:         25×25 with critical assets")
    print("  Budgets:      heli=8, crew=20")
    print()

    env_kwargs: dict = {}
    if scenario_dataset_path:
        records = load_scenario_parameter_records(scenario_dataset_path)
        env_kwargs["scenario_parameter_records"] = records
        print(f"  Scenario records: {len(records)} from {scenario_dataset_path}")
    else:
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
    from src.models.fire_env import WildfireEnv as Env

    eval_kwargs = dict(env_kwargs)
    eval_env = Env(**eval_kwargs)
    returns = []
    assets_lost_total = []
    for ep in range(5):
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

    print(f"  Mean return:      {sum(returns) / len(returns):.1f}")
    print(f"  Mean assets lost: {sum(assets_lost_total) / len(assets_lost_total):.1f}")
    print(f"\nTraining complete. Model ready at {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO wildfire tactical agent")
    parser.add_argument(
        "--timesteps", type=int, default=200_000, help="Total training timesteps (default: 200000)"
    )
    parser.add_argument(
        "--spread-rate", type=float, default=15.0, help="Fire spread rate in m/min (default: 15.0)"
    )
    parser.add_argument(
        "--envs", type=int, default=4, help="Number of parallel environments (default: 4)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--scenario-dataset",
        type=str,
        default=None,
        help="Path to cached scenario parameter JSON dataset",
    )
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        spread_rate_m_per_min=args.spread_rate,
        n_envs=args.envs,
        seed=args.seed,
        scenario_dataset_path=args.scenario_dataset,
    )
