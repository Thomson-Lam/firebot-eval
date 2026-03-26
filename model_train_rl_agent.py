"""
train_rl_agent.py — PPO tactical agent training script.

Trains a PPO agent on the WildfireEnv gymnasium environment.
Saves the trained model to src/models/tactical_ppo_agent.zip.

Run from backend/:
    uv run python -m src.models.train_rl_agent

Training takes ~2–4 minutes on CPU for 50,000 timesteps.
For a quick test, use --timesteps 10000 (~30 seconds).
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # suppress SB3 verbosity
logger = logging.getLogger(__name__)

MODEL_SAVE_PATH = Path(__file__).parent / "tactical_ppo_agent"


def train(
    total_timesteps: int = 50_000,
    spread_rate_m_per_min: float = 15.0,
    n_envs: int = 4,
    seed: int = 42,
) -> None:
    """
    Train the PPO tactical agent.

    Args:
        total_timesteps:      Total env steps to train for. 50k ≈ 3min on CPU.
        spread_rate_m_per_min: Fire spread rate (from XGBoost) for training conditions.
        n_envs:               Parallel environments (improves sample efficiency).
        seed:                 Random seed for reproducibility.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
        from src.models.fire_env import WildfireEnv
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
    print()

    # Create vectorized parallel environments
    env_kwargs = {"base_spread_rate_m_per_min": spread_rate_m_per_min}
    vec_env = make_vec_env(
        WildfireEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )

    # PPO with MlpPolicy — processes the flat (2502,) observation vector
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,            # steps per env before update
        batch_size=64,
        n_epochs=10,
        gamma=0.995,            # high discount — rewards future containment
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # encourages exploration
        seed=seed,
        device="cpu",
    )

    print("Training PPO agent (this takes ~2-4 minutes)...\n")
    model.learn(total_timesteps=total_timesteps)

    model.save(str(MODEL_SAVE_PATH))
    print(f"\nPPO model saved -> {MODEL_SAVE_PATH}.zip")

    # Quick evaluation
    print("\nRunning quick inference test...")
    from src.models.rl_agent import get_tactical_recommendations
    demo_fire = {
        "fire_id": "BC-2026-001",
        "latitude": 49.9071,
        "longitude": -119.496,
        "area_hectares": 12450,
    }
    demo_spread = {"spread_1h_m": 900, "spread_3h_m": 2700}
    waypoints = get_tactical_recommendations("BC-2026-001", demo_fire, demo_spread)

    print(f"\nGenerated {len(waypoints)} tactical waypoints:")
    for i, wp in enumerate(waypoints, 1):
        print(f"  {i}. [{wp['asset_type'].upper()}] "
              f"({wp['latitude']}, {wp['longitude']}) — {wp['rationale']}")

    print(f"\nTraining complete. Model ready at {MODEL_SAVE_PATH}.zip")
    print("   Test via API: Invoke-RestMethod http://localhost:8000/api/v1/choke_points/BC-2026-001")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO wildfire tactical agent")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Total training timesteps (default: 50000)")
    parser.add_argument("--spread-rate", type=float, default=15.0,
                        help="Fire spread rate in m/min for training (default: 15.0)")
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        spread_rate_m_per_min=args.spread_rate,
        n_envs=args.envs,
    )
