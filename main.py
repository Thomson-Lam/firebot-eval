import argparse
import os
import pickle

import numpy as np

from src.config import ExperimentConfig
from src.train import train_layer1, train_layer2, train_layer3


def main():
    parser = argparse.ArgumentParser(description="Dynamic Reward Decomposition")
    parser.add_argument("--layer", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = ExperimentConfig()
    config.train.seed = args.seed
    config.train.use_wandb = not args.no_wandb
    config.train.device = args.device
    if args.timesteps:
        config.train.total_timesteps = args.timesteps

    if args.layer == 1:
        results = train_layer1(config)
        print(f"\nBest static: {results['best_static']['label']} "
              f"(return={results['best_static']['final_mean_return']:.2f})")
        print(f"Recurrent: return={results['recurrent_result']['final_mean_return']:.2f}")
        print(f"Oracle: return={results['oracle_result']['final_mean_return']:.2f}")
    elif args.layer == 2:
        results = train_layer2(config)
        print(f"\nDRD final return: {results['final_mean_return']:.2f}")
        # Save results for visualization
        os.makedirs("plots", exist_ok=True)
        pickle.dump({
            "all_returns": results["all_returns"],
            "final_mean_return": results["final_mean_return"],
            "weight_history": results["weight_history"],
            "regime_history": results["regime_history"],
        }, open("plots/drd_results.pkl", "wb"))
        print("Results saved to plots/drd_results.pkl")
    elif args.layer == 3:
        results = train_layer3(config)
        for k, v in results.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
