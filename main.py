import argparse

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
    elif args.layer == 2:
        results = train_layer2(config)
        print(f"\nDRD final return: {results['final_mean_return']:.2f}")
    elif args.layer == 3:
        results = train_layer3(config)
        for k, v in results.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
