"""Unified wildfire benchmark trainer for PPO/A2C/DQN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.models.benchmarking import (
    RUN_LABELS,
    build_default_splits,
    canonical_train_preset,
    evaluate_agent_on_split,
    load_model_for_algo,
    load_records,
)
from src.models.fire_env import benchmark_env_kwargs

ALGO_CHOICES = ("ppo", "a2c", "dqn")


def _default_hyperparameters(algo: str) -> dict[str, Any]:
    if algo == "ppo":
        return {
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        }
    if algo == "a2c":
        return {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.995,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
        }
    if algo == "dqn":
        return {
            "learning_rate": 1e-4,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 64,
            "gamma": 0.995,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1_000,
            "exploration_fraction": 0.2,
            "exploration_final_eps": 0.05,
        }
    raise ValueError(f"Unsupported algorithm '{algo}'")


def _build_model(*, algo: str, env, seed: int, device: str, hyperparams: dict[str, Any]):
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO("MlpPolicy", env, seed=seed, device=device, verbose=1, **hyperparams)
    if algo == "a2c":
        from stable_baselines3 import A2C

        return A2C("MlpPolicy", env, seed=seed, device=device, verbose=1, **hyperparams)
    if algo == "dqn":
        from stable_baselines3 import DQN

        return DQN("MlpPolicy", env, seed=seed, device=device, verbose=1, **hyperparams)
    raise ValueError(f"Unsupported algorithm '{algo}'")


def _create_train_env(*, algo: str, env_kwargs: dict[str, Any], n_envs: int, seed: int):
    from src.models.fire_env import WildfireEnv

    if algo in {"ppo", "a2c"}:
        from stable_baselines3.common.env_util import make_vec_env

        return make_vec_env(
            WildfireEnv,
            n_envs=n_envs,
            seed=seed,
            env_kwargs=env_kwargs,
        )

    return WildfireEnv(**env_kwargs)


def _selects_better_checkpoint(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    cand_val = candidate["splits"]["val"]
    inc_val = incumbent["splits"]["val"]

    cand_primary = float(cand_val["asset_survival_rate"])
    inc_primary = float(inc_val["asset_survival_rate"])
    if cand_primary != inc_primary:
        return cand_primary > inc_primary

    cand_tie = float(cand_val["mean_return"])
    inc_tie = float(inc_val["mean_return"])
    return cand_tie > inc_tie


def _single_seed_split_summary(split_eval: dict[str, Any]) -> dict[str, Any]:
    seed_summary = dict(split_eval["seed_metrics"][0])
    seed_summary.pop("seed", None)
    return seed_summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _resolve_hyperparameters(args, algo: str) -> dict[str, Any]:
    hyperparams = _default_hyperparameters(algo)

    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate

    if algo in {"ppo", "a2c"} and args.n_steps is not None:
        hyperparams["n_steps"] = args.n_steps
    if algo in {"ppo", "a2c"} and args.ent_coef is not None:
        hyperparams["ent_coef"] = args.ent_coef

    if algo == "dqn":
        if args.exploration_fraction is not None:
            hyperparams["exploration_fraction"] = args.exploration_fraction
        if args.exploration_final_eps is not None:
            hyperparams["exploration_final_eps"] = args.exploration_final_eps
        if args.target_update_interval is not None:
            hyperparams["target_update_interval"] = args.target_update_interval
        if args.replay_buffer_size is not None:
            hyperparams["buffer_size"] = args.replay_buffer_size

    return hyperparams


def _families_to_jsonable(families: list[tuple[str, str, str]]) -> list[list[str]]:
    return [list(family) for family in families]


def _parse_family_spec(value: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            f"Invalid family spec '{value}'. Expected format: ignition,severity,layout"
        )
    return (parts[0], parts[1], parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train benchmark RL agents for wildfire task")
    parser.add_argument("--algo", type=str, default="ppo", choices=ALGO_CHOICES)
    parser.add_argument(
        "--benchmark-preset",
        type=str,
        default="canonical",
        choices=("canonical",),
        help="Benchmark-safe training preset",
    )
    parser.add_argument("--run-label", type=str, default="smoke", choices=RUN_LABELS)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/benchmark"))

    parser.add_argument("--train-dataset", type=Path, default=None)
    parser.add_argument("--val-dataset", type=Path, default=None)
    parser.add_argument("--holdout-dataset", type=Path, default=None)
    parser.add_argument(
        "--train-family",
        type=str,
        default=None,
        help="Optional single-family override for train split (ignition,severity,layout)",
    )
    parser.add_argument(
        "--val-family",
        type=str,
        default=None,
        help="Optional single-family override for val split (ignition,severity,layout)",
    )
    parser.add_argument(
        "--family-holdout-family",
        type=str,
        default=None,
        help="Optional single-family override for family holdout split (ignition,severity,layout)",
    )

    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=None)
    parser.add_argument("--final-eval-episodes", type=int, default=None)
    parser.add_argument("--include-family-holdout-checkpoints", action="store_true")
    parser.add_argument("--include-family-holdout-final", action="store_true")
    parser.add_argument("--include-temporal-holdout-final", action="store_true")
    parser.add_argument("--no-normalized-burn-final", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--exploration-fraction", type=float, default=None)
    parser.add_argument("--exploration-final-eps", type=float, default=None)
    parser.add_argument("--target-update-interval", type=int, default=None)
    parser.add_argument("--replay-buffer-size", type=int, default=None)

    args = parser.parse_args()

    try:
        preset = canonical_train_preset(args.algo)
    except Exception as exc:
        print(f"Failed to load benchmark preset: {exc}")
        sys.exit(1)

    total_timesteps = args.timesteps or int(preset["total_timesteps"])
    checkpoint_interval = args.checkpoint_interval or int(preset["checkpoint_interval_steps"])
    checkpoint_eval_episodes = args.checkpoint_eval_episodes or int(
        preset["checkpoint_eval_episodes"]
    )
    final_eval_episodes = args.final_eval_episodes or int(preset["final_eval_episodes"])

    train_dataset = args.train_dataset or Path(preset["train_dataset"])
    val_dataset = args.val_dataset or Path(preset["val_dataset"])
    holdout_dataset = args.holdout_dataset or Path(preset["holdout_dataset"])

    train_families = list(preset["train_families"])
    val_families = list(preset["val_families"])
    family_holdout_families = list(preset["family_holdout_families"])

    if args.train_family is not None:
        train_families = [_parse_family_spec(args.train_family)]
    if args.val_family is not None:
        val_families = [_parse_family_spec(args.val_family)]
    if args.family_holdout_family is not None:
        family_holdout_families = [_parse_family_spec(args.family_holdout_family)]

    train_records = load_records(train_dataset, expected_split="train")
    val_records = load_records(val_dataset, expected_split="val")
    holdout_records = load_records(holdout_dataset, expected_split="holdout")

    run_dir = args.artifact_root / args.run_label / args.algo / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    hyperparams = _resolve_hyperparameters(args, args.algo)

    train_env_kwargs = benchmark_env_kwargs(
        expected_split="train",
        scenario_parameter_records=train_records,
    )
    train_env_kwargs["scenario_families"] = train_families

    print("=" * 72)
    print(f"Wildfire benchmark training | algo={args.algo} | seed={args.seed}")
    print("=" * 72)
    print(f"run_label={args.run_label}")
    print(f"timesteps={total_timesteps:,}")
    print(f"checkpoint_interval={checkpoint_interval:,}")
    print(f"checkpoint_eval_episodes={checkpoint_eval_episodes}")
    print(f"final_eval_episodes={final_eval_episodes}")
    print(f"artifacts={run_dir}")

    try:
        train_env = _create_train_env(
            algo=args.algo,
            env_kwargs=train_env_kwargs,
            n_envs=args.envs,
            seed=args.seed,
        )
        model = _build_model(
            algo=args.algo,
            env=train_env,
            seed=args.seed,
            device=args.device,
            hyperparams=hyperparams,
        )
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Run: uv sync")
        sys.exit(1)

    checkpoint_splits = build_default_splits(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        holdout_dataset=holdout_dataset,
        include_family_holdout=args.include_family_holdout_checkpoints,
        include_temporal_holdout=False,
        train_families=train_families,
        val_families=val_families,
        family_holdout_families=family_holdout_families,
        temporal_holdout_families=train_families,
    )

    records_by_split = {
        "train": train_records,
        "val": val_records,
        "family_holdout": val_records,
        "temporal_holdout_diagnostic": holdout_records,
    }

    config_payload = {
        "algo": args.algo,
        "run_label": args.run_label,
        "seed": args.seed,
        "timesteps": total_timesteps,
        "n_envs": args.envs,
        "device": args.device,
        "datasets": {
            "train": str(train_dataset),
            "val": str(val_dataset),
            "holdout": str(holdout_dataset),
        },
        "record_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "holdout": len(holdout_records),
        },
        "families": {
            "train": _families_to_jsonable(train_families),
            "val": _families_to_jsonable(val_families),
            "family_holdout": _families_to_jsonable(family_holdout_families),
        },
        "checkpoint": {
            "interval_steps": checkpoint_interval,
            "episodes": checkpoint_eval_episodes,
            "visible_splits": [split.name for split in checkpoint_splits],
            "selection_metric": "val.asset_survival_rate",
            "tie_breaker": "val.mean_return",
            "temporal_holdout_visible": False,
        },
        "final_evaluation": {
            "episodes": final_eval_episodes,
            "include_family_holdout": args.include_family_holdout_final,
            "include_temporal_holdout_diagnostic": args.include_temporal_holdout_final,
            "compute_normalized_burn_ratio": not args.no_normalized_burn_final,
        },
        "hyperparameters": hyperparams,
    }
    _write_json(run_dir / "config.json", config_payload)

    checkpoint_entries: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None
    best_index: int | None = None

    while int(model.num_timesteps) < total_timesteps:
        remaining = total_timesteps - int(model.num_timesteps)
        learn_chunk = min(checkpoint_interval, remaining)
        model.learn(total_timesteps=learn_chunk, reset_num_timesteps=False, progress_bar=False)
        current_steps = int(model.num_timesteps)

        split_metrics = {}
        for split in checkpoint_splits:
            split_eval = evaluate_agent_on_split(
                agent_name=args.algo,
                model=model,
                records=records_by_split[split.name],
                expected_split=split.expected_split,
                scenario_families=split.scenario_families,
                seeds=[args.seed],
                episodes_per_seed=checkpoint_eval_episodes,
                compute_normalized_burn_ratio=False,
            )
            split_metrics[split.name] = _single_seed_split_summary(split_eval)

        entry = {
            "algo": args.algo,
            "seed": args.seed,
            "train_steps": current_steps,
            "selected_for_best": False,
            "splits": split_metrics,
        }
        checkpoint_entries.append(entry)

        if _selects_better_checkpoint(entry, best_entry):
            best_entry = entry
            best_index = len(checkpoint_entries) - 1
            model.save(str(run_dir / "best_model"))

        print(
            f"checkpoint step={current_steps:,} "
            f"val.asset_survival={entry['splits']['val']['asset_survival_rate']:.3f} "
            f"val.return={entry['splits']['val']['mean_return']:.2f}"
        )

    model.save(str(run_dir / "last_model"))

    if best_index is None:
        msg = "No checkpoint evaluations were produced; cannot select best checkpoint"
        raise RuntimeError(msg)

    checkpoint_entries[best_index]["selected_for_best"] = True
    best_entry = checkpoint_entries[best_index]

    _write_json(run_dir / "checkpoint_metrics.json", checkpoint_entries)
    _write_json(
        run_dir / "best_checkpoint.json",
        {
            "algo": args.algo,
            "seed": args.seed,
            "selected_train_steps": best_entry["train_steps"],
            "selection_metric": "val.asset_survival_rate",
            "tie_breaker": "val.mean_return",
            "val_metrics": best_entry["splits"]["val"],
            "best_checkpoint_entry": best_entry,
        },
    )

    best_model = load_model_for_algo(args.algo, run_dir / "best_model.zip")
    final_splits = build_default_splits(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        holdout_dataset=holdout_dataset,
        include_family_holdout=args.include_family_holdout_final,
        include_temporal_holdout=args.include_temporal_holdout_final,
        train_families=train_families,
        val_families=val_families,
        family_holdout_families=family_holdout_families,
        temporal_holdout_families=train_families,
    )

    final_split_metrics = {}
    for split in final_splits:
        final_eval = evaluate_agent_on_split(
            agent_name=args.algo,
            model=best_model,
            records=records_by_split[split.name],
            expected_split=split.expected_split,
            scenario_families=split.scenario_families,
            seeds=[args.seed],
            episodes_per_seed=final_eval_episodes,
            compute_normalized_burn_ratio=not args.no_normalized_burn_final,
        )
        final_split_metrics[split.name] = _single_seed_split_summary(final_eval)

    final_eval_payload: dict[str, Any] = {
        "algo": args.algo,
        "seed": args.seed,
        "model_artifact": str(run_dir / "best_model.zip"),
        "episodes_per_split": final_eval_episodes,
        "splits": final_split_metrics,
    }
    if args.include_temporal_holdout_final and len(holdout_records) <= 1:
        final_eval_payload["temporal_holdout_note"] = (
            "Temporal holdout contains one record and is reported as diagnostic-only evidence."
        )

    _write_json(run_dir / "final_eval_best.json", final_eval_payload)

    print("\nTraining complete")
    print(f"best checkpoint step={best_entry['train_steps']:,}")
    print(f"artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
