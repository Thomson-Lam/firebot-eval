"""Unified benchmark evaluation runner for learned and heuristic agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models.benchmarking import (
    RUN_LABELS,
    build_default_splits,
    canonical_eval_preset,
    evaluate_agent_on_split,
    heldout_performance_drop,
    load_model_for_algo,
    load_records,
)

SUPPORTED_AGENTS = ("ppo", "a2c", "dqn", "greedy", "random")
LEARNED_AGENTS = {"ppo", "a2c", "dqn"}


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one seed must be provided")
    return values


def _parse_agents(raw: str) -> list[str]:
    agents = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not agents:
        raise ValueError("At least one agent must be provided")
    invalid = [agent for agent in agents if agent not in SUPPORTED_AGENTS]
    if invalid:
        msg = f"Unsupported agents {invalid}; expected any of {sorted(SUPPORTED_AGENTS)}"
        raise ValueError(msg)
    return agents


def _resolve_model_paths(args, agents: list[str]) -> dict[str, Path]:
    learned_agents = [agent for agent in agents if agent in LEARNED_AGENTS]
    model_paths: dict[str, Path] = {}

    for agent in learned_agents:
        path = getattr(args, f"{agent}_model")
        if path is not None:
            model_paths[agent] = path

    if args.model_path is not None and len(learned_agents) == 1:
        model_paths[learned_agents[0]] = args.model_path

    missing = [agent for agent in learned_agents if agent not in model_paths]
    if missing:
        msg = (
            "Missing model paths for learned agents "
            f"{missing}. Provide --model-path (single learned agent only) or --ppo-model/--a2c-model/--dqn-model."
        )
        raise ValueError(msg)

    return model_paths


def _compute_performance_drops(agent_result: dict[str, Any]) -> dict[str, float]:
    drops: dict[str, float] = {}
    train_summary = agent_result.get("train", {}).get("aggregate", {})
    train_asset_survival = train_summary.get("asset_survival_rate")
    if train_asset_survival is None:
        return drops

    for split in ("val", "family_holdout", "temporal_holdout_diagnostic"):
        split_summary = agent_result.get(split, {}).get("aggregate", {})
        heldout_value = split_summary.get("asset_survival_rate")
        if heldout_value is None:
            continue
        drops[f"{split}_asset_survival_drop"] = heldout_performance_drop(
            float(train_asset_survival),
            float(heldout_value),
        )
    return drops


def main() -> None:
    preset = canonical_eval_preset()

    parser = argparse.ArgumentParser(
        description="Evaluate benchmark agents on canonical wildfire splits"
    )
    parser.add_argument(
        "--benchmark-preset",
        type=str,
        default="canonical",
        choices=("canonical",),
        help="Benchmark-safe eval preset",
    )
    parser.add_argument("--agents", type=str, default="ppo,a2c,dqn,greedy,random")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--ppo-model", type=Path, default=None)
    parser.add_argument("--a2c-model", type=Path, default=None)
    parser.add_argument("--dqn-model", type=Path, default=None)
    parser.add_argument("--train-dataset", type=Path, default=preset["train_dataset"])
    parser.add_argument("--val-dataset", type=Path, default=preset["val_dataset"])
    parser.add_argument("--holdout-dataset", type=Path, default=preset["holdout_dataset"])
    parser.add_argument(
        "--episodes",
        type=int,
        default=preset["episodes"],
        help="Episodes per seed per split",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in preset["seeds"]),
        help="Comma-separated seed list",
    )
    parser.add_argument(
        "--include-family-holdout",
        action="store_true",
        help="Evaluate validation records with HELD_OUT_FAMILIES",
    )
    parser.add_argument(
        "--include-temporal-holdout",
        action="store_true",
        help="Evaluate temporal holdout as diagnostic output",
    )
    parser.add_argument("--no-normalized-burn", action="store_true")
    parser.add_argument(
        "--run-label",
        type=str,
        default="final",
        choices=RUN_LABELS,
        help="Run label attached to output metadata",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    del args.benchmark_preset  # canonical is currently the only supported preset

    seeds = _parse_csv_ints(args.seeds)
    agents = _parse_agents(args.agents)
    model_paths = _resolve_model_paths(args, agents)

    split_configs = build_default_splits(
        train_dataset=args.train_dataset,
        val_dataset=args.val_dataset,
        holdout_dataset=args.holdout_dataset,
        include_family_holdout=args.include_family_holdout,
        include_temporal_holdout=args.include_temporal_holdout,
        train_families=preset["train_families"],
        val_families=preset["val_families"],
        family_holdout_families=preset["family_holdout_families"],
        temporal_holdout_families=preset["temporal_holdout_families"],
    )

    split_records = {
        split.name: load_records(split.dataset_path, expected_split=split.expected_split)
        for split in split_configs
    }

    results: dict[str, Any] = {
        "config": {
            "run_label": args.run_label,
            "agents": agents,
            "seeds": seeds,
            "episodes_per_seed": args.episodes,
            "compute_normalized_burn_ratio": not args.no_normalized_burn,
            "splits": [split.name for split in split_configs],
            "datasets": {
                "train": str(args.train_dataset),
                "val": str(args.val_dataset),
                "holdout": str(args.holdout_dataset),
            },
            "model_paths": {name: str(path) for name, path in model_paths.items()},
        },
        "results": {},
    }

    loaded_models = {agent: load_model_for_algo(agent, path) for agent, path in model_paths.items()}

    for agent in agents:
        agent_result: dict[str, Any] = {}
        model = loaded_models.get(agent)
        for split in split_configs:
            split_eval = evaluate_agent_on_split(
                agent_name=agent,
                model=model,
                records=split_records[split.name],
                expected_split=split.expected_split,
                scenario_families=split.scenario_families,
                seeds=seeds,
                episodes_per_seed=args.episodes,
                compute_normalized_burn_ratio=not args.no_normalized_burn,
            )
            agent_result[split.name] = split_eval
        agent_result["heldout_performance_drop"] = _compute_performance_drops(agent_result)
        results["results"][agent] = agent_result

    print("\nBenchmark Summary")
    print("=" * 92)
    for agent, split_payload in results["results"].items():
        for split in split_configs:
            aggregate = split_payload[split.name]["aggregate"]
            stds = aggregate["std_across_seeds"]
            print(
                f"{agent:>8} | {split.name:<26} "
                f"| return={aggregate['mean_return']:.2f}±{stds['mean_return']:.2f} "
                f"| asset_survival={aggregate['asset_survival_rate']:.3f}±{stds['asset_survival_rate']:.3f} "
                f"| containment={aggregate['containment_success_rate']:.3f}±{stds['containment_success_rate']:.3f} "
                f"| burned_frac={aggregate['mean_burned_area_fraction']:.3f}±{stds['mean_burned_area_fraction']:.3f}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
