from __future__ import annotations

from pathlib import Path

from src.models.benchmarking import (
    CANONICAL_CHECKPOINT_EVAL_EPISODES,
    CANONICAL_CHECKPOINT_INTERVAL_STEPS,
    CANONICAL_FINAL_EVAL_EPISODES,
    CANONICAL_TIMESTEPS_BY_ALGO,
    aggregate_seed_summaries,
    canonical_train_preset,
    evaluate_agent_on_split,
    summarize_episodes,
)


def _record(**overrides):
    record = {
        "record_id": "AB-2020-001__20200101",
        "split": "train",
        "base_spread_prob": 0.14,
        "severity_bucket": "medium",
        "wind_direction": "E",
        "wind_strength": 0.35,
        "ignition_seed": 101,
        "layout_seed": 202,
    }
    record.update(overrides)
    return record


def test_canonical_train_preset_matches_frozen_protocol():
    preset = canonical_train_preset("ppo")

    assert preset["total_timesteps"] == CANONICAL_TIMESTEPS_BY_ALGO["ppo"]
    assert preset["checkpoint_interval_steps"] == CANONICAL_CHECKPOINT_INTERVAL_STEPS
    assert preset["checkpoint_eval_episodes"] == CANONICAL_CHECKPOINT_EVAL_EPISODES
    assert preset["final_eval_episodes"] == CANONICAL_FINAL_EVAL_EPISODES
    assert Path(preset["train_dataset"]).name == "scenario_parameter_records_seeded_train.json"
    assert preset["holdout_checkpoint_visibility"] is False


def test_summarize_episodes_handles_absent_containment_times():
    summary = summarize_episodes(
        [
            {
                "return": 10.0,
                "assets_lost": 1,
                "containment_success": 0.0,
                "burned_area_fraction": 0.20,
                "time_to_containment": None,
                "resource_efficiency": 0.0,
                "wasted_deployment_rate": 1.0,
                "final_burned_area_cells": 125,
            },
            {
                "return": -5.0,
                "assets_lost": 2,
                "containment_success": 0.0,
                "burned_area_fraction": 0.25,
                "time_to_containment": None,
                "resource_efficiency": 0.5,
                "wasted_deployment_rate": 0.25,
                "final_burned_area_cells": 150,
            },
        ]
    )

    assert summary["episodes"] == 2
    assert summary["mean_return"] == 2.5
    assert summary["containment_success_rate"] == 0.0
    assert summary["mean_time_to_containment"] is None
    assert summary["mean_resource_efficiency"] == 0.25


def test_evaluate_agent_on_split_reports_seed_and_aggregate_schema():
    result = evaluate_agent_on_split(
        agent_name="random",
        model=None,
        records=[_record()],
        expected_split="train",
        scenario_families=[("center", "medium", "A")],
        seeds=[11, 22],
        episodes_per_seed=2,
        compute_normalized_burn_ratio=True,
    )

    assert len(result["seed_metrics"]) == 2
    assert "aggregate" in result
    aggregate = result["aggregate"]
    assert aggregate["episodes_per_seed"] == 2
    assert aggregate["num_seeds"] == 2
    assert "mean_return" in aggregate
    assert "asset_survival_rate" in aggregate
    assert "mean_burned_area_fraction" in aggregate
    assert "mean_normalized_burn_ratio" in aggregate
    assert "std_across_seeds" in aggregate
    assert "mean_return" in aggregate["std_across_seeds"]


def test_aggregate_seed_summaries_handles_optional_none_metrics():
    aggregate = aggregate_seed_summaries(
        [
            {
                "seed": 1,
                "episodes": 3,
                "mean_return": 10.0,
                "asset_survival_rate": 0.5,
                "containment_success_rate": 0.25,
                "mean_burned_area_fraction": 0.3,
                "mean_time_to_containment": None,
                "mean_resource_efficiency": 0.2,
                "wasted_deployment_rate": 0.7,
            },
            {
                "seed": 2,
                "episodes": 3,
                "mean_return": 20.0,
                "asset_survival_rate": 0.75,
                "containment_success_rate": 0.5,
                "mean_burned_area_fraction": 0.1,
                "mean_time_to_containment": None,
                "mean_resource_efficiency": 0.4,
                "wasted_deployment_rate": 0.6,
            },
        ]
    )

    assert aggregate["mean_return"] == 15.0
    assert aggregate["std_across_seeds"]["mean_return"] == 5.0
    assert aggregate["mean_time_to_containment"] is None
    assert aggregate["std_across_seeds"]["mean_time_to_containment"] is None
