from __future__ import annotations

import json

import pytest

from src.models.fire_env import WildfireEnv, load_scenario_parameter_records


def _record(**overrides):
    base = {
        "record_id": "AB-2020-001__20200101",
        "split": "train",
        "base_spread_prob": 0.14,
        "severity_bucket": "medium",
        "wind_dir_deg": 90.0,
        "wind_strength": 0.35,
        "fire_id": "AB-2020-001",
        "year": 2020,
        "source": "AB_HISTORICAL_WILDFIRE",
        "province": "AB",
        "record_quality_flag": "measured",
        "spread_rate_1h_m": 120.0,
        "spread_score": 0.51,
        "weather_score": 0.42,
        "cffdrs_dryness_score": 0.3,
        "size_factor": 1.02,
        "fire_type_factor": 1.0,
        "fuel_factor": 1.08,
        "rain_factor": 0.96,
    }
    base.update(overrides)
    return base


def _write_records(path, records):
    path.write_text(json.dumps({"records": records}))


def test_schema_validation_requires_core_fields(tmp_path):
    path = tmp_path / "records.json"
    _write_records(path, [_record(record_id=None)])

    with pytest.raises(ValueError, match="missing required fields"):
        load_scenario_parameter_records(path, benchmark_mode=True, expected_split="train")


def test_schema_validation_checks_domains_and_ranges(tmp_path):
    path = tmp_path / "records.json"
    _write_records(path, [_record(wind_strength=1.5)])

    with pytest.raises(ValueError, match="wind_strength"):
        load_scenario_parameter_records(path, benchmark_mode=True, expected_split="train")


def test_dev_mode_warns_and_skips_invalid_records(tmp_path, caplog):
    path = tmp_path / "records.json"
    _write_records(path, [_record(), _record(record_id="bad", split="unexpected")])

    records = load_scenario_parameter_records(path, benchmark_mode=False, expected_split="train")

    assert len(records) == 1
    assert records[0]["record_id"] == "AB-2020-001__20200101"
    assert "invalid record" in caplog.text.lower() or "split-mismatched" in caplog.text.lower()


def test_benchmark_mode_requires_records_on_env_creation():
    with pytest.raises(ValueError, match="requires non-empty scenario_parameter_records"):
        WildfireEnv(scenario_parameter_records=[], benchmark_mode=True)


def test_benchmark_mode_reset_keeps_record_driven_path():
    env = WildfireEnv(
        scenario_parameter_records=[_record(record_id="AB-2020-keep")],
        benchmark_mode=True,
        expected_split="train",
    )

    _obs, info = env.reset(seed=11)

    assert info["parameter_record"] is not None
    assert info["record_id"] == "AB-2020-keep"


def test_active_scenario_uses_cached_parameter_values():
    record = _record(
        severity_bucket="high",
        wind_dir_deg=123.0,
        wind_strength=0.57,
        base_spread_prob=0.2,
    )
    env = WildfireEnv(
        scenario_parameter_records=[record],
        scenario_families=[("center", "medium", "A")],
        benchmark_mode=True,
        expected_split="train",
    )

    env.reset(seed=7)

    assert env.scenario.severity == "high"
    assert env.scenario.wind_dir_deg == pytest.approx(123.0)
    assert env.scenario.wind_strength == pytest.approx(0.57)
    assert env.scenario.spread_prob == pytest.approx(0.2)


def test_split_isolation_on_loader_expected_split(tmp_path):
    path = tmp_path / "records.json"
    _write_records(path, [_record(split="val")])

    with pytest.raises(ValueError, match="Split consistency check failed"):
        load_scenario_parameter_records(path, benchmark_mode=True, expected_split="train")


def test_split_isolation_from_filename_hint(tmp_path):
    path = tmp_path / "scenario_parameter_records_train.json"
    _write_records(path, [_record(split="val")])

    with pytest.raises(ValueError, match="Split consistency check failed"):
        load_scenario_parameter_records(path, benchmark_mode=True)


def test_split_isolation_on_env_creation_expected_split_mismatch():
    with pytest.raises(ValueError, match="expected split 'train'"):
        WildfireEnv(
            scenario_parameter_records=[_record(split="val")],
            benchmark_mode=True,
            expected_split="train",
        )


def test_reset_and_step_info_include_record_and_split_metadata():
    record = _record(
        record_id="AB-2020-info",
        split="train",
        fire_id="AB-2020-777",
        year=2020,
        source="AB_HISTORICAL_WILDFIRE",
        province="AB",
        record_quality_flag="measured",
    )
    env = WildfireEnv(
        scenario_parameter_records=[record],
        scenario_families=[("center", "medium", "A")],
        benchmark_mode=True,
        expected_split="train",
    )

    _obs, reset_info = env.reset(seed=21)
    _obs, _reward, _done, _trunc, step_info = env.step(0)

    assert reset_info["record_id"] == "AB-2020-info"
    assert reset_info["split"] == "train"
    assert reset_info["ignition_seed"] is not None
    assert reset_info["layout_seed"] is not None
    assert reset_info["parameter_record_meta"]["fire_id"] == "AB-2020-777"
    assert "spread_score" in reset_info["parameter_audit"]

    assert step_info["record_id"] == "AB-2020-info"
    assert step_info["split"] == "train"
    assert step_info["ignition_seed"] == reset_info["ignition_seed"]
    assert step_info["layout_seed"] == reset_info["layout_seed"]
    assert step_info["parameter_record_meta"]["record_quality_flag"] == "measured"
    assert "cffdrs_dryness_score" in step_info["parameter_audit"]


def test_record_provided_initialization_seeds_make_spatial_setup_replayable():
    record = _record(record_id="AB-2020-seeded", ignition_seed=12345, layout_seed=54321)
    env = WildfireEnv(
        scenario_parameter_records=[record],
        scenario_families=[("center", "medium", "A")],
        benchmark_mode=True,
        expected_split="train",
    )

    env.reset(seed=1)
    grid_a = env.grid.copy()
    first_info_seed_pair = (env._ignition_seed_used, env._layout_seed_used)

    env.reset(seed=999)
    grid_b = env.grid.copy()
    second_info_seed_pair = (env._ignition_seed_used, env._layout_seed_used)

    assert first_info_seed_pair == (12345, 54321)
    assert second_info_seed_pair == (12345, 54321)
    assert (grid_a == grid_b).all()
