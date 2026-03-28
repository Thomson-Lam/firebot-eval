# Environment Changelog (Pipeline Alignment)

This document records what was implemented for environment alignment with the frozen static data pipeline.

Primary implementation files:
- `src/models/fire_env.py`
- `src/models/train_rl_agent.py`
- `src/models/evaluate_agents.py`
- `tests/models/test_fire_env_setup_contract.py`

---

## 1) Frozen scenario records became the canonical setup path

- [x] Add an explicit benchmark/frozen mode flag in `WildfireEnv`.
  - Implemented in `WildfireEnv.__init__` via `benchmark_mode` (`src/models/fire_env.py`).
  - Benchmark mode is now the canonical constructor path for train/eval.
- [x] In benchmark mode, require `scenario_parameter_records` and fail fast if missing or empty.
  - Added hard checks in `WildfireEnv.__init__` that raise `ValueError` when records are missing.
  - Added additional reset-time hard guard for impossible benchmark state.
- [x] Remove silent fallback to `random_scenario()` in benchmark mode.
  - `WildfireEnv.reset` now raises in benchmark mode if no cached records are available.
  - Silent fallback to `random_scenario()` remains available only when `benchmark_mode=False`.
- [x] Keep heuristic/random scenario generation as explicit dev/ablation mode only.
  - Dev path kept through `random_scenario()` and non-benchmark env setup.
  - Error messages explicitly state benchmark mode cannot use dev fallback paths.
- [x] Keep or remove `base_spread_rate_m_per_min` legacy path intentionally.
  - Legacy spread-rate path retained but explicitly dev-only.
  - Benchmark mode rejects `base_spread_rate_m_per_min`.
  - Training script now requires explicit `--allow-legacy-dev-fallback` for this path.

Changed functions/files:
- `WildfireEnv.__init__`, `WildfireEnv.reset`, `random_scenario` in `src/models/fire_env.py`
- `train` and CLI args in `src/models/train_rl_agent.py`
- benchmark env creation usage in `src/models/evaluate_agents.py`

---

## 2) Scenario-record schema validation was enforced

- [x] Validate required fields on load (`record_id`, `split`, `base_spread_prob`, `severity_bucket`, `wind_direction`, `wind_strength`, `ignition_seed`, `layout_seed` in benchmark mode).
  - Added required-field checks in `load_scenario_parameter_records`.
  - Normalizes validated records (typed numeric fields, lowercase enum/split).
- [x] Validate value domains (severity enum, numeric ranges, finite floats).
  - Added severity and split enum checks.
  - Added finite/range checks for spread and wind numeric values.
  - Added optional seed validation for `ignition_seed` and `layout_seed`.
- [x] Fail in benchmark mode; warn-and-skip in dev mode.
  - Benchmark mode raises actionable `ValueError` with sampled invalid rows.
  - Dev mode logs warnings and skips invalid records.
- [x] Hard-reject missing/invalid `split` in benchmark mode.
  - Decision implemented at loader and env-construction levels.

Changed functions/files:
- `load_scenario_parameter_records` in `src/models/fire_env.py`
- benchmark load call sites in `src/models/train_rl_agent.py` and `src/models/evaluate_agents.py`

---

## 3) Reset-time selection moved from severity matching to record sampling

- [x] Stop selecting cached records by `severity_bucket` only.
  - Removed severity-only filtering from `WildfireEnv.reset`.
  - Reset now samples from full validated records.
- [x] Use deterministic/seed-stable record sampling.
  - Added `_sample_parameter_record` with shuffled index order and cursor.
  - Sampling uses env RNG and is reshuffled on seed-driven resets.
- [x] Keep `severity_bucket` as metadata, not primary selector.
  - `severity_bucket` is consumed from selected record in `scenario_from_parameter_record`.
  - It now affects scenario state only through the selected record.
- [x] Track sampled `record_id` per episode.
  - Added `_active_record_id` and included `record_id` in reset and step `info`.

Changed functions/files:
- `WildfireEnv.reset`, `_sample_parameter_record`, `scenario_from_parameter_record` in `src/models/fire_env.py`

---

## 4) Environment variable ingestion aligned with pipeline contract

- [x] Canonical runtime uses cached `base_spread_prob`, `wind_direction` (8-direction string), `wind_strength`, and `severity_bucket` directly.
  - `scenario_from_parameter_record` now maps required cached fields directly into `ScenarioConfig`.
  - Removed permissive fallback defaults for canonical record mapping.
- [x] Keep audit fields as logging/debug unless promoted.
  - Added `PARAMETER_AUDIT_FIELDS` and exposed them via `parameter_audit` in `info`.
  - Audit fields are not used in transition dynamics.
- [x] Surface selected metadata in episode info.
  - Added `PARAMETER_METADATA_FIELDS` and surfaced `parameter_record_meta`.
  - `record_id` and `split` are included directly in reset and step `info`.
- [x] Ensure optional CFFDRS fields do not imply runtime fetches.
  - Benchmark runtime only consumes cached records.
  - Added explicit module-level note that benchmark env does not fetch FIRMS/CWFIS/Open-Meteo/CFFDRS at runtime.

Changed functions/files:
- `scenario_from_parameter_record`, `WildfireEnv.reset`, `WildfireEnv.step`, `_parameter_metadata`, `_parameter_audit` in `src/models/fire_env.py`

---

## 5) Train/val/holdout setup semantics were tightened

- [x] Standardize env construction by split.
  - Train path uses expected split `train`.
  - Eval path uses expected split `train`/`val`/`holdout` per dataset.
- [x] Add internal guardrails against split mixing.
  - Added split checks in loader and `WildfireEnv.__init__`.
  - Benchmark mode rejects mixed-split records unless a single split is enforced.
- [x] Decision: trust both dataset filenames and record `split` values.
  - Added `_split_hint_from_path` and cross-check logic against `expected_split` and record payload.
- [x] Add split-consistency checks at env creation.
  - Added `expected_split` to env constructor.
  - Constructor validates record splits and fails fast in benchmark mode.
  - Canonical train/eval defaults now use seeded split artifacts (`scenario_parameter_records_seeded_{split}.json`).

Changed functions/files:
- `_split_hint_from_path`, `load_scenario_parameter_records`, `WildfireEnv.__init__` in `src/models/fire_env.py`
- `_load_split_records`, `_evaluate_agent_on_split` in `src/models/evaluate_agents.py`
- train/eval load paths in `src/models/train_rl_agent.py`

---

## 6) Stale live-ingestion assumptions were removed from env setup

- [x] Canonical env no longer assumes FIRMS/CWFIS live-fire ingestion.
  - Clarified benchmark runtime behavior in `fire_env` module docstring.
- [x] Canonical env no longer depends on Open-Meteo/runtime CFFDRS fetches.
  - Benchmark runtime now strictly requires frozen records unless explicit dev fallback is enabled.
- [x] Spread/weather features treated as precomputed offline inputs.
  - Training canonical path requires scenario dataset.
  - Legacy spread-rate path requires explicit `--allow-legacy-dev-fallback`.

Changed functions/files:
- module docs and benchmark checks in `src/models/fire_env.py`
- fallback gating in `src/models/train_rl_agent.py`

---

## 7) Targeted tests were added for the setup contract

- [x] Add tests for schema validation.
  - Added tests for missing required fields and invalid numeric ranges.
- [x] Add tests for benchmark fail-fast behavior.
  - Added test ensuring env creation fails in benchmark mode without records.
- [x] Add tests that active scenario parameters match selected record.
  - Added test asserting severity/wind/spread values match cached record.
- [x] Add tests for split isolation.
  - Added tests for loader expected split mismatch, filename hint mismatch, and env split mismatch.
- [x] Add tests for `record_id` and split metadata in `info`.
  - Added reset/step info assertions for `record_id`, `split`, metadata, and audit payloads.

Changed functions/files:
- New suite in `tests/models/test_fire_env_setup_contract.py`
- Test path bootstrap in `tests/conftest.py`

---

## 8) Cleanup and naming consistency were completed

- [x] Remove/rename heuristic-first identifiers.
  - Renamed `SEVERITY_SPREAD_PROB` to `LEGACY_SEVERITY_SPREAD_PROB`.
  - Updated comments/docstrings to mark dev/ablation semantics.
- [x] Keep dev and benchmark paths clearly separated.
  - Benchmark mode stays strict.
  - Dev fallback remains explicit and opt-in in training.
- [x] Add a single benchmark env factory/helper.
  - Added `benchmark_env_kwargs` and `create_benchmark_env`.
  - Updated train/eval code paths to use this centralized helper.

Changed functions/files:
- constants and helpers in `src/models/fire_env.py`
- helper adoption in `src/models/train_rl_agent.py` and `src/models/evaluate_agents.py`

---

## 9) Final decision: keep ignition/layout simulator-side, add replay seeds

- [x] Decide on ignition controls in dataset.
  - Decision: do not move ignition controls into cached dataset schema now.
- [x] Decide on asset layout controls in dataset.
  - Decision: do not move asset layout controls into cached dataset schema now.
- [x] If moved, define dataset fields and reset logic.
  - Not applied in this phase due to ROI decision.
- [x] Document simulator-side ignition/layout controls.
  - Added reset-time comment clarifying that ignition/layout remain simulator-side.
- [x] Add optional `ignition_seed` / `layout_seed` for replayability.
  - Added seeded parameter artifact generation in pipeline (`scenario_parameter_records_seeded*.json`).
  - Benchmark mode now requires per-record `ignition_seed` and `layout_seed`.
  - Added `_stable_seed` deterministic fallback from `record_id + reset_seed` for non-benchmark/dev compatibility.
  - Added `_configure_initialization_rngs` and separate ignition/layout RNGs.
  - `_ignite` and `_place_assets` now use those RNGs.
  - Seeds are surfaced in reset and step `info`.
  - Seeded holdout artifact is intentionally reduced to a single unique held-out record for now.

Changed functions/files:
- `_stable_seed`, `_configure_initialization_rngs`, `_ignite`, `_place_assets`, loader validation, metadata fields in `src/models/fire_env.py`
- Replayability tests in `tests/models/test_fire_env_setup_contract.py`

---

## Verification status

- `uv run python -m py_compile src/models/fire_env.py src/models/train_rl_agent.py src/models/evaluate_agents.py`
- `uv run pytest tests/models/test_fire_env_setup_contract.py`

Current targeted setup-contract test status: passing.
