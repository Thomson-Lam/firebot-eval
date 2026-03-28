# Environment Checklist (Incremental Pipeline Alignment)

This checklist replaces the previous one and focuses on incremental updates to the currently implemented environment setup in:

- `src/models/fire_env.py`
- `src/models/train_rl_agent.py`
- `src/models/evaluate_agents.py`

It is aligned to the current static data pipeline outputs in `src/ingestion/static_dataset.py`, including new record fields and split-aware datasets.

---

## 1) Make frozen scenario records the canonical setup path

- [x] Add an explicit benchmark/frozen mode flag in `WildfireEnv`.
- [x] In benchmark mode, require `scenario_parameter_records` and fail fast if missing or empty.
- [x] Remove silent fallback to `random_scenario()` in benchmark mode.
- [x] Keep heuristic/random scenario generation as explicit dev/ablation mode only.
- [x] Keep or remove `base_spread_rate_m_per_min` legacy path intentionally; do not let it remain an implicit canonical fallback.

## 2) Enforce schema validation for ingested scenario records

- [x] Validate required fields on load: `record_id`, `split`, `base_spread_prob`, `severity_bucket`, `wind_dir_deg`, `wind_strength`.
- [x] Validate value domains (severity enum, numeric ranges, finite floats).
- [x] Fail with actionable errors in benchmark mode; warn-and-skip in dev mode if needed.
- [x] Decide whether to hard-reject records with missing/invalid `split` even when dataset files are split-specific (decision: yes in benchmark mode).

## 3) Replace severity-only matching in reset-time record selection

- [x] Stop selecting cached records by `severity_bucket` only.
- [x] Use a deterministic or seed-stable record sampling strategy from the provided split dataset.
- [x] Keep `severity_bucket` as observation/reporting metadata, not as the primary selector.
- [x] Ensure sampled `record_id` is tracked for traceability each episode.

## 4) Align environment variable ingestion with pipeline contract

- [ ] Confirm canonical runtime uses cached `base_spread_prob`, `wind_dir_deg`, `wind_strength`, and `severity_bucket` directly.
- [ ] Keep builder audit fields (`spread_rate_1h_m`, `spread_score`, `weather_score`, `cffdrs_dryness_score`, `size_factor`, `fire_type_factor`, `fuel_factor`, `rain_factor`, `record_quality_flag`) as logging/debug fields unless explicitly promoted.
- [ ] Surface selected metadata in episode info for debugging (at minimum `record_id`, `split`; optionally `fire_id`, `year`, `source`, `province`, `record_quality_flag`).
- [ ] Ensure optional CFFDRS-derived fields do not trigger runtime network dependencies.

## 5) Tighten train/val/holdout setup semantics

- [ ] Standardize env construction so training uses train records, validation uses val records, holdout uses holdout records.
- [ ] Add internal guardrails to prevent accidental split mixing/leakage.
- [ ] Decide whether to trust dataset filenames for split identity, record `split` values, or both.
- [ ] Add lightweight split-consistency checks at environment creation time.

## 6) Remove stale live-ingestion assumptions from env setup

- [ ] Ensure canonical env setup does not assume FIRMS or CWFIS live-fire ingestion.
- [ ] Ensure canonical env setup does not depend on Open-Meteo or runtime CFFDRS fetches.
- [ ] Treat all spread/weather features as precomputed offline inputs.

## 7) Add targeted tests for the new setup contract

- [ ] Add tests for scenario-record schema validation.
- [ ] Add tests that benchmark mode fails fast instead of silently falling back.
- [ ] Add tests that active scenario parameters in env match the selected cached record.
- [ ] Add tests for split isolation between train/val/holdout datasets.
- [ ] Add tests that reset/step info includes `record_id` and split metadata.

## 8) Cleanup and naming consistency

- [ ] Remove or rename comments/identifiers that imply heuristic-first or live-data-first canonical behavior.
- [ ] Keep dev/ablation paths clearly separated from benchmark/frozen paths.
- [ ] Add a single benchmark env factory/helper to centralize canonical env creation.

## 9) Final items (defer until core alignment is complete)

- [ ] Decide whether ignition controls should move into cached dataset records.
- [ ] Decide whether asset layout controls should move into cached dataset records.
- [ ] If moved, define new dataset fields and update reset logic accordingly.
- [ ] If not moved, explicitly document that ignition/layout remain simulator-side randomized controls.
- [ ] Consider optional `ignition_seed` / `layout_seed` for exact episode replay once benchmark mode is stable.

---

## Suggested incremental order

1. Benchmark mode + fail-fast behavior.
2. Schema validation for scenario records.
3. Replace severity-only selection.
4. Split guardrails and setup consistency.
5. Tests for no-fallback + split isolation + record traceability.
6. Defer ignition/layout dataset migration decisions to final items.
