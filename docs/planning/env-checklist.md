# Environment Checklist

This checklist captures the remaining environment changes needed to align `src/models/fire_env.py` with the current benchmark design in `docs/envspec.md` and `docs/planning/impl-plan.md`.

---

## 1) Replace remaining heuristic-only environment defaults

- [ ] Remove the old assumption that severity alone determines spread through `SEVERITY_SPREAD_PROB` when a cached `base_spread_prob` record is available.
- [ ] Make the canonical path use static scenario parameter records first, with severity acting as reporting/observation metadata rather than the main spread heuristic.
- [ ] Keep severity heuristics only as a fallback dev mode, not as the benchmark-default path.
- [ ] Decide whether canonical benchmark mode should hard-fail if no cached parameter records are provided.

## 2) Make reset-time episode construction more dataset-driven

- [ ] Decide what belongs in the cached scenario record versus what remains randomized inside the simulator.
- [ ] If desired, extend cached records to include reset-time metadata such as ignition family, asset layout, and optional size/dryness tags.
- [ ] Stop sampling scenario families and parameter records independently if that can create inconsistent pairings.
- [ ] Replace the current severity-only record matching with a stronger record-selection rule tied to the frozen train/held-out split.

## 3) Freeze the canonical benchmark mode more strictly

- [ ] Add an explicit benchmark mode flag to `FireEnv` so train/eval runs cannot silently fall back to ad hoc random scenario generation.
- [ ] In benchmark mode, fail fast on missing or malformed parameter records.
- [ ] Keep legacy `base_spread_rate_m_per_min` support only for backward compatibility or remove it entirely once the static dataset path is stable.
- [ ] Ensure benchmark mode never depends on runtime live ingestion.

## 4) Align environment parameters with the static dataset builder

- [ ] Confirm the cached parameter schema used by `FireEnv` matches the output of `src/ingestion/static_dataset.py`.
- [ ] Use `base_spread_prob`, `wind_dir_deg`, and `wind_strength` directly from cached records in the canonical path.
- [ ] Keep extra builder fields such as `spread_rate_1h_m`, `spread_score`, `dryness_score`, and `record_quality_flag` for logging/debugging only unless promoted into the canonical env contract.
- [ ] Decide whether `severity_bucket` should be fully precomputed offline rather than inferred from old hard-coded spread heuristics.

## 5) Tighten the reward and transition accounting

- [ ] Check whether `new_burned` is currently measuring the intended quantity; it now tracks the change in burning cells rather than newly burned cells strictly.
- [ ] Verify the reward matches the frozen coefficients and intended semantics in `docs/envspec.md`.
- [ ] Confirm wasted-action logic matches the benchmark wording for blocked and zero-effect deployments.
- [ ] Confirm asset-loss accounting is correct when assets transition into burning cells.

## 6) Decide what remains randomized inside the simulator

- [ ] Keep ignition coordinates randomized within a frozen family if that is the intended benchmark design.
- [ ] Keep asset coordinates randomized within layout `A` and `B` if that is the intended benchmark design.
- [ ] If more reproducibility is needed, precompute reset seeds or exact placements in the cached scenario dataset.
- [ ] Document clearly that the benchmark is a fixed environment family with randomized episode instances, not one single fixed map.

## 7) Improve scenario-family integration

- [ ] Ensure train families and held-out families are sampled exactly as frozen in `docs/planning/impl-plan.md`.
- [ ] Prevent held-out family leakage during training.
- [ ] Consider storing a `split` field or family tag directly in cached scenario records.
- [ ] Confirm layout `A` and `B` generation in code really matches the written definitions.

## 8) Clean up legacy or transitional code paths

- [ ] Audit whether `random_scenario()` should remain part of the canonical benchmark path or only support smoke tests and ablations.
- [ ] Remove or isolate older spread-rate override code once the static parameter dataset path is fully working.
- [ ] Clarify whether `ScenarioConfig` should remain the main reset object or become a thin wrapper over cached parameter records.
- [ ] Remove comments or naming that still imply the older XGBoost-centered flow.

## 9) Add validation and tests

- [ ] Add tests for loading cached scenario parameter records.
- [ ] Add tests that benchmark-mode reset uses cached parameters and does not fall back silently.
- [ ] Add tests that observation shape remains stable at `636` unless the observation contract intentionally changes.
- [ ] Add tests that severity one-hot and wind bias in the observation match the active cached parameter record.
- [ ] Add tests that train/held-out family filtering works as intended.

## 10) Nice-to-have improvements after canonical alignment

- [ ] Add richer info logging so each episode returns the active `record_id` and scenario family tags.
- [ ] Add optional per-record diagnostics for spread calibration sanity checks.
- [ ] Consider adding a cached `ignition_seed` or `layout_seed` field for exact episode replay.
- [ ] Add a dedicated benchmark env factory that always builds the environment from frozen cached records.

---

## Suggested order

1. Make cached parameter records the canonical reset path.
2. Remove silent fallback behavior in benchmark mode.
3. Align reward/transition accounting with the written spec.
4. Freeze family sampling and held-out split handling.
5. Add tests around cached-record loading and reset behavior.
