# DRD Canonical Alignment Backlog

This backlog lists the key changes required to align implementation with `impl-plan.md` (canonical frozen plan).

## P0 - Must Fix Before Any Canonical Claims

- [ ] **Enforce benchmark invariant: only objective priority shifts across regimes**
  - Keep reward component formulas semantically identical across regimes.
  - Move regime dependence to weighting profiles only (not component definitions).
  - Update `src/env.py` so `r_progress`, `r_safety`, `r_efficiency` are each defined consistently in A/B.

- [ ] **Replace fixed every-K switching with stochastic variable-duration switching**
  - Remove deterministic `regime_switch_interval` as the core train/eval mechanism.
  - Implement stochastic switch timing / hazard-rate or sampled dwell times.
  - Keep fixed schedule only behind an explicit debug/sanity flag.

- [ ] **Reset recurrent hidden states at episode boundaries**
  - In rollout collection, reset GRU hidden state on `done`.
  - Never carry hidden states across episodes or env instances.

- [ ] **Switch DRD training to concurrent end-to-end**
  - Remove pretrain-then-co-train as default method schedule.
  - Keep pretraining only as optional debug scaffold.
  - Ensure final reported DRD results come from concurrent training only.

- [ ] **Remove privileged regime supervision from method training**
  - Remove/disable regime-classification and directed losses that consume true regime labels.
  - Keep true regime for logging/evaluation only.

- [ ] **Implement canonical critic architecture: K per-component value heads**
  - Replace scalar value head with one head per sub-reward component.
  - Compute PPO scalar targets/advantages via weighted combination when needed.

- [ ] **Correct baseline reward training objectives**
  - Scalar PPO and recurrent-scalar PPO must optimize environment scalar reward directly.
  - Only decomposition-based methods should rely on decomposed/internal effective reward pathways.

## P1 - High Priority for Fairness and Attribution

- [ ] **Implement missing canonical baselines**
  - Add scalar PPO (no recurrence, no decomposition).
  - Add recurrent PPO (memory only, scalar reward).
  - Keep recurrent static decomposition baseline matching DRD backbone except dynamic weights.

- [ ] **Align weight network with canonical form**
  - Implement `w_hat_t = softmax(MLP(h_t))` as primary architecture.
  - Keep current hierarchical-sigmoid design only as an ablation.

- [ ] **Include `done_{t-1}` in GRU input tuple**
  - Update GRU input construction and config dimensions to `(o_t, a_{t-1}, r_{t-1}^{vec}, done_{t-1})`.

- [ ] **Fix oracle baseline definition**
  - Implement true oracle regime-access agent (policy gets true regime or true weights).
  - Use it as an upper-bound ceiling baseline.

- [ ] **Add rollout-edge bootstrap value for GAE**
  - Replace hardcoded `last_val = 0` with bootstrap from value function when rollout ends mid-episode.

## P2 - Evaluation and Reporting Completeness

- [ ] **Wire a real evaluation pipeline into train/CLI**
  - Run periodic evaluation episodes from training loop.
  - Ensure `src/evaluate.py` uses correct DRD policy inputs (`state + context`).

- [ ] **Implement canonical core metrics**
  - Average episodic return.
  - Post-switch performance drop.
  - Recovery time.
  - Variance across seeds.
  - Adaptation lag + similarity of `w_hat_t` to true regime weighting.

- [ ] **Run multi-seed protocol consistently across methods**
  - Same environment generator/schedule family, steps, seeds, and compute budget.

- [ ] **Separate canonical results from exploratory ablations**
  - Keep exploratory settings (e.g., per-episode random regime) out of main claims.
  - Report them as diagnostics only.

## P3 - Reproducibility and Configuration Hygiene

- [ ] **Add config guardrails**
  - Validate incompatible settings (e.g., pretrain budget > total timesteps).
  - Surface canonical/debug mode explicitly in config/CLI.

- [ ] **Make stack constraints consistent with plan**
  - Revisit Python requirement and dependency pins for reproducibility with the planned stack.
