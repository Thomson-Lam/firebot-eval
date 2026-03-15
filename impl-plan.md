# Unified Implementation Plan (Canonical + Cross-Referenced)

This file is the single source of truth for implementation. It combines the canonical technical plan from `plan.md` with safe, aligned content from the other draft notes.

## plan.md

### Problem definition
We study non-stationary episodic RL where objectives remain fixed but their relative importance changes over time. We use decomposed rewards (progress, safety, time), infer hidden regime context from trajectory history, and learn dynamic weights over reward components to produce an interpretable priority vector.

Project invariant (must not be violated):
- Keep the objective set fixed across all regimes.
- Model only objective-priority shifts over time.
- Do not change environment dynamics/transition behavior across regimes.
- Do not change which objectives are rewarded across regimes; only change their relative importance.

### Core method
- Use a recurrent encoder (GRU) to infer latent context from `(o_t, a_{t-1}, r_{t-1}^{vec}, done_{t-1})`.
- Use a weighting network `w_hat_t = softmax(MLP(h_t))` over normalized sub-rewards.
- Policy must be context-conditioned, with minimum form `pi(a_t | s_t, h_t)`.
- Optional policy variant for testing: `pi(a_t | s_t, h_t, w_hat_t)`.
- Train policy and critic heads with PPO using online rollouts.
- Use `K` critic value heads, where each head predicts the value for exactly one sub-reward component.
- Combine per-component values with `w_hat_t` when an effective scalar value is needed for PPO targets/analysis.

### Effective reward and normalization
- Normalize each reward component online:
  - `r_tilde[t,k] = (r[t,k] - mu[k]) / (sigma[k] + eps)`
  - purpose: avoid scale artifacts so learned weights reflect priority, not magnitude mismatch
- Compute effective scalar reward:
  - `r_eff[t] = sum_k w_hat[t,k] * r_tilde[t,k]`
- The environment defines the ground-truth task. All methods are evaluated on the same ground-truth task return.
- DRD additionally uses learned weights to form an internal effective reward for training/analysis, while scalar baselines use the environment scalar reward directly.

### Environment and benchmark design
- Prefer a partially observable regime-switching environment with hidden regime variable `z_t`.
- Keep objectives fixed; only regime-dependent weighting changes.
- Regime shifts in this project represent objective-priority changes only.
- Do not implement regime-dependent physics/dynamics changes or objective-definition changes.
- Keep reward components semantically stable across regimes; vary only the weighting profile over those same components.
- Use stochastic switch timing with variable regime durations to prevent overfitting to time-based periodicity.
- Episodes must be long enough to contain at least one, and preferably multiple, within-episode regime switches.
- Do not use fixed every-`K`-steps switching for actual training/evaluation.
- If fixed every-`K` switching is used at all, use it only as scaffolding for debugging/sanity checks.
- Log true `z_t` only for evaluation, never as policy input.

### Training protocol
- Train one end-to-end model concurrently.
- PPO cycle: collect fresh on-policy rollouts from parallel envs, then perform multiple SGD epochs over that batch.
- Store and handle recurrent state carefully; reset hidden states at episode boundaries.
- Never preserve or carry GRU hidden states across episodes or across different environment instances.
- Fairness protocol: use the same environment generator/schedule family, training steps, seeds, and compute budget where possible across all baselines.
- Start from a controlled setup and follow staged complexity.
- Do not use pretrain-then-co-train as the actual method training schedule.
- If pretraining is used at all, use it only as temporary scaffolding for debugging or sanity checks, and report final results from concurrent training only.

### Baselines and comparison models (canonical)
- Scalar PPO (no recurrence, no decomposition, one scalar reward)
  - basic standard RL
  - question answered: Does anything about context or decomposition matter at all?
- Recurrent PPO (adds memory with one scalar reward)
  - question answered: Is recurrence alone enough? Do we need dynamic reward weighting?
- Recurrent static decomposition baseline
  - same recurrent backbone, context-conditioned policy, and per-component critic structure as full DRD
  - decomposed rewards
  - fixed reward weights (not learned dynamically)
  - this baseline differs from full DRD only by using fixed weights instead of learned dynamic weights
  - question answered: If I already have recurrence and decomposed rewards, do I still need learned dynamic weights?
- Oracle regime-access agent
  - receives true hidden regime or true environment weights
  - cheapest to train among comparison models
  - role: performance ceiling
  - question answered: How close can we get to an agent with perfect regime-shift awareness?
- Full DRD model (primary method under test)
  - recurrent encoder
  - dynamic weight network
  - decomposed rewards
  - context-conditioned policy and critic
  - per-component critic heads in canonical architecture

Not considered in this project scope:
- RL^2
- PEARL
- LILAC

### Evaluation metrics (canonical)
Core metrics to definitely report:
1. average episodic return
2. post-switch performance drop
3. recovery time
4. variance across seeds
5. adaptation lag and similarity of `w_hat_t` to true regime weighting

Good secondary metrics:
- AUC / return over training
- post-switch regret vs oracle
- hazard violations / success rate (if naturally exposed by the environment)
- collapse/failure rate

Optional diagnostics only:
- mode classification accuracy / linear probe
- window length `L` sensitivity (only as optional comparative diagnostic)
- forgetting score
- Pareto front plots

### Suggested implementation stack
- Python 3.10+
- PyTorch
- Gymnasium (+ custom env)
- MiniGrid and/or NS-Gym for environment support
- CleanRL-style PPO for custom architecture
- SB3-Contrib RecurrentPPO for baseline
- TensorBoard/WandB for logging

### Hardware and compute requirements (for current environment design)
The current benchmark environment (10x10 gridworld, small MLP/GRU policies, on-policy PPO) is lightweight and can run on CPU. Recommended budgets:

- Minimum local development:
  - CPU: 4 logical cores
  - RAM: 8 GB
  - GPU: not required
  - Storage: 2-5 GB free (checkpoints, logs, plots)

- Recommended single-run training (one method, one seed):
  - CPU: 8 logical cores preferred
  - RAM: 16 GB
  - GPU: optional (NVIDIA >= 6 GB VRAM) for faster recurrent updates
  - Storage: 10 GB free

- Recommended canonical experiment bundle (all baselines + DRD, multi-seed):
  - CPU: 8-16 logical cores
  - RAM: 16-32 GB
  - GPU: optional but helpful for throughput (single consumer GPU is sufficient)
  - Storage: 20-50 GB free for logs/artifacts across seeds

Notes:
- Because rollouts are currently single-environment and not heavily vectorized, CPU throughput can be a bottleneck before GPU memory becomes a bottleneck.
- For fair comparisons, run all methods under the same hardware class and report wall-clock/runtime settings with seeds.

### Build order
1. Implement environment and verify reward decomposition/regime logging.
2. Train scalar PPO.
3. Train recurrent PPO.
4. Add fixed-weight decomposition.
5. Add GRU + dynamic weighting.
6. Add `K` per-component critic heads (one head per sub-reward).
7. Add interpretability and adaptation-lag evaluation.
8. Run multi-seed evaluations and harder variants.
