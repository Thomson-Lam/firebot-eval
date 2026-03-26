# Proposal: RL Benchmark for Wildfire Asset Protection (Option A)

## Working Title

**Protecting Critical Assets Under Limited Suppression Budget: An Empirical RL Benchmark in a Custom Wildfire Simulator**

---

## 1) Problem Statement

Emergency wildfire response requires fast tactical decisions under uncertainty: where to move, when to deploy suppression, and which limited resource to spend first. Real-world evaluation is expensive, risky, and not reproducible.

This project studies a controlled version of that problem in a custom simulator with one concrete objective:

**Protect critical assets under limited suppression budget.**

---

## 2) Why Existing Techniques Are Not Fully Satisfying

- Heuristic approaches are often brittle across different fire layouts and spread severities.
- A single PPO implementation does not establish robust algorithm suitability.
- The current hackathon codebase lacks a clean multi-seed, multi-algorithm benchmark harness.
- Existing docs overstate operational realism relative to implementation; paper claims must match implemented evidence.

Therefore, a defensible one-week contribution is an empirical benchmark of standard RL methods on an improved simulator with real control tradeoffs.

---

## 3) Intuition Behind the Developed Technique

The developed technique is not a new RL algorithm. It is an enhanced benchmark environment and evaluation protocol designed to produce meaningful tactical tradeoffs:

1. **Prioritization under risk**: critical assets can be lost if not protected.
2. **Planning under scarcity**: helicopter/crew actions are limited and costly.
3. **Spatial reasoning**: non-uniform spread field (flammability map or wind bias).
4. **Robustness testing**: multiple scenario families and held-out test families.

Core intuition: better benchmark structure and rigorous evaluation produce more defensible RL evidence than adding algorithmic novelty under time pressure.

---

## 4) Techniques to Tackle the Problem

## 4.1 Algorithms to compare (with rationale)

- **DQN**: value-based baseline for discrete tactical control.
- **A2C**: lightweight on-policy actor-critic baseline.
- **PPO**: stronger policy-gradient baseline and current repo baseline.
- **Greedy heuristic**: domain-inspired non-RL tactical baseline.
- **Random policy**: floor sanity check.

Optional only if hidden regime shifts are added and time permits:

- **Recurrent PPO baseline** for partial observability robustness.

## 4.2 Environment enhancements to keep (minimum viable strong paper)

1. **Scenario diversity**
   - ignition patterns: center, edge, corner, multi-cluster
   - severity levels: low/medium/high

2. **Finite resources with costs/cooldowns**
   - limited helicopter drops and crew deployments
   - resource cost and cooldown penalties

3. **Heterogeneous spread field (choose one)**
   - flammability map, or
   - directional wind bias

4. **Clean benchmark harness**
   - fixed train/eval protocol
   - multi-seed runs
   - no fallback heuristic contamination during evaluation

## 4.3 High-value additions (if time permits)

1. **Critical assets (recommended, highest value)**
   - place 2-5 assets on map
   - strong penalty when assets burn

2. **Travel/action latency**
   - deployment requires position or delayed effect

3. **Simple hidden shift test**
   - stochastic wind shift mid-episode

4. **Train/test split on scenario families**
   - evaluate on held-out ignition/severity combinations

---

## 5) Planned Related Work Review (4-8 papers)

The report will cover:

1. Core RL algorithm papers:
   - DQN
   - A3C/A2C
   - PPO

2. RL benchmark/reproducibility papers:
   - seed sensitivity, fair comparison protocols

3. Wildfire decision-support/spread-modeling papers:
   - decision-support framing
   - spread prediction and response strategy

The literature section will support an empirical benchmarking contribution, not a novel algorithm claim.

---

## 6) Experimental Plan

## 6.1 Fixed protocol

- Same environment generator family for all algorithms.
- Same timestep budget per algorithm.
- Same seed set (3-5 seeds).
- Same evaluation episodes/checkpoints.
- Fallback heuristic disabled during RL benchmark runs.

## 6.2 Metrics

Primary:

- mean episodic return
- critical asset survival rate
- containment success rate
- final burned area (cells)

Secondary:

- resource efficiency (suppression impact per resource spent)
- wasted deployment rate
- time to containment
- variance across seeds

## 6.3 Generalization evaluation

- Train on subset of scenario families.
- Test on held-out ignition/severity combinations.
- Compare performance drop and robustness ranking across methods.

---

## 7) Data Pipeline Positioning in the Paper

Data pipeline remains **supporting context**, not the central empirical claim.

Based on audit findings, claims must stay realistic:

- implemented ingestion: FIRMS, CWFIS active fires, Open-Meteo, CFFDRS
- not fully implemented as production ETL: CIFFC, BC/AB ArcGIS full pipeline, ECCC Datamart orchestration, broad historical validated spread labels

Paper wording will avoid operational overclaim and state:

"We benchmark RL methods in an enhanced custom wildfire simulator inspired by wildfire decision-support structure."

---

## 8) One-Week Execution Plan

Day 1:
- Freeze objective and benchmark protocol.
- Add critical assets + resource budgets to environment.

Day 2:
- Add scenario generator and heterogeneous spread field.
- Add eval mode without fallback contamination.

Day 3:
- Implement DQN/A2C runners alongside PPO.
- Standardize logs and output schema.

Day 4:
- Pilot runs and reward sanity checks.
- Fix instability and calibration issues.

Day 5:
- Full multi-seed train/eval runs.

Day 6:
- Aggregate results, figures, and tables.
- Draft evaluation/discussion.

Day 7:
- Final report polish with limitations and future work.

---

## 9) Expected Contribution and Defensibility

This proposal is defensible because it:

- focuses on a single tangible objective,
- introduces genuine tactical tradeoffs,
- compares standard baselines fairly,
- reports multi-seed results under controlled scenario families,
- avoids overclaiming real-world deployment readiness.

Expected claim:

"We design an enhanced wildfire tactical suppression benchmark with protected assets, limited suppression budget, and heterogeneous spread, then compare standard RL and heuristic baselines under controlled scenario families."

---

## 10) Future Work (Out of Scope This Week)

- multi-agent coordination
- real GIS terrain integration
- historical replay validation
- complex dispatch logistics
- continuous-action aircraft routing
