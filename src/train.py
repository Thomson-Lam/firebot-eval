from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import numpy as np
import torch

from src.config import ExperimentConfig
from src.env import RegimeSwitchingGridWorld
from src.networks import DRDAgent, RecurrentStaticAgent, StaticWeightAgent
from src.ppo import (
    RolloutBuffer,
    ppo_update_drd,
    ppo_update_recurrent,
    ppo_update_static,
)
from src.utils import RunningNormalizer, set_seed


# ── wandb helpers ─────────────────────────────────────────────────────────────


def _wandb_init(config: ExperimentConfig, run_name: str) -> Any:
    if not config.train.use_wandb:
        return None
    try:
        import wandb

        return wandb.init(
            project=config.train.wandb_project,
            name=run_name,
            config=dataclasses.asdict(config),
        )
    except Exception:
        return None


def _wandb_log(metrics: dict, step: int, use_wandb: bool) -> None:
    if not use_wandb:
        return
    try:
        import wandb

        wandb.log(metrics, step=step)
    except Exception:
        pass


def _wandb_finish(use_wandb: bool) -> None:
    if not use_wandb:
        return
    try:
        import wandb

        wandb.finish()
    except Exception:
        pass


# ── Rollout collection ───────────────────────────────────────────────────────


def collect_rollout_static(
    env: RegimeSwitchingGridWorld,
    agent: StaticWeightAgent,
    buffer: RolloutBuffer,
    rollout_length: int,
    device: torch.device,
    obs: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Collect rollout for non-recurrent static agent. Returns (last_obs, episode_stats)."""
    buffer.clear()
    if obs is None:
        obs, _info = env.reset()

    episode_returns: list[float] = []
    episode_return = 0.0
    ep_safety_violations = 0
    ep_steps = 0
    regime_returns: dict[str, list[float]] = {"A": [], "B": []}
    current_regime_return = 0.0
    current_regime = env.get_regime()

    for _ in range(rollout_length):
        state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, log_prob, value = agent.act(state_t)

        a = action.item()
        obs_next, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        buffer.states.append(obs)
        buffer.actions.append(a)
        buffer.log_probs.append(log_prob.item())
        buffer.sub_rewards.append(info["sub_rewards"])
        buffer.values.append(value.item())
        buffer.dones.append(done)
        buffer.regimes.append(info["regime"])
        buffer.on_hazards.append(info.get("on_hazard", False))

        episode_return += reward
        current_regime_return += reward
        if info.get("on_hazard", False):
            ep_safety_violations += 1
        ep_steps += 1

        new_regime = env.get_regime()
        if new_regime != current_regime:
            regime_returns[current_regime].append(current_regime_return)
            current_regime_return = 0.0
            current_regime = new_regime

        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            ep_safety_violations = 0
            ep_steps = 0
            current_regime_return = 0.0
            obs_next, _info = env.reset()
            current_regime = env.get_regime()

        obs = obs_next

    stats = {
        "episode_returns": episode_returns,
        "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
        "regime_A_returns": regime_returns["A"],
        "regime_B_returns": regime_returns["B"],
    }
    return obs, stats


def collect_rollout_recurrent(
    env: RegimeSwitchingGridWorld,
    agent: RecurrentStaticAgent | DRDAgent,
    buffer: RolloutBuffer,
    rollout_length: int,
    device: torch.device,
    obs: np.ndarray | None = None,
    is_drd: bool = False,
) -> tuple[np.ndarray, dict]:
    """Collect rollout for recurrent agents (RecurrentStatic or DRD)."""
    buffer.clear()
    if obs is None:
        obs, _info = env.reset()

    h = agent.init_hidden(1, device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)
    prev_sub_rewards = torch.zeros(1, 3, dtype=torch.float32, device=device)

    episode_returns: list[float] = []
    episode_return = 0.0
    regime_returns: dict[str, list[float]] = {"A": [], "B": []}
    current_regime_return = 0.0
    current_regime = env.get_regime()

    for _ in range(rollout_length):
        state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if is_drd:
            action, log_prob, value, h_new, weights = agent.act(
                state_t, prev_action, prev_sub_rewards, h
            )
            buffer.weights.append(weights.squeeze(0).cpu().numpy())
        else:
            action, log_prob, value, h_new = agent.act(
                state_t, prev_action, prev_sub_rewards, h
            )

        a = action.item()
        obs_next, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        buffer.states.append(obs)
        buffer.actions.append(a)
        buffer.log_probs.append(log_prob.item())
        buffer.sub_rewards.append(info["sub_rewards"])
        buffer.values.append(value.item())
        buffer.dones.append(done)
        buffer.regimes.append(info["regime"])
        buffer.on_hazards.append(info.get("on_hazard", False))
        buffer.gru_hiddens.append(h.detach().cpu())
        buffer.prev_actions.append(prev_action.item())
        buffer.prev_sub_rewards.append(prev_sub_rewards.squeeze(0).cpu().numpy())

        episode_return += reward
        current_regime_return += reward

        new_regime = env.get_regime()
        if new_regime != current_regime:
            regime_returns[current_regime].append(current_regime_return)
            current_regime_return = 0.0
            current_regime = new_regime

        prev_action = action
        prev_sub_rewards = torch.tensor(
            info["sub_rewards"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        h = h_new

        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            current_regime_return = 0.0
            obs_next, _info = env.reset()
            # NOTE: Do NOT reset GRU hidden state at episode boundaries.
            # The GRU must carry state across episodes within a rollout so it
            # can accumulate evidence about regime switches.
            prev_action = torch.zeros(1, dtype=torch.long, device=device)
            prev_sub_rewards = torch.zeros(1, 3, dtype=torch.float32, device=device)
            current_regime = env.get_regime()

        obs = obs_next

    stats = {
        "episode_returns": episode_returns,
        "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
        "regime_A_returns": regime_returns["A"],
        "regime_B_returns": regime_returns["B"],
    }
    return obs, stats


# ── Internal training loops (no wandb — caller handles logging) ──────────────


def _train_static_single(
    config: ExperimentConfig,
    weights: np.ndarray,
    label: str,
    log_fn,
) -> dict:
    """Train a single static-weight PPO agent. Calls log_fn(prefix, metrics, step)."""
    set_seed(config.train.seed)
    device = torch.device(config.train.device)
    env = RegimeSwitchingGridWorld(config.env)
    agent = StaticWeightAgent(config.net).to(device)
    normalizer = RunningNormalizer(config.net.num_sub_rewards)
    buffer = RolloutBuffer()

    obs = None
    all_returns: list[float] = []
    global_step = 0
    num_updates = config.train.total_timesteps // config.ppo.rollout_length

    for update in range(num_updates):
        obs, stats = collect_rollout_static(
            env, agent, buffer, config.ppo.rollout_length, device, obs
        )
        all_returns.extend(stats["episode_returns"])

        losses = ppo_update_static(agent, buffer, weights, normalizer, config.ppo, device)
        global_step += config.ppo.rollout_length

        metrics = {
            **losses,
            "mean_return": stats["mean_return"],
            "mean_return_regime_A": np.mean(stats["regime_A_returns"]) if stats["regime_A_returns"] else 0,
            "mean_return_regime_B": np.mean(stats["regime_B_returns"]) if stats["regime_B_returns"] else 0,
        }
        log_fn(label, metrics, global_step)

        if (update + 1) % 10 == 0:
            recent = all_returns[-20:] if all_returns else [0]
            print(f"  [{label}] step {global_step}: mean_return={np.mean(recent):.2f}")

    return {
        "label": label,
        "weights": weights,
        "all_returns": all_returns,
        "final_mean_return": np.mean(all_returns[-50:]) if len(all_returns) >= 50 else np.mean(all_returns) if all_returns else 0,
    }


def _train_recurrent_static(
    config: ExperimentConfig,
    weights: np.ndarray,
    label: str,
    log_fn,
) -> dict:
    """Train a recurrent PPO agent with fixed weights."""
    set_seed(config.train.seed)
    device = torch.device(config.train.device)
    env = RegimeSwitchingGridWorld(config.env)
    agent = RecurrentStaticAgent(config.net).to(device)
    normalizer = RunningNormalizer(config.net.num_sub_rewards)
    buffer = RolloutBuffer()

    obs = None
    all_returns: list[float] = []
    global_step = 0
    num_updates = config.train.total_timesteps // config.ppo.rollout_length

    for update in range(num_updates):
        obs, stats = collect_rollout_recurrent(
            env, agent, buffer, config.ppo.rollout_length, device, obs, is_drd=False
        )
        all_returns.extend(stats["episode_returns"])

        losses = ppo_update_recurrent(agent, buffer, weights, normalizer, config.ppo, device)
        global_step += config.ppo.rollout_length

        metrics = {
            **losses,
            "mean_return": stats["mean_return"],
            "mean_return_regime_A": np.mean(stats["regime_A_returns"]) if stats["regime_A_returns"] else 0,
            "mean_return_regime_B": np.mean(stats["regime_B_returns"]) if stats["regime_B_returns"] else 0,
        }
        log_fn(label, metrics, global_step)

        if (update + 1) % 10 == 0:
            recent = all_returns[-20:] if all_returns else [0]
            print(f"  [{label}] step {global_step}: mean_return={np.mean(recent):.2f}")

    return {
        "label": label,
        "weights": weights,
        "all_returns": all_returns,
        "final_mean_return": np.mean(all_returns[-50:]) if len(all_returns) >= 50 else np.mean(all_returns) if all_returns else 0,
    }


# ── Layer 1: Baselines ───────────────────────────────────────────────────────


def train_layer1(config: ExperimentConfig) -> dict:
    """Layer 1: Train static-weight baselines and recurrent baseline. Single wandb run."""
    print("=" * 60)
    print("Layer 1: Static Weight Grid Search + Recurrent Baseline")
    print("=" * 60)

    _wandb_init(config, "layer1_baselines")

    # Accumulated step offset so each variant's steps don't overlap in wandb
    step_offset = 0

    def log_fn(prefix: str, metrics: dict, step: int) -> None:
        prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}
        _wandb_log(prefixed, step + step_offset, config.train.use_wandb)

    # Grid search over weight combinations
    w_progress_vals = [0.2, 0.4, 0.6, 0.8]
    w_safety_vals = [0.1, 0.3, 0.5]

    weight_combos = []
    for wp, ws in itertools.product(w_progress_vals, w_safety_vals):
        we = 1.0 - wp - ws
        if we > 0.01:
            weight_combos.append((wp, ws, we))

    # Uniform baseline
    weight_combos.append((1 / 3, 1 / 3, 1 / 3))

    results = []

    for wp, ws, we in weight_combos:
        weights = np.array([wp, ws, we], dtype=np.float32)
        label = f"w{wp:.1f}_{ws:.1f}_{we:.1f}"
        print(f"\nTraining static weights: progress={wp:.1f}, safety={ws:.1f}, efficiency={we:.1f}")
        result = _train_static_single(config, weights, label, log_fn)
        results.append(result)
        print(f"  Final mean return: {result['final_mean_return']:.2f}")
        step_offset += config.train.total_timesteps

    # Find best static weights
    best = max(results, key=lambda r: r["final_mean_return"])
    print(f"\nBest static weights: {best['label']} (return={best['final_mean_return']:.2f})")

    # Train recurrent baseline with best static weights
    print(f"\nTraining recurrent baseline with best static weights...")
    recurrent_result = _train_recurrent_static(
        config, best["weights"], "recurrent", log_fn
    )
    print(f"  Recurrent final mean return: {recurrent_result['final_mean_return']:.2f}")

    # Log summary table
    if config.train.use_wandb:
        try:
            import wandb

            summary = {r["label"]: r["final_mean_return"] for r in results}
            summary["recurrent"] = recurrent_result["final_mean_return"]
            summary["best_static_label"] = best["label"]
            wandb.summary.update(summary)
        except Exception:
            pass

    _wandb_finish(config.train.use_wandb)

    return {
        "static_results": results,
        "best_static": best,
        "recurrent_result": recurrent_result,
    }


# ── Layer 2: DRD Core ────────────────────────────────────────────────────────


def train_layer2(config: ExperimentConfig, best_static_weights: np.ndarray | None = None) -> dict:
    """Layer 2: DRD with pretrain-then-co-train. Single wandb run."""
    print("=" * 60)
    print("Layer 2: Dynamic Reward Decomposition")
    print("=" * 60)

    if best_static_weights is None:
        best_static_weights = np.array([0.8, 0.1, 0.1], dtype=np.float32)

    set_seed(config.train.seed)
    device = torch.device(config.train.device)
    env = RegimeSwitchingGridWorld(config.env)

    # Warm-start weight network: compute logits so softmax ≈ best_static_weights
    # log(w) shifted to have mean 0 gives the right softmax output
    log_w = np.log(best_static_weights + 1e-8)
    init_logits = torch.tensor(log_w - log_w.mean(), dtype=torch.float32)
    agent = DRDAgent(config.net, init_weight_logits=init_logits, min_weight=config.train.min_weight).to(device)
    normalizer = RunningNormalizer(config.net.num_sub_rewards)
    buffer = RolloutBuffer()

    _wandb_init(config, "layer2_drd")

    obs = None
    all_returns: list[float] = []
    weight_history: list[np.ndarray] = []
    regime_history: list[str] = []
    global_step = 0
    num_updates = config.train.total_timesteps // config.ppo.rollout_length
    pretrain_updates = config.train.pretrain_timesteps // config.ppo.rollout_length

    for update in range(num_updates):
        freeze_weights = update < pretrain_updates
        phase = "pretrain" if freeze_weights else "co-train"

        obs, stats = collect_rollout_recurrent(
            env, agent, buffer, config.ppo.rollout_length, device, obs, is_drd=True
        )
        all_returns.extend(stats["episode_returns"])

        # If pretraining, override buffer weights with static
        if freeze_weights:
            T = len(buffer)
            buffer.weights = [best_static_weights.copy() for _ in range(T)]

        losses = ppo_update_drd(
            agent, buffer, normalizer, config.ppo, config.train, device,
            freeze_weights=freeze_weights,
        )
        global_step += config.ppo.rollout_length

        # Track weight trajectories
        if buffer.weights:
            weight_history.extend(buffer.weights)
            regime_history.extend(buffer.regimes)

        mean_weights = np.mean(buffer.weights, axis=0) if buffer.weights else np.zeros(3)

        # Per-regime weight breakdown
        weights_arr = np.array(buffer.weights)
        regimes_arr = np.array(buffer.regimes)
        regime_a_mask = regimes_arr == "A"
        regime_b_mask = regimes_arr == "B"
        mean_w_a = np.mean(weights_arr[regime_a_mask], axis=0) if regime_a_mask.any() else np.zeros(3)
        mean_w_b = np.mean(weights_arr[regime_b_mask], axis=0) if regime_b_mask.any() else np.zeros(3)

        metrics = {
            **losses,
            "mean_return": stats["mean_return"],
            "mean_return_regime_A": np.mean(stats["regime_A_returns"]) if stats["regime_A_returns"] else 0,
            "mean_return_regime_B": np.mean(stats["regime_B_returns"]) if stats["regime_B_returns"] else 0,
            "weight_progress": mean_weights[0],
            "weight_safety": mean_weights[1],
            "weight_efficiency": mean_weights[2],
            "weight_A_progress": mean_w_a[0],
            "weight_A_safety": mean_w_a[1],
            "weight_A_efficiency": mean_w_a[2],
            "weight_B_progress": mean_w_b[0],
            "weight_B_safety": mean_w_b[1],
            "weight_B_efficiency": mean_w_b[2],
            "phase": 0 if freeze_weights else 1,
        }
        _wandb_log(metrics, global_step, config.train.use_wandb)

        if (update + 1) % 10 == 0:
            recent = all_returns[-20:] if all_returns else [0]
            print(
                f"  [DRD {phase}] step {global_step}: mean_return={np.mean(recent):.2f} "
                f"w=[{mean_weights[0]:.2f}, {mean_weights[1]:.2f}, {mean_weights[2]:.2f}] "
                f"A=[{mean_w_a[0]:.2f}, {mean_w_a[1]:.2f}, {mean_w_a[2]:.2f}] "
                f"B=[{mean_w_b[0]:.2f}, {mean_w_b[1]:.2f}, {mean_w_b[2]:.2f}]"
            )

    _wandb_finish(config.train.use_wandb)

    return {
        "all_returns": all_returns,
        "weight_history": weight_history,
        "regime_history": regime_history,
        "agent": agent,
        "final_mean_return": np.mean(all_returns[-50:]) if len(all_returns) >= 50 else np.mean(all_returns) if all_returns else 0,
    }


# ── Layer 3: Ablations ───────────────────────────────────────────────────────


def train_layer3(config: ExperimentConfig) -> dict:
    """Layer 3: Ablation sweeps. Single wandb run with prefixed metrics."""
    print("=" * 60)
    print("Layer 3: Ablation Studies")
    print("=" * 60)

    # Disable wandb in sub-calls — we log from here
    results = {}

    _wandb_init(config, "layer3_ablations")
    step_counter = 0

    def _run_ablation(label: str, cfg: ExperimentConfig) -> dict:
        nonlocal step_counter
        # Disable wandb in the sub-call (we log summary ourselves)
        cfg.train = dataclasses.replace(cfg.train, use_wandb=False)
        result = train_layer2(cfg)
        step_counter += 1
        _wandb_log(
            {f"{label}/final_mean_return": result["final_mean_return"]},
            step_counter,
            config.train.use_wandb,
        )
        return result

    # Smoothness lambda sweep
    print("\n--- Smoothness Lambda Sweep ---")
    for lam in [0.0, 0.01, 0.1, 1.0]:
        cfg = dataclasses.replace(config)
        cfg.train = dataclasses.replace(config.train, smoothness_lambda=lam)
        label = f"smooth_{lam}"
        print(f"\n  smoothness_lambda={lam}")
        result = _run_ablation(label, cfg)
        results[label] = {
            "final_mean_return": result["final_mean_return"],
            "param": lam,
        }

    # GRU hidden dim sweep
    print("\n--- GRU Hidden Dim Sweep ---")
    for dim in [16, 32, 64, 128]:
        cfg = dataclasses.replace(config)
        cfg.net = dataclasses.replace(config.net, gru_hidden_dim=dim)
        label = f"gru_dim_{dim}"
        print(f"\n  gru_hidden_dim={dim}")
        result = _run_ablation(label, cfg)
        results[label] = {
            "final_mean_return": result["final_mean_return"],
            "param": dim,
        }

    # Weight LR sweep
    print("\n--- Weight LR Sweep ---")
    for lr in [1e-5, 1e-4, 1e-3]:
        cfg = dataclasses.replace(config)
        cfg.ppo = dataclasses.replace(config.ppo, lr_weight=lr)
        label = f"weight_lr_{lr}"
        print(f"\n  lr_weight={lr}")
        result = _run_ablation(label, cfg)
        results[label] = {
            "final_mean_return": result["final_mean_return"],
            "param": lr,
        }

    # Per-episode randomized regime
    print("\n--- Per-Episode Randomized Regime ---")
    cfg = dataclasses.replace(config)
    cfg.env = dataclasses.replace(config.env, randomize_regime_per_episode=True)
    result = _run_ablation("randomized_regime", cfg)
    results["randomized_regime"] = {
        "final_mean_return": result["final_mean_return"],
    }

    # Log summary
    if config.train.use_wandb:
        try:
            import wandb

            wandb.summary.update({k: v["final_mean_return"] for k, v in results.items()})
        except Exception:
            pass

    _wandb_finish(config.train.use_wandb)

    return results
