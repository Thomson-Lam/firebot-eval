from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from src.config import PPOConfig, TrainConfig
from src.utils import RunningNormalizer


@dataclass
class RolloutBuffer:
    """Stores trajectory data for one rollout."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    sub_rewards: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    regimes: list[str] = field(default_factory=list)
    on_hazards: list[bool] = field(default_factory=list)

    # For recurrent agents
    gru_hiddens: list[torch.Tensor] = field(default_factory=list)
    weights: list[np.ndarray] = field(default_factory=list)
    prev_actions: list[int] = field(default_factory=list)
    prev_sub_rewards: list[np.ndarray] = field(default_factory=list)

    def clear(self) -> None:
        for f in self.__dataclass_fields__:
            getattr(self, f).clear()

    def __len__(self) -> int:
        return len(self.states)


def compute_effective_rewards(
    sub_rewards: np.ndarray,
    weights: np.ndarray,
    normalizer: RunningNormalizer,
) -> np.ndarray:
    """Compute effective reward from sub-rewards and weights.

    sub_rewards: (T, k), weights: (T, k) -> effective: (T,)
    """
    normed = normalizer.normalize(sub_rewards)
    return (normed * weights).sum(axis=-1)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns.

    rewards, values, dones: (T,)
    Returns: (advantages, returns) each (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def ppo_update_static(
    agent,
    buffer: RolloutBuffer,
    static_weights: np.ndarray,
    normalizer: RunningNormalizer,
    ppo_config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    """PPO update for StaticWeightAgent (non-recurrent)."""
    T = len(buffer)
    sub_rews = np.array(buffer.sub_rewards)  # (T, 3)
    normalizer.update(sub_rews)

    weights_tiled = np.tile(static_weights, (T, 1))
    eff_rewards = compute_effective_rewards(sub_rews, weights_tiled, normalizer)

    values_arr = np.array(buffer.values)
    dones_arr = np.array(buffer.dones, dtype=np.float32)

    # Bootstrap last value
    last_val = 0.0  # terminal or truncated

    advantages, returns = compute_gae(
        eff_rewards, values_arr, dones_arr, last_val,
        ppo_config.gamma, ppo_config.gae_lambda,
    )

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states_t = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=device)
    actions_t = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_log_probs_t = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    optimizers = agent.get_optimizers(ppo_config)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for _ in range(ppo_config.num_epochs):
        indices = torch.randperm(T, device=device)
        for start in range(0, T, ppo_config.batch_size):
            end = min(start + ppo_config.batch_size, T)
            idx = indices[start:end]

            log_probs, values, entropy = agent.evaluate(states_t[idx], actions_t[idx])

            ratio = torch.exp(log_probs - old_log_probs_t[idx])
            adv = advantages_t[idx]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - ppo_config.clip_eps, 1 + ppo_config.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, returns_t[idx])
            entropy_loss = -entropy.mean()

            loss = policy_loss + ppo_config.value_coef * value_loss + ppo_config.entropy_coef * entropy_loss

            for opt in optimizers.values():
                opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), ppo_config.max_grad_norm)
            for opt in optimizers.values():
                opt.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_loss.item()
            num_updates += 1

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
    }


def ppo_update_recurrent(
    agent,
    buffer: RolloutBuffer,
    static_weights: np.ndarray,
    normalizer: RunningNormalizer,
    ppo_config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    """PPO update for RecurrentStaticAgent (recurrent, fixed weights)."""
    T = len(buffer)
    sub_rews = np.array(buffer.sub_rewards)
    normalizer.update(sub_rews)

    weights_tiled = np.tile(static_weights, (T, 1))
    eff_rewards = compute_effective_rewards(sub_rews, weights_tiled, normalizer)

    values_arr = np.array(buffer.values)
    dones_arr = np.array(buffer.dones, dtype=np.float32)
    last_val = 0.0

    advantages, returns = compute_gae(
        eff_rewards, values_arr, dones_arr, last_val,
        ppo_config.gamma, ppo_config.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Chunk the rollout for truncated BPTT
    chunk_len = ppo_config.chunk_length
    num_chunks = T // chunk_len
    if num_chunks == 0:
        num_chunks = 1
        chunk_len = T

    # Prepare tensors
    states_arr = np.array(buffer.states)
    actions_arr = np.array(buffer.actions)
    old_lp_arr = np.array(buffer.log_probs)
    prev_actions_arr = np.array(buffer.prev_actions)
    prev_sub_rews_arr = np.array(buffer.prev_sub_rewards)

    optimizers = agent.get_optimizers(ppo_config)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for _ in range(ppo_config.num_epochs):
        # Process chunks sequentially
        chunk_order = torch.randperm(num_chunks).tolist()
        for ci in chunk_order:
            s = ci * chunk_len
            e = min(s + chunk_len, T)

            states_t = torch.tensor(states_arr[s:e], dtype=torch.float32, device=device).unsqueeze(0)
            actions_t = torch.tensor(actions_arr[s:e], dtype=torch.long, device=device).unsqueeze(0)
            prev_act_t = torch.tensor(prev_actions_arr[s:e], dtype=torch.long, device=device).unsqueeze(0)
            prev_sr_t = torch.tensor(prev_sub_rews_arr[s:e], dtype=torch.float32, device=device).unsqueeze(0)
            old_lp_t = torch.tensor(old_lp_arr[s:e], dtype=torch.float32, device=device)
            adv_t = torch.tensor(advantages[s:e], dtype=torch.float32, device=device)
            ret_t = torch.tensor(returns[s:e], dtype=torch.float32, device=device)

            h_init = buffer.gru_hiddens[s].to(device)  # (1, 1, hidden)

            log_probs, values, entropy = agent.evaluate(
                states_t, actions_t, prev_act_t, prev_sr_t, h_init
            )
            log_probs = log_probs.squeeze(0)
            values = values.squeeze(0)
            entropy = entropy.squeeze(0)

            ratio = torch.exp(log_probs - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - ppo_config.clip_eps, 1 + ppo_config.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, ret_t)
            entropy_loss = -entropy.mean()

            loss = policy_loss + ppo_config.value_coef * value_loss + ppo_config.entropy_coef * entropy_loss

            for opt in optimizers.values():
                opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), ppo_config.max_grad_norm)
            for opt in optimizers.values():
                opt.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_loss.item()
            num_updates += 1

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
    }


def ppo_update_drd(
    agent,
    buffer: RolloutBuffer,
    normalizer: RunningNormalizer,
    ppo_config: PPOConfig,
    train_config: TrainConfig,
    device: torch.device,
    freeze_weights: bool = False,
) -> dict[str, float]:
    """PPO update for DRDAgent with learned dynamic weights."""
    T = len(buffer)
    sub_rews = np.array(buffer.sub_rewards)
    normalizer.update(sub_rews)

    # Use stored weights from rollout to compute effective rewards for GAE
    weights_arr = np.array(buffer.weights)  # (T, 3)
    eff_rewards = compute_effective_rewards(sub_rews, weights_arr, normalizer)

    values_arr = np.array(buffer.values)
    dones_arr = np.array(buffer.dones, dtype=np.float32)
    last_val = 0.0

    advantages, returns = compute_gae(
        eff_rewards, values_arr, dones_arr, last_val,
        ppo_config.gamma, ppo_config.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    chunk_len = ppo_config.chunk_length
    num_chunks = T // chunk_len
    if num_chunks == 0:
        num_chunks = 1
        chunk_len = T

    states_arr = np.array(buffer.states)
    actions_arr = np.array(buffer.actions)
    old_lp_arr = np.array(buffer.log_probs)
    prev_actions_arr = np.array(buffer.prev_actions)
    prev_sub_rews_arr = np.array(buffer.prev_sub_rewards)
    # Regime labels: A=0, B=1 for binary classification
    regime_labels = np.array([1.0 if r == "B" else 0.0 for r in buffer.regimes], dtype=np.float32)

    optimizers = agent.get_optimizers(ppo_config)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_smoothness = 0.0
    total_regime_cls = 0.0
    total_weight_directed = 0.0
    direction_accum = np.zeros(3, dtype=np.float64)
    direction_count = 0
    num_updates = 0

    for _ in range(ppo_config.num_epochs):
        chunk_order = torch.randperm(num_chunks).tolist()
        for ci in chunk_order:
            s = ci * chunk_len
            e = min(s + chunk_len, T)

            states_t = torch.tensor(states_arr[s:e], dtype=torch.float32, device=device).unsqueeze(0)
            actions_t = torch.tensor(actions_arr[s:e], dtype=torch.long, device=device).unsqueeze(0)
            prev_act_t = torch.tensor(prev_actions_arr[s:e], dtype=torch.long, device=device).unsqueeze(0)
            prev_sr_t = torch.tensor(prev_sub_rews_arr[s:e], dtype=torch.float32, device=device).unsqueeze(0)
            old_lp_t = torch.tensor(old_lp_arr[s:e], dtype=torch.float32, device=device)
            adv_t = torch.tensor(advantages[s:e], dtype=torch.float32, device=device)
            ret_t = torch.tensor(returns[s:e], dtype=torch.float32, device=device)
            regime_t = torch.tensor(regime_labels[s:e], dtype=torch.float32, device=device)

            h_init = buffer.gru_hiddens[s].to(device)

            log_probs, values, entropy, new_weights, regime_logits = agent.evaluate(
                states_t, actions_t, prev_act_t, prev_sr_t, h_init
            )
            log_probs = log_probs.squeeze(0)
            values = values.squeeze(0)
            entropy = entropy.squeeze(0)
            new_weights = new_weights.squeeze(0)  # (seq_len, k)
            regime_logits = regime_logits.squeeze(0)  # (seq_len,)

            ratio = torch.exp(log_probs - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - ppo_config.clip_eps, 1 + ppo_config.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, ret_t)
            entropy_loss = -entropy.mean()

            # Weight smoothness penalty
            smoothness_loss = agent.weight_smoothness_loss(new_weights.unsqueeze(0))

            # Auxiliary regime classification loss: forces the GRU to encode
            # regime information using privileged labels during training.
            regime_cls_loss = nn.functional.binary_cross_entropy_with_logits(
                regime_logits, regime_t
            )

            # Directed contrastive on SAFETY and EFFICIENCY only (indices 1,2).
            # Progress (index 0) is excluded: its importance is similar in both
            # regimes under a healthy policy, and including it creates a feedback
            # loop (high safety weight in A → agent stops moving → progress drops
            # in A → direction says "progress belongs in B" → collapse).
            # Safety/efficiency have robust regime-dependent signals: barrier
            # guarantees safety hits in A; step cost ramps in B regardless of policy.
            mask_a = (regime_t < 0.5)
            mask_b = (regime_t > 0.5)
            if mask_a.any() and mask_b.any():
                sub_rews_chunk = torch.tensor(
                    sub_rews[s:e], dtype=torch.float32, device=device
                )
                # Per-regime importance: |mean| + std for each component
                mean_a = sub_rews_chunk[mask_a].mean(dim=0)
                std_a = sub_rews_chunk[mask_a].std(dim=0) + 1e-8
                imp_a = mean_a.abs() + std_a  # (k,)

                mean_b = sub_rews_chunk[mask_b].mean(dim=0)
                std_b = sub_rews_chunk[mask_b].std(dim=0) + 1e-8
                imp_b = mean_b.abs() + std_b  # (k,)

                # Full direction (for logging)
                direction = (imp_a - imp_b) / (imp_a + imp_b + 1e-8)  # (k,)
                direction_accum += direction.detach().cpu().numpy()
                direction_count += 1

                # Only apply contrastive to safety (1) and efficiency (2)
                direction_se = direction[1:]  # (2,)
                mean_w_a = new_weights[mask_a].mean(dim=0)  # (k,)
                mean_w_b = new_weights[mask_b].mean(dim=0)  # (k,)
                w_diff_se = (mean_w_a - mean_w_b)[1:]  # (2,)

                weight_directed_loss = -(w_diff_se * direction_se.detach()).sum()
            else:
                weight_directed_loss = torch.tensor(0.0, device=device)

            loss = (
                policy_loss
                + ppo_config.value_coef * value_loss
                + ppo_config.entropy_coef * entropy_loss
                + train_config.smoothness_lambda * smoothness_loss
                + train_config.regime_cls_coef * regime_cls_loss
                + train_config.weight_reward_coef * weight_directed_loss
            )

            for opt in optimizers.values():
                opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), ppo_config.max_grad_norm)

            optimizers["policy"].step()
            optimizers["value"].step()
            optimizers["gru"].step()  # GRU always learns (value + regime cls gradients)
            if not freeze_weights:
                optimizers["weight"].step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_loss.item()
            total_smoothness += smoothness_loss.item()
            total_regime_cls += regime_cls_loss.item()
            total_weight_directed += weight_directed_loss.item()
            num_updates += 1

    n = max(num_updates, 1)
    avg_direction = direction_accum / max(direction_count, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "entropy": total_entropy / n,
        "smoothness_loss": total_smoothness / n,
        "regime_cls_loss": total_regime_cls / n,
        "weight_directed_loss": total_weight_directed / n,
        "direction_progress": avg_direction[0],
        "direction_safety": avg_direction[1],
        "direction_efficiency": avg_direction[2],
    }
