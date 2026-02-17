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

    # Pre-normalize sub-rewards for the weight-reward alignment loss
    sub_rews_normed = normalizer.normalize(sub_rews)

    optimizers = agent.get_optimizers(ppo_config)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_smoothness = 0.0
    total_weight_reward = 0.0
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
            sub_rews_t = torch.tensor(sub_rews_normed[s:e], dtype=torch.float32, device=device)

            h_init = buffer.gru_hiddens[s].to(device)

            log_probs, values, entropy, new_weights = agent.evaluate(
                states_t, actions_t, prev_act_t, prev_sr_t, h_init
            )
            log_probs = log_probs.squeeze(0)
            values = values.squeeze(0)
            entropy = entropy.squeeze(0)
            new_weights = new_weights.squeeze(0)  # (seq_len, k)

            ratio = torch.exp(log_probs - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - ppo_config.clip_eps, 1 + ppo_config.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, ret_t)
            entropy_loss = -entropy.mean()

            # Weight smoothness penalty (via agent method)
            smoothness_loss = agent.weight_smoothness_loss(new_weights.unsqueeze(0))

            # Weight-reward alignment loss: REINFORCE-style gradient for weights.
            # Compute differentiable effective reward using new weights, then
            # weight by advantages — this trains the weight network to prioritize
            # sub-rewards that correlate with good outcomes.
            new_r_eff = (new_weights * sub_rews_t).sum(dim=-1)  # (seq_len,)
            weight_reward_loss = -(new_r_eff * adv_t.detach()).mean()

            loss = (
                policy_loss
                + ppo_config.value_coef * value_loss
                + ppo_config.entropy_coef * entropy_loss
                + train_config.smoothness_lambda * smoothness_loss
                + train_config.weight_reward_coef * weight_reward_loss
            )

            for opt in optimizers.values():
                opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), ppo_config.max_grad_norm)

            optimizers["policy"].step()
            optimizers["value"].step()
            optimizers["gru"].step()  # GRU always learns (value loss provides gradient)
            if not freeze_weights:
                optimizers["weight"].step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_loss.item()
            total_smoothness += smoothness_loss.item()
            total_weight_reward += weight_reward_loss.item()
            num_updates += 1

    n = max(num_updates, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "entropy": total_entropy / n,
        "smoothness_loss": total_smoothness / n,
        "weight_reward_loss": total_weight_reward / n,
    }
