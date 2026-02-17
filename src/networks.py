from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.config import NetworkConfig, PPOConfig


class PolicyNetwork(nn.Module):
    """MLP policy: state -> action distribution."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> Categorical:
        logits = self.net(state)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """Value function: (state, optional h_t) -> scalar value."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GRUHistoryEncoder(nn.Module):
    """Encodes trajectory (state, action_onehot, sub_rewards) via GRU."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, input_dim), h: (1, batch, hidden) -> output, h_new."""
        output, h_new = self.gru(x, h)
        return output, h_new

    def forward_step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single step: x: (batch, input_dim), h: (1, batch, hidden)."""
        output, h_new = self.gru(x.unsqueeze(1), h)
        return output.squeeze(1), h_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)


class WeightNetwork(nn.Module):
    """Maps GRU hidden state to sub-reward weights on the simplex."""

    def __init__(
        self,
        hidden_dim: int,
        num_sub_rewards: int,
        weight_hidden: int,
        init_logits: torch.Tensor | None = None,
        min_weight: float = 0.0,
    ):
        super().__init__()
        self.min_weight = min_weight
        self.num_sub_rewards = num_sub_rewards
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, weight_hidden),
            nn.ReLU(),
            nn.Linear(weight_hidden, num_sub_rewards),
        )
        # Warm-start: set bias so initial output ≈ softmax(init_logits).
        # Scale weights to 0.5x — enough input sensitivity for regime
        # differentiation while keeping the warm-start stable.
        if init_logits is not None:
            with torch.no_grad():
                self.net[-1].weight.mul_(0.5)
                self.net[-1].bias.copy_(init_logits)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (batch, hidden_dim) -> weights: (batch, num_sub_rewards), sums to 1.

        Applies a minimum weight floor to prevent collapse, then renormalizes.
        """
        raw = torch.softmax(self.net(h), dim=-1)
        if self.min_weight > 0:
            # Mix with uniform: w' = (1 - k*min) * softmax + min
            floor = self.min_weight
            k = self.num_sub_rewards
            raw = raw * (1 - k * floor) + floor
        return raw


class DRDAgent(nn.Module):
    """Composite agent wrapping all components."""

    def __init__(
        self,
        config: NetworkConfig,
        init_weight_logits: torch.Tensor | None = None,
        min_weight: float = 0.0,
    ):
        super().__init__()
        self.config = config
        self.gru = GRUHistoryEncoder(config.gru_input_dim, config.gru_hidden_dim)
        # Policy takes state + GRU hidden so it can learn regime-dependent behavior.
        # The weight network also conditions on GRU hidden for reward weighting.
        self.policy = PolicyNetwork(
            config.state_dim + config.gru_hidden_dim, config.action_dim, config.policy_hidden
        )
        self.weight_net = WeightNetwork(
            config.gru_hidden_dim, config.num_sub_rewards, config.weight_hidden,
            init_logits=init_weight_logits,
            min_weight=min_weight,
        )
        self.value_net = ValueNetwork(config.state_dim + config.gru_hidden_dim, config.value_hidden)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.gru.init_hidden(batch_size, device)

    def _build_gru_input(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        sub_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Build GRU input from (state, action_onehot, sub_rewards)."""
        action_oh = torch.nn.functional.one_hot(action.long(), self.config.action_dim).float()
        return torch.cat([state, action_oh, sub_rewards], dim=-1)

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_sub_rewards: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout step. Returns (action, log_prob, value, h_new, weights)."""
        gru_input = self._build_gru_input(state, prev_action, prev_sub_rewards)
        gru_out, h_new = self.gru.forward_step(gru_input, h)

        weights = self.weight_net(gru_out)
        policy_input = torch.cat([state, gru_out], dim=-1)
        dist = self.policy(policy_input)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        value = self.value_net(policy_input)

        return action, log_prob, value, h_new, weights

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_sub_rewards: torch.Tensor,
        h_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a sequence for PPO update.

        All inputs: (batch, seq_len, ...) except h_init: (1, batch, hidden).
        Returns (log_probs, values, entropy, weights) each (batch, seq_len, ...).
        """
        batch, seq_len = states.shape[:2]

        gru_inputs = self._build_gru_input(
            states.reshape(-1, self.config.state_dim),
            prev_actions.reshape(-1),
            prev_sub_rewards.reshape(-1, self.config.num_sub_rewards),
        ).reshape(batch, seq_len, -1)

        gru_out, _ = self.gru(gru_inputs, h_init)  # (batch, seq_len, hidden)

        weights = self.weight_net(gru_out.reshape(-1, self.config.gru_hidden_dim))
        weights = weights.reshape(batch, seq_len, self.config.num_sub_rewards)

        flat_states = states.reshape(-1, self.config.state_dim)
        flat_gru = gru_out.reshape(-1, self.config.gru_hidden_dim)
        policy_input = torch.cat([flat_states, flat_gru], dim=-1)

        dists = self.policy(policy_input)
        flat_actions = actions.reshape(-1)
        log_probs = dists.log_prob(flat_actions).reshape(batch, seq_len)
        entropy = dists.entropy().reshape(batch, seq_len)

        values = self.value_net(policy_input).reshape(batch, seq_len)

        return log_probs, values, entropy, weights

    def weight_smoothness_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Penalize rapid weight changes between consecutive timesteps.

        weights: (batch, seq_len, num_sub_rewards) -> scalar penalty.
        """
        if weights.shape[1] <= 1:
            return torch.tensor(0.0, device=weights.device)
        diff = weights[:, 1:, :] - weights[:, :-1, :]
        return (diff ** 2).mean()

    def get_optimizers(self, ppo_config: PPOConfig) -> dict[str, torch.optim.Adam]:
        return {
            "policy": torch.optim.Adam(self.policy.parameters(), lr=ppo_config.lr_policy),
            "value": torch.optim.Adam(self.value_net.parameters(), lr=ppo_config.lr_value),
            "gru": torch.optim.Adam(self.gru.parameters(), lr=ppo_config.lr_value),
            "weight": torch.optim.Adam(self.weight_net.parameters(), lr=ppo_config.lr_weight),
        }


class StaticWeightAgent(nn.Module):
    """Non-recurrent PPO agent with fixed sub-reward weights (Layer 1 baseline)."""

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.policy = PolicyNetwork(config.state_dim, config.action_dim, config.policy_hidden)
        self.value_net = ValueNetwork(config.state_dim, config.value_hidden)

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, value)."""
        dist = self.policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(state)
        return action, log_prob, value

    def evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (log_probs, values, entropy)."""
        dists = self.policy(states)
        log_probs = dists.log_prob(actions)
        entropy = dists.entropy()
        values = self.value_net(states)
        return log_probs, values, entropy

    def get_optimizers(self, ppo_config: PPOConfig) -> dict[str, torch.optim.Adam]:
        return {
            "policy": torch.optim.Adam(self.policy.parameters(), lr=ppo_config.lr_policy),
            "value": torch.optim.Adam(self.value_net.parameters(), lr=ppo_config.lr_value),
        }


class RecurrentStaticAgent(nn.Module):
    """Recurrent PPO agent with GRU in policy but fixed weights (Layer 1 baseline).

    This isolates the contribution of recurrence from dynamic weighting.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.gru = GRUHistoryEncoder(config.gru_input_dim, config.gru_hidden_dim)
        # Policy takes state + gru hidden
        self.policy = PolicyNetwork(
            config.state_dim + config.gru_hidden_dim, config.action_dim, config.policy_hidden
        )
        self.value_net = ValueNetwork(config.state_dim + config.gru_hidden_dim, config.value_hidden)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.gru.init_hidden(batch_size, device)

    def _build_gru_input(
        self, state: torch.Tensor, action: torch.Tensor, sub_rewards: torch.Tensor
    ) -> torch.Tensor:
        action_oh = torch.nn.functional.one_hot(action.long(), self.config.action_dim).float()
        return torch.cat([state, action_oh, sub_rewards], dim=-1)

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_sub_rewards: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, value, h_new)."""
        gru_input = self._build_gru_input(state, prev_action, prev_sub_rewards)
        gru_out, h_new = self.gru.forward_step(gru_input, h)

        policy_input = torch.cat([state, gru_out], dim=-1)
        dist = self.policy(policy_input)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(policy_input)

        return action, log_prob, value, h_new

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_sub_rewards: torch.Tensor,
        h_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (log_probs, values, entropy)."""
        batch, seq_len = states.shape[:2]

        gru_inputs = self._build_gru_input(
            states.reshape(-1, self.config.state_dim),
            prev_actions.reshape(-1),
            prev_sub_rewards.reshape(-1, self.config.num_sub_rewards),
        ).reshape(batch, seq_len, -1)

        gru_out, _ = self.gru(gru_inputs, h_init)

        flat_states = states.reshape(-1, self.config.state_dim)
        flat_gru = gru_out.reshape(-1, self.config.gru_hidden_dim)
        policy_input = torch.cat([flat_states, flat_gru], dim=-1)

        dists = self.policy(policy_input)
        flat_actions = actions.reshape(-1)
        log_probs = dists.log_prob(flat_actions).reshape(batch, seq_len)
        entropy = dists.entropy().reshape(batch, seq_len)
        values = self.value_net(policy_input).reshape(batch, seq_len)

        return log_probs, values, entropy

    def get_optimizers(self, ppo_config: PPOConfig) -> dict[str, torch.optim.Adam]:
        return {
            "policy": torch.optim.Adam(
                list(self.policy.parameters()) + list(self.gru.parameters()),
                lr=ppo_config.lr_policy,
            ),
            "value": torch.optim.Adam(self.value_net.parameters(), lr=ppo_config.lr_value),
        }
