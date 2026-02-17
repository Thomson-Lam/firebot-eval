from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import ExperimentConfig
from src.env import RegimeSwitchingGridWorld
from src.networks import DRDAgent


def evaluate_agent(
    env: RegimeSwitchingGridWorld,
    agent: DRDAgent,
    num_episodes: int,
    device: torch.device,
) -> dict:
    """Run DRD agent greedily for evaluation. Returns metrics and trajectories."""
    agent.eval()
    episode_returns = []
    episode_steps_to_goal = []
    safety_violations = []
    weight_trajectories = []
    regime_trajectories = []

    for _ in range(num_episodes):
        obs, _info = env.reset()
        h = agent.init_hidden(1, device)
        prev_action = torch.zeros(1, dtype=torch.long, device=device)
        prev_sub_rewards = torch.zeros(1, 3, dtype=torch.float32, device=device)

        ep_return = 0.0
        ep_violations = 0
        ep_weights = []
        ep_regimes = []
        done = False

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                # Use greedy action (mode of policy)
                gru_input = agent._build_gru_input(state_t, prev_action, prev_sub_rewards)
                gru_out, h = agent.gru.forward_step(gru_input, h)
                weights = agent.weight_net(gru_out)
                dist = agent.policy(state_t)
                action = dist.probs.argmax(dim=-1)

            a = action.item()
            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            ep_return += reward
            if info.get("on_hazard", False):
                ep_violations += 1
            ep_weights.append(weights.squeeze(0).cpu().numpy())
            ep_regimes.append(info["regime"])

            prev_action = action
            prev_sub_rewards = torch.tensor(
                info["sub_rewards"], dtype=torch.float32, device=device
            ).unsqueeze(0)

        episode_returns.append(ep_return)
        safety_violations.append(ep_violations)
        if info.get("goal_reached", False):
            episode_steps_to_goal.append(info["steps_to_goal"])
        weight_trajectories.append(np.array(ep_weights))
        regime_trajectories.append(ep_regimes)

    agent.train()

    return {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_safety_violations": np.mean(safety_violations),
        "mean_steps_to_goal": np.mean(episode_steps_to_goal) if episode_steps_to_goal else float("inf"),
        "goal_rate": len(episode_steps_to_goal) / num_episodes,
        "weight_trajectories": weight_trajectories,
        "regime_trajectories": regime_trajectories,
    }


def train_regime_probe(
    agent: DRDAgent,
    env: RegimeSwitchingGridWorld,
    num_episodes: int,
    device: torch.device,
) -> float:
    """Train a linear probe on GRU hidden states to predict regime.

    Returns classification accuracy.
    """
    agent.eval()

    hidden_states = []
    regime_labels = []

    # Collect (h_t, regime) pairs
    for _ in range(num_episodes):
        obs, _info = env.reset()
        h = agent.init_hidden(1, device)
        prev_action = torch.zeros(1, dtype=torch.long, device=device)
        prev_sub_rewards = torch.zeros(1, 3, dtype=torch.float32, device=device)
        done = False

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                gru_input = agent._build_gru_input(state_t, prev_action, prev_sub_rewards)
                gru_out, h = agent.gru.forward_step(gru_input, h)

            hidden_states.append(gru_out.squeeze(0).cpu())
            regime_labels.append(1.0 if env.get_regime() == "A" else 0.0)

            with torch.no_grad():
                dist = agent.policy(state_t)
                action = dist.sample()

            a = action.item()
            obs, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            prev_action = action
            prev_sub_rewards = torch.tensor(
                info["sub_rewards"], dtype=torch.float32, device=device
            ).unsqueeze(0)

    agent.train()

    # Train linear probe
    X = torch.stack(hidden_states)  # (N, hidden_dim)
    y = torch.tensor(regime_labels, dtype=torch.float32)  # (N,)

    # 80/20 split
    n = len(X)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    X_train, X_test = X[perm[:split]], X[perm[split:]]
    y_train, y_test = y[perm[:split]], y[perm[split:]]

    probe = nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for _ in range(200):
        logits = probe(X_train).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        preds = (torch.sigmoid(probe(X_test).squeeze(-1)) > 0.5).float()
        accuracy = (preds == y_test).float().mean().item()

    return accuracy


def compute_pareto_metrics(
    results: dict[str, dict],
) -> dict[str, tuple[float, float]]:
    """Extract (safety_violations, time_to_goal) pairs for Pareto frontier."""
    pareto = {}
    for name, r in results.items():
        violations = r.get("mean_safety_violations", 0)
        time_to_goal = r.get("mean_steps_to_goal", float("inf"))
        pareto[name] = (violations, time_to_goal)
    return pareto
