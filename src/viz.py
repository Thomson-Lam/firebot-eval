from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no GUI window

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_weight_trajectories(
    weight_traj: np.ndarray,
    regime_traj: list[str],
    save_path: str | None = None,
    title: str = "Dynamic Weight Trajectories",
) -> None:
    """Plot w_progress, w_safety, w_efficiency over time with regime shading."""
    T = len(weight_traj)
    fig, ax = plt.subplots(figsize=(14, 5))

    # Shade regime backgrounds
    regime_colors = {"A": "#ffcccc", "B": "#cce5ff"}
    start = 0
    current = regime_traj[0]
    for t in range(1, T):
        if regime_traj[t] != current or t == T - 1:
            end = t if regime_traj[t] != current else t + 1
            ax.axvspan(start, end, alpha=0.3, color=regime_colors.get(current, "#eeeeee"))
            start = t
            current = regime_traj[t]

    labels = ["Progress", "Safety", "Efficiency"]
    colors = ["#2196F3", "#F44336", "#4CAF50"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(weight_traj[:, i], label=label, color=color, linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")

    # Regime legend
    patches = [
        mpatches.Patch(color="#ffcccc", alpha=0.3, label="Regime A (Minefield)"),
        mpatches.Patch(color="#cce5ff", alpha=0.3, label="Regime B (Time Pressure)"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + patches, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_learning_curves(
    metrics_dict: dict[str, list[float]],
    window: int = 50,
    save_path: str | None = None,
    title: str = "Learning Curves",
) -> None:
    """Plot smoothed return curves for multiple agents."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, returns in metrics_dict.items():
        if len(returns) < window:
            ax.plot(returns, label=label, alpha=0.8)
            continue
        # Smoothed with rolling mean
        smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label=label, alpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_regime_specific_returns(
    metrics_dict: dict[str, tuple[float, float]],
    save_path: str | None = None,
    title: str = "Per-Regime Returns",
) -> None:
    """Bar chart of regime A vs B returns for each method."""
    labels = list(metrics_dict.keys())
    regime_a = [v[0] for v in metrics_dict.values()]
    regime_b = [v[1] for v in metrics_dict.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, regime_a, width, label="Regime A (Minefield)", color="#F44336", alpha=0.7)
    ax.bar(x + width / 2, regime_b, width, label="Regime B (Time Pressure)", color="#2196F3", alpha=0.7)

    ax.set_xlabel("Method")
    ax.set_ylabel("Mean Return")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_pareto_frontier(
    pareto_data: dict[str, tuple[float, float]],
    save_path: str | None = None,
    title: str = "Pareto Frontier: Safety vs Efficiency",
) -> None:
    """Scatter plot of (safety violations, time to goal) with Pareto frontier."""
    fig, ax = plt.subplots(figsize=(8, 6))

    names = list(pareto_data.keys())
    violations = [pareto_data[n][0] for n in names]
    times = [pareto_data[n][1] for n in names]

    ax.scatter(violations, times, s=100, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (violations[i], times[i]), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)

    # Draw Pareto frontier
    points = sorted(zip(violations, times))
    pareto_front = [points[0]]
    for p in points[1:]:
        if p[1] <= pareto_front[-1][1]:
            pareto_front.append(p)
    if len(pareto_front) > 1:
        pf_x, pf_y = zip(*pareto_front)
        ax.plot(pf_x, pf_y, "r--", alpha=0.5, label="Pareto frontier")

    ax.set_xlabel("Safety Violations (lower is better)")
    ax.set_ylabel("Steps to Goal (lower is better)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_ablation_heatmap(
    sweep_results: dict[str, dict],
    param_name: str,
    metric_name: str = "final_mean_return",
    save_path: str | None = None,
) -> None:
    """Bar chart for ablation sweep results."""
    labels = []
    values = []
    for name, result in sweep_results.items():
        labels.append(str(result.get("param", name)))
        values.append(result.get(metric_name, 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(labels)), values, color="#4CAF50", alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Ablation: {param_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
