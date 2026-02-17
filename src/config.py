from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    grid_size: int = 10
    regime_switch_interval: int = 50
    max_episode_steps: int = 200
    hazard_penalty_full: float = -1.0
    hazard_penalty_reduced: float = -0.02
    step_cost_base: float = -0.01
    step_cost_ramp: float = -0.005
    goal_reward: float = 1.0
    num_hazard_cells: int = 20
    randomize_regime_per_episode: bool = False
    seed: int | None = None


@dataclass
class NetworkConfig:
    state_dim: int = 4
    action_dim: int = 4
    num_sub_rewards: int = 3
    gru_hidden_dim: int = 64
    gru_input_dim: int = 11  # state_dim + action_dim(onehot) + num_sub_rewards
    policy_hidden: int = 64
    value_hidden: int = 64
    weight_hidden: int = 32


@dataclass
class PPOConfig:
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    lr_weight: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    batch_size: int = 64
    rollout_length: int = 2048
    chunk_length: int = 16


@dataclass
class TrainConfig:
    total_timesteps: int = 500_000
    pretrain_timesteps: int = 100_000
    smoothness_lambda: float = 0.5
    weight_reward_coef: float = 0.05
    min_weight: float = 0.05
    eval_interval: int = 10_000
    eval_episodes: int = 20
    seed: int = 42
    use_wandb: bool = True
    wandb_project: str = "drd-gridworld"
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    net: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
