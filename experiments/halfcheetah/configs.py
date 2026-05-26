# configs.py
import dataclasses
from typing import List

@dataclasses.dataclass
class PPOConfig:
    """
    Arguments for HalfCheetah tasks.
    """
    # hidden dimension
    hidden_dims: tuple = (256, 256)
    leaky_relu_slope: float = 0.2
    
    # learning rates
    learning_rate: float = 0.0003 
    
    # PPO hyperparameters
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    
    env_steps_per_task: int = 1e6
    rollout_steps: int = 2048
    epochs: int = 10
    minibatch_size: int = 256