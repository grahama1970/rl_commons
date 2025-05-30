"""RL Commons - Shared Reinforcement Learning Components"""

from .core.base import RLAgent, RLState, RLAction, RLReward
from .core.replay_buffer import ReplayBuffer
from .monitoring.tracker import RLTracker
from .algorithms.bandits import ContextualBandit, ThompsonSamplingBandit
from .algorithms.dqn import DQNAgent, DoubleDQNAgent, DQNetwork
from .algorithms.hierarchical import (
    Option,
    OptionPolicy,
    MetaController,
    HierarchicalRLAgent,
    HierarchicalReplayBuffer
)
from .algorithms.ppo import PPOAgent
from .algorithms.a3c import A3CAgent
from .integrations import ArangoDBOptimizer

__version__ = "0.1.0"

__all__ = [
    # Core components
    "RLAgent",
    "RLState", 
    "RLAction",
    "RLReward",
    "ReplayBuffer",
    "RLTracker",
    # Bandit algorithms
    "ContextualBandit",
    "ThompsonSamplingBandit",
    # DQN algorithms
    "DQNAgent",
    "DoubleDQNAgent",
    "DQNetwork",
    # Hierarchical RL
    "Option",
    "OptionPolicy",
    "MetaController",
    "HierarchicalRLAgent",
    "HierarchicalReplayBuffer",
    # PPO
    "PPOAgent",
    # A3C
    "A3CAgent",
    # Integrations
    "ArangoDBOptimizer"
]
