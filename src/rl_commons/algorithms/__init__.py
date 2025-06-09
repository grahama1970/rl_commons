"""RL algorithms available in RL Commons

Module: __init__.py
"""

from .dqn.vanilla_dqn import DQNAgent
from .bandits.contextual import ContextualBandit
from .hierarchical import HierarchicalRLAgent
from .ppo.ppo import PPOAgent
from .a3c.a3c import A3CAgent

# Curriculum Learning
from .curriculum import (
    CurriculumScheduler,
    Task,
    TaskDifficulty,
    PerformanceTracker,
    AutomaticCurriculum,
    ProgressiveCurriculum,
    AdaptiveCurriculum,
    MetaCurriculum
)

__all__ = [
    'DQNAgent',
    'ContextualBandit', 
    'HierarchicalRLAgent',
    'PPOAgent',
    'A3CAgent',
    # Curriculum Learning
    'CurriculumScheduler',
    'Task',
    'TaskDifficulty',
    'PerformanceTracker',
    'AutomaticCurriculum',
    'ProgressiveCurriculum',
    'AdaptiveCurriculum',
    'MetaCurriculum'
]
