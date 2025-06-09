"""Meta-learning algorithms for reinforcement learning.

This module provides implementations of various meta-learning algorithms
that can quickly adapt to new tasks.
"""

from .maml import MAML, MAMLAgent
from .reptile import Reptile, ReptileAgent
from .task_distribution import TaskDistribution, TaskSampler

__all__ = [
    "MAML",
    "MAMLAgent", 
    "Reptile",
    "ReptileAgent",
    "TaskDistribution",
    "TaskSampler",
]
