"""
Meta-Learning algorithms for rapid adaptation to new tasks.
# Module: __init__.py
# Description: Package initialization and exports

This module provides Model-Agnostic Meta-Learning (MAML) and related algorithms
for few-shot learning in RL contexts.
"""

from .maml import MAML, MAMLAgent
from .reptile import Reptile
from .task_distribution import (
    TaskDistribution, TaskSampler, Task,
    SyntheticTaskDistribution, ModuleTaskDistribution
)

__all__ = [
    'MAML',
    'MAMLAgent',
    'Reptile',
    'TaskDistribution',
    'TaskSampler',
    'Task',
    'SyntheticTaskDistribution',
    'ModuleTaskDistribution'
]