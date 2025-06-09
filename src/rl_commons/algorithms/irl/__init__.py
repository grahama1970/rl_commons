"""
Inverse Reinforcement Learning algorithms for learning from demonstrations.
# Module: __init__.py
# Description: Package initialization and exports

This module provides IRL algorithms to infer reward functions from expert demonstrations,
useful for learning from human behavior or existing module interactions.
"""

from .base_irl import IRLBase, Demonstration, RewardFunction
from .max_entropy_irl import MaxEntropyIRL
from .behavioral_cloning import BehavioralCloning
from .gail import GAIL

__all__ = [
    'IRLBase',
    'Demonstration', 
    'RewardFunction',
    'MaxEntropyIRL',
    'BehavioralCloning',
    'GAIL'
]