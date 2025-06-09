"""
Multi-Objective Reinforcement Learning algorithms.
# Module: __init__.py
# Description: Package initialization and exports

This module provides MORL algorithms for balancing multiple competing objectives,
essential for module orchestration with diverse performance metrics.
"""

from .base_morl import MORLBase, MultiObjectiveReward, ParetoFront
from .weighted_sum import WeightedSumMORL
from .pareto_q_learning import ParetoQLearning
from .mo_ppo import MultiObjectivePPO
from .entropy_aware_morl import EntropyAwareMORL

__all__ = [
    'MORLBase',
    'MultiObjectiveReward',
    'ParetoFront',
    'WeightedSumMORL',
    'ParetoQLearning',
    'MultiObjectivePPO',
    'EntropyAwareMORL'
]