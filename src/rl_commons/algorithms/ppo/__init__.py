"""Proximal Policy Optimization (PPO) algorithm for continuous control

Module: __init__.py
"""

from .ppo import PPOAgent
from .entropy_aware_ppo import EntropyAwarePPO
from .kl_cov_ppo import KLCovPPO

__all__ = ['PPOAgent', 'EntropyAwarePPO', 'KLCovPPO']
