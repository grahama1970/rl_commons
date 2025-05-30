"""
Module: a3c/__init__.py
Purpose: Asynchronous Advantage Actor-Critic algorithm for distributed RL

External Dependencies:
- None

Example Usage:
>>> from graham_rl_commons.algorithms.a3c import A3CAgent
>>> agent = A3CAgent(state_dim=10, action_dim=4)
"""

from .a3c import A3CAgent, A3CWorker, SharedAdam, A3CConfig, ActorCriticNetwork

__all__ = ['A3CAgent', 'A3CWorker', 'SharedAdam', 'A3CConfig', 'ActorCriticNetwork']