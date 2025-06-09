"""
Module: optimization_agent.py
Description: General optimization agent

External Dependencies:
- None (standard library only)
"""

from typing import List, Dict, Any
from .contextual_bandit import ContextualBandit


class OptimizationAgent:
    """General purpose optimization agent using RL"""
    
    def __init__(self, name: str, actions: List[str], features: List[str]):
        """Initialize optimization agent"""
        self.name = name
        self.bandit = ContextualBandit(
            actions=actions,
            context_features=features,
            exploration_rate=0.15
        )
    
    def optimize(self, context: Dict[str, Any]) -> str:
        """Make optimized decision"""
        return self.bandit.select_action(context)
    
    def learn(self, action: str, reward: float, context: Dict[str, Any]):
        """Learn from outcome"""
        self.bandit.update(action, reward, context)
