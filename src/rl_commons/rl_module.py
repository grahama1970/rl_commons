"""RL Commons module"""

import random
from typing import List, Dict, Any
from collections import defaultdict


class ContextualBandit:
    """Contextual multi-armed bandit"""
    
    def __init__(self, actions: List[str], context_features: List[str], exploration_rate: float = 0.1):
        if not actions:
            raise ValueError("Actions list cannot be empty")
        
        if exploration_rate < 0 or exploration_rate > 1:
            raise ValueError(f"Exploration rate must be between 0 and 1, got {exploration_rate}")
        
        self.actions = actions
        self.context_features = context_features
        self.exploration_rate = exploration_rate
        self.action_values = defaultdict(float)
        self.action_counts = defaultdict(int)
        self.last_was_exploration = False
        self.total_decisions = 0
    
    def select_action(self, context: Dict[str, Any]) -> str:
        """Select an action"""
        self.total_decisions += 1
        
        if random.random() < self.exploration_rate:
            self.last_was_exploration = True
            return random.choice(self.actions)
        
        self.last_was_exploration = False
        
        # Choose best action
        if self.action_counts:
            best_action = max(
                self.actions,
                key=lambda a: self.action_values[a] if self.action_counts[a] > 0 else 0
            )
            return best_action
        
        return random.choice(self.actions)
    
    def update(self, action: str, reward: float, context: Dict[str, Any] = None):
        """Update action values"""
        if action not in self.actions:
            return
        
        self.action_counts[action] += 1
        count = self.action_counts[action]
        current_value = self.action_values[action]
        
        # Incremental average
        self.action_values[action] = current_value + (reward - current_value) / count


class OptimizationAgent:
    """Optimization agent using RL"""
    
    def __init__(self, name: str, actions: List[str], features: List[str]):
        self.name = name
        self.bandit = ContextualBandit(actions, features, 0.15)
    
    def optimize(self, context: Dict[str, Any]) -> str:
        return self.bandit.select_action(context)
    
    def learn(self, action: str, reward: float, context: Dict[str, Any]):
        self.bandit.update(action, reward, context)
