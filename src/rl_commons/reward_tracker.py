"""
Module: reward_tracker.py
Description: Track and analyze rewards

External Dependencies:
- None (standard library only)
"""

from typing import Dict, List, Tuple
from collections import defaultdict, deque


class RewardTracker:
    """Track rewards and performance metrics"""
    
    def __init__(self, window_size: int = 100):
        """Initialize reward tracker"""
        self.window_size = window_size
        self.rewards = defaultdict(lambda: deque(maxlen=window_size))
        self.total_rewards = defaultdict(float)
        self.counts = defaultdict(int)
    
    def add_reward(self, action: str, reward: float):
        """Add a reward observation"""
        self.rewards[action].append(reward)
        self.total_rewards[action] += reward
        self.counts[action] += 1
    
    def get_average_reward(self, action: str) -> float:
        """Get average reward for an action"""
        if self.counts[action] == 0:
            return 0.0
        return self.total_rewards[action] / self.counts[action]
    
    def get_recent_average(self, action: str) -> float:
        """Get recent average reward"""
        recent = self.rewards[action]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)
