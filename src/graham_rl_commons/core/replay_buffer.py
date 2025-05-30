"""Experience replay buffer for RL algorithms"""

from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import random
from dataclasses import dataclass

from .base import RLState, RLAction, RLReward


@dataclass
class Experience:
    """Single experience tuple"""
    state: RLState
    action: RLAction
    reward: RLReward
    next_state: RLState
    done: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
            "reward": self.reward.to_dict(),
            "next_state": self.next_state.to_dict(),
            "done": self.done
        }


class ReplayBuffer:
    """Standard experience replay buffer"""
    
    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(self, 
             state: RLState, 
             action: RLAction, 
             reward: RLReward, 
             next_state: RLState, 
             done: bool) -> None:
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough experiences in buffer: {len(self.buffer)} < {batch_size}")
        
        return random.sample(self.buffer, batch_size)
    
    def sample_tensors(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch and return as numpy arrays
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = self.sample(batch_size)
        
        states = np.array([exp.state.features for exp in batch])
        actions = np.array([exp.action.action_id for exp in batch])
        rewards = np.array([exp.reward.value for exp in batch])
        next_states = np.array([exp.next_state.features for exp in batch])
        dones = np.array([exp.done for exp in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples"""
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """Clear all experiences"""
        self.buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0
            }
        
        rewards = [exp.reward.value for exp in self.buffer]
        
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "avg_reward": np.mean(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "done_ratio": sum(exp.done for exp in self.buffer) / len(self.buffer)
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer"""
    
    def __init__(self, 
                 capacity: int, 
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum number of experiences
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: Beta increment per sample
            seed: Random seed
        """
        super().__init__(capacity, seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, 
             state: RLState, 
             action: RLAction, 
             reward: RLReward, 
             next_state: RLState, 
             done: bool) -> None:
        """Add experience with maximum priority"""
        super().push(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """
        Sample batch with priorities
        
        Returns:
            Tuple of (experiences, importance_weights, indices)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough experiences in buffer: {len(self.buffer)} < {batch_size}")
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, weights, indices.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon for numerical stability
            self.max_priority = max(self.max_priority, priority)
