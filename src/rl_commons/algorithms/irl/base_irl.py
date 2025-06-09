"""
Module: base_irl.py
Purpose: Base classes for Inverse Reinforcement Learning

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.irl import IRLBase, Demonstration
>>> demo = Demonstration(states, actions, rewards)
>>> irl = IRLBase()
>>> reward_fn = irl.learn_reward(demonstrations)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ...core.base import RLState, RLAction, RLReward


@dataclass
class Demonstration:
    """Container for expert demonstration data."""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: Optional[torch.Tensor] = None
    next_states: Optional[torch.Tensor] = None
    dones: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Validate demonstration data."""
        assert len(self.states) == len(self.actions), "States and actions must have same length"
        if self.rewards is not None:
            assert len(self.rewards) == len(self.states), "Rewards must match states length"
        if self.next_states is not None:
            assert len(self.next_states) == len(self.states), "Next states must match states length"
        if self.dones is not None:
            assert len(self.dones) == len(self.states), "Dones must match states length"
    
    def to_device(self, device: torch.device) -> 'Demonstration':
        """Move demonstration to device."""
        return Demonstration(
            states=self.states.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device) if self.rewards is not None else None,
            next_states=self.next_states.to(device) if self.next_states is not None else None,
            dones=self.dones.to(device) if self.dones is not None else None
        )
    
    def split(self, ratio: float = 0.8) -> Tuple['Demonstration', 'Demonstration']:
        """Split demonstration into train/test sets."""
        n = len(self.states)
        split_idx = int(n * ratio)
        
        train = Demonstration(
            states=self.states[:split_idx],
            actions=self.actions[:split_idx],
            rewards=self.rewards[:split_idx] if self.rewards is not None else None,
            next_states=self.next_states[:split_idx] if self.next_states is not None else None,
            dones=self.dones[:split_idx] if self.dones is not None else None
        )
        
        test = Demonstration(
            states=self.states[split_idx:],
            actions=self.actions[split_idx:],
            rewards=self.rewards[split_idx:] if self.rewards is not None else None,
            next_states=self.next_states[split_idx:] if self.next_states is not None else None,
            dones=self.dones[split_idx:] if self.dones is not None else None
        )
        
        return train, test


class RewardFunction(nn.Module):
    """Neural network reward function."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        """Initialize reward function.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension (optional, for state-action rewards)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network
        input_dim = state_dim + (action_dim if action_dim else 0)
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))  # Single reward output
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reward for states (and optionally actions).
        
        Args:
            states: State tensor [batch_size, state_dim]
            actions: Action tensor [batch_size, action_dim] (optional)
            
        Returns:
            Reward tensor [batch_size, 1]
        """
        if actions is not None and self.action_dim is not None:
            # Concatenate states and actions
            inputs = torch.cat([states, actions], dim=-1)
        else:
            inputs = states
        
        return self.network(inputs)
    
    def get_rewards(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get rewards without gradient computation."""
        with torch.no_grad():
            return self.forward(states, actions).squeeze(-1)


class IRLBase(ABC):
    """Base class for Inverse Reinforcement Learning algorithms."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        reward_hidden_dim: int = 128,
        device: str = None
    ):
        """Initialize IRL algorithm.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            reward_hidden_dim: Hidden dimension for reward network
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize reward function
        self.reward_function = RewardFunction(
            state_dim,
            action_dim,
            hidden_dim=reward_hidden_dim
        ).to(self.device)
        
        # Training history
        self.training_history = {
            'losses': [],
            'rewards': [],
            'metrics': []
        }
    
    @abstractmethod
    def learn_reward(
        self,
        demonstrations: List[Demonstration],
        num_iterations: int = 1000
    ) -> RewardFunction:
        """Learn reward function from demonstrations.
        
        Args:
            demonstrations: List of expert demonstrations
            num_iterations: Number of training iterations
            
        Returns:
            Learned reward function
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        demonstration: Demonstration,
        policy: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Evaluate learned reward function.
        
        Args:
            demonstration: Test demonstration
            policy: Policy to evaluate (optional)
            
        Returns:
            Evaluation metrics
        """
        pass
    
    def save(self, path: str):
        """Save IRL model."""
        torch.save({
            'reward_function': self.reward_function.state_dict(),
            'training_history': self.training_history,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
    
    def load(self, path: str):
        """Load IRL model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.reward_function.load_state_dict(checkpoint['reward_function'])
        self.training_history = checkpoint['training_history']
    
    def compute_feature_expectations(
        self,
        demonstrations: List[Demonstration],
        feature_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """Compute feature expectations from demonstrations.
        
        Args:
            demonstrations: List of demonstrations
            feature_fn: Feature function (uses states if None)
            
        Returns:
            Feature expectations
        """
        features = []
        
        for demo in demonstrations:
            demo = demo.to_device(self.device)
            
            if feature_fn is None:
                # Use raw states as features
                features.append(demo.states)
            else:
                # Apply feature function
                features.append(feature_fn(demo.states))
        
        # Compute expectations
        all_features = torch.cat(features, dim=0)
        return all_features.mean(dim=0)


if __name__ == "__main__":
    # Validation test
    # Create dummy demonstration
    states = torch.randn(100, 8)
    actions = torch.randint(0, 4, (100,))
    rewards = torch.randn(100)
    
    demo = Demonstration(states, actions, rewards)
    assert len(demo.states) == 100, "Wrong demonstration length"
    
    # Test train/test split
    train_demo, test_demo = demo.split(ratio=0.8)
    assert len(train_demo.states) == 80, "Wrong train split"
    assert len(test_demo.states) == 20, "Wrong test split"
    
    # Test reward function
    reward_fn = RewardFunction(state_dim=8, action_dim=4)
    rewards = reward_fn.get_rewards(states, actions)
    assert rewards.shape == (100,), f"Wrong reward shape: {rewards.shape}"
    
    print(" IRL base validation passed")