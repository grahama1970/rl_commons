"""
Module: base_morl.py
Purpose: Base classes for Multi-Objective Reinforcement Learning

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.morl import MORLBase, MultiObjectiveReward
>>> reward = MultiObjectiveReward([0.5, 0.8, 0.3])
>>> morl = MORLBase(num_objectives=3)
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from collections import defaultdict

from ...core.base import RLState, RLAction, RLReward, RLAgent


@dataclass
class MultiObjectiveReward:
    """Container for multi-objective rewards."""
    values: np.ndarray  # Shape: (num_objectives,)
    objectives: Optional[List[str]] = None  # Names of objectives
    
    def __post_init__(self):
        """Validate reward structure."""
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values)
        
        if self.objectives is not None:
            assert len(self.objectives) == len(self.values), \
                f"Number of objectives {len(self.objectives)} != reward dimensions {len(self.values)}"
    
    def scalarize(self, weights: np.ndarray) -> float:
        """Convert to scalar reward using weights.
        
        Args:
            weights: Weight vector for objectives
            
        Returns:
            Scalar reward
        """
        assert len(weights) == len(self.values), "Weight dimension mismatch"
        return float(np.dot(self.values, weights))
    
    def dominates(self, other: 'MultiObjectiveReward') -> bool:
        """Check if this reward Pareto-dominates another.
        
        Args:
            other: Another multi-objective reward
            
        Returns:
            True if this dominates other
        """
        return (self.values >= other.values).all() and (self.values > other.values).any()
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor."""
        return torch.FloatTensor(self.values)


class ParetoFront:
    """Maintains Pareto-optimal solutions."""
    
    def __init__(self, num_objectives: int):
        """Initialize Pareto front.
        
        Args:
            num_objectives: Number of objectives
        """
        self.num_objectives = num_objectives
        self.solutions = []  # List of (state, action, reward) tuples
        self.rewards = []    # List of MultiObjectiveReward
    
    def update(
        self,
        state: RLState,
        action: RLAction,
        reward: MultiObjectiveReward
    ) -> bool:
        """Update Pareto front with new solution.
        
        Args:
            state: State where action was taken
            action: Action taken
            reward: Multi-objective reward received
            
        Returns:
            True if solution was added to front
        """
        # Check if dominated by existing solutions
        for existing_reward in self.rewards:
            if existing_reward.dominates(reward):
                return False
        
        # Remove solutions dominated by new one
        non_dominated = []
        non_dominated_rewards = []
        
        for i, existing_reward in enumerate(self.rewards):
            if not reward.dominates(existing_reward):
                non_dominated.append(self.solutions[i])
                non_dominated_rewards.append(existing_reward)
        
        # Add new solution
        non_dominated.append((state, action, reward))
        non_dominated_rewards.append(reward)
        
        self.solutions = non_dominated
        self.rewards = non_dominated_rewards
        
        return True
    
    def get_coverage(self) -> float:
        """Compute hypervolume coverage of Pareto front.
        
        Returns:
            Hypervolume metric
        """
        if not self.rewards:
            return 0.0
        
        # Simple approximation - better algorithms exist
        reward_matrix = np.array([r.values for r in self.rewards])
        
        # Normalize to [0, 1]
        min_vals = reward_matrix.min(axis=0)
        max_vals = reward_matrix.max(axis=0)
        normalized = (reward_matrix - min_vals) / (max_vals - min_vals + 1e-8)
        
        # Approximate hypervolume
        volume = 0.0
        for point in normalized:
            # Volume of box from origin to point
            volume += np.prod(point)
        
        return volume / len(self.rewards)
    
    def plot(self, objective_indices: Tuple[int, int] = (0, 1)):
        """Plot 2D projection of Pareto front.
        
        Args:
            objective_indices: Which two objectives to plot
        """
        if not self.rewards:
            print("No solutions in Pareto front")
            return
        
        i, j = objective_indices
        x_vals = [r.values[i] for r in self.rewards]
        y_vals = [r.values[j] for r in self.rewards]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(x_vals, y_vals, s=100, alpha=0.7)
        plt.xlabel(f"Objective {i}")
        plt.ylabel(f"Objective {j}")
        plt.title("Pareto Front")
        plt.grid(True, alpha=0.3)
        plt.show()


class MORLBase(ABC):
    """Base class for Multi-Objective RL algorithms."""
    
    def __init__(
        self,
        num_objectives: int,
        state_dim: int,
        action_dim: int,
        objective_names: Optional[List[str]] = None,
        device: str = None
    ):
        """Initialize MORL algorithm.
        
        Args:
            num_objectives: Number of objectives to optimize
            state_dim: State space dimension
            action_dim: Action space dimension
            objective_names: Names of objectives
            device: Device for computation
        """
        self.num_objectives = num_objectives
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.objective_names = objective_names or [f"Obj_{i}" for i in range(num_objectives)]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pareto front tracking
        self.pareto_front = ParetoFront(num_objectives)
        
        # Training history
        self.training_history = defaultdict(list)
    
    @abstractmethod
    def select_action(
        self,
        state: RLState,
        preference: Optional[np.ndarray] = None
    ) -> RLAction:
        """Select action based on state and preference.
        
        Args:
            state: Current state
            preference: Preference weights for objectives
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(
        self,
        state: RLState,
        action: RLAction,
        reward: MultiObjectiveReward,
        next_state: RLState,
        done: bool
    ):
        """Update policy with multi-objective feedback.
        
        Args:
            state: Current state
            action: Action taken
            reward: Multi-objective reward
            next_state: Next state
            done: Episode done flag
        """
        pass
    
    @abstractmethod
    def train(
        self,
        env: Any,
        num_episodes: int,
        preference_sampling: str = 'uniform'
    ) -> Dict[str, Any]:
        """Train MORL agent.
        
        Args:
            env: Environment with multi-objective rewards
            num_episodes: Number of training episodes
            preference_sampling: How to sample preferences ('uniform', 'corner', 'adaptive')
            
        Returns:
            Training results
        """
        pass
    
    def sample_preference(self, method: str = 'uniform') -> np.ndarray:
        """Sample preference weights.
        
        Args:
            method: Sampling method
            
        Returns:
            Preference weight vector
        """
        if method == 'uniform':
            # Random weights that sum to 1
            weights = np.random.dirichlet(np.ones(self.num_objectives))
        elif method == 'corner':
            # Focus on corner cases (single objectives)
            if np.random.random() < 0.5:
                # Pure single objective
                weights = np.zeros(self.num_objectives)
                weights[np.random.randint(self.num_objectives)] = 1.0
            else:
                # Uniform
                weights = np.random.dirichlet(np.ones(self.num_objectives))
        elif method == 'adaptive':
            # Adaptive based on Pareto front coverage
            # Simple version - sample where coverage is low
            weights = np.random.dirichlet(np.ones(self.num_objectives) * 0.5)
        else:
            raise ValueError(f"Unknown preference sampling method: {method}")
        
        return weights
    
    def evaluate_policy(
        self,
        env: Any,
        preference: np.ndarray,
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate policy for given preference.
        
        Args:
            env: Environment
            preference: Preference weights
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        total_rewards = np.zeros(self.num_objectives)
        episode_lengths = []
        
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_length = 0
            episode_rewards = np.zeros(self.num_objectives)
            
            while not done:
                action = self.select_action(RLState(features=state), preference)
                next_state, reward, done, info = env.step(action.action_id)
                
                # Extract multi-objective reward
                if isinstance(reward, MultiObjectiveReward):
                    mo_reward = reward
                elif isinstance(reward, (list, tuple, np.ndarray)):
                    mo_reward = MultiObjectiveReward(reward)
                else:
                    # Single objective - convert
                    mo_reward = MultiObjectiveReward([reward] * self.num_objectives)
                
                episode_rewards += mo_reward.values
                state = next_state
                episode_length += 1
            
            total_rewards += episode_rewards
            episode_lengths.append(episode_length)
        
        avg_rewards = total_rewards / num_episodes
        
        return {
            'avg_rewards': avg_rewards.tolist(),
            'scalarized_reward': float(np.dot(avg_rewards, preference)),
            'avg_episode_length': np.mean(episode_lengths),
            'preference': preference.tolist()
        }
    
    def save(self, path: str):
        """Save MORL model."""
        torch.save({
            'num_objectives': self.num_objectives,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'objective_names': self.objective_names,
            'pareto_front': self.pareto_front,
            'training_history': dict(self.training_history)
        }, path)
    
    def load(self, path: str):
        """Load MORL model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.num_objectives = checkpoint['num_objectives']
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.objective_names = checkpoint['objective_names']
        self.pareto_front = checkpoint['pareto_front']
        self.training_history = defaultdict(list, checkpoint['training_history'])


if __name__ == "__main__":
    # Validation test
    # Test MultiObjectiveReward
    reward1 = MultiObjectiveReward([0.8, 0.2, 0.5], objectives=['latency', 'throughput', 'cost'])
    reward2 = MultiObjectiveReward([0.6, 0.4, 0.7])
    
    # Test dominance
    reward3 = MultiObjectiveReward([0.9, 0.3, 0.6])
    assert reward3.dominates(reward1), "Should dominate"
    assert not reward1.dominates(reward3), "Should not dominate"
    
    # Test scalarization
    weights = np.array([0.5, 0.3, 0.2])
    scalar = reward1.scalarize(weights)
    expected = 0.8*0.5 + 0.2*0.3 + 0.5*0.2
    assert abs(scalar - expected) < 1e-6, f"Wrong scalarization: {scalar} vs {expected}"
    
    # Test Pareto front
    front = ParetoFront(num_objectives=3)
    
    # Add solutions
    state = RLState(features=[0, 0])
    action = RLAction(action_type='discrete', action_id=0)
    
    assert front.update(state, action, reward1), "Should add first solution"
    assert front.update(state, action, reward2), "Should add non-dominated solution"
    assert not front.update(state, action, reward1), "Should not re-add same solution"
    assert front.update(state, action, reward3), "Should add dominating solution"
    
    # Check front size
    assert len(front.solutions) == 2, f"Wrong front size: {len(front.solutions)}"
    
    # Test coverage
    coverage = front.get_coverage()
    assert 0 <= coverage <= 1, f"Invalid coverage: {coverage}"
    
    print(" MORL base validation passed")