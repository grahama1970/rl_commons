"""
Module: entropy_aware_morl.py
Purpose: Multi-objective RL with entropy as implicit objective

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.morl import EntropyAwareMORL
>>> morl = EntropyAwareMORL(num_objectives=2, state_dim=8, action_dim=4)
>>> morl.train(env, num_episodes=1000)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from collections import deque

from .mo_ppo import MultiObjectivePPO, MOActorCritic
from .base_morl import MultiObjectiveReward
from ...core.base import RLState, RLAction
from ...monitoring.entropy_tracker import EntropyTracker

logger = logging.getLogger(__name__)


class EntropyAwareMOActorCritic(MOActorCritic):
    """Extended Actor-Critic that tracks and reports entropy."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add entropy tracking
        self.entropy_history = deque(maxlen=100)
    
    def forward_with_entropy(self, states: torch.Tensor, preferences: torch.Tensor) -> Tuple[Any, torch.Tensor, float]:
        """Forward pass with entropy calculation.
        
        Returns:
            Action distribution, values, and entropy
        """
        action_dist, values = self.forward(states, preferences)
        entropy = action_dist.entropy().mean()
        self.entropy_history.append(entropy.item())
        return action_dist, values, entropy


class EntropyAwareMORL(MultiObjectivePPO):
    """
    Multi-Objective RL that treats entropy as an implicit objective.
    
    Automatically balances task objectives with entropy preservation to prevent
    policy collapse and maintain exploration.
    """
    
    def __init__(
        self,
        num_objectives: int,
        state_dim: int,
        action_dim: int,
        entropy_weight: float = 0.1,
        min_entropy: float = 0.5,
        adaptive_entropy_weight: bool = True,
        entropy_target: float = 1.0,
        pareto_include_entropy: bool = True,
        **kwargs
    ):
        """Initialize entropy-aware MORL.
        
        Args:
            num_objectives: Number of task objectives (not including entropy)
            state_dim: State dimension
            action_dim: Action dimension
            entropy_weight: Initial weight for entropy objective
            min_entropy: Minimum acceptable entropy
            adaptive_entropy_weight: Whether to adapt entropy weight
            entropy_target: Target entropy level
            pareto_include_entropy: Include entropy in Pareto front
            **kwargs: Additional arguments for MultiObjectivePPO
        """
        # Entropy is added as an implicit objective
        self.explicit_objectives = num_objectives
        self.entropy_objective_idx = num_objectives  # Last objective
        
        # Initialize with extended objectives
        super().__init__(
            num_objectives=num_objectives + 1,  # +1 for entropy
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs
        )
        
        # Entropy-specific configuration
        self.entropy_weight = entropy_weight
        self.min_entropy = min_entropy
        self.adaptive_entropy_weight = adaptive_entropy_weight
        self.entropy_target = entropy_target
        self.pareto_include_entropy = pareto_include_entropy
        
        # Replace actor-critic with entropy-aware version
        self.ac_network = EntropyAwareMOActorCritic(
            state_dim, action_dim, self.num_objectives,
            kwargs.get('hidden_dim', 128),
            kwargs.get('continuous_actions', False)
        ).to(self.device)
        
        # Re-initialize optimizer with new network
        self.optimizer = optim.Adam(
            self.ac_network.parameters(),
            lr=self.learning_rate
        )
        
        # Entropy tracker
        self.entropy_tracker = EntropyTracker(
            window_size=100,
            collapse_threshold=min_entropy,
            min_healthy_entropy=min_entropy
        )
        
        # Entropy adaptation
        self.entropy_weight_history = deque(maxlen=100)
        self.entropy_weight_history.append(entropy_weight)
        
        logger.info(f"Initialized EntropyAwareMORL with {self.explicit_objectives} task objectives + entropy")
    
    def _extend_preference_with_entropy(self, preference: np.ndarray) -> np.ndarray:
        """Add entropy weight to preference vector.
        
        Args:
            preference: Task preference weights
            
        Returns:
            Extended preference including entropy weight
        """
        if len(preference) == self.explicit_objectives:
            # Normalize task preferences to sum to (1 - entropy_weight)
            task_sum = preference.sum()
            if task_sum > 0:
                task_preferences = preference * (1 - self.entropy_weight) / task_sum
            else:
                task_preferences = np.ones(self.explicit_objectives) * (1 - self.entropy_weight) / self.explicit_objectives
            
            # Add entropy weight
            extended = np.zeros(self.num_objectives)
            extended[:self.explicit_objectives] = task_preferences
            extended[self.entropy_objective_idx] = self.entropy_weight
            
            return extended
        else:
            # Already extended
            return preference
    
    def select_action(
        self,
        state: RLState,
        preference: Optional[np.ndarray] = None
    ) -> RLAction:
        """Select action with entropy awareness.
        
        Args:
            state: Current state
            preference: Task preference weights (entropy added automatically)
            
        Returns:
            Selected action
        """
        if preference is None:
            preference = np.ones(self.explicit_objectives) / self.explicit_objectives
        
        # Extend preference with entropy
        extended_preference = self._extend_preference_with_entropy(preference)
        
        # Use parent's action selection with extended preference
        return super().select_action(state, extended_preference)
    
    def update(
        self,
        state: RLState,
        action: RLAction,
        reward: MultiObjectiveReward,
        next_state: RLState,
        done: bool,
        preference: Optional[np.ndarray] = None
    ):
        """Update with entropy as objective.
        
        Args:
            state: Current state
            action: Action taken
            reward: Multi-objective reward (task objectives only)
            next_state: Next state
            done: Episode done flag
            preference: Task preference weights
        """
        # Calculate current entropy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
            pref_tensor = torch.FloatTensor(
                self._extend_preference_with_entropy(
                    preference if preference is not None else self.current_preference
                )
            ).unsqueeze(0).to(self.device)
            
            action_dist, _, entropy = self.ac_network.forward_with_entropy(state_tensor, pref_tensor)
            current_entropy = entropy.item()
        
        # Update entropy tracker
        metrics = self.entropy_tracker.update(current_entropy)
        
        # Adapt entropy weight if needed
        if self.adaptive_entropy_weight:
            self._adapt_entropy_weight(metrics)
        
        # Extend reward with entropy bonus
        if hasattr(reward, 'values'):
            task_rewards = reward.values
        else:
            task_rewards = np.array([reward] * self.explicit_objectives)
        
        # Calculate entropy reward (positive for high entropy)
        entropy_reward = current_entropy - self.min_entropy
        
        # Create extended reward
        extended_reward = np.zeros(self.num_objectives)
        extended_reward[:self.explicit_objectives] = task_rewards
        extended_reward[self.entropy_objective_idx] = entropy_reward
        
        mo_reward_extended = MultiObjectiveReward(extended_reward)
        
        # Store with extended preference
        extended_pref = self._extend_preference_with_entropy(
            preference if preference is not None else self.current_preference[:self.explicit_objectives]
        )
        
        # Call parent update with extended values
        super().update(state, action, mo_reward_extended, next_state, done, extended_pref)
    
    def _adapt_entropy_weight(self, metrics):
        """Adapt entropy weight based on entropy health.
        
        Args:
            metrics: Entropy metrics from tracker
        """
        # Check if metrics is EntropyMetrics object or dict
        if hasattr(metrics, 'collapse_risk'):
            collapse_risk = metrics.collapse_risk
            current_entropy = metrics.current
        else:
            collapse_risk = metrics.get('collapse_risk', 0)
            current_entropy = metrics.get('current', 1.0)
        
        if collapse_risk > 0.7:
            # High collapse risk - increase entropy weight
            self.entropy_weight = min(0.5, self.entropy_weight * 1.1)
            logger.warning(f"High entropy collapse risk ({collapse_risk:.2f}), "
                         f"increased entropy weight to {self.entropy_weight:.3f}")
        elif current_entropy > self.entropy_target * 1.2:
            # Entropy too high - slightly decrease weight
            self.entropy_weight = max(0.01, self.entropy_weight * 0.95)
        
        self.entropy_weight_history.append(self.entropy_weight)
    
    def get_pareto_solutions_with_entropy(self) -> List[Dict[str, Any]]:
        """Get Pareto solutions including entropy information.
        
        Returns:
            List of solutions with objectives and entropy
        """
        solutions = []
        
        for i, (state, action, reward) in enumerate(self.pareto_front.solutions):
            # Extract task objectives and entropy
            reward_values = self.pareto_front.rewards[i].values
            task_objectives = reward_values[:self.explicit_objectives]
            entropy_objective = reward_values[self.entropy_objective_idx]
            
            solutions.append({
                'task_objectives': task_objectives,
                'entropy': entropy_objective + self.min_entropy,  # Convert back to actual entropy
                'scalarized': np.dot(task_objectives, 
                                   np.ones(self.explicit_objectives) / self.explicit_objectives),
                'state': state,
                'action': action
            })
        
        # Sort by entropy to show diversity
        solutions.sort(key=lambda x: x['entropy'], reverse=True)
        
        return solutions
    
    def visualize_entropy_pareto(self) -> Optional[Any]:
        """Visualize Pareto front with entropy dimension.
        
        Returns:
            Matplotlib figure or None
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            return None
        
        solutions = self.get_pareto_solutions_with_entropy()
        
        if not solutions:
            logger.warning("No Pareto solutions to visualize")
            return None
        
        fig = plt.figure(figsize=(12, 5))
        
        if self.explicit_objectives == 2:
            # 3D plot: obj1, obj2, entropy
            ax1 = fig.add_subplot(121, projection='3d')
            
            obj1 = [s['task_objectives'][0] for s in solutions]
            obj2 = [s['task_objectives'][1] for s in solutions]
            entropy = [s['entropy'] for s in solutions]
            
            scatter = ax1.scatter(obj1, obj2, entropy, c=entropy, cmap='viridis', s=50)
            ax1.set_xlabel('Objective 1')
            ax1.set_ylabel('Objective 2')
            ax1.set_zlabel('Entropy')
            ax1.set_title('Pareto Front with Entropy')
            
            # 2D projection
            ax2 = fig.add_subplot(122)
            scatter2 = ax2.scatter(obj1, obj2, c=entropy, cmap='viridis', s=50)
            ax2.set_xlabel('Objective 1')
            ax2.set_ylabel('Objective 2')
            ax2.set_title('Task Objectives (colored by entropy)')
            
            # Add colorbar
            cbar = plt.colorbar(scatter2, ax=ax2)
            cbar.set_label('Entropy')
            
            # Mark high-entropy solutions
            high_entropy_mask = np.array(entropy) > self.entropy_target
            if any(high_entropy_mask):
                ax2.scatter(np.array(obj1)[high_entropy_mask],
                          np.array(obj2)[high_entropy_mask],
                          marker='*', s=200, c='red', label='High Entropy')
                ax2.legend()
        
        else:
            # For >2 objectives, show entropy vs scalarized reward
            ax = fig.add_subplot(111)
            
            scalarized = [s['scalarized'] for s in solutions]
            entropy = [s['entropy'] for s in solutions]
            
            ax.scatter(scalarized, entropy, s=50)
            ax.axhline(y=self.min_entropy, color='r', linestyle='--', label='Min Entropy')
            ax.axhline(y=self.entropy_target, color='g', linestyle='--', label='Target Entropy')
            ax.set_xlabel('Scalarized Task Reward')
            ax.set_ylabel('Policy Entropy')
            ax.set_title('Entropy vs Task Performance Trade-off')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics including entropy health."""
        # Get base metrics from parent if available
        if hasattr(super(), 'get_metrics'):
            metrics = super().get_metrics()
        else:
            metrics = {
                'training_steps': getattr(self, 'training_steps', 0),
                'episodes': getattr(self, 'episodes', 0)
            }
        
        # Add entropy-specific metrics
        # Get last entropy metrics by updating with current value
        if self.ac_network.entropy_history:
            current_entropy = self.ac_network.entropy_history[-1]
        else:
            current_entropy = 1.0
            
        entropy_metrics = self.entropy_tracker.update(current_entropy)
        
        # Handle both dict and EntropyMetrics object
        if hasattr(entropy_metrics, '__dict__'):
            # It's an object, convert to dict
            em = entropy_metrics.__dict__
        else:
            em = entropy_metrics
            
        metrics.update({
            'current_entropy': em.get('current', 0),
            'mean_entropy': em.get('mean', 0),
            'min_entropy': em.get('min', 0),
            'entropy_trend': em.get('trend', 0),
            'entropy_collapse_risk': em.get('collapse_risk', 0),
            'entropy_weight': self.entropy_weight,
            'avg_entropy_weight': np.mean(list(self.entropy_weight_history))
        })
        
        # Pareto analysis
        solutions = self.get_pareto_solutions_with_entropy()
        if solutions:
            entropy_values = [s['entropy'] for s in solutions]
            metrics['pareto_entropy_diversity'] = np.std(entropy_values)
            metrics['pareto_high_entropy_ratio'] = sum(e > self.entropy_target for e in entropy_values) / len(entropy_values)
        
        return metrics
    
    def save_model(self, path: str):
        """Save model including entropy configuration."""
        import pickle
        
        save_dict = {
            'state_dict': self.ac_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'entropy_weight': self.entropy_weight,
            'entropy_weight_history': list(self.entropy_weight_history),
            'entropy_tracker_state': self.entropy_tracker.get_metrics(),
            'config': {
                'explicit_objectives': self.explicit_objectives,
                'entropy_weight': self.entropy_weight,
                'min_entropy': self.min_entropy,
                'adaptive_entropy_weight': self.adaptive_entropy_weight,
                'entropy_target': self.entropy_target
            }
        }
        
        torch.save(save_dict, path)
        logger.info(f"Saved EntropyAwareMORL model to {path}")
    
    def load_model(self, path: str):
        """Load model including entropy configuration."""
        save_dict = torch.load(path, map_location=self.device)
        
        self.ac_network.load_state_dict(save_dict['state_dict'])
        self.optimizer.load_state_dict(save_dict['optimizer_state'])
        self.entropy_weight = save_dict['entropy_weight']
        self.entropy_weight_history = deque(save_dict['entropy_weight_history'], maxlen=100)
        
        logger.info(f"Loaded EntropyAwareMORL model from {path}")


if __name__ == "__main__":
    # Validation test
    print("Testing EntropyAwareMORL...")
    
    # Test 1: Basic initialization
    morl = EntropyAwareMORL(
        num_objectives=2,  # Task objectives
        state_dim=8,
        action_dim=4,
        entropy_weight=0.1,
        min_entropy=0.5,
        continuous_actions=False
    )
    
    assert morl.num_objectives == 3, f"Expected 3 objectives (2 task + 1 entropy), got {morl.num_objectives}"
    assert morl.entropy_objective_idx == 2, "Entropy should be last objective"
    
    # Test 2: Preference extension
    task_pref = np.array([0.7, 0.3])
    extended = morl._extend_preference_with_entropy(task_pref)
    
    assert len(extended) == 3, "Extended preference should have 3 components"
    assert abs(extended.sum() - 1.0) < 1e-6, "Extended preference should sum to 1"
    assert extended[2] == morl.entropy_weight, "Entropy weight not properly set"
    
    # Test 3: Action selection
    state = RLState(features=np.random.randn(8))
    action = morl.select_action(state, preference=task_pref)
    
    assert isinstance(action, RLAction), "Invalid action type"
    assert 'log_prob' in action.parameters, "Missing log probability"
    
    # Test 4: Update with entropy tracking
    reward = MultiObjectiveReward([0.5, 0.8])  # Task rewards only
    next_state = RLState(features=np.random.randn(8))
    
    morl.update(state, action, reward, next_state, False, task_pref)
    
    assert len(morl.buffer) == 1, "Experience not stored"
    stored = morl.buffer[0]
    assert len(stored['reward']) == 3, "Reward should be extended with entropy"
    
    # Test 5: Entropy metrics
    metrics = morl.get_metrics()
    
    assert 'current_entropy' in metrics, "Missing entropy metric"
    assert 'entropy_collapse_risk' in metrics, "Missing collapse risk"
    assert 'entropy_weight' in metrics, "Missing entropy weight"
    
    # Test 6: Pareto solutions with entropy
    # Add some fake solutions
    for i in range(5):
        fake_reward = MultiObjectiveReward([
            np.random.rand(), 
            np.random.rand(),
            np.random.randn()  # Entropy component
        ])
        morl.pareto_front.update(state, action, fake_reward)
    
    solutions = morl.get_pareto_solutions_with_entropy()
    assert len(solutions) > 0, "No Pareto solutions found"
    assert 'entropy' in solutions[0], "Missing entropy in solution"
    assert 'task_objectives' in solutions[0], "Missing task objectives"
    
    print(f" EntropyAwareMORL validation complete!")
    print(f"   - Entropy weight: {morl.entropy_weight:.3f}")
    print(f"   - Min entropy: {morl.min_entropy}")
    print(f"   - Pareto solutions: {len(solutions)}")
    
    # Test visualization (if matplotlib available)
    try:
        fig = morl.visualize_entropy_pareto()
        if fig:
            print("   - Visualization successful")
    except Exception as e:
        print(f"   - Visualization skipped: {e}")