"""Contextual Bandit implementation for provider/strategy selection"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging

from ...core.base import RLAgent, RLState, RLAction, RLReward

logger = logging.getLogger(__name__)


class ContextualBandit(RLAgent):
    """
    Contextual Bandit using linear upper confidence bound (LinUCB)
    Suitable for provider selection, A/B testing, and similar problems
    """
    
    def __init__(self, 
                 name: str,
                 n_arms: int,
                 n_features: int,
                 alpha: float = 1.0,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize contextual bandit
        
        Args:
            name: Agent name
            n_arms: Number of arms (actions) to choose from
            n_features: Dimension of context features
            alpha: Exploration parameter (higher = more exploration)
            config: Additional configuration
        """
        super().__init__(name, config)
        
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # Initialize model parameters for each arm
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset bandit parameters"""
        # A_i = I (identity matrix) for each arm
        self.A = [np.eye(self.n_features) for _ in range(self.n_arms)]
        # b_i = 0 vector for each arm
        self.b = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        
        # Track arm selection counts
        self.arm_counts = np.zeros(self.n_arms)
        self.total_reward = 0.0
        
    def select_action(self, state: RLState, explore: bool = True) -> RLAction:
        """
        Select an arm using LinUCB algorithm
        
        Args:
            state: Current state with context features
            explore: Whether to explore (True) or exploit (False)
            
        Returns:
            Selected action
        """
        context = state.features
        
        # Ensure context has correct dimension
        if len(context) != self.n_features:
            raise ValueError(f"Context dimension {len(context)} != expected {self.n_features}")
        
        # Calculate upper confidence bound for each arm
        ucb_values = []
        
        for arm in range(self.n_arms):
            # Solve for theta: A_i * theta_i = b_i
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            
            # Calculate UCB
            # mean + alpha * sqrt(context^T * A_inv * context)
            mean = context @ theta
            
            if explore and self.training:
                variance = context @ A_inv @ context
                confidence = self.alpha * np.sqrt(variance)
                ucb = mean + confidence
            else:
                # Pure exploitation
                ucb = mean
            
            ucb_values.append(ucb)
        
        # Select arm with highest UCB
        selected_arm = int(np.argmax(ucb_values))
        
        self.logger.debug(f"UCB values: {ucb_values}, selected: {selected_arm}")
        
        return RLAction(
            action_type="select_arm",
            action_id=selected_arm,
            parameters={
                "ucb_values": [float(v) for v in ucb_values],
                "exploration": explore and self.training
            }
        )
    
    def update(self, 
               state: RLState, 
               action: RLAction, 
               reward: RLReward, 
               next_state: RLState, 
               done: bool = False) -> Dict[str, float]:
        """
        Update bandit parameters based on observed reward
        
        Args:
            state: State when action was taken
            action: Action taken (arm selected)
            reward: Observed reward
            next_state: Not used for bandits
            done: Not used for bandits
            
        Returns:
            Training metrics
        """
        context = state.features
        arm = action.action_id
        r = reward.value
        
        # Update parameters for selected arm
        # A_i = A_i + context * context^T
        self.A[arm] += np.outer(context, context)
        # b_i = b_i + reward * context
        self.b[arm] += r * context
        
        # Update statistics
        self.arm_counts[arm] += 1
        self.total_reward += r
        self.training_steps += 1
        
        # Calculate current performance metrics
        metrics = {
            "reward": r,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(1, self.training_steps),
            "arm_selected": arm,
            "exploration_rate": self.alpha,
        }
        
        # Add arm selection distribution
        if self.training_steps > 0:
            for i in range(self.n_arms):
                metrics[f"arm_{i}_percentage"] = self.arm_counts[i] / self.training_steps
        
        return metrics
    
    def get_arm_weights(self) -> List[np.ndarray]:
        """Get learned weights for each arm"""
        weights = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            weights.append(theta)
        return weights
    
    def save(self, path: Path) -> None:
        """Save bandit state"""
        path = Path(path)
        
        state_dict = {
            "name": self.name,
            "n_arms": self.n_arms,
            "n_features": self.n_features,
            "alpha": self.alpha,
            "A": [A.tolist() for A in self.A],
            "b": [b.tolist() for b in self.b],
            "arm_counts": self.arm_counts.tolist(),
            "total_reward": self.total_reward,
            "training_steps": self.training_steps,
            "config": self.config
        }
        
        path.write_text(json.dumps(state_dict, indent=2))
        self.logger.info(f"Saved contextual bandit to {path}")
    
    def load(self, path: Path) -> None:
        """Load bandit state"""
        path = Path(path)
        state_dict = json.loads(path.read_text())
        
        # Validate compatibility
        if state_dict["n_arms"] != self.n_arms:
            raise ValueError(f"Incompatible n_arms: {state_dict['n_arms']} != {self.n_arms}")
        if state_dict["n_features"] != self.n_features:
            raise ValueError(f"Incompatible n_features: {state_dict['n_features']} != {self.n_features}")
        
        # Load parameters
        self.name = state_dict["name"]
        self.alpha = state_dict["alpha"]
        self.A = [np.array(A) for A in state_dict["A"]]
        self.b = [np.array(b) for b in state_dict["b"]]
        self.arm_counts = np.array(state_dict["arm_counts"])
        self.total_reward = state_dict["total_reward"]
        self.training_steps = state_dict["training_steps"]
        self.config = state_dict.get("config", {})
        
        self.logger.info(f"Loaded contextual bandit from {path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = super().get_metrics()
        
        # Add bandit-specific metrics
        metrics.update({
            "n_arms": self.n_arms,
            "n_features": self.n_features,
            "alpha": self.alpha,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(1, self.training_steps),
            "arm_selection_counts": self.arm_counts.tolist(),
        })
        
        # Add percentage for each arm
        if self.training_steps > 0:
            arm_percentages = self.arm_counts / self.training_steps
            metrics["arm_selection_percentages"] = arm_percentages.tolist()
            
            # Find most/least selected arms
            metrics["most_selected_arm"] = int(np.argmax(self.arm_counts))
            metrics["least_selected_arm"] = int(np.argmin(self.arm_counts))
        
        return metrics


class ThompsonSamplingBandit(ContextualBandit):
    """
    Thompson Sampling variant for contextual bandits
    Uses Bayesian approach with parameter sampling
    """
    
    def __init__(self, 
                 name: str,
                 n_arms: int,
                 n_features: int,
                 regularization: float = 1.0,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Thompson Sampling bandit
        
        Args:
            name: Agent name
            n_arms: Number of arms
            n_features: Feature dimension
            regularization: Regularization parameter
            config: Additional config
        """
        # Thompson Sampling doesn't use alpha, but we pass 0 to parent
        super().__init__(name, n_arms, n_features, alpha=0.0, config=config)
        self.regularization = regularization
        
    def select_action(self, state: RLState, explore: bool = True) -> RLAction:
        """Select arm using Thompson Sampling"""
        context = state.features
        
        # Sample from posterior for each arm
        sampled_means = []
        
        for arm in range(self.n_arms):
            # Posterior mean and covariance
            A_inv = np.linalg.inv(self.A[arm] + self.regularization * np.eye(self.n_features))
            mean = A_inv @ self.b[arm]
            
            if explore and self.training:
                # Sample from posterior
                cov = A_inv
                sampled_theta = np.random.multivariate_normal(mean, cov)
                sampled_mean = context @ sampled_theta
            else:
                # Use posterior mean
                sampled_mean = context @ mean
            
            sampled_means.append(sampled_mean)
        
        # Select arm with highest sampled value
        selected_arm = int(np.argmax(sampled_means))
        
        return RLAction(
            action_type="select_arm",
            action_id=selected_arm,
            parameters={
                "sampled_values": [float(v) for v in sampled_means],
                "exploration": explore and self.training
            }
        )
