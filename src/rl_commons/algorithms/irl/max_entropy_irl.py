"""
Module: max_entropy_irl.py
Purpose: Maximum Entropy Inverse Reinforcement Learning

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.irl import MaxEntropyIRL
>>> irl = MaxEntropyIRL(state_dim=8, action_dim=4)
>>> reward_fn = irl.learn_reward(demonstrations)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict

from .base_irl import IRLBase, Demonstration, RewardFunction
from ...core.base import RLState, RLAction


class MaxEntropyIRL(IRLBase):
    """Maximum Entropy IRL for learning from demonstrations."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        reward_hidden_dim: int = 128,
        learning_rate: float = 0.001,
        entropy_weight: float = 1.0,
        discount_factor: float = 0.99,
        device: str = None
    ):
        """Initialize MaxEnt IRL.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            reward_hidden_dim: Hidden dimension for reward network
            learning_rate: Learning rate for optimization
            entropy_weight: Weight for entropy regularization
            discount_factor: Discount factor for future rewards
            device: Device for computation
        """
        super().__init__(state_dim, action_dim, reward_hidden_dim, device)
        
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        self.discount_factor = discount_factor
        
        # Optimizer for reward function
        self.optimizer = optim.Adam(self.reward_function.parameters(), lr=learning_rate)
        
        # Policy network for computing state visitation
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
    
    def learn_reward(
        self,
        demonstrations: List[Demonstration],
        num_iterations: int = 1000,
        batch_size: int = 32
    ) -> RewardFunction:
        """Learn reward function using Maximum Entropy IRL.
        
        Args:
            demonstrations: Expert demonstrations
            num_iterations: Number of training iterations
            batch_size: Batch size for training
            
        Returns:
            Learned reward function
        """
        # Compute expert feature expectations
        expert_features = self.compute_feature_expectations(demonstrations)
        
        for iteration in range(num_iterations):
            # Sample batch of demonstrations
            batch_demos = np.random.choice(demonstrations, min(batch_size, len(demonstrations)))
            
            # Forward pass through demonstrations
            total_loss = 0.0
            
            for demo in batch_demos:
                demo = demo.to_device(self.device)
                
                # Compute rewards for expert trajectory
                if self.reward_function.action_dim is not None:
                    # Convert discrete actions to one-hot if needed
                    if len(demo.actions.shape) == 1:
                        action_one_hot = nn.functional.one_hot(demo.actions.long(), num_classes=self.action_dim).float()
                        rewards = self.reward_function(demo.states, action_one_hot)
                    else:
                        rewards = self.reward_function(demo.states, demo.actions)
                else:
                    rewards = self.reward_function(demo.states)
                
                # Compute policy distribution
                logits = self.policy_network(demo.states)
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                
                # Select log probabilities for taken actions
                if len(demo.actions.shape) == 1:
                    action_log_probs = log_probs.gather(1, demo.actions.unsqueeze(-1)).squeeze(-1)
                else:
                    action_log_probs = (log_probs * demo.actions).sum(dim=-1)
                
                # Maximum entropy objective
                # Maximize: sum(rewards) + entropy_weight * sum(log_probs)
                trajectory_reward = rewards.squeeze().sum()
                entropy = -action_log_probs.sum()
                
                loss = -(trajectory_reward + self.entropy_weight * entropy)
                total_loss += loss
            
            # Backward pass
            total_loss = total_loss / len(batch_demos)
            
            self.optimizer.zero_grad()
            self.policy_optimizer.zero_grad()
            
            total_loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.reward_function.parameters(), 1.0)
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
            
            self.optimizer.step()
            self.policy_optimizer.step()
            
            # Record training progress
            self.training_history['losses'].append(total_loss.item())
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
        
        return self.reward_function
    
    def compute_state_visitation_frequency(
        self,
        demonstrations: List[Demonstration]
    ) -> Dict[int, float]:
        """Compute state visitation frequency from demonstrations.
        
        Args:
            demonstrations: List of demonstrations
            
        Returns:
            State visitation frequencies
        """
        state_counts = defaultdict(float)
        total_steps = 0
        
        for demo in demonstrations:
            # Discretize states for counting (in practice, use function approximation)
            for state in demo.states:
                # Simple discretization - in practice, use more sophisticated methods
                state_key = tuple(state.cpu().numpy().round(2))
                state_counts[state_key] += 1
                total_steps += 1
        
        # Normalize to get frequencies
        state_visitation = {
            state: count / total_steps 
            for state, count in state_counts.items()
        }
        
        return state_visitation
    
    def evaluate(
        self,
        demonstration: Demonstration,
        policy: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Evaluate learned reward function.
        
        Args:
            demonstration: Test demonstration
            policy: Policy to evaluate (uses learned policy if None)
            
        Returns:
            Evaluation metrics
        """
        demo = demonstration.to_device(self.device)
        
        with torch.no_grad():
            # Compute rewards for demonstration
            if self.reward_function.action_dim is not None:
                # Convert discrete actions to one-hot if needed
                if len(demo.actions.shape) == 1:
                    action_one_hot = nn.functional.one_hot(demo.actions.long(), num_classes=self.action_dim).float()
                    rewards = self.reward_function(demo.states, action_one_hot)
                else:
                    rewards = self.reward_function(demo.states, demo.actions)
            else:
                rewards = self.reward_function(demo.states)
            
            # Compute policy predictions
            logits = self.policy_network(demo.states)
            predicted_actions = logits.argmax(dim=-1)
            
            # Compute accuracy
            if len(demo.actions.shape) == 1:
                accuracy = (predicted_actions == demo.actions).float().mean().item()
            else:
                # For continuous actions
                accuracy = nn.functional.cosine_similarity(
                    logits, demo.actions, dim=-1
                ).mean().item()
            
            # Compute average reward
            rewards = rewards.squeeze()
            avg_reward = rewards.mean().item()
            
            # Compute entropy
            probs = nn.functional.softmax(logits, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
        
        return {
            'accuracy': accuracy,
            'avg_reward': avg_reward,
            'entropy': entropy,
            'total_reward': rewards.sum().item()
        }
    
    def generate_policy(self) -> Callable:
        """Generate policy from learned reward function.
        
        Returns:
            Policy function
        """
        def policy(state: RLState) -> RLAction:
            """Policy based on learned rewards."""
            state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.policy_network(state_tensor)
                probs = nn.functional.softmax(logits, dim=-1)
                
                # Sample action from distribution
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample().item()
            
            return RLAction(
                action_type='discrete',
                action_id=action,
                parameters={'probs': probs.cpu().numpy()}
            )
        
        return policy


if __name__ == "__main__":
    # Validation test
    # Create expert demonstrations
    demonstrations = []
    for _ in range(5):
        states = torch.randn(50, 8)
        actions = torch.randint(0, 4, (50,))
        demo = Demonstration(states, actions)
        demonstrations.append(demo)
    
    # Initialize MaxEnt IRL
    irl = MaxEntropyIRL(state_dim=8, action_dim=4)
    
    # Learn reward function
    print("Learning reward function...")
    reward_fn = irl.learn_reward(demonstrations, num_iterations=100)
    
    # Evaluate on test demonstration
    test_demo = Demonstration(
        states=torch.randn(20, 8),
        actions=torch.randint(0, 4, (20,))
    )
    
    metrics = irl.evaluate(test_demo)
    print(f"Evaluation metrics: {metrics}")
    
    assert 'accuracy' in metrics, "Missing accuracy metric"
    assert 'avg_reward' in metrics, "Missing reward metric"
    assert len(irl.training_history['losses']) > 0, "No training history"
    
    # Test policy generation
    policy = irl.generate_policy()
    state = RLState(features=np.random.randn(8))
    action = policy(state)
    
    assert isinstance(action, RLAction), "Invalid action type"
    assert 0 <= action.action_id < 4, f"Invalid action: {action.action_id}"
    
    print(" MaxEntropy IRL validation passed")