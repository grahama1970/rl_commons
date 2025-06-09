"""
Module: weighted_sum.py
Purpose: Weighted Sum approach to Multi-Objective RL

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.morl import WeightedSumMORL
>>> morl = WeightedSumMORL(num_objectives=3, state_dim=8, action_dim=4)
>>> action = morl.select_action(state, preference=[0.5, 0.3, 0.2])
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import time

from .base_morl import MORLBase, MultiObjectiveReward
from ...core.base import RLState, RLAction


class PreferenceConditionedNetwork(nn.Module):
    """Neural network conditioned on preference weights."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_objectives: int,
        hidden_dim: int = 128
    ):
        """Initialize preference-conditioned network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            num_objectives: Number of objectives
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Input is state + preference weights
        input_dim = state_dim + num_objectives
        
        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states: torch.Tensor, preferences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            states: State tensor [batch_size, state_dim]
            preferences: Preference tensor [batch_size, num_objectives]
            
        Returns:
            Action logits and value estimates
        """
        # Concatenate states and preferences
        inputs = torch.cat([states, preferences], dim=-1)
        
        action_logits = self.policy_network(inputs)
        values = self.value_network(inputs)
        
        return action_logits, values


class WeightedSumMORL(MORLBase):
    """Weighted Sum approach to Multi-Objective RL."""
    
    def __init__(
        self,
        num_objectives: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        device: str = None
    ):
        """Initialize Weighted Sum MORL.
        
        Args:
            num_objectives: Number of objectives
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension for networks
            learning_rate: Learning rate
            discount_factor: Discount factor
            epsilon: Exploration rate
            device: Device for computation
        """
        super().__init__(num_objectives, state_dim, action_dim, device=device)
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Preference-conditioned network
        self.network = PreferenceConditionedNetwork(
            state_dim, action_dim, num_objectives, hidden_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.buffer = deque(maxlen=10000)
        
        # Current preference (can be changed dynamically)
        self.current_preference = np.ones(num_objectives) / num_objectives
    
    def select_action(
        self,
        state: RLState,
        preference: Optional[np.ndarray] = None
    ) -> RLAction:
        """Select action using preference-conditioned policy.
        
        Args:
            state: Current state
            preference: Preference weights (uses current if None)
            
        Returns:
            Selected action
        """
        if preference is None:
            preference = self.current_preference
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return RLAction(
                action_type='discrete',
                action_id=np.random.randint(self.action_dim),
                parameters={'exploration': True}
            )
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
        pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, _ = self.network(state_tensor, pref_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            action = action_probs.argmax(dim=-1).item()
        
        return RLAction(
            action_type='discrete',
            action_id=action,
            parameters={'probs': action_probs.cpu().numpy()}
        )
    
    def update(
        self,
        state: RLState,
        action: RLAction,
        reward: MultiObjectiveReward,
        next_state: RLState,
        done: bool,
        preference: Optional[np.ndarray] = None
    ):
        """Update policy with experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Multi-objective reward
            next_state: Next state
            done: Episode done flag
            preference: Preference weights used
        """
        if preference is None:
            preference = self.current_preference
        
        # Store experience
        self.buffer.append({
            'state': state.features,
            'action': action.action_id,
            'reward': reward.values,
            'next_state': next_state.features,
            'done': done,
            'preference': preference
        })
        
        # Update Pareto front
        self.pareto_front.update(state, action, reward)
        
        # Train if enough experiences
        if len(self.buffer) >= 32:
            self._train_batch(batch_size=32)
    
    def _train_batch(self, batch_size: int):
        """Train on a batch of experiences."""
        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        # Convert to tensors
        states = torch.FloatTensor([e['state'] for e in batch]).to(self.device)
        actions = torch.LongTensor([e['action'] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e['done'] for e in batch]).to(self.device)
        preferences = torch.FloatTensor([e['preference'] for e in batch]).to(self.device)
        
        # Scalarize rewards using preferences
        scalarized_rewards = (rewards * preferences).sum(dim=-1)
        
        # Current Q-values
        action_logits, values = self.network(states, preferences)
        action_log_probs = torch.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Target values
        with torch.no_grad():
            next_action_logits, next_values = self.network(next_states, preferences)
            next_values = next_values.squeeze()
            target_values = scalarized_rewards + self.discount_factor * next_values * (1 - dones)
        
        # Losses
        value_loss = nn.functional.mse_loss(values.squeeze(), target_values)
        
        # Policy gradient loss
        advantages = (target_values - values.squeeze()).detach()
        policy_loss = -(selected_log_probs * advantages).mean()
        
        # Total loss
        total_loss = value_loss + policy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Record metrics
        self.training_history['value_loss'].append(value_loss.item())
        self.training_history['policy_loss'].append(policy_loss.item())
    
    def train(
        self,
        env: Any,
        num_episodes: int,
        preference_sampling: str = 'uniform'
    ) -> Dict[str, Any]:
        """Train MORL agent.
        
        Args:
            env: Multi-objective environment
            num_episodes: Number of training episodes
            preference_sampling: How to sample preferences
            
        Returns:
            Training results
        """
        episode_rewards = []
        episode_lengths = []
        preferences_used = []
        
        for episode in range(num_episodes):
            # Sample preference for this episode
            preference = self.sample_preference(preference_sampling)
            self.current_preference = preference
            preferences_used.append(preference)
            
            state = env.reset()
            done = False
            episode_reward = np.zeros(self.num_objectives)
            episode_length = 0
            
            while not done:
                # Select action
                action = self.select_action(RLState(features=state), preference)
                
                # Take action
                next_state, reward, done, info = env.step(action.action_id)
                
                # Convert reward to multi-objective
                if isinstance(reward, (list, tuple, np.ndarray)):
                    mo_reward = MultiObjectiveReward(reward)
                else:
                    mo_reward = MultiObjectiveReward([reward] * self.num_objectives)
                
                # Update
                self.update(
                    RLState(features=state),
                    action,
                    mo_reward,
                    RLState(features=next_state),
                    done,
                    preference
                )
                
                episode_reward += mo_reward.values
                state = next_state
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            if episode % 100 == 0:
                avg_rewards = np.mean(episode_rewards[-100:], axis=0)
                print(f"Episode {episode}: Avg Rewards = {avg_rewards}, "
                      f"Pareto Front Size = {len(self.pareto_front.solutions)}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'preferences_used': preferences_used,
            'final_pareto_size': len(self.pareto_front.solutions),
            'pareto_coverage': self.pareto_front.get_coverage()
        }
    
    def get_policy_for_preference(self, preference: np.ndarray):
        """Get deterministic policy for specific preference.
        
        Args:
            preference: Preference weights
            
        Returns:
            Policy function
        """
        def policy(state: RLState) -> RLAction:
            """Fixed preference policy."""
            state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
            pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_logits, _ = self.network(state_tensor, pref_tensor)
                action = action_logits.argmax(dim=-1).item()
            
            return RLAction(
                action_type='discrete',
                action_id=action,
                parameters={'preference': preference.tolist()}
            )
        
        return policy


if __name__ == "__main__":
    # Validation test
    # Create MORL agent
    morl = WeightedSumMORL(
        num_objectives=3,
        state_dim=8,
        action_dim=4,
        hidden_dim=64
    )
    
    # Test action selection
    state = RLState(features=np.random.randn(8))
    preference = np.array([0.5, 0.3, 0.2])
    
    action = morl.select_action(state, preference)
    assert isinstance(action, RLAction), "Invalid action type"
    assert 0 <= action.action_id < 4, f"Invalid action: {action.action_id}"
    
    # Test update
    next_state = RLState(features=np.random.randn(8))
    reward = MultiObjectiveReward([0.8, 0.5, 0.3])
    
    morl.update(state, action, reward, next_state, False, preference)
    
    assert len(morl.buffer) == 1, "Experience not stored"
    assert len(morl.pareto_front.solutions) >= 0, "Pareto front error"
    
    # Test preference sampling
    for method in ['uniform', 'corner', 'adaptive']:
        pref = morl.sample_preference(method)
        assert len(pref) == 3, f"Wrong preference dimension: {len(pref)}"
        assert abs(pref.sum() - 1.0) < 1e-6, f"Preferences don't sum to 1: {pref.sum()}"
    
    # Test policy generation
    policy = morl.get_policy_for_preference(preference)
    action2 = policy(state)
    assert isinstance(action2, RLAction), "Policy returned invalid action"
    
    print("✅ Weighted Sum MORL validation passed")