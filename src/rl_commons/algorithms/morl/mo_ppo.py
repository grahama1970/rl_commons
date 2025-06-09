"""
Module: mo_ppo.py  
Purpose: Multi-Objective Proximal Policy Optimization

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.morl import MultiObjectivePPO
>>> mo_ppo = MultiObjectivePPO(num_objectives=3, state_dim=8, action_dim=4)
>>> mo_ppo.train(env, num_episodes=1000)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import time

from .base_morl import MORLBase, MultiObjectiveReward
from ...core.base import RLState, RLAction


class MOActorCritic(nn.Module):
    """Actor-Critic network for multi-objective PPO."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_objectives: int,
        hidden_dim: int = 128,
        continuous_actions: bool = False
    ):
        """Initialize MO Actor-Critic.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension  
            num_objectives: Number of objectives
            hidden_dim: Hidden dimension
            continuous_actions: Whether actions are continuous
        """
        super().__init__()
        
        self.continuous_actions = continuous_actions
        self.num_objectives = num_objectives
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + num_objectives, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        if continuous_actions:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Multi-objective value heads
        self.value_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_objectives)
        ])
    
    def forward(self, states: torch.Tensor, preferences: torch.Tensor) -> Tuple[dist.Distribution, torch.Tensor]:
        """Forward pass.
        
        Args:
            states: State tensor
            preferences: Preference weights
            
        Returns:
            Action distribution and multi-objective values
        """
        # Concatenate states and preferences
        inputs = torch.cat([states, preferences], dim=-1)
        features = self.feature_net(inputs)
        
        # Actor output
        if self.continuous_actions:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            action_dist = dist.Normal(mean, std)
        else:
            logits = self.actor(features)
            action_dist = dist.Categorical(logits=logits)
        
        # Value outputs
        values = []
        for value_head in self.value_heads:
            values.append(value_head(features))
        
        mo_values = torch.cat(values, dim=-1)  # [batch, num_objectives]
        
        return action_dist, mo_values


class MultiObjectivePPO(MORLBase):
    """Multi-Objective PPO implementation."""
    
    def __init__(
        self,
        num_objectives: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        continuous_actions: bool = False,
        device: str = None
    ):
        """Initialize MO-PPO.
        
        Args:
            num_objectives: Number of objectives
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension
            learning_rate: Learning rate
            discount_factor: Discount factor
            gae_lambda: GAE lambda
            clip_ratio: PPO clipping ratio
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            continuous_actions: Whether actions are continuous
            device: Device for computation
        """
        super().__init__(num_objectives, state_dim, action_dim, device=device)
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Actor-Critic network
        self.ac_network = MOActorCritic(
            state_dim, action_dim, num_objectives, 
            hidden_dim, continuous_actions
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = []
        
        # Current preference
        self.current_preference = np.ones(num_objectives) / num_objectives
    
    def select_action(
        self,
        state: RLState,
        preference: Optional[np.ndarray] = None
    ) -> RLAction:
        """Select action using current policy.
        
        Args:
            state: Current state
            preference: Preference weights
            
        Returns:
            Selected action
        """
        if preference is None:
            preference = self.current_preference
        
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
        pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_dist, _ = self.ac_network(state_tensor, pref_tensor)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        if self.ac_network.continuous_actions:
            return RLAction(
                action_type='continuous',
                action_id=0,
                parameters={
                    'values': action.cpu().numpy().flatten(),
                    'log_prob': log_prob.sum().cpu().item()  # Sum log probs for multi-dim actions
                }
            )
        else:
            return RLAction(
                action_type='discrete',
                action_id=action.cpu().item(),
                parameters={'log_prob': log_prob.cpu().item()}
            )
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        preferences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Multi-objective rewards [seq_len, num_objectives]
            values: Multi-objective values [seq_len, num_objectives]
            dones: Done flags [seq_len]
            preferences: Preference weights [seq_len, num_objectives]
            
        Returns:
            Advantages and returns
        """
        seq_len = len(rewards)
        
        # Scalarize using preferences
        scalarized_rewards = (rewards * preferences).sum(dim=-1)
        scalarized_values = (values * preferences).sum(dim=-1)
        
        advantages = torch.zeros(seq_len).to(self.device)
        returns = torch.zeros(seq_len).to(self.device)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = scalarized_values[t + 1]
            
            delta = scalarized_rewards[t] + self.discount_factor * next_value * (1 - dones[t]) - scalarized_values[t]
            gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + scalarized_values[t]
        
        # Also compute multi-objective returns for value function training
        mo_returns = torch.zeros(seq_len, self.num_objectives).to(self.device)
        mo_next_return = torch.zeros(self.num_objectives).to(self.device)
        
        for t in reversed(range(seq_len)):
            mo_returns[t] = rewards[t] + self.discount_factor * mo_next_return * (1 - dones[t])
            mo_next_return = mo_returns[t]
        
        return advantages, mo_returns
    
    def update(
        self,
        state: RLState,
        action: RLAction,
        reward: MultiObjectiveReward,
        next_state: RLState,
        done: bool,
        preference: Optional[np.ndarray] = None
    ):
        """Store experience (actual update happens in train_on_batch).
        
        Args:
            state: Current state
            action: Action taken
            reward: Multi-objective reward
            next_state: Next state
            done: Episode done flag
            preference: Preference used
        """
        if preference is None:
            preference = self.current_preference
        
        self.buffer.append({
            'state': state.features,
            'action': action.action_id if action.action_type == 'discrete' else action.parameters['values'],
            'reward': reward.values,
            'log_prob': action.parameters['log_prob'],
            'done': done,
            'preference': preference
        })
        
        # Update Pareto front
        self.pareto_front.update(state, action, reward)
    
    def train_on_batch(self, epochs: int = 4, batch_size: int = 64):
        """Train on collected batch of experiences.
        
        Args:
            epochs: Number of epochs to train
            batch_size: Mini-batch size
        """
        if not self.buffer:
            return
        
        # Convert buffer to tensors
        states = torch.FloatTensor([e['state'] for e in self.buffer]).to(self.device)
        actions = torch.LongTensor([e['action'] for e in self.buffer]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in self.buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([e['log_prob'] for e in self.buffer]).to(self.device)
        dones = torch.FloatTensor([e['done'] for e in self.buffer]).to(self.device)
        preferences = torch.FloatTensor([e['preference'] for e in self.buffer]).to(self.device)
        
        # Compute values and advantages
        with torch.no_grad():
            _, values = self.ac_network(states, preferences)
            advantages, mo_returns = self.compute_gae(rewards, values, dones, preferences)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training loop
        dataset_size = len(states)
        
        for _ in range(epochs):
            # Shuffle indices
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_mo_returns = mo_returns[batch_indices]
                batch_preferences = preferences[batch_indices]
                
                # Forward pass
                action_dist, values = self.ac_network(batch_states, batch_preferences)
                
                # Policy loss
                if self.ac_network.continuous_actions:
                    log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)
                else:
                    log_probs = action_dist.log_prob(batch_actions)
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                # Value loss (multi-objective)
                value_loss = nn.functional.mse_loss(values, batch_mo_returns)
                
                # Entropy bonus
                entropy = action_dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_network.parameters(), 0.5)
                self.optimizer.step()
                
                # Record
                self.training_history['policy_loss'].append(policy_loss.item())
                self.training_history['value_loss'].append(value_loss.item())
                self.training_history['entropy'].append(entropy.item())
        
        # Clear buffer
        self.buffer.clear()
    
    def train(
        self,
        env: Any,
        num_episodes: int,
        preference_sampling: str = 'uniform',
        steps_per_update: int = 2048
    ) -> Dict[str, Any]:
        """Train MO-PPO.
        
        Args:
            env: Multi-objective environment
            num_episodes: Number of episodes
            preference_sampling: How to sample preferences
            steps_per_update: Steps before policy update
            
        Returns:
            Training results
        """
        episode_rewards = []
        episode_lengths = []
        total_steps = 0
        
        for episode in range(num_episodes):
            # Sample preference for episode
            preference = self.sample_preference(preference_sampling)
            self.current_preference = preference
            
            state = env.reset()
            done = False
            episode_reward = np.zeros(self.num_objectives)
            episode_length = 0
            
            while not done:
                # Select action
                action = self.select_action(RLState(features=state), preference)
                
                # Take action
                next_state, reward, done, info = env.step(
                    action.action_id if action.action_type == 'discrete' else action.parameters['values']
                )
                
                # Convert reward
                if isinstance(reward, (list, tuple, np.ndarray)):
                    mo_reward = MultiObjectiveReward(reward)
                else:
                    mo_reward = MultiObjectiveReward([reward] * self.num_objectives)
                
                # Store experience
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
                total_steps += 1
                
                # Update policy
                if total_steps % steps_per_update == 0:
                    self.train_on_batch()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if episode % 100 == 0:
                avg_rewards = np.mean(episode_rewards[-100:], axis=0)
                print(f"Episode {episode}: Avg Rewards = {avg_rewards}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_pareto_size': len(self.pareto_front.solutions),
            'pareto_coverage': self.pareto_front.get_coverage()
        }


if __name__ == "__main__":
    # Validation test
    mo_ppo = MultiObjectivePPO(
        num_objectives=3,
        state_dim=8,
        action_dim=4,
        continuous_actions=False
    )
    
    # Test action selection
    state = RLState(features=np.random.randn(8))
    preference = np.array([0.6, 0.3, 0.1])
    
    action = mo_ppo.select_action(state, preference)
    assert isinstance(action, RLAction), "Invalid action type"
    assert 'log_prob' in action.parameters, "Missing log prob"
    
    # Test experience storage
    reward = MultiObjectiveReward([0.5, 0.8, 0.2])
    next_state = RLState(features=np.random.randn(8))
    
    mo_ppo.update(state, action, reward, next_state, False, preference)
    assert len(mo_ppo.buffer) == 1, "Experience not stored"
    
    # Test with continuous actions
    mo_ppo_cont = MultiObjectivePPO(
        num_objectives=2,
        state_dim=6,
        action_dim=3,
        continuous_actions=True
    )
    
    action_cont = mo_ppo_cont.select_action(state=RLState(features=np.random.randn(6)))
    assert action_cont.action_type == 'continuous', "Wrong action type"
    assert len(action_cont.parameters['values']) == 3, "Wrong action dimension"
    
    print(" Multi-Objective PPO validation passed")