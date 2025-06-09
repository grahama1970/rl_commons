"""
Module: pareto_q_learning.py
Purpose: Pareto Q-Learning for finding Pareto-optimal policies

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.morl import ParetoQLearning
>>> pql = ParetoQLearning(num_objectives=3, state_dim=8, action_dim=4)
>>> pql.train(env, num_episodes=1000)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import time

from .base_morl import MORLBase, MultiObjectiveReward, ParetoFront
from ...core.base import RLState, RLAction


class MultiObjectiveQNetwork(nn.Module):
    """Q-network that outputs Q-values for each objective."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_objectives: int,
        hidden_dim: int = 128
    ):
        """Initialize multi-objective Q-network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            num_objectives: Number of objectives
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.num_objectives = num_objectives
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for each objective
        self.objective_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim)
            for _ in range(num_objectives)
        ])
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            states: State tensor [batch_size, state_dim]
            
        Returns:
            Q-values tensor [batch_size, num_objectives, action_dim]
        """
        features = self.feature_net(states)
        
        q_values = []
        for head in self.objective_heads:
            q_values.append(head(features))
        
        # Stack along objective dimension
        return torch.stack(q_values, dim=1)


class ParetoQLearning(MORLBase):
    """Pareto Q-Learning for multi-objective RL."""
    
    def __init__(
        self,
        num_objectives: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        pareto_selection: str = 'hypervolume',
        device: str = None
    ):
        """Initialize Pareto Q-Learning.
        
        Args:
            num_objectives: Number of objectives
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension
            learning_rate: Learning rate
            discount_factor: Discount factor
            epsilon: Exploration rate
            pareto_selection: How to select from Pareto set ('random', 'hypervolume', 'crowding')
            device: Device for computation
        """
        super().__init__(num_objectives, state_dim, action_dim, device=device)
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.pareto_selection = pareto_selection
        
        # Q-networks
        self.q_network = MultiObjectiveQNetwork(
            state_dim, action_dim, num_objectives, hidden_dim
        ).to(self.device)
        
        self.target_network = MultiObjectiveQNetwork(
            state_dim, action_dim, num_objectives, hidden_dim
        ).to(self.device)
        
        # Copy weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.buffer = []
        self.buffer_size = 10000
        
        # Track non-dominated Q-values per state-action
        self.nd_q_sets = defaultdict(list)  # (state_hash, action) -> list of Q-vectors
    
    def _get_pareto_actions(self, q_values: torch.Tensor) -> List[int]:
        """Get Pareto-optimal actions for given Q-values.
        
        Args:
            q_values: Q-values [num_objectives, action_dim]
            
        Returns:
            List of Pareto-optimal action indices
        """
        q_np = q_values.cpu().numpy()
        pareto_actions = []
        
        for action in range(self.action_dim):
            dominated = False
            for other_action in range(self.action_dim):
                if action != other_action:
                    # Check if other_action dominates action
                    if (q_np[:, other_action] >= q_np[:, action]).all() and \
                       (q_np[:, other_action] > q_np[:, action]).any():
                        dominated = True
                        break
            
            if not dominated:
                pareto_actions.append(action)
        
        return pareto_actions
    
    def _select_from_pareto(
        self,
        pareto_actions: List[int],
        q_values: torch.Tensor,
        preference: Optional[np.ndarray] = None
    ) -> int:
        """Select action from Pareto set.
        
        Args:
            pareto_actions: List of Pareto-optimal actions
            q_values: Q-values for all actions
            preference: Optional preference weights
            
        Returns:
            Selected action
        """
        if len(pareto_actions) == 1:
            return pareto_actions[0]
        
        if preference is not None:
            # Use preference to select
            pref_tensor = torch.FloatTensor(preference).to(self.device)
            scalarized = (q_values * pref_tensor.unsqueeze(-1)).sum(dim=0)
            best_action = pareto_actions[0]
            best_value = scalarized[best_action]
            
            for action in pareto_actions[1:]:
                if scalarized[action] > best_value:
                    best_value = scalarized[action]
                    best_action = action
            
            return best_action
        
        elif self.pareto_selection == 'random':
            return np.random.choice(pareto_actions)
        
        elif self.pareto_selection == 'hypervolume':
            # Select action that maximizes hypervolume contribution
            best_action = pareto_actions[0]
            best_hv = np.prod(q_values[:, best_action].cpu().numpy())
            
            for action in pareto_actions[1:]:
                hv = np.prod(q_values[:, action].cpu().numpy())
                if hv > best_hv:
                    best_hv = hv
                    best_action = action
            
            return best_action
        
        else:  # crowding distance or other methods
            return np.random.choice(pareto_actions)
    
    def select_action(
        self,
        state: RLState,
        preference: Optional[np.ndarray] = None
    ) -> RLAction:
        """Select action using Pareto Q-learning.
        
        Args:
            state: Current state
            preference: Optional preference for tie-breaking
            
        Returns:
            Selected action
        """
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return RLAction(
                action_type='discrete',
                action_id=np.random.randint(self.action_dim),
                parameters={'exploration': True}
            )
        
        # Get Q-values
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)  # [num_objectives, action_dim]
        
        # Find Pareto-optimal actions
        pareto_actions = self._get_pareto_actions(q_values)
        
        # Select from Pareto set
        action = self._select_from_pareto(pareto_actions, q_values, preference)
        
        return RLAction(
            action_type='discrete',
            action_id=action,
            parameters={
                'pareto_set_size': len(pareto_actions),
                'q_values': q_values.cpu().numpy()
            }
        )
    
    def update(
        self,
        state: RLState,
        action: RLAction,
        reward: MultiObjectiveReward,
        next_state: RLState,
        done: bool
    ):
        """Update Q-values with experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Multi-objective reward
            next_state: Next state
            done: Episode done flag
        """
        # Store experience
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        
        self.buffer.append({
            'state': state.features,
            'action': action.action_id,
            'reward': reward.values,
            'next_state': next_state.features,
            'done': done
        })
        
        # Update Pareto front
        self.pareto_front.update(state, action, reward)
        
        # Train if enough experiences
        if len(self.buffer) >= 32:
            self._train_batch(32)
    
    def _train_batch(self, batch_size: int):
        """Train on batch of experiences."""
        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        # Convert to tensors
        states = torch.FloatTensor([e['state'] for e in batch]).to(self.device)
        actions = torch.LongTensor([e['action'] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e['done'] for e in batch]).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states)  # [batch, objectives, actions]
        current_q_selected = current_q.gather(
            2, actions.unsqueeze(1).unsqueeze(2).expand(-1, self.num_objectives, -1)
        ).squeeze(2)  # [batch, objectives]
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states)  # [batch, objectives, actions]
            
            # For each sample, find Pareto-optimal actions
            next_q_max = torch.zeros(batch_size, self.num_objectives).to(self.device)
            
            for i in range(batch_size):
                if not dones[i]:
                    # Get Pareto actions for next state
                    pareto_actions = self._get_pareto_actions(next_q[i])
                    
                    if pareto_actions:
                        # Use optimistic assumption - take max over Pareto set
                        pareto_q = next_q[i, :, pareto_actions]
                        next_q_max[i] = pareto_q.max(dim=1)[0]
        
        # Compute targets
        targets = rewards + self.discount_factor * next_q_max * (1 - dones.unsqueeze(1))
        
        # Loss
        loss = nn.functional.mse_loss(current_q_selected, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Record
        self.training_history['loss'].append(loss.item())
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(
        self,
        env: Any,
        num_episodes: int,
        preference_sampling: str = 'uniform'
    ) -> Dict[str, Any]:
        """Train Pareto Q-Learning.
        
        Args:
            env: Multi-objective environment
            num_episodes: Number of episodes
            preference_sampling: Not used directly, but kept for compatibility
            
        Returns:
            Training results
        """
        episode_rewards = []
        episode_lengths = []
        pareto_set_sizes = []
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = np.zeros(self.num_objectives)
            episode_length = 0
            episode_pareto_sizes = []
            
            while not done:
                # Select action
                action = self.select_action(RLState(features=state))
                episode_pareto_sizes.append(action.parameters.get('pareto_set_size', 1))
                
                # Take action
                next_state, reward, done, info = env.step(action.action_id)
                
                # Convert reward
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
                    done
                )
                
                episode_reward += mo_reward.values
                state = next_state
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            pareto_set_sizes.append(np.mean(episode_pareto_sizes))
            
            # Update target network
            if episode % 10 == 0:
                self.update_target_network()
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            if episode % 100 == 0:
                avg_rewards = np.mean(episode_rewards[-100:], axis=0)
                avg_pareto_size = np.mean(pareto_set_sizes[-100:])
                print(f"Episode {episode}: Avg Rewards = {avg_rewards}, "
                      f"Avg Pareto Actions = {avg_pareto_size:.2f}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'pareto_set_sizes': pareto_set_sizes,
            'final_pareto_front_size': len(self.pareto_front.solutions),
            'pareto_coverage': self.pareto_front.get_coverage()
        }


if __name__ == "__main__":
    # Validation test
    pql = ParetoQLearning(
        num_objectives=3,
        state_dim=8,
        action_dim=4
    )
    
    # Test Pareto action detection
    q_values = torch.tensor([
        [0.5, 0.8, 0.3, 0.6],  # Obj 1
        [0.9, 0.2, 0.7, 0.4],  # Obj 2
        [0.1, 0.6, 0.8, 0.2]   # Obj 3
    ])
    
    pareto_actions = pql._get_pareto_actions(q_values)
    print(f"Pareto actions: {pareto_actions}")
    
    # Actions 0, 1, 2 should be Pareto-optimal
    assert 0 in pareto_actions, "Action 0 should be Pareto-optimal"
    assert 1 in pareto_actions, "Action 1 should be Pareto-optimal"
    assert 2 in pareto_actions, "Action 2 should be Pareto-optimal"
    
    # Test action selection
    state = RLState(features=np.random.randn(8))
    action = pql.select_action(state)
    
    assert isinstance(action, RLAction), "Invalid action type"
    assert 0 <= action.action_id < 4, f"Invalid action: {action.action_id}"
    
    print(" Pareto Q-Learning validation passed")