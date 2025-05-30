"""Proximal Policy Optimization (PPO) implementation for continuous control"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import json
import logging
from collections import deque

from ...core.base import RLAgent, RLState, RLAction, RLReward

logger = logging.getLogger(__name__)


class ActorCriticNetwork(nn.Module):
    """Combined Actor-Critic network for PPO"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = None,
                 continuous: bool = True,
                 action_bounds: Optional[Tuple[float, float]] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.continuous = continuous
        self.action_bounds = action_bounds
        
        # Shared layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Actor (policy) head
        self.actor_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        # Critic (value) head
        self.critic_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # For continuous actions, we also need log_std
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Special initialization for output layers
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Returns:
            mean: Mean of action distribution
            std: Standard deviation of action distribution (continuous) or None
            value: State value estimate
        """
        shared_features = self.shared(state)
        
        # Actor output
        action_mean = self.actor_head(shared_features)
        
        # Apply action bounds if specified
        if self.continuous and self.action_bounds is not None:
            action_mean = torch.tanh(action_mean)
            low, high = self.action_bounds
            action_mean = low + (high - low) * (action_mean + 1) / 2
        
        # Standard deviation for continuous actions
        if self.continuous:
            action_std = torch.exp(self.log_std)
        else:
            action_std = None
        
        # Critic output
        value = self.critic_head(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy"""
        mean, std, value = self.forward(state)
        
        if self.continuous:
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                
            # Clip actions to bounds
            if self.action_bounds is not None:
                low, high = self.action_bounds
                action = torch.clamp(action, low, high)
                
            return action, dist.log_prob(action).sum(-1), value
        else:
            # Discrete actions
            probs = F.softmax(mean, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
            
            return action, dist.log_prob(action), value


class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset the buffer"""
        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.position = 0
        self.full = False
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            value: float, log_prob: float, done: bool):
        """Add experience to buffer"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.buffer_size
        if self.position == 0:
            self.full = True
    
    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation"""
        last_gae_lam = 0
        n_steps = self.buffer_size if self.full else self.position
        
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            last_gae_lam = self.advantages[step]
        
        self.returns[:n_steps] = self.advantages[:n_steps] + self.values[:n_steps]
    
    def get_samples(self, batch_size: int):
        """Get random samples from buffer"""
        n_steps = self.buffer_size if self.full else self.position
        indices = np.random.choice(n_steps, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'old_log_probs': torch.FloatTensor(self.log_probs[indices]).to(self.device),
            'advantages': torch.FloatTensor(self.advantages[indices]).to(self.device),
            'returns': torch.FloatTensor(self.returns[indices]).to(self.device)
        }


class PPOAgent(RLAgent):
    """
    Proximal Policy Optimization agent for continuous control
    Suitable for parameter optimization in systems like ArangoDB
    """
    
    def __init__(self,
                 name: str,
                 state_dim: int,
                 action_dim: int,
                 continuous: bool = True,
                 action_bounds: Optional[Tuple[float, float]] = None,
                 learning_rate: float = 3e-4,
                 discount_factor: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 buffer_size: int = 2048,
                 hidden_dims: List[int] = None,
                 device: str = "cpu",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize PPO agent
        
        Args:
            name: Agent name
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            continuous: Whether action space is continuous
            action_bounds: Optional bounds for continuous actions (low, high)
            learning_rate: Learning rate for optimizer
            discount_factor: Gamma for future rewards
            gae_lambda: Lambda for GAE
            clip_ratio: Epsilon for PPO clipping
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of epochs per update
            batch_size: Batch size for training
            buffer_size: Rollout buffer size
            hidden_dims: Hidden layer dimensions
            device: Device to run on (cpu/cuda)
            config: Additional configuration
        """
        super().__init__(name, config)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.action_bounds = action_bounds
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else cpu)
        
        # Network
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_dims, continuous, action_bounds
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size, state_dim, action_dim, self.device
        )
        
        # Training stats
        self.training_steps = 0
        self.episodes = 0
        self.training = True
        
        # Metrics tracking
        self.metrics_history = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'actor_losses': deque(maxlen=100),
            'critic_losses': deque(maxlen=100),
            'entropy_losses': deque(maxlen=100),
            'total_losses': deque(maxlen=100)
        }
        
        # Episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        logger.info(f"Initialized PPO agent {name} with state_dim={state_dim}, action_dim={action_dim}, continuous={continuous}")
    
    def select_action(self, state: RLState, deterministic: bool = False) -> RLAction:
        """Select action using current policy"""
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
        
        # Get action from network
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)
        
        # Convert to numpy
        action_np = action.cpu().numpy().squeeze()
        log_prob_np = log_prob.cpu().numpy().item()
        value_np = value.cpu().numpy().item()
        
        # Store for buffer if training
        if self.training and hasattr(self, '_last_state'):
            self.rollout_buffer.add(
                self._last_state.features,
                action_np if self.continuous else np.array([action_np]),
                0.0,  # Reward will be updated in update method
                value_np,
                log_prob_np,
                False  # Done will be updated
            )
        
        self._last_state = state
        self._last_value = value_np
        
        # Create action object
        if self.continuous:
            return RLAction(
                action_type="continuous",
                action_id=0,
                parameters={"values": action_np.tolist()}
            )
        else:
            return RLAction(
                action_type="discrete",
                action_id=int(action_np),
                parameters=None
            )
    
    def update(self, state: RLState, action: RLAction, reward: RLReward, 
               next_state: RLState, done: bool) -> Dict[str, Any]:
        """Update agent with experience"""
        # Update episode tracking
        self.current_episode_reward += reward.value
        self.current_episode_length += 1
        
        # Update buffer with actual reward and done
        if self.rollout_buffer.position > 0:
            self.rollout_buffer.rewards[self.rollout_buffer.position - 1] = reward.value
            self.rollout_buffer.dones[self.rollout_buffer.position - 1] = done
        
        # If episode ended or buffer full, perform update
        if done:
            self.episodes += 1
            self.metrics_history['episode_rewards'].append(self.current_episode_reward)
            self.metrics_history['episode_lengths'].append(self.current_episode_length)
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Compute GAE and train
            if self.training and self.rollout_buffer.position > self.batch_size:
                losses = self._train_on_rollout()
                return losses
        elif self.rollout_buffer.full:
            # Buffer is full, need to train
            if self.training:
                # Compute value of next state for GAE
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(next_state.features).unsqueeze(0).to(self.device)
                    _, _, next_value = self.network(next_state_tensor)
                    next_value = next_value.cpu().numpy().item()
                
                losses = self._train_on_rollout(last_value=next_value)
                return losses
        
        return {}
    
    def _train_on_rollout(self, last_value: float = 0.0) -> Dict[str, float]:
        """Train on collected rollout data"""
        # Compute GAE
        self.rollout_buffer.compute_gae(last_value, self.discount_factor, self.gae_lambda)
        
        # Normalize advantages
        n_steps = self.rollout_buffer.buffer_size if self.rollout_buffer.full else self.rollout_buffer.position
        advantages = self.rollout_buffer.advantages[:n_steps]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.rollout_buffer.advantages[:n_steps] = advantages
        
        # Training loop
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            # Sample from buffer
            batch = self.rollout_buffer.get_samples(min(self.batch_size, n_steps))
            
            # Forward pass
            mean, std, values = self.network(batch['states'])
            
            # Calculate log probabilities
            if self.continuous:
                dist = Normal(mean, std)
                log_probs = dist.log_prob(batch['actions']).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
            else:
                probs = F.softmax(mean, dim=-1)
                dist = Categorical(probs)
                log_probs = dist.log_prob(batch['actions'].squeeze(-1))
                entropy = dist.entropy().mean()
            
            # PPO loss
            ratio = torch.exp(log_probs - batch['old_log_probs'])
            surr1 = ratio * batch['advantages']
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch['advantages']
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values_clipped = batch['returns']  # Could add value clipping here
            critic_loss = F.mse_loss(values.squeeze(-1), batch['returns'])
            
            # Total loss
            loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Track losses
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy.item()
            total_loss += loss.item()
            n_updates += 1
        
        # Average losses
        avg_actor_loss = total_actor_loss / n_updates
        avg_critic_loss = total_critic_loss / n_updates
        avg_entropy_loss = total_entropy_loss / n_updates
        avg_total_loss = total_loss / n_updates
        
        # Update metrics
        self.metrics_history['actor_losses'].append(avg_actor_loss)
        self.metrics_history['critic_losses'].append(avg_critic_loss)
        self.metrics_history['entropy_losses'].append(avg_entropy_loss)
        self.metrics_history['total_losses'].append(avg_total_loss)
        
        # Reset buffer
        self.rollout_buffer.reset()
        
        # Update training steps
        self.training_steps += n_steps
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy_loss,
            'total_loss': avg_total_loss,
            'training_steps': self.training_steps
        }
    
    def save(self, path: Path):
        """Save agent state"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save network
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episodes': self.episodes
        }, path / f"{self.name}_model.pt")
        
        # Save configuration
        config = {
            'name': self.name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'continuous': self.continuous,
            'action_bounds': self.action_bounds,
            'config': self.config,
            'metrics_history': {k: list(v) for k, v in self.metrics_history.items()}
        }
        
        with open(path / f"{self.name}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved PPO agent {self.name} to {path}")
    
    def load(self, path: Path):
        """Load agent state"""
        path = Path(path)
        
        # Load network
        checkpoint = torch.load(path / f"{self.name}_model.pt", map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.episodes = checkpoint['episodes']
        
        # Load configuration
        with open(path / f"{self.name}_config.json", 'r') as f:
            config = json.load(f)
        
        # Restore metrics history
        for k, v in config.get('metrics_history', {}).items():
            self.metrics_history[k] = deque(v, maxlen=100)
        
        logger.info(f"Loaded PPO agent {self.name} from {path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = {
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'training': self.training
        }
        
        # Add average metrics if available
        for key, values in self.metrics_history.items():
            if len(values) > 0:
                metrics[f'avg_{key}'] = np.mean(list(values))
                metrics[f'recent_{key}'] = list(values)[-10:]
        
        return metrics
    
    def reset(self):
        """Reset agent for new episode"""
        self.current_episode_reward = 0
        self.current_episode_length = 0
        if hasattr(self, '_last_state'):
            delattr(self, '_last_state')
        if hasattr(self, '_last_value'):
            delattr(self, '_last_value')
