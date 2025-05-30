"""Deep Q-Network (DQN) implementation for sequential decision making"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import logging

from ...core.base import RLAgent, RLState, RLAction, RLReward
from ...core.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNetwork(nn.Module):
    """Neural network for Q-value approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent(RLAgent):
    """
    Deep Q-Network agent for discrete action spaces
    Suitable for sequential decision making like processing strategy selection
    """
    
    def __init__(self,
                 name: str,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-3,
                 discount_factor: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 buffer_size: int = 10000,
                 target_update_freq: int = 100,
                 hidden_dims: List[int] = None,
                 device: str = "cpu",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize DQN agent
        
        Args:
            name: Agent name
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for optimizer
            discount_factor: Gamma for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            batch_size: Batch size for training
            buffer_size: Replay buffer capacity
            target_update_freq: Steps between target network updates
            hidden_dims: Hidden layer dimensions
            device: Device to run on (cpu/cuda)
            config: Additional configuration
        """
        super().__init__(name, config)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
        self.q_values = []
        
    def select_action(self, state: RLState, explore: bool = True) -> RLAction:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            explore: Whether to explore (training mode)
            
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if explore and self.training and np.random.random() < self.epsilon:
            action_id = np.random.randint(self.action_dim)
            exploration_used = True
        else:
            # Use Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action_id = q_values.argmax().item()
                exploration_used = False
                
                # Track Q-values for monitoring
                self.q_values.append(q_values.cpu().numpy().flatten())
        
        return RLAction(
            action_type="discrete_action",
            action_id=int(action_id),
            parameters={
                "epsilon": self.epsilon,
                "exploration": exploration_used,
                "q_values": self.q_values[-1].tolist() if self.q_values else None
            }
        )
    
    def update(self,
               state: RLState,
               action: RLAction,
               reward: RLReward,
               next_state: RLState,
               done: bool = False) -> Dict[str, float]:
        """
        Update DQN using experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            
        Returns:
            Training metrics
        """
        # Store experience
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Only train if we have enough experiences
        if not self.replay_buffer.is_ready(self.batch_size):
            return {"buffer_size": len(self.replay_buffer)}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state.features for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action.action_id for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward.value for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state.features for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Track metrics
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        metrics = {
            "loss": loss_value,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "avg_q_value": current_q_values.mean().item(),
            "training_steps": self.training_steps
        }
        
        return metrics
    
    def reset_episode(self) -> None:
        """Reset for new episode"""
        super().reset_episode()
        
        # Decay epsilon
        if self.training:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: Path) -> None:
        """Save DQN state"""
        path = Path(path)
        
        checkpoint = {
            "name": self.name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "episodes": self.episodes,
            "config": self.config
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved DQN to {path}")
        
        # Save metrics separately
        metrics_path = path.parent / f"{path.stem}_metrics.json"
        metrics = {
            "losses": self.losses[-1000:],  # Last 1000 losses
            "avg_loss": np.mean(self.losses[-100:]) if self.losses else 0,
            "epsilon": self.epsilon,
            "training_steps": self.training_steps
        }
        metrics_path.write_text(json.dumps(metrics, indent=2))
    
    def load(self, path: Path) -> None:
        """Load DQN state"""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        # Validate compatibility
        if checkpoint["state_dim"] != self.state_dim:
            raise ValueError(f"Incompatible state_dim: {checkpoint['state_dim']} != {self.state_dim}")
        if checkpoint["action_dim"] != self.action_dim:
            raise ValueError(f"Incompatible action_dim: {checkpoint['action_dim']} != {self.action_dim}")
        
        # Load states
        self.name = checkpoint["name"]
        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.episodes = checkpoint["episodes"]
        self.config = checkpoint.get("config", {})
        
        self.logger.info(f"Loaded DQN from {path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = super().get_metrics()
        
        # Add DQN-specific metrics
        metrics.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "avg_loss": np.mean(self.losses[-100:]) if self.losses else 0,
            "avg_q_value": np.mean([q.mean() for q in self.q_values[-100:]]) if self.q_values else 0
        })
        
        return metrics


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN variant that reduces overestimation bias
    Uses Q-network to select actions and target network to evaluate them
    """
    
    def update(self,
               state: RLState,
               action: RLAction,
               reward: RLReward,
               next_state: RLState,
               done: bool = False) -> Dict[str, float]:
        """Update using Double DQN algorithm"""
        # Store experience
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if not self.replay_buffer.is_ready(self.batch_size):
            return {"buffer_size": len(self.replay_buffer)}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state.features for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action.action_id for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward.value for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state.features for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use Q-network to select actions, target network to evaluate
        with torch.no_grad():
            # Select best actions using Q-network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate those actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Track metrics
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return {
            "loss": loss_value,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "avg_q_value": current_q_values.mean().item(),
            "training_steps": self.training_steps
        }
