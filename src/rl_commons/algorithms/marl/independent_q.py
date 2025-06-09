"""
Module: independent_q.py
Purpose: Independent Q-Learning for decentralized multi-agent RL

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- torch: https://pytorch.org/docs/stable/

Example Usage:
>>> from rl_commons.algorithms.marl import IndependentQLearning, MARLEnvironment
>>> agent = IndependentQLearning(agent_id=0, obs_dim=4, action_dim=4)
>>> obs = AgentObservation(agent_id=0, local_obs=np.zeros(4))
>>> action = agent.act(obs)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from collections import deque
import random

from .base_marl import MARLAgent, AgentObservation, MARLTransition
from ...core.base import RLAction
from collections import deque
import random as random_module


class QNetwork(nn.Module):
    """Q-network for value estimation."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class IndependentQLearning(MARLAgent):
    """Independent Q-Learning agent for decentralized MARL."""
    
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 32,
        update_frequency: int = 4,
        hidden_dim: int = 128
    ):
        """Initialize Independent Q-Learning agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_dim: Dimension of observation space
            action_dim: Number of discrete actions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            update_frequency: Steps between network updates
            hidden_dim: Hidden layer dimension
        """
        super().__init__(agent_id, obs_dim, action_dim, learning_rate)
        
        # Q-learning parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        # Networks
        self.q_network = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Simple replay buffer for MARL
        self.replay_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        
        # Training tracking
        self.steps = 0
        self.update_count = 0
        self.losses = deque(maxlen=100)
    
    def act(self, observation: AgentObservation, explore: bool = True) -> RLAction:
        """Select action using epsilon-greedy policy.
        
        Args:
            observation: Current agent observation
            explore: Whether to use exploration
            
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            action_id = random.randint(0, self.action_dim - 1)
        else:
            # Get Q-values
            with torch.no_grad():
                obs_tensor = observation.to_tensor(self.device)
                q_values = self.q_network(obs_tensor.unsqueeze(0))
                action_id = q_values.argmax(dim=1).item()
        
        # Check for messages from other agents
        if self.communication_inbox:
            # Process messages (could influence action selection)
            recent_messages = self.communication_inbox[-5:]  # Last 5 messages
            # In a more sophisticated implementation, we could use messages
            # to coordinate actions
        
        return RLAction(
            action_type='discrete',
            action_id=action_id,
            parameters={'agent_id': self.agent_id}
        )
    
    def update(self, transition: MARLTransition):
        """Update Q-network based on transition.
        
        Args:
            transition: Multi-agent transition data
        """
        # Extract this agent's data
        agent_idx = next(
            i for i, obs in enumerate(transition.observations) 
            if obs.agent_id == self.agent_id
        )
        
        obs = transition.observations[agent_idx].local_obs
        action = transition.actions[agent_idx].action_id
        reward = transition.rewards[agent_idx]
        next_obs = transition.next_observations[agent_idx].local_obs
        done = transition.dones[agent_idx]
        
        # Store in replay buffer
        self.replay_buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        })
        
        # Update network
        if len(self.replay_buffer) >= self.batch_size:
            if self.steps % self.update_frequency == 0:
                self._update_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
    
    def _update_network(self):
        """Perform network update using replay buffer."""
        # Sample batch
        batch = random_module.sample(self.replay_buffer, self.batch_size)
        
        states = torch.FloatTensor([b['obs'] for b in batch]).to(self.device)
        actions = torch.LongTensor([b['action'] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b['reward'] for b in batch]).to(self.device)
        next_states = torch.FloatTensor([b['next_obs'] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b['done'] for b in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Track loss
        self.losses.append(loss.item())
        self.update_count += 1
        
        # Update target network
        if self.update_count % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics."""
        return {
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'buffer_size': len(self.replay_buffer),
            'update_count': self.update_count
        }
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.update_count = checkpoint['update_count']


if __name__ == "__main__":
    # Validation test
    agent = IndependentQLearning(
        agent_id=0,
        obs_dim=4,
        action_dim=4,
        learning_rate=0.001
    )
    
    # Test action selection
    obs = AgentObservation(
        agent_id=0,
        local_obs=np.random.randn(4).astype(np.float32)
    )
    action = agent.act(obs, explore=True)
    
    assert isinstance(action, RLAction), "Invalid action type"
    assert 0 <= action.action_id < 4, f"Invalid action id: {action.action_id}"
    
    # Test update with dummy transition
    transition = MARLTransition(
        observations=[obs],
        actions=[action],
        rewards=[0.5],
        next_observations=[
            AgentObservation(
                agent_id=0,
                local_obs=np.random.randn(4).astype(np.float32)
            )
        ],
        dones=[False],
        info={}
    )
    
    agent.update(transition)
    metrics = agent.get_metrics()
    
    assert 'epsilon' in metrics, "Missing epsilon in metrics"
    assert metrics['buffer_size'] == 1, f"Expected buffer size 1, got {metrics['buffer_size']}"
    
    print(" Independent Q-Learning validation passed")