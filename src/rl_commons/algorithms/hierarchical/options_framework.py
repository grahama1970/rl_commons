"""Hierarchical RL using Options Framework for complex orchestration

Module: options_framework.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import random

from ...core.base import RLAgent, RLState, RLAction, RLReward
from ...core.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class Option:
    """
    An option in hierarchical RL represents a temporally extended action
    
    Attributes:
        id: Unique identifier for the option
        name: Human-readable name
        initiation_set: States where this option can be initiated
        termination_condition: Function that determines when option terminates
        policy: The intra-option policy (low-level actions)
    """
    id: int
    name: str
    initiation_set: Callable[[np.ndarray], bool]
    termination_condition: Callable[[np.ndarray], float]  # Returns termination probability
    policy: Optional['OptionPolicy'] = None
    
    def is_available(self, state: np.ndarray) -> bool:
        """Check if option can be initiated in current state"""
        return self.initiation_set(state)
        
    def should_terminate(self, state: np.ndarray) -> bool:
        """Sample termination based on termination probability"""
        term_prob = self.termination_condition(state)
        return np.random.random() < term_prob


class OptionPolicy(nn.Module):
    """Neural network policy for executing an option"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 64]
            
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self.action_dim = action_dim
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities"""
        logits = self.network(state)
        return F.softmax(logits, dim=-1)
        
    def sample_action(self, state: np.ndarray) -> int:
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.forward(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item()


class MetaController(nn.Module):
    """High-level controller that selects options"""
    
    def __init__(self, state_dim: int, num_options: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
            
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
            
        # Output Q-values for each option
        layers.append(nn.Linear(input_dim, num_options))
        
        self.network = nn.Sequential(*layers)
        self.num_options = num_options
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all options"""
        return self.network(state)
        
    def select_option(self, state: np.ndarray, available_options: List[int], epsilon: float = 0.1) -> int:
        """Select option using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            # Random selection from available options
            return np.random.choice(available_options)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state_tensor).squeeze().numpy()
            
        # Mask unavailable options
        masked_q_values = np.full(self.num_options, -np.inf)
        for opt in available_options:
            masked_q_values[opt] = q_values[opt]
            
        return int(np.argmax(masked_q_values))


class HierarchicalReplayBuffer:
    """Replay buffer for hierarchical RL"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.meta_buffer = deque(maxlen=capacity)
        self.option_buffers = {}  # Separate buffer for each option
        
    def add_meta_transition(self, 
                           state: np.ndarray,
                           option: int,
                           reward: float,
                           next_state: np.ndarray,
                           done: bool,
                           steps: int):
        """Add meta-level transition"""
        self.meta_buffer.append({
            'state': state,
            'option': option,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'steps': steps
        })
        
    def add_option_transition(self,
                            option_id: int,
                            state: np.ndarray,
                            action: int,
                            reward: float,
                            next_state: np.ndarray,
                            done: bool):
        """Add option-level transition"""
        if option_id not in self.option_buffers:
            self.option_buffers[option_id] = deque(maxlen=self.capacity)
            
        self.option_buffers[option_id].append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
    def sample_meta_batch(self, batch_size: int) -> List[Dict]:
        """Sample batch of meta-level transitions"""
        return random.sample(self.meta_buffer, min(batch_size, len(self.meta_buffer)))
        
    def sample_option_batch(self, option_id: int, batch_size: int) -> List[Dict]:
        """Sample batch of option-level transitions"""
        if option_id not in self.option_buffers:
            return []
        buffer = self.option_buffers[option_id]
        return random.sample(buffer, min(batch_size, len(buffer)))


class HierarchicalRLAgent(RLAgent):
    """
    Hierarchical RL agent using Options Framework
    Suitable for complex orchestration tasks like module coordination
    """
    
    def __init__(self,
                 state_dim: int,
                 primitive_actions: int,
                 options: List[Option],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 meta_epsilon: float = 0.1,
                 option_epsilon: float = 0.1,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 update_frequency: int = 4):
        """
        Initialize hierarchical RL agent
        
        Args:
            state_dim: Dimension of state space
            primitive_actions: Number of primitive (low-level) actions
            options: List of available options
            learning_rate: Learning rate for networks
            gamma: Discount factor
            meta_epsilon: Exploration rate for meta-controller
            option_epsilon: Exploration rate for option policies
            buffer_size: Size of replay buffers
            batch_size: Batch size for training
            update_frequency: Steps between network updates
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.primitive_actions = primitive_actions
        self.options = options
        self.num_options = len(options)
        self.gamma = gamma
        self.meta_epsilon = meta_epsilon
        self.option_epsilon = option_epsilon
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        # Create meta-controller
        self.meta_controller = MetaController(state_dim, self.num_options)
        self.meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=learning_rate)
        
        # Create option policies
        for i, option in enumerate(self.options):
            if option.policy is None:
                option.policy = OptionPolicy(state_dim, primitive_actions)
        
        self.option_optimizers = {
            i: optim.Adam(opt.policy.parameters(), lr=learning_rate)
            for i, opt in enumerate(self.options)
            if opt.policy is not None
        }
        
        # Replay buffer
        self.replay_buffer = HierarchicalReplayBuffer(buffer_size)
        
        # Tracking
        self.current_option = None
        self.option_start_state = None
        self.option_cumulative_reward = 0
        self.option_steps = 0
        self.total_steps = 0
        
    def select_action(self, state: RLState) -> RLAction:
        """
        Select action using hierarchical policy
        
        Returns primitive action based on current option
        """
        state_array = self._state_to_array(state)
        
        # Check if we need to select a new option
        if self.current_option is None or self.options[self.current_option].should_terminate(state_array):
            # If option terminated, store transition
            if self.current_option is not None:
                self.replay_buffer.add_meta_transition(
                    self.option_start_state,
                    self.current_option,
                    self.option_cumulative_reward,
                    state_array,
                    False,  # Episode not done
                    self.option_steps
                )
                
            # Select new option
            available_options = [
                i for i, opt in enumerate(self.options)
                if opt.is_available(state_array)
            ]
            
            if not available_options:
                # Fallback to random action if no options available
                logger.warning("No options available, selecting random action")
                return RLAction(value=np.random.randint(self.primitive_actions))
                
            self.current_option = self.meta_controller.select_option(
                state_array, available_options, self.meta_epsilon
            )
            
            self.option_start_state = state_array
            self.option_cumulative_reward = 0
            self.option_steps = 0
            
            logger.debug(f"Selected option: {self.options[self.current_option].name}")
            
        # Execute current option's policy
        option = self.options[self.current_option]
        if option.policy is not None:
            action = option.policy.sample_action(state_array)
        else:
            # Random action if no policy defined
            action = np.random.randint(self.primitive_actions)
            
        return RLAction(value=action)
        
    def update(self, 
              state: RLState,
              action: RLAction,
              reward: RLReward,
              next_state: RLState,
              done: bool = False):
        """Update both meta-controller and option policies"""
        state_array = self._state_to_array(state)
        next_state_array = self._state_to_array(next_state)
        reward_value = self._reward_to_float(reward)
        action_value = self._action_to_int(action)
        
        # Update option-level info
        self.option_cumulative_reward += reward_value
        self.option_steps += 1
        
        # Store option-level transition
        if self.current_option is not None:
            self.replay_buffer.add_option_transition(
                self.current_option,
                state_array,
                action_value,
                reward_value,
                next_state_array,
                done
            )
            
        # If episode ended, store final meta transition
        if done and self.current_option is not None:
            self.replay_buffer.add_meta_transition(
                self.option_start_state,
                self.current_option,
                self.option_cumulative_reward,
                next_state_array,
                done,
                self.option_steps
            )
            self.current_option = None
            
        # Update networks
        self.total_steps += 1
        if self.total_steps % self.update_frequency == 0:
            self._update_networks()
            
    def _update_networks(self):
        """Update meta-controller and option policies"""
        # Update meta-controller
        if len(self.replay_buffer.meta_buffer) >= self.batch_size:
            self._update_meta_controller()
            
        # Update option policies
        for option_id in range(self.num_options):
            if (option_id in self.replay_buffer.option_buffers and 
                len(self.replay_buffer.option_buffers[option_id]) >= self.batch_size):
                self._update_option_policy(option_id)
                
    def _update_meta_controller(self):
        """Update meta-controller using Q-learning"""
        batch = self.replay_buffer.sample_meta_batch(self.batch_size)
        
        states = torch.FloatTensor([t['state'] for t in batch])
        options = torch.LongTensor([t['option'] for t in batch])
        rewards = torch.FloatTensor([t['reward'] for t in batch])
        next_states = torch.FloatTensor([t['next_state'] for t in batch])
        dones = torch.FloatTensor([t['done'] for t in batch])
        
        # Current Q-values
        current_q_values = self.meta_controller(states)
        current_q_values = current_q_values.gather(1, options.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.meta_controller(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Update
        self.meta_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_controller.parameters(), 1.0)
        self.meta_optimizer.step()
        
    def _update_option_policy(self, option_id: int):
        """Update option policy using policy gradient"""
        if self.options[option_id].policy is None:
            return
            
        batch = self.replay_buffer.sample_option_batch(option_id, self.batch_size)
        
        states = torch.FloatTensor([t['state'] for t in batch])
        actions = torch.LongTensor([t['action'] for t in batch])
        rewards = torch.FloatTensor([t['reward'] for t in batch])
        
        # Get action probabilities
        policy = self.options[option_id].policy
        action_probs = policy(states)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1))
        
        # Compute policy gradient loss
        loss = -torch.mean(torch.log(selected_action_probs) * rewards)
        
        # Update
        optimizer = self.option_optimizers[option_id]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
    def save(self, path: str):
        """Save agent state"""
        state_dict = {
            'meta_controller': self.meta_controller.state_dict(),
            'option_policies': {
                i: opt.policy.state_dict() 
                for i, opt in enumerate(self.options) 
                if opt.policy is not None
            },
            'config': {
                'state_dim': self.state_dim,
                'primitive_actions': self.primitive_actions,
                'num_options': self.num_options,
                'gamma': self.gamma,
                'meta_epsilon': self.meta_epsilon,
                'option_epsilon': self.option_epsilon
            }
        }
        torch.save(state_dict, path)
        
    def load(self, path: str):
        """Load agent state"""
        state_dict = torch.load(path)
        self.meta_controller.load_state_dict(state_dict['meta_controller'])
        
        for option_id, policy_state in state_dict['option_policies'].items():
            if self.options[int(option_id)].policy is not None:
                self.options[int(option_id)].policy.load_state_dict(policy_state)
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        option_usage = {}
        for i, opt in enumerate(self.options):
            usage_count = sum(1 for t in self.replay_buffer.meta_buffer if t['option'] == i)
            option_usage[opt.name] = usage_count
            
        return {
            "total_steps": self.total_steps,
            "meta_buffer_size": len(self.replay_buffer.meta_buffer),
            "option_usage": option_usage,
            "meta_epsilon": self.meta_epsilon,
            "option_epsilon": self.option_epsilon
        }
        
    def _state_to_array(self, state: RLState) -> np.ndarray:
        """Convert RLState to numpy array"""
        if isinstance(state.value, np.ndarray):
            return state.value
        elif isinstance(state.value, list):
            return np.array(state.value)
        else:
            return np.array([state.value])
            
    def _action_to_int(self, action: RLAction) -> int:
        """Convert RLAction to integer"""
        return int(action.value)
        
    def _reward_to_float(self, reward: RLReward) -> float:
        """Convert RLReward to float"""
        return float(reward.value)
