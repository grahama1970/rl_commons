"""
Module: base_marl.py
Purpose: Base classes for Multi-Agent Reinforcement Learning

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- torch: https://pytorch.org/docs/stable/

Example Usage:
>>> from rl_commons.algorithms.marl import MARLAgent, MARLEnvironment
>>> env = MARLEnvironment(num_agents=3)
>>> agents = [MARLAgent(agent_id=i) for i in range(3)]
>>> observations = env.reset()
>>> for agent, obs in zip(agents, observations):
...     action = agent.act(obs)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import time

from ...core.base import RLState, RLAction, RLReward, RLAgent


@dataclass
class AgentObservation:
    """Observation for a single agent in multi-agent setting."""
    agent_id: int
    local_obs: np.ndarray
    global_state: Optional[np.ndarray] = None
    other_agents_visible: Optional[List[int]] = None
    communication_buffer: Optional[List[Any]] = None
    
    def to_tensor(self, device='cpu'):
        """Convert observation to PyTorch tensor."""
        return torch.FloatTensor(self.local_obs).to(device)


@dataclass 
class MARLTransition:
    """Transition data for multi-agent experience."""
    observations: List[AgentObservation]
    actions: List[RLAction]
    rewards: List[float]
    next_observations: List[AgentObservation]
    dones: List[bool]
    info: Dict[str, Any]


class MARLAgent(RLAgent):
    """Base class for agents in multi-agent settings."""
    
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, 
                 learning_rate: float = 0.001):
        """Initialize MARL agent.
        
        Args:
            agent_id: Unique identifier for this agent
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            learning_rate: Learning rate for updates
        """
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Track other agents
        self.other_agents = {}
        self.communication_inbox = []
        self.episode_count = 0
        
    def observe_other_agents(self, other_agents: Dict[int, 'MARLAgent']):
        """Register other agents for coordination."""
        self.other_agents = {
            aid: agent for aid, agent in other_agents.items() 
            if aid != self.agent_id
        }
    
    def send_message(self, recipient_id: int, message: Any):
        """Send message to another agent."""
        if recipient_id in self.other_agents:
            self.other_agents[recipient_id].receive_message(
                self.agent_id, message
            )
    
    def receive_message(self, sender_id: int, message: Any):
        """Receive message from another agent."""
        self.communication_inbox.append({
            'sender': sender_id,
            'message': message,
            'timestamp': time.time()
        })
    
    def broadcast_message(self, message: Any):
        """Broadcast message to all other agents."""
        for agent_id in self.other_agents:
            self.send_message(agent_id, message)
    
    @abstractmethod
    def act(self, observation: AgentObservation, explore: bool = True) -> RLAction:
        """Select action based on observation."""
        pass
    
    @abstractmethod
    def update(self, transition: MARLTransition):
        """Update agent based on experience."""
        pass
    
    # Base RLAgent abstract methods
    def select_action(self, state: RLState) -> RLAction:
        """RLAgent interface - delegates to act()."""
        obs = AgentObservation(
            agent_id=self.agent_id,
            local_obs=state.features
        )
        return self.act(obs, explore=True)
    
    def update_policy(self, state: RLState, action: RLAction, 
                      reward: RLReward, next_state: RLState, done: bool):
        """RLAgent interface - not used in MARL, update via transitions."""
        pass
    
    def clear_messages(self):
        """Clear communication inbox."""
        self.communication_inbox = []


class MARLEnvironment(ABC):
    """Base environment for multi-agent settings."""
    
    def __init__(self, num_agents: int):
        """Initialize multi-agent environment.
        
        Args:
            num_agents: Number of agents in the environment
        """
        self.num_agents = num_agents
        self.agents = []
        self.step_count = 0
        self.episode_count = 0
        
    @abstractmethod
    def reset(self) -> List[AgentObservation]:
        """Reset environment and return initial observations."""
        pass
    
    @abstractmethod 
    def step(self, actions: List[RLAction]) -> Tuple[
        List[AgentObservation], List[float], List[bool], Dict
    ]:
        """Execute actions and return results.
        
        Args:
            actions: List of actions from each agent
            
        Returns:
            observations: Next observations for each agent
            rewards: Rewards for each agent
            dones: Whether episode ended for each agent
            info: Additional information
        """
        pass
    
    def get_global_state(self) -> Optional[np.ndarray]:
        """Get global state if available (for centralized training)."""
        return None
    
    def render(self, mode='human'):
        """Render the environment."""
        pass


class DecentralizedMARLTrainer:
    """Trainer for decentralized multi-agent RL."""
    
    def __init__(self, env: MARLEnvironment, agents: List[MARLAgent]):
        """Initialize trainer.
        
        Args:
            env: Multi-agent environment
            agents: List of agents to train
        """
        self.env = env
        self.agents = agents
        self.episode_rewards = defaultdict(list)
        self.training_start_time = time.time()
        
        # Let agents observe each other
        agent_dict = {agent.agent_id: agent for agent in agents}
        for agent in agents:
            agent.observe_other_agents(agent_dict)
    
    def train_episode(self) -> Dict[str, float]:
        """Train agents for one episode.
        
        Returns:
            Dictionary of metrics from the episode
        """
        observations = self.env.reset()
        episode_rewards = defaultdict(float)
        episode_length = 0
        
        # Clear communication buffers
        for agent in self.agents:
            agent.clear_messages()
        
        done = False
        while not done:
            # Collect actions from all agents
            actions = []
            for agent, obs in zip(self.agents, observations):
                action = agent.act(obs, explore=True)
                actions.append(action)
            
            # Environment step
            next_observations, rewards, dones, info = self.env.step(actions)
            
            # Create transition
            transition = MARLTransition(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones,
                info=info
            )
            
            # Update each agent
            for agent in self.agents:
                agent.update(transition)
            
            # Track rewards
            for i, (agent, reward) in enumerate(zip(self.agents, rewards)):
                episode_rewards[f'agent_{agent.agent_id}'] += reward
            
            observations = next_observations
            done = all(dones)
            episode_length += 1
            
            # Clear old messages periodically
            if episode_length % 10 == 0:
                for agent in self.agents:
                    agent.clear_messages()
        
        # Record episode metrics
        metrics = {
            'episode_length': episode_length,
            'total_reward': sum(episode_rewards.values()),
            **episode_rewards
        }
        
        return metrics
    
    def train(self, num_episodes: int, verbose: bool = True) -> Dict[str, List[float]]:
        """Train agents for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to train
            verbose: Whether to print progress
            
        Returns:
            Dictionary of training metrics over time
        """
        all_metrics = defaultdict(list)
        
        for episode in range(num_episodes):
            metrics = self.train_episode()
            
            # Record metrics
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            if verbose and episode % 100 == 0:
                avg_reward = np.mean(all_metrics['total_reward'][-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
        
        return dict(all_metrics)


if __name__ == "__main__":
    # Validation test with simple multi-agent environment
    class SimpleMAEnvironment(MARLEnvironment):
        """Simple grid world for testing."""
        
        def __init__(self, num_agents=3):
            super().__init__(num_agents)
            self.grid_size = 5
            self.agent_positions = None
            
        def reset(self):
            self.agent_positions = [
                np.random.randint(0, self.grid_size, 2) 
                for _ in range(self.num_agents)
            ]
            return [
                AgentObservation(
                    agent_id=i,
                    local_obs=pos.astype(np.float32),
                    other_agents_visible=list(range(self.num_agents))
                )
                for i, pos in enumerate(self.agent_positions)
            ]
        
        def step(self, actions):
            # Simple movement in grid
            rewards = []
            observations = []
            
            for i, (pos, action) in enumerate(zip(self.agent_positions, actions)):
                # Move based on action
                if action.action_id == 0:  # up
                    pos[0] = max(0, pos[0] - 1)
                elif action.action_id == 1:  # down
                    pos[0] = min(self.grid_size - 1, pos[0] + 1)
                elif action.action_id == 2:  # left
                    pos[1] = max(0, pos[1] - 1)
                elif action.action_id == 3:  # right
                    pos[1] = min(self.grid_size - 1, pos[1] + 1)
                
                # Reward for reaching center
                center = self.grid_size // 2
                distance = np.abs(pos - center).sum()
                reward = -distance / self.grid_size
                rewards.append(reward)
                
                observations.append(
                    AgentObservation(
                        agent_id=i,
                        local_obs=pos.astype(np.float32),
                        other_agents_visible=list(range(self.num_agents))
                    )
                )
            
            self.step_count += 1
            dones = [self.step_count >= 100] * self.num_agents
            
            return observations, rewards, dones, {}
    
    # Test environment creation
    env = SimpleMAEnvironment(num_agents=3)
    obs = env.reset()
    
    assert len(obs) == 3, f"Expected 3 observations, got {len(obs)}"
    assert all(isinstance(o, AgentObservation) for o in obs), "Invalid observation type"
    
    print(" MARL base module validation passed")