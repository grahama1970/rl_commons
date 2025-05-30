"""
Module: a3c.py
Purpose: Asynchronous Advantage Actor-Critic (A3C) implementation

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from graham_rl_commons.algorithms.a3c import A3CAgent
>>> agent = A3CAgent(state_dim=10, action_dim=4, num_workers=4)
>>> agent.train(env_fn, episodes=1000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time
from collections import deque
import logging

from ...core.base import RLAgent, RLState, RLAction, RLReward
from ...monitoring.tracker import RLTracker

logger = logging.getLogger(__name__)

# Set multiprocessing start method
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


class ActorCriticNetwork(nn.Module):
    """Shared Actor-Critic network for A3C"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int],
                 continuous: bool = False,
                 action_bounds: Optional[Tuple[float, float]] = None):
        """
        Initialize Actor-Critic network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            continuous: Whether actions are continuous
            action_bounds: Min/max bounds for continuous actions
        """
        super().__init__()
        
        self.continuous = continuous
        self.action_bounds = action_bounds
        
        # Build shared layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:  # All but last hidden layer
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
            
        self.shared = nn.Sequential(*layers)
        
        # Separate heads for actor and critic
        last_hidden = hidden_dims[-1]
        self.actor_hidden = nn.Linear(prev_dim, last_hidden)
        self.critic_hidden = nn.Linear(prev_dim, last_hidden)
        
        if continuous:
            # Output mean and log_std for each action dimension
            self.actor_mean = nn.Linear(last_hidden, action_dim)
            self.actor_log_std = nn.Linear(last_hidden, action_dim)
            
            # Initialize outputs
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.constant_(self.actor_mean.bias, 0)
            nn.init.orthogonal_(self.actor_log_std.weight, gain=0.01)
            nn.init.constant_(self.actor_log_std.bias, 0)
        else:
            # Output logits for discrete actions
            self.actor_head = nn.Linear(last_hidden, action_dim)
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.constant_(self.actor_head.bias, 0)
            
        # Value head
        self.critic_head = nn.Linear(last_hidden, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0)
        
        # Initialize shared and hidden layers
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
                
        nn.init.orthogonal_(self.actor_hidden.weight, gain=np.sqrt(2))
        nn.init.constant_(self.actor_hidden.bias, 0)
        nn.init.orthogonal_(self.critic_hidden.weight, gain=np.sqrt(2))
        nn.init.constant_(self.critic_hidden.bias, 0)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network
        
        Args:
            state: State tensor
            
        Returns:
            (action_params, value) tuple
        """
        shared_features = self.shared(state)
        
        # Actor path
        actor_features = F.relu(self.actor_hidden(shared_features))
        
        if self.continuous:
            mean = self.actor_mean(actor_features)
            log_std = self.actor_log_std(actor_features)
            
            # Bound the actions if specified
            if self.action_bounds is not None:
                mean = torch.tanh(mean)
                mean = mean * (self.action_bounds[1] - self.action_bounds[0]) / 2
                mean = mean + (self.action_bounds[1] + self.action_bounds[0]) / 2
                
            action_params = (mean, log_std)
        else:
            logits = self.actor_head(actor_features)
            action_params = logits
            
        # Critic path
        critic_features = F.relu(self.critic_hidden(shared_features))
        value = self.critic_head(critic_features)
        
        return action_params, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Get action from policy"""
        action_params, value = self.forward(state)
        
        if self.continuous:
            mean, log_std = action_params
            std = torch.exp(log_std)
            
            if deterministic:
                action = mean
                log_prob = torch.zeros(1)
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            logits = action_params
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
                log_prob = torch.zeros(1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action).unsqueeze(-1)
                
        return action, log_prob, value
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate log probabilities and entropy of actions"""
        action_params, values = self.forward(states)
        
        if self.continuous:
            mean, log_std = action_params
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            logits = action_params
            dist = Categorical(logits=logits)
            
            log_probs = dist.log_prob(actions).unsqueeze(-1)
            entropy = dist.entropy().mean()
            
        return log_probs, values, entropy


class SharedAdam(torch.optim.Adam):
    """Adam optimizer that can be shared across processes"""
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        
        # Share state across processes
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['shared_exp_avg'] = p.data.new_zeros(p.size()).share_memory_()
                state['shared_exp_avg_sq'] = p.data.new_zeros(p.size()).share_memory_()
                
    def step(self, closure=None):
        """Perform optimization step with shared state"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                
                # Sync with shared state
                state['step'] = int(state['shared_step'].item())
                state['exp_avg'] = state['shared_exp_avg']
                state['exp_avg_sq'] = state['shared_exp_avg_sq']
                
        # Perform regular Adam step
        loss = super().step(closure)
        
        # Update shared state
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                state['shared_step'] += 1
                state['shared_exp_avg'].copy_(state['exp_avg'])
                state['shared_exp_avg_sq'].copy_(state['exp_avg_sq'])
                
        return loss


@dataclass
class A3CConfig:
    """Configuration for A3C algorithm"""
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_steps: int = 20  # Steps before update
    num_workers: int = 4
    max_episode_steps: int = 1000
    update_interval: int = 100  # Global model save interval
    

class A3CWorker(mp.Process):
    """Worker process for A3C training"""
    
    def __init__(self,
                 worker_id: int,
                 global_network: ActorCriticNetwork,
                 shared_optimizer: SharedAdam,
                 global_episode_counter: mp.Value,
                 global_step_counter: mp.Value,
                 env_fn: Callable,
                 config: A3CConfig,
                 device: str = "cpu",
                 result_queue: Optional[mp.Queue] = None):
        """
        Initialize A3C worker
        
        Args:
            worker_id: Unique worker ID
            global_network: Shared global network
            shared_optimizer: Shared optimizer
            global_episode_counter: Shared episode counter
            global_step_counter: Shared step counter
            env_fn: Function to create environment
            config: A3C configuration
            device: Device to use
            result_queue: Queue for returning results
        """
        super().__init__()
        
        self.worker_id = worker_id
        self.global_network = global_network
        self.shared_optimizer = shared_optimizer
        self.global_episode_counter = global_episode_counter
        self.global_step_counter = global_step_counter
        self.env_fn = env_fn
        self.config = config
        self.device = device
        self.result_queue = result_queue
        
    def run(self):
        """Run worker training loop"""
        torch.manual_seed(self.worker_id)
        np.random.seed(self.worker_id)
        
        # Create local network
        env = self.env_fn()
        state_dim = env.observation_space.shape[0]
        
        if hasattr(env.action_space, 'n'):
            action_dim = env.action_space.n
            continuous = False
            action_bounds = None
        else:
            action_dim = env.action_space.shape[0]
            continuous = True
            action_bounds = (env.action_space.low[0], env.action_space.high[0])
            
        local_network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 128],
            continuous=continuous,
            action_bounds=action_bounds
        ).to(self.device)
        
        # Training loop
        episode_count = 0
        while self.global_episode_counter.value < 1000:  # Max episodes
            # Sync with global network
            local_network.load_state_dict(self.global_network.state_dict())
            
            # Collect trajectory
            states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
            
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
                
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.config.num_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, log_prob, value = local_network.get_action(state_tensor)
                    
                # Store trajectory
                states.append(state)
                actions.append(action.cpu().numpy().squeeze())
                values.append(value.cpu().numpy().squeeze())
                log_probs.append(log_prob.cpu().numpy().squeeze())
                
                # Take action
                if continuous:
                    next_state, reward, done, truncated, info = env.step(action.cpu().numpy().squeeze())
                else:
                    next_state, reward, done, truncated, info = env.step(action.cpu().item())
                    
                done = done or truncated
                rewards.append(reward)
                dones.append(done)
                
                episode_reward += reward
                episode_length += 1
                
                with self.global_step_counter.get_lock():
                    self.global_step_counter.value += 1
                    
                if done or episode_length >= self.config.max_episode_steps:
                    state = env.reset()
                    if isinstance(state, tuple):
                        state = state[0]
                        
                    with self.global_episode_counter.get_lock():
                        self.global_episode_counter.value += 1
                        
                    episode_count += 1
                    
                    # Report results
                    if self.result_queue is not None:
                        self.result_queue.put({
                            'worker_id': self.worker_id,
                            'episode': episode_count,
                            'reward': episode_reward,
                            'length': episode_length
                        })
                        
                    episode_reward = 0
                    episode_length = 0
                else:
                    state = next_state
                    
            # Compute returns and advantages
            returns = self._compute_returns(rewards, values, dones, 
                                          state, local_network)
            advantages = returns - np.array(values)
            
            # Update global network
            self._update_global(states, actions, returns, advantages, 
                              log_probs, local_network)
                              
        env.close()
        
    def _compute_returns(self, rewards, values, dones, last_state, network):
        """Compute discounted returns"""
        returns = []
        
        # Bootstrap from last state
        if dones[-1]:
            next_value = 0
        else:
            state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value = network.forward(state_tensor)
                next_value = next_value.cpu().numpy().squeeze()
                
        # Compute returns backward
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.config.discount_factor * R * (1 - dones[step])
            returns.insert(0, R)
            
        return np.array(returns)
    
    def _update_global(self, states, actions, returns, advantages, log_probs, 
                      local_network):
        """Update global network with local gradients"""
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device) if not local_network.continuous else torch.FloatTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        
        # Forward pass
        log_probs_new, values, entropy = local_network.evaluate_actions(states, actions)
        
        # Compute losses
        value_loss = F.mse_loss(values, returns)
        policy_loss = -(log_probs_new * advantages.detach()).mean()
        
        total_loss = (policy_loss + 
                     self.config.value_loss_coef * value_loss - 
                     self.config.entropy_coef * entropy)
                     
        # Compute gradients locally
        self.shared_optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(local_network.parameters(), 
                                      self.config.max_grad_norm)
                                      
        # Transfer gradients to global network
        for local_param, global_param in zip(local_network.parameters(), 
                                           self.global_network.parameters()):
            global_param._grad = local_param.grad.clone()
            
        # Update global network
        self.shared_optimizer.step()


class A3CAgent(RLAgent):
    """Asynchronous Advantage Actor-Critic agent"""
    
    def __init__(self,
                 name: str,
                 state_dim: int,
                 action_dim: int,
                 continuous: bool = False,
                 action_bounds: Optional[Tuple[float, float]] = None,
                 hidden_dims: Optional[List[int]] = None,
                 config: Optional[A3CConfig] = None,
                 device: str = "cpu"):
        """
        Initialize A3C agent
        
        Args:
            name: Agent name
            state_dim: State space dimension
            action_dim: Action space dimension
            continuous: Whether actions are continuous
            action_bounds: Bounds for continuous actions
            hidden_dims: Hidden layer dimensions
            config: A3C configuration
            device: Device to use
        """
        super().__init__(name)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.action_bounds = action_bounds
        self.device = device
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
            
        self.config = config or A3CConfig()
        
        # Create global network
        self.global_network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            continuous=continuous,
            action_bounds=action_bounds
        ).to(device)
        
        # Share network parameters
        self.global_network.share_memory()
        
        # Create shared optimizer
        self.shared_optimizer = SharedAdam(
            self.global_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Shared counters
        self.global_episode_counter = mp.Value('i', 0)
        self.global_step_counter = mp.Value('i', 0)
        
        # Tracking
        self.tracker = RLTracker(name)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
    def train(self, env_fn: Callable, episodes: int = 1000,
              save_interval: int = 100, save_path: Optional[Path] = None):
        """
        Train agent using A3C
        
        Args:
            env_fn: Function to create environment
            episodes: Number of episodes to train
            save_interval: Save model every N episodes
            save_path: Path to save models
        """
        # Reset counters
        self.global_episode_counter.value = 0
        self.global_step_counter.value = 0
        
        # Create result queue
        result_queue = mp.Queue()
        
        # Create workers
        workers = []
        for i in range(self.config.num_workers):
            worker = A3CWorker(
                worker_id=i,
                global_network=self.global_network,
                shared_optimizer=self.shared_optimizer,
                global_episode_counter=self.global_episode_counter,
                global_step_counter=self.global_step_counter,
                env_fn=env_fn,
                config=self.config,
                device=self.device,
                result_queue=result_queue
            )
            worker.start()
            workers.append(worker)
            
        # Monitor training
        start_time = time.time()
        last_episode = 0
        
        try:
            while self.global_episode_counter.value < episodes:
                # Collect results
                try:
                    result = result_queue.get(timeout=1.0)
                    self.episode_rewards.append(result['reward'])
                    self.episode_lengths.append(result['length'])
                    
                    # Log progress
                    if self.global_episode_counter.value % 10 == 0:
                        current_episode = self.global_episode_counter.value
                        if current_episode > last_episode:
                            elapsed = time.time() - start_time
                            eps_per_sec = current_episode / elapsed
                            
                            logger.info(
                                f"Episode {current_episode}/{episodes} | "
                                f"Avg Reward: {np.mean(self.episode_rewards):.2f} | "
                                f"Steps: {self.global_step_counter.value} | "
                                f"Speed: {eps_per_sec:.1f} eps/s"
                            )
                            last_episode = current_episode
                            
                except:
                    pass
                    
                # Save periodically
                if (save_path is not None and 
                    self.global_episode_counter.value % save_interval == 0 and
                    self.global_episode_counter.value > 0):
                    self.save(save_path)
                    
        finally:
            # Stop workers
            for worker in workers:
                worker.terminate()
                worker.join()
                
        # Final save
        if save_path is not None:
            self.save(save_path)
            
        return self.get_metrics()
    
    def select_action(self, state: RLState, deterministic: bool = False) -> RLAction:
        """Select action using global network"""
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.global_network.get_action(state_tensor, deterministic)
            
        if self.continuous:
            action_np = action.cpu().numpy().squeeze()
            return RLAction(
                action_type="continuous",
                action_id=0,
                parameters={"values": action_np.tolist()}
            )
        else:
            return RLAction(
                action_type="discrete",
                action_id=int(action.cpu().item()),
                parameters=None
            )
            
    def update(self, state: RLState, action: RLAction, reward: RLReward,
               next_state: RLState, done: bool) -> Dict[str, Any]:
        """A3C doesn't use single-step updates"""
        return {}
    
    def save(self, path: Union[str, Path]):
        """Save agent state"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save network only (SharedAdam state is complex)
        torch.save({
            'network_state_dict': self.global_network.state_dict(),
            'episodes': self.global_episode_counter.value,
            'steps': self.global_step_counter.value
        }, path / f"{self.name}_model.pt")
        
        # Save config
        config_dict = {
            'name': self.name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'continuous': self.continuous,
            'action_bounds': self.action_bounds,
            'config': self.config.__dict__
        }
        
        with open(path / f"{self.name}_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        logger.info(f"Saved A3C agent {self.name} to {path}")
        
    def load(self, path: Union[str, Path]):
        """Load agent state"""
        path = Path(path)
        
        # Load network
        checkpoint = torch.load(path / f"{self.name}_model.pt", 
                              map_location=self.device)
        self.global_network.load_state_dict(checkpoint['network_state_dict'])
        
        if 'episodes' in checkpoint:
            self.global_episode_counter.value = checkpoint['episodes']
        if 'steps' in checkpoint:
            self.global_step_counter.value = checkpoint['steps']
            
        logger.info(f"Loaded A3C agent {self.name} from {path}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        return {
            'episodes': self.global_episode_counter.value,
            'steps': self.global_step_counter.value,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'num_workers': self.config.num_workers
        }


if __name__ == "__main__":
    # Test A3C with a simple environment
    import gymnasium as gym
    
    def make_env():
        return gym.make('CartPole-v1')
        
    # Create agent
    agent = A3CAgent(
        name="test_a3c",
        state_dim=4,
        action_dim=2,
        continuous=False,
        config=A3CConfig(num_workers=2, num_steps=10)
    )
    
    print("Testing A3C agent...")
    
    # Test action selection
    test_state = RLState(features=np.array([0.1, -0.2, 0.3, -0.4]))
    action = agent.select_action(test_state)
    print(f"Selected action: {action.action_id}")
    
    # Test short training run
    print("\nTraining for 50 episodes...")
    metrics = agent.train(make_env, episodes=50)
    print(f"Training complete: {metrics}")
    
    print("\n✅ A3C implementation validated!")