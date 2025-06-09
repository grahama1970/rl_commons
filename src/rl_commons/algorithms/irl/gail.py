"""
Module: gail.py
Purpose: Generative Adversarial Imitation Learning

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.irl import GAIL
>>> gail = GAIL(state_dim=8, action_dim=4)
>>> gail.train(expert_demos, environment)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import deque
import time

from .base_irl import Demonstration
from ...core.base import RLState, RLAction, RLReward


class Discriminator(nn.Module):
    """Discriminator network for GAIL."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialize discriminator.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            states: State tensor
            actions: Action tensor
            
        Returns:
            Probability of being expert data
        """
        # Handle discrete actions
        if len(actions.shape) == 1:
            actions = nn.functional.one_hot(actions.long(), num_classes=self.action_dim).float()
        
        inputs = torch.cat([states, actions], dim=-1)
        return self.network(inputs)
    
    def get_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute reward signal for generator.
        
        Args:
            states: State tensor
            actions: Action tensor
            
        Returns:
            Reward signal
        """
        with torch.no_grad():
            d_values = self.forward(states, actions)
            # Reward = log(D(s,a)) - log(1-D(s,a))
            rewards = torch.log(d_values + 1e-8) - torch.log(1 - d_values + 1e-8)
        return rewards.squeeze()


class PolicyNetwork(nn.Module):
    """Policy network for GAIL generator."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialize policy network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            states: State tensor
            
        Returns:
            Action logits
        """
        return self.network(states)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from policy.
        
        Args:
            state: State tensor
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action and log probability
        """
        logits = self.forward(state)
        probs = nn.functional.softmax(logits, dim=-1)
        
        if deterministic:
            action = logits.argmax(dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1))
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob


class GAIL:
    """Generative Adversarial Imitation Learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = None
    ):
        """Initialize GAIL.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension for networks
            learning_rate: Learning rate
            discount_factor: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.discriminator = Discriminator(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # Optimizers
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = deque(maxlen=10000)
        
        # Training history
        self.training_history = {
            'd_losses': [],
            'policy_losses': [],
            'value_losses': [],
            'rewards': []
        }
    
    def train_discriminator(
        self,
        expert_demos: List[Demonstration],
        batch_size: int = 64
    ) -> float:
        """Train discriminator to distinguish expert from generated data.
        
        Args:
            expert_demos: Expert demonstrations
            batch_size: Batch size
            
        Returns:
            Discriminator loss
        """
        if len(self.buffer) < batch_size:
            return 0.0
        
        # Sample expert data
        expert_batch = np.random.choice(expert_demos, min(batch_size, len(expert_demos)))
        expert_states = []
        expert_actions = []
        
        for demo in expert_batch:
            idx = np.random.randint(len(demo.states))
            expert_states.append(demo.states[idx])
            expert_actions.append(demo.actions[idx])
        
        expert_states = torch.stack(expert_states).to(self.device)
        expert_actions = torch.stack(expert_actions).to(self.device)
        
        # Sample generated data
        gen_batch = np.random.choice(len(self.buffer), batch_size)
        gen_data = [self.buffer[i] for i in gen_batch]
        gen_states = torch.stack([d['state'] for d in gen_data]).to(self.device)
        gen_actions = torch.stack([d['action'] for d in gen_data]).to(self.device)
        
        # Train discriminator
        self.discriminator.train()
        
        # Expert data should have label 1
        expert_preds = self.discriminator(expert_states, expert_actions)
        expert_loss = nn.functional.binary_cross_entropy(expert_preds, torch.ones_like(expert_preds))
        
        # Generated data should have label 0
        gen_preds = self.discriminator(gen_states, gen_actions)
        gen_loss = nn.functional.binary_cross_entropy(gen_preds, torch.zeros_like(gen_preds))
        
        # Total loss
        d_loss = expert_loss + gen_loss
        
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self, batch_size: int = 64) -> Tuple[float, float]:
        """Train generator (policy) using PPO with discriminator rewards.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Policy loss and value loss
        """
        if len(self.buffer) < batch_size:
            return 0.0, 0.0
        
        # Sample batch
        batch_indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in batch_indices]
        
        states = torch.stack([d['state'] for d in batch]).to(self.device)
        actions = torch.stack([d['action'] for d in batch]).to(self.device)
        rewards = torch.stack([d['reward'] for d in batch]).to(self.device)
        old_log_probs = torch.stack([d['log_prob'] for d in batch]).to(self.device)
        returns = torch.stack([d['return'] for d in batch]).to(self.device)
        advantages = torch.stack([d['advantage'] for d in batch]).to(self.device)
        
        # Update policy
        self.policy.train()
        _, new_log_probs = self.policy.get_action(states, deterministic=True)
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value function
        self.value_network.train()
        values = self.value_network(states).squeeze()
        value_loss = nn.functional.mse_loss(values, returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def collect_experience(self, env: Any, num_steps: int = 1000):
        """Collect experience using current policy.
        
        Args:
            env: Environment to interact with
            num_steps: Number of steps to collect
        """
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_log_probs = []
        
        for step in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob = self.policy.get_action(state_tensor)
                value = self.value_network(state_tensor)
            
            action_value = action.item()
            
            # Take action in environment
            next_state, env_reward, done, _ = env.step(action_value)
            
            # Get discriminator reward
            with torch.no_grad():
                d_reward = self.discriminator.get_reward(state_tensor, action)
            
            # Store experience
            episode_states.append(state_tensor.squeeze())
            episode_actions.append(action.squeeze())
            episode_rewards.append(d_reward.squeeze())
            episode_log_probs.append(log_prob.squeeze())
            
            if done or step == num_steps - 1:
                # Compute returns and advantages
                returns = []
                advantages = []
                
                R = 0
                for r in reversed(episode_rewards):
                    R = r + self.discount_factor * R
                    returns.insert(0, R)
                
                returns = torch.tensor(returns).to(self.device)
                
                # Add to buffer
                for i in range(len(episode_states)):
                    self.buffer.append({
                        'state': episode_states[i],
                        'action': episode_actions[i],
                        'reward': episode_rewards[i],
                        'log_prob': episode_log_probs[i],
                        'return': returns[i],
                        'advantage': returns[i]  # Simplified - should use GAE
                    })
                
                # Reset episode
                state = env.reset()
                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_log_probs = []
            else:
                state = next_state
    
    def train(
        self,
        expert_demos: List[Demonstration],
        env: Any,
        num_iterations: int = 1000,
        d_steps: int = 5,
        g_steps: int = 5
    ) -> Dict[str, Any]:
        """Train GAIL.
        
        Args:
            expert_demos: Expert demonstrations
            env: Environment
            num_iterations: Number of training iterations
            d_steps: Discriminator steps per iteration
            g_steps: Generator steps per iteration
            
        Returns:
            Training results
        """
        for iteration in range(num_iterations):
            # Collect experience
            self.collect_experience(env, num_steps=100)
            
            # Train discriminator
            d_losses = []
            for _ in range(d_steps):
                d_loss = self.train_discriminator(expert_demos)
                d_losses.append(d_loss)
            
            # Train generator
            policy_losses = []
            value_losses = []
            for _ in range(g_steps):
                p_loss, v_loss = self.train_generator()
                policy_losses.append(p_loss)
                value_losses.append(v_loss)
            
            # Record metrics
            avg_d_loss = np.mean(d_losses)
            avg_p_loss = np.mean(policy_losses)
            avg_v_loss = np.mean(value_losses)
            
            self.training_history['d_losses'].append(avg_d_loss)
            self.training_history['policy_losses'].append(avg_p_loss)
            self.training_history['value_losses'].append(avg_v_loss)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: D Loss: {avg_d_loss:.4f}, "
                      f"Policy Loss: {avg_p_loss:.4f}, Value Loss: {avg_v_loss:.4f}")
        
        return {
            'final_d_loss': self.training_history['d_losses'][-1],
            'final_policy_loss': self.training_history['policy_losses'][-1],
            'final_value_loss': self.training_history['value_losses'][-1]
        }
    
    def get_policy(self) -> Callable:
        """Get learned policy function.
        
        Returns:
            Policy function
        """
        def policy(state: RLState) -> RLAction:
            """Learned GAIL policy."""
            state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _ = self.policy.get_action(state_tensor, deterministic=True)
            
            return RLAction(
                action_type='discrete',
                action_id=action.item(),
                parameters={'source': 'gail'}
            )
        
        return policy


if __name__ == "__main__":
    # Validation test
    # Create dummy demonstrations
    demonstrations = []
    for _ in range(5):
        states = torch.randn(50, 8)
        actions = torch.randint(0, 4, (50,))
        demo = Demonstration(states, actions)
        demonstrations.append(demo)
    
    # Initialize GAIL
    gail = GAIL(state_dim=8, action_dim=4)
    
    # Test discriminator
    d_loss = gail.train_discriminator(demonstrations, batch_size=16)
    print(f"Initial discriminator loss: {d_loss}")
    
    # Add some fake experience
    for _ in range(100):
        gail.buffer.append({
            'state': torch.randn(8),
            'action': torch.randint(0, 4, (1,)).squeeze(),
            'reward': torch.randn(1).squeeze(),
            'log_prob': torch.randn(1).squeeze(),
            'return': torch.randn(1).squeeze(),
            'advantage': torch.randn(1).squeeze()
        })
    
    # Test generator training
    p_loss, v_loss = gail.train_generator(batch_size=32)
    print(f"Policy loss: {p_loss}, Value loss: {v_loss}")
    
    # Test policy
    policy = gail.get_policy()
    state = RLState(features=np.random.randn(8))
    action = policy(state)
    
    assert isinstance(action, RLAction), "Invalid action type"
    assert 0 <= action.action_id < 4, f"Invalid action: {action.action_id}"
    
    print(" GAIL validation passed")