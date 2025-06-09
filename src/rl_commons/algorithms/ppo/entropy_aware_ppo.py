"""
Module: entropy_aware_ppo.py
Purpose: Enhanced PPO with Clip-Cov for preventing entropy collapse

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.ppo import EntropyAwarePPO
>>> agent = EntropyAwarePPO(
...     name="entropy_ppo",
...     state_dim=4,
...     action_dim=2,
...     clip_cov_percentile=99.8,
...     asymmetric_clipping=True
... )
>>> action = agent.select_action(state)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import json
import logging

try:
    from .ppo import PPOAgent, ActorCriticNetwork, RolloutBuffer
    from ...core.base import RLState, RLAction, RLReward
    from ...core.covariance_analyzer import CovarianceAnalyzer
    from ...monitoring.entropy_tracker import EntropyTracker
except ImportError:
    # For standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from rl_commons.algorithms.ppo.ppo import PPOAgent, ActorCriticNetwork, RolloutBuffer
    from rl_commons.core.base import RLState, RLAction, RLReward
    from rl_commons.core.covariance_analyzer import CovarianceAnalyzer
    from rl_commons.monitoring.entropy_tracker import EntropyTracker

logger = logging.getLogger(__name__)


class EntropyAwarePPO(PPOAgent):
    """
    Enhanced PPO with Clip-Cov mechanism to prevent entropy collapse.
    
    Based on research showing that gradient detachment for high-covariance
    tokens can maintain healthy entropy levels throughout training.
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
                 clip_ratio_upper: Optional[float] = None,  # For asymmetric clipping
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 buffer_size: int = 2048,
                 hidden_dims: List[int] = None,
                 device: str = "cpu",
                 # Entropy-aware specific parameters
                 clip_cov_percentile: float = 99.8,
                 asymmetric_clipping: bool = True,
                 asymmetric_factor: float = 1.4,
                 adaptive_entropy_coef: bool = True,
                 min_entropy_coef: float = 0.001,
                 max_entropy_coef: float = 0.1,
                 entropy_target: float = 0.5,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Entropy-Aware PPO agent.
        
        Additional Args:
            clip_ratio_upper: Upper clip ratio (if None, uses asymmetric_factor)
            clip_cov_percentile: Percentile for identifying high-covariance tokens
            asymmetric_clipping: Whether to use asymmetric clipping bounds
            asymmetric_factor: Factor for upper bound (clip_upper = factor * clip_lower)
            adaptive_entropy_coef: Whether to adapt entropy coefficient based on current entropy
            min_entropy_coef: Minimum entropy coefficient
            max_entropy_coef: Maximum entropy coefficient
            entropy_target: Target entropy level to maintain
        """
        # Ensure config is not None
        if config is None:
            config = {}
        
        # Initialize base PPO
        super().__init__(
            name=name,
            state_dim=state_dim,
            action_dim=action_dim,
            continuous=continuous,
            action_bounds=action_bounds,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            n_epochs=n_epochs,
            batch_size=batch_size,
            buffer_size=buffer_size,
            hidden_dims=hidden_dims,
            device=device,
            config=config
        )
        
        # Entropy-aware parameters
        self.clip_cov_percentile = clip_cov_percentile
        self.asymmetric_clipping = asymmetric_clipping
        self.asymmetric_factor = asymmetric_factor
        self.adaptive_entropy_coef = adaptive_entropy_coef
        self.min_entropy_coef = min_entropy_coef
        self.max_entropy_coef = max_entropy_coef
        self.entropy_target = entropy_target
        
        # Set clipping bounds
        if asymmetric_clipping:
            self.clip_ratio_lower = clip_ratio
            self.clip_ratio_upper = clip_ratio_upper or (clip_ratio * asymmetric_factor)
        else:
            self.clip_ratio_lower = clip_ratio
            self.clip_ratio_upper = clip_ratio
        
        # Initialize covariance analyzer
        self.covariance_analyzer = CovarianceAnalyzer(
            percentile_threshold=clip_cov_percentile,
            device=str(device)
        )
        
        # Entropy tracking (inherited from base, but ensure it's enabled)
        if not self.entropy_tracking_enabled:
            self.entropy_tracking_enabled = True
            self.entropy_tracker = EntropyTracker(
                collapse_threshold=entropy_target * 0.5,
                min_healthy_entropy=entropy_target
            )
        
        # Additional metrics
        self.metrics_history['policy_entropy'] = []
        self.metrics_history['high_cov_ratio'] = []
        self.metrics_history['effective_entropy_coef'] = []
        
        logger.info(f"Initialized EntropyAwarePPO with clip_cov={clip_cov_percentile}%, "
                   f"asymmetric={asymmetric_clipping}, clip_bounds=[{self.clip_ratio_lower}, {self.clip_ratio_upper}]")
    
    def _train_on_rollout(self, last_value: float = 0.0) -> Dict[str, float]:
        """Enhanced training with Clip-Cov mechanism"""
        # Compute GAE (same as base)
        self.rollout_buffer.compute_gae(last_value, self.discount_factor, self.gae_lambda)
        
        # Normalize advantages
        n_steps = self.rollout_buffer.buffer_size if self.rollout_buffer.full else self.rollout_buffer.position
        advantages = self.rollout_buffer.advantages[:n_steps]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.rollout_buffer.advantages[:n_steps] = advantages
        
        # Training metrics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_loss = 0
        total_high_cov_ratio = 0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            # Sample from buffer
            batch = self.rollout_buffer.get_samples(min(self.batch_size, n_steps))
            
            # Forward pass
            mean, std, values = self.network(batch['states'])
            
            # Calculate distributions and entropy
            if self.continuous:
                dist = Normal(mean, std)
                log_probs = dist.log_prob(batch['actions']).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
            else:
                probs = F.softmax(mean, dim=-1)
                dist = Categorical(probs)
                log_probs = dist.log_prob(batch['actions'].squeeze(-1))
                entropy = dist.entropy().mean()
            
            # Log entropy
            if self.entropy_tracker:
                entropy_metrics = self.log_entropy(entropy.item())
            
            # Identify high-covariance tokens
            with torch.no_grad():
                high_cov_mask, cov_metrics = self.covariance_analyzer.identify_outliers(
                    log_probs.detach(),
                    batch['advantages'],
                    return_metrics=True
                )
                high_cov_mask = torch.from_numpy(high_cov_mask).to(self.device)
                high_cov_ratio = cov_metrics.outlier_ratio
            
            # PPO loss with Clip-Cov
            ratio = torch.exp(log_probs - batch['old_log_probs'])
            
            # Asymmetric clipping
            if self.asymmetric_clipping:
                # Different bounds for positive and negative advantages
                positive_adv = batch['advantages'] > 0
                negative_adv = ~positive_adv
                
                # Apply different clipping for positive/negative advantages
                ratio_clipped = torch.zeros_like(ratio)
                ratio_clipped[positive_adv] = torch.clamp(
                    ratio[positive_adv], 
                    1 - self.clip_ratio_lower, 
                    1 + self.clip_ratio_upper
                )
                ratio_clipped[negative_adv] = torch.clamp(
                    ratio[negative_adv],
                    1 - self.clip_ratio_upper,
                    1 + self.clip_ratio_lower
                )
            else:
                ratio_clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            # Standard PPO objectives
            surr1 = ratio * batch['advantages']
            surr2 = ratio_clipped * batch['advantages']
            
            # Clip-Cov: Detach gradients for high-covariance tokens
            if high_cov_mask.any():
                # For high-covariance tokens, use detached gradients
                surr1_detached = ratio.detach() * batch['advantages']
                surr2_detached = ratio_clipped.detach() * batch['advantages']
                
                # Combine: use detached for high-cov, normal for others
                surr1 = torch.where(high_cov_mask, surr1_detached, surr1)
                surr2 = torch.where(high_cov_mask, surr2_detached, surr2)
                
                logger.debug(f"Detached gradients for {high_cov_mask.sum().item()}/{len(high_cov_mask)} high-cov tokens")
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (unchanged)
            critic_loss = F.mse_loss(values.squeeze(-1), batch['returns'])
            
            # Adaptive entropy coefficient
            if self.adaptive_entropy_coef and self.entropy_tracker:
                # Increase entropy coef if entropy is too low
                current_entropy = entropy.item()
                if current_entropy < self.entropy_target:
                    # Scale up entropy coefficient
                    entropy_scale = self.entropy_target / (current_entropy + 1e-8)
                    entropy_scale = min(entropy_scale, 5.0)  # Cap scaling
                    effective_entropy_coef = min(
                        self.entropy_coef * entropy_scale,
                        self.max_entropy_coef
                    )
                else:
                    # Gradually reduce if entropy is healthy
                    effective_entropy_coef = max(
                        self.entropy_coef * 0.95,
                        self.min_entropy_coef
                    )
            else:
                effective_entropy_coef = self.entropy_coef
            
            # Total loss with adaptive entropy
            loss = actor_loss + self.value_loss_coef * critic_loss - effective_entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Track metrics
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            total_loss += loss.item()
            total_high_cov_ratio += high_cov_ratio
            n_updates += 1
        
        # Average metrics
        avg_actor_loss = total_actor_loss / n_updates
        avg_critic_loss = total_critic_loss / n_updates
        avg_entropy = total_entropy / n_updates
        avg_total_loss = total_loss / n_updates
        avg_high_cov_ratio = total_high_cov_ratio / n_updates
        
        # Update metrics history
        self.metrics_history['actor_losses'].append(avg_actor_loss)
        self.metrics_history['critic_losses'].append(avg_critic_loss)
        self.metrics_history['entropy_losses'].append(avg_entropy)
        self.metrics_history['total_losses'].append(avg_total_loss)
        self.metrics_history['policy_entropy'].append(avg_entropy)
        self.metrics_history['high_cov_ratio'].append(avg_high_cov_ratio)
        self.metrics_history['effective_entropy_coef'].append(effective_entropy_coef)
        
        # Reset buffer
        self.rollout_buffer.reset()
        
        # Update training steps
        self.training_steps += n_steps
        
        # Check for entropy warnings
        if self.entropy_tracker and self.entropy_tracker.detect_collapse():
            logger.warning(f"Entropy collapse detected! Current entropy: {avg_entropy:.4f}")
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy,
            'total_loss': avg_total_loss,
            'high_cov_ratio': avg_high_cov_ratio,
            'effective_entropy_coef': effective_entropy_coef,
            'training_steps': self.training_steps
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics including entropy-specific ones"""
        metrics = super().get_metrics()
        
        # Add entropy-aware metrics
        if 'policy_entropy' in self.metrics_history and len(self.metrics_history['policy_entropy']) > 0:
            metrics['current_entropy'] = self.metrics_history['policy_entropy'][-1]
            metrics['avg_entropy'] = np.mean(self.metrics_history['policy_entropy'][-10:])
        
        if 'high_cov_ratio' in self.metrics_history and len(self.metrics_history['high_cov_ratio']) > 0:
            metrics['avg_high_cov_ratio'] = np.mean(self.metrics_history['high_cov_ratio'][-10:])
        
        # Add covariance analyzer stats
        cov_stats = self.covariance_analyzer.analyze_distribution()
        if cov_stats:
            metrics['covariance_stats'] = cov_stats
        
        return metrics


if __name__ == "__main__":
    # Validation example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running EntropyAwarePPO validation...")
    
    # Create agent
    agent = EntropyAwarePPO(
        name="test_entropy_ppo",
        state_dim=4,
        action_dim=2,
        continuous=True,
        clip_cov_percentile=99.8,
        asymmetric_clipping=True,
        adaptive_entropy_coef=True,
        entropy_target=1.0
    )
    
    # Simulate training step
    for i in range(10):
        state = RLState(features=np.random.randn(4))
        action = agent.select_action(state)
        reward = RLReward(value=np.random.randn())
        next_state = RLState(features=np.random.randn(4))
        done = i == 9
        
        metrics = agent.update(state, action, reward, next_state, done)
        
        if metrics:
            logger.info(f"Step {i}: {metrics}")
    
    # Get final metrics
    final_metrics = agent.get_metrics()
    logger.info(f"Final metrics: {final_metrics}")
    
    logger.info("✅ EntropyAwarePPO validation complete!")