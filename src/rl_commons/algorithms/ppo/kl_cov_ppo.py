"""
Module: kl_cov_ppo.py
Purpose: Enhanced PPO with KL-Cov penalty for high-covariance tokens

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.ppo import KLCovPPO
>>> agent = KLCovPPO(
...     name="kl_ppo",
...     state_dim=4,
...     action_dim=2,
...     kl_coef=0.1,
...     kl_cov_percentile=99.8
... )
>>> action = agent.select_action(state)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, kl_divergence
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


class KLCovPPO(PPOAgent):
    """
    Enhanced PPO with KL-Cov penalty for high-covariance tokens.
    
    Instead of detaching gradients like Clip-Cov, KL-Cov applies a 
    KL divergence penalty proportional to covariance magnitude.
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
                 # KL-Cov specific parameters
                 kl_coef: float = 0.1,
                 kl_cov_percentile: float = 99.8,
                 kl_scale_by_covariance: bool = True,
                 kl_max_penalty: float = 1.0,
                 dynamic_kl_coef: bool = True,
                 kl_target: float = 0.01,
                 kl_adaptation_rate: float = 1.5,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize KL-Cov PPO agent.
        
        Additional Args:
            kl_coef: Base KL penalty coefficient
            kl_cov_percentile: Percentile for identifying high-covariance tokens
            kl_scale_by_covariance: Scale KL penalty by covariance magnitude
            kl_max_penalty: Maximum KL penalty to apply
            dynamic_kl_coef: Adapt KL coefficient based on actual KL divergence
            kl_target: Target KL divergence for dynamic adaptation
            kl_adaptation_rate: Rate of KL coefficient adjustment
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
        
        # KL-Cov parameters
        self.kl_coef = kl_coef
        self.kl_cov_percentile = kl_cov_percentile
        self.kl_scale_by_covariance = kl_scale_by_covariance
        self.kl_max_penalty = kl_max_penalty
        self.dynamic_kl_coef = dynamic_kl_coef
        self.kl_target = kl_target
        self.kl_adaptation_rate = kl_adaptation_rate
        
        # Initialize covariance analyzer
        self.covariance_analyzer = CovarianceAnalyzer(
            percentile_threshold=kl_cov_percentile,
            device=str(device)
        )
        
        # Store old network for KL calculation
        self.old_network = ActorCriticNetwork(
            state_dim, action_dim, hidden_dims, continuous, action_bounds
        ).to(self.device)
        self._update_old_network()
        
        # Entropy tracking (inherited from base)
        if not self.entropy_tracking_enabled:
            self.entropy_tracking_enabled = True
            self.entropy_tracker = EntropyTracker(
                collapse_threshold=0.5,
                min_healthy_entropy=0.5
            )
        
        # Additional metrics
        self.metrics_history['policy_entropy'] = []
        self.metrics_history['kl_divergence'] = []
        self.metrics_history['high_cov_ratio'] = []
        self.metrics_history['effective_kl_coef'] = []
        
        logger.info(f"Initialized KLCovPPO with kl_coef={kl_coef}, "
                   f"kl_cov_percentile={kl_cov_percentile}%, dynamic={dynamic_kl_coef}")
    
    def _update_old_network(self):
        """Update old network with current network parameters"""
        self.old_network.load_state_dict(self.network.state_dict())
    
    def _calculate_kl_divergence(self, states: torch.Tensor, 
                                old_network: nn.Module,
                                new_network: nn.Module) -> torch.Tensor:
        """Calculate KL divergence between old and new policies"""
        # Get distributions from both networks
        with torch.no_grad():
            old_mean, old_std, _ = old_network(states)
            if self.continuous:
                old_dist = Normal(old_mean, old_std)
            else:
                old_probs = F.softmax(old_mean, dim=-1)
                old_dist = Categorical(old_probs)
        
        new_mean, new_std, _ = new_network(states)
        if self.continuous:
            new_dist = Normal(new_mean, new_std)
        else:
            new_probs = F.softmax(new_mean, dim=-1)
            new_dist = Categorical(new_probs)
        
        # Calculate KL divergence
        kl_div = kl_divergence(old_dist, new_dist)
        
        # Sum over action dimensions for continuous
        if self.continuous:
            kl_div = kl_div.sum(-1)
            
        return kl_div
    
    def _train_on_rollout(self, last_value: float = 0.0) -> Dict[str, float]:
        """Enhanced training with KL-Cov penalty"""
        # Update old network before training
        self._update_old_network()
        
        # Compute GAE
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
        total_kl = 0
        total_loss = 0
        total_high_cov_ratio = 0
        n_updates = 0
        
        # Current KL coefficient (may be adapted)
        current_kl_coef = self.kl_coef
        
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
            
            # Calculate KL divergence
            kl_div = self._calculate_kl_divergence(
                batch['states'], self.old_network, self.network
            )
            mean_kl = kl_div.mean()
            
            # Identify high-covariance tokens
            with torch.no_grad():
                # Get old log probs for covariance calculation
                old_log_probs = batch['old_log_probs']
                
                # Calculate covariance with advantages
                high_cov_mask, cov_metrics = self.covariance_analyzer.identify_outliers(
                    old_log_probs.cpu().numpy(),
                    batch['advantages'].cpu().numpy(),
                    return_metrics=True
                )
                high_cov_mask = torch.from_numpy(high_cov_mask).to(self.device)
                high_cov_ratio = cov_metrics.outlier_ratio
                
                # Get covariance magnitudes for scaling
                if self.kl_scale_by_covariance:
                    covariances = self.covariance_analyzer.calculate_token_covariances(
                        old_log_probs.cpu().numpy(),
                        batch['advantages'].cpu().numpy()
                    )
                    cov_magnitudes = torch.from_numpy(np.abs(covariances)).to(self.device)
                    # Normalize to [0, 1]
                    cov_magnitudes = cov_magnitudes / (cov_magnitudes.max() + 1e-8)
            
            # Standard PPO loss
            ratio = torch.exp(log_probs - batch['old_log_probs'])
            surr1 = ratio * batch['advantages']
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch['advantages']
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # KL-Cov penalty: Apply KL penalty weighted by covariance
            if high_cov_mask.any():
                if self.kl_scale_by_covariance:
                    # Scale KL penalty by covariance magnitude for high-cov tokens
                    kl_penalty = kl_div.clone()
                    # Apply stronger penalty to high-cov tokens
                    kl_penalty[high_cov_mask] = kl_div[high_cov_mask] * (1 + cov_magnitudes[high_cov_mask] * 2)
                    kl_penalty = torch.clamp(kl_penalty, max=self.kl_max_penalty)
                else:
                    # Fixed penalty multiplier for high-cov tokens
                    kl_penalty = kl_div.clone()
                    kl_penalty[high_cov_mask] = kl_div[high_cov_mask] * 2.0
                
                kl_loss = current_kl_coef * kl_penalty.mean()
                
                logger.debug(f"Applied KL penalty to {high_cov_mask.sum().item()}/{len(high_cov_mask)} high-cov tokens")
            else:
                # Standard KL penalty
                kl_loss = current_kl_coef * mean_kl
            
            # Value loss
            critic_loss = F.mse_loss(values.squeeze(-1), batch['returns'])
            
            # Total loss
            loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy + kl_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Track metrics
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            total_kl += mean_kl.item()
            total_loss += loss.item()
            total_high_cov_ratio += high_cov_ratio
            n_updates += 1
            
            # Adapt KL coefficient if enabled
            if self.dynamic_kl_coef and epoch == 0:  # Adapt once per rollout
                if mean_kl > self.kl_target * 1.5:
                    current_kl_coef *= self.kl_adaptation_rate
                elif mean_kl < self.kl_target / 1.5:
                    current_kl_coef /= self.kl_adaptation_rate
                # Clamp to reasonable range
                current_kl_coef = np.clip(current_kl_coef, 0.001, 1.0)
        
        # Update KL coefficient for next time
        if self.dynamic_kl_coef:
            self.kl_coef = current_kl_coef
        
        # Average metrics
        avg_actor_loss = total_actor_loss / n_updates
        avg_critic_loss = total_critic_loss / n_updates
        avg_entropy = total_entropy / n_updates
        avg_kl = total_kl / n_updates
        avg_total_loss = total_loss / n_updates
        avg_high_cov_ratio = total_high_cov_ratio / n_updates
        
        # Update metrics history
        self.metrics_history['actor_losses'].append(avg_actor_loss)
        self.metrics_history['critic_losses'].append(avg_critic_loss)
        self.metrics_history['entropy_losses'].append(avg_entropy)
        self.metrics_history['total_losses'].append(avg_total_loss)
        self.metrics_history['policy_entropy'].append(avg_entropy)
        self.metrics_history['kl_divergence'].append(avg_kl)
        self.metrics_history['high_cov_ratio'].append(avg_high_cov_ratio)
        self.metrics_history['effective_kl_coef'].append(current_kl_coef)
        
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
            'kl_divergence': avg_kl,
            'total_loss': avg_total_loss,
            'high_cov_ratio': avg_high_cov_ratio,
            'effective_kl_coef': current_kl_coef,
            'training_steps': self.training_steps
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics including KL-specific ones"""
        metrics = super().get_metrics()
        
        # Add KL-Cov metrics
        if 'policy_entropy' in self.metrics_history and len(self.metrics_history['policy_entropy']) > 0:
            metrics['current_entropy'] = self.metrics_history['policy_entropy'][-1]
            metrics['avg_entropy'] = np.mean(self.metrics_history['policy_entropy'][-10:])
        
        if 'kl_divergence' in self.metrics_history and len(self.metrics_history['kl_divergence']) > 0:
            metrics['avg_kl_divergence'] = np.mean(self.metrics_history['kl_divergence'][-10:])
        
        if 'high_cov_ratio' in self.metrics_history and len(self.metrics_history['high_cov_ratio']) > 0:
            metrics['avg_high_cov_ratio'] = np.mean(self.metrics_history['high_cov_ratio'][-10:])
        
        if 'effective_kl_coef' in self.metrics_history and len(self.metrics_history['effective_kl_coef']) > 0:
            metrics['current_kl_coef'] = self.metrics_history['effective_kl_coef'][-1]
        
        # Add covariance analyzer stats
        cov_stats = self.covariance_analyzer.analyze_distribution()
        if cov_stats:
            metrics['covariance_stats'] = cov_stats
        
        return metrics


if __name__ == "__main__":
    # Validation example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running KLCovPPO validation...")
    
    # Create agent
    agent = KLCovPPO(
        name="test_kl_ppo",
        state_dim=4,
        action_dim=2,
        continuous=True,
        kl_coef=0.1,
        kl_cov_percentile=99.8,
        kl_scale_by_covariance=True,
        dynamic_kl_coef=True,
        kl_target=0.01
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
            logger.info(f"Step {i}: Entropy={metrics.get('entropy', 0):.3f}, "
                       f"KL={metrics.get('kl_divergence', 0):.3f}, "
                       f"KL_coef={metrics.get('effective_kl_coef', 0):.3f}")
    
    # Get final metrics
    final_metrics = agent.get_metrics()
    logger.info(f"Final metrics: {final_metrics}")
    
    logger.info("✅ KLCovPPO validation complete!")