"""
Module: arangodb_optimizer.py
Purpose: PPO-based optimizer for ArangoDB parameter tuning

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- torch: https://pytorch.org/docs/stable/

Example Usage:
>>> from graham_rl_commons.integrations import ArangoDBOptimizer
>>> optimizer = ArangoDBOptimizer()
>>> state = optimizer.get_current_state(db_metrics)
>>> action = optimizer.optimize(state)
>>> optimizer.update_with_feedback(state, action, new_metrics)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from ..algorithms.ppo import PPOAgent
from ..core.base import RLState, RLAction, RLReward
from ..monitoring.tracker import RLTracker


@dataclass
class ArangoDBMetrics:
    """Metrics collected from ArangoDB instance"""
    query_latency_ms: float
    throughput_qps: float
    memory_usage_fraction: float
    cpu_usage_fraction: float
    cache_hit_ratio: float
    connection_count: int
    error_rate: float
    response_time_p99: float


@dataclass
class ArangoDBParameters:
    """Tunable ArangoDB parameters"""
    cache_size_mb: int
    timeout_ms: int
    thread_pool_size: int
    connection_pool_size: int
    
    def to_normalized(self) -> np.ndarray:
        """Convert parameters to normalized values [0, 1]"""
        return np.array([
            (self.cache_size_mb - 128) / (4096 - 128),
            (self.timeout_ms - 1000) / (30000 - 1000),
            (self.thread_pool_size - 4) / (64 - 4),
            (self.connection_pool_size - 10) / (100 - 10)
        ])
    
    @classmethod
    def from_normalized(cls, values: np.ndarray) -> 'ArangoDBParameters':
        """Create parameters from normalized values"""
        return cls(
            cache_size_mb=int(values[0] * (4096 - 128) + 128),
            timeout_ms=int(values[1] * (30000 - 1000) + 1000),
            thread_pool_size=int(values[2] * (64 - 4) + 4),
            connection_pool_size=int(values[3] * (100 - 10) + 10)
        )


class ArangoDBOptimizer:
    """PPO-based optimizer for ArangoDB performance tuning"""
    
    def __init__(
        self,
        name: str = "arangodb_optimizer",
        learning_rate: float = 1e-4,
        hidden_dims: List[int] = None,
        target_latency_ms: float = 50.0,
        target_throughput_qps: float = 2000.0,
        max_resource_usage: float = 0.8
    ):
        """
        Initialize the ArangoDB optimizer
        
        Args:
            name: Name for the optimizer instance
            learning_rate: Learning rate for PPO
            hidden_dims: Hidden layer dimensions for neural network
            target_latency_ms: Target query latency in milliseconds
            target_throughput_qps: Target throughput in queries per second
            max_resource_usage: Maximum allowed resource usage (0-1)
        """
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
            
        self.agent = PPOAgent(
            name=name,
            state_dim=8,  # 4 params + 4 key metrics
            action_dim=4,  # 4 parameters to optimize
            continuous=True,
            action_bounds=(-0.1, 0.1),  # Small adjustments per step
            learning_rate=learning_rate,
            discount_factor=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            buffer_size=2048,
            hidden_dims=hidden_dims
        )
        
        self.target_latency = target_latency_ms
        self.target_throughput = target_throughput_qps
        self.max_resource_usage = max_resource_usage
        
        self.tracker = RLTracker(name)
        self.current_params = ArangoDBParameters(
            cache_size_mb=1024,
            timeout_ms=5000,
            thread_pool_size=16,
            connection_pool_size=50
        )
        
    def get_current_state(
        self,
        metrics: ArangoDBMetrics,
        params: Optional[ArangoDBParameters] = None
    ) -> RLState:
        """
        Convert database metrics and parameters into RL state
        
        Args:
            metrics: Current database metrics
            params: Current parameters (uses self.current_params if None)
            
        Returns:
            RLState with 8 features
        """
        if params is None:
            params = self.current_params
            
        # Normalize parameters
        param_features = params.to_normalized()
        
        # Normalize metrics (key performance indicators)
        metric_features = np.array([
            metrics.query_latency_ms / 1000.0,  # Normalize to seconds
            metrics.throughput_qps / 10000.0,    # Normalize to 10k scale
            metrics.memory_usage_fraction,       # Already 0-1
            metrics.cpu_usage_fraction          # Already 0-1
        ])
        
        # Combine into state vector
        state_vector = np.concatenate([param_features, metric_features])
        
        return RLState(
            features=state_vector,
            context={
                'timestamp': time.time(),
                'cache_hit_ratio': metrics.cache_hit_ratio,
                'error_rate': metrics.error_rate,
                'p99_latency': metrics.response_time_p99
            }
        )
    
    def optimize(self, state: RLState) -> Tuple[RLAction, ArangoDBParameters]:
        """
        Get optimized parameters based on current state
        
        Args:
            state: Current database state
            
        Returns:
            Tuple of (action, new_parameters)
        """
        # Get action from PPO agent
        action = self.agent.select_action(state)
        
        # Extract parameter adjustments
        adjustments = action.parameters['values']
        
        # Apply adjustments to current parameters
        current_normalized = self.current_params.to_normalized()
        new_normalized = np.clip(current_normalized + adjustments, 0, 1)
        
        # Convert back to actual parameters
        new_params = ArangoDBParameters.from_normalized(new_normalized)
        
        # Log the optimization step
        self.tracker.log_training_metrics({
            'cache_size_mb': new_params.cache_size_mb,
            'timeout_ms': new_params.timeout_ms,
            'thread_pool_size': new_params.thread_pool_size,
            'connection_pool_size': new_params.connection_pool_size
        })
        
        return action, new_params
    
    def calculate_reward(
        self,
        old_metrics: ArangoDBMetrics,
        new_metrics: ArangoDBMetrics
    ) -> RLReward:
        """
        Calculate reward based on performance improvement
        
        Args:
            old_metrics: Metrics before parameter change
            new_metrics: Metrics after parameter change
            
        Returns:
            RLReward with component breakdown
        """
        # Latency improvement (lower is better)
        latency_ratio = old_metrics.query_latency_ms / max(new_metrics.query_latency_ms, 1.0)
        latency_score = np.clip((latency_ratio - 0.5) * 2, -1, 1)
        
        # Bonus for meeting target
        if new_metrics.query_latency_ms <= self.target_latency:
            latency_score += 0.2
            
        # Throughput improvement (higher is better)
        throughput_ratio = new_metrics.throughput_qps / max(old_metrics.throughput_qps, 1.0)
        throughput_score = np.clip((throughput_ratio - 0.5) * 2, -1, 1)
        
        # Bonus for meeting target
        if new_metrics.throughput_qps >= self.target_throughput:
            throughput_score += 0.2
            
        # Resource usage penalty
        resource_penalty = 0.0
        if new_metrics.memory_usage_fraction > self.max_resource_usage:
            resource_penalty -= (new_metrics.memory_usage_fraction - self.max_resource_usage)
        if new_metrics.cpu_usage_fraction > self.max_resource_usage:
            resource_penalty -= (new_metrics.cpu_usage_fraction - self.max_resource_usage)
            
        # Error rate penalty
        error_penalty = -new_metrics.error_rate * 5.0  # Heavy penalty for errors
        
        # Cache efficiency bonus
        cache_bonus = new_metrics.cache_hit_ratio * 0.3
        
        # Calculate total reward
        total_reward = (
            latency_score * 0.35 +
            throughput_score * 0.35 +
            resource_penalty * 0.15 +
            error_penalty * 0.1 +
            cache_bonus * 0.05
        )
        
        return RLReward(
            value=np.clip(total_reward, -1, 1),
            components={
                'latency': latency_score,
                'throughput': throughput_score,
                'resources': resource_penalty,
                'errors': error_penalty,
                'cache': cache_bonus
            }
        )
    
    def update_with_feedback(
        self,
        old_state: RLState,
        action: RLAction,
        new_metrics: ArangoDBMetrics,
        new_params: ArangoDBParameters
    ):
        """
        Update the agent with observed results
        
        Args:
            old_state: State before optimization
            action: Action taken
            new_metrics: Resulting metrics
            new_params: New parameters applied
        """
        # Get new state
        new_state = self.get_current_state(new_metrics, new_params)
        
        # Extract old metrics from state for reward calculation
        old_features = old_state.features
        old_metrics = ArangoDBMetrics(
            query_latency_ms=old_features[4] * 1000,
            throughput_qps=old_features[5] * 10000,
            memory_usage_fraction=old_features[6],
            cpu_usage_fraction=old_features[7],
            cache_hit_ratio=old_state.context.get('cache_hit_ratio', 0.5),
            connection_count=50,  # Default
            error_rate=old_state.context.get('error_rate', 0.0),
            response_time_p99=old_state.context.get('p99_latency', 100.0)
        )
        
        # Calculate reward
        reward = self.calculate_reward(old_metrics, new_metrics)
        
        # Update agent
        self.agent.update(old_state, action, reward, new_state, done=False)
        
        # Update current parameters
        self.current_params = new_params
        
        # Log training metrics
        agent_metrics = self.agent.get_metrics()
        self.tracker.log_training_metrics({
            'reward': reward.value,
            'policy_loss': agent_metrics.get('policy_loss', 0),
            'value_loss': agent_metrics.get('value_loss', 0),
            'training_steps': agent_metrics.get('training_steps', 0)
        })
    
    def save(self, path: str):
        """Save the optimizer model"""
        self.agent.save(path)
        
    def load(self, path: str):
        """Load a pre-trained optimizer model"""
        self.agent.load(path)
        
    def get_metrics(self) -> Dict:
        """Get current training metrics"""
        return {
            **self.agent.get_metrics(),
            **self.tracker.get_current_stats()
        }


if __name__ == "__main__":
    # Test module with simulated data
    optimizer = ArangoDBOptimizer()
    
    # Simulate current metrics
    current_metrics = ArangoDBMetrics(
        query_latency_ms=120.0,
        throughput_qps=1500.0,
        memory_usage_fraction=0.6,
        cpu_usage_fraction=0.7,
        cache_hit_ratio=0.85,
        connection_count=45,
        error_rate=0.001,
        response_time_p99=250.0
    )
    
    # Get current state
    state = optimizer.get_current_state(current_metrics)
    
    # Optimize parameters
    action, new_params = optimizer.optimize(state)
    
    print("Current parameters:", optimizer.current_params)
    print("Optimized parameters:", new_params)
    
    # Simulate improved metrics
    improved_metrics = ArangoDBMetrics(
        query_latency_ms=95.0,
        throughput_qps=1800.0,
        memory_usage_fraction=0.65,
        cpu_usage_fraction=0.68,
        cache_hit_ratio=0.88,
        connection_count=50,
        error_rate=0.0008,
        response_time_p99=200.0
    )
    
    # Update with feedback
    optimizer.update_with_feedback(state, action, improved_metrics, new_params)
    
    print("✅ ArangoDBOptimizer validation passed")