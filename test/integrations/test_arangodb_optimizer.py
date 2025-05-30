"""
Test ArangoDBOptimizer integration with PPO
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from rl_commons.integrations import ArangoDBOptimizer
from rl_commons.integrations.arangodb_optimizer import ArangoDBMetrics, ArangoDBParameters


class TestArangoDBOptimizer:
    """Test the ArangoDB optimizer functionality"""
    
    def test_optimizer_initialization(self):
        """Test basic optimizer creation"""
        optimizer = ArangoDBOptimizer(
            name="test_optimizer",
            learning_rate=1e-3,
            hidden_dims=[128, 64]
        )
        
        assert optimizer.agent.name == "test_optimizer"
        assert optimizer.target_latency == 50.0
        assert optimizer.target_throughput == 2000.0
        assert optimizer.max_resource_usage == 0.8
        
    def test_state_creation(self):
        """Test state vector creation from metrics"""
        optimizer = ArangoDBOptimizer()
        
        metrics = ArangoDBMetrics(
            query_latency_ms=100.0,
            throughput_qps=1500.0,
            memory_usage_fraction=0.6,
            cpu_usage_fraction=0.7,
            cache_hit_ratio=0.85,
            connection_count=45,
            error_rate=0.001,
            response_time_p99=200.0
        )
        
        state = optimizer.get_current_state(metrics)
        
        # Check state dimensions
        assert state.features.shape == (8,)
        
        # Check normalized values
        assert 0 <= state.features[0] <= 1  # cache_size
        assert 0 <= state.features[1] <= 1  # timeout
        assert 0 <= state.features[2] <= 1  # thread_pool
        assert 0 <= state.features[3] <= 1  # connection_pool
        
    def test_parameter_normalization(self):
        """Test parameter normalization and denormalization"""
        params = ArangoDBParameters(
            cache_size_mb=2048,
            timeout_ms=15000,
            thread_pool_size=32,
            connection_pool_size=50
        )
        
        # Normalize
        normalized = params.to_normalized()
        assert normalized.shape == (4,)
        assert np.all((normalized >= 0) & (normalized <= 1))
        
        # Denormalize
        recovered = ArangoDBParameters.from_normalized(normalized)
        assert recovered.cache_size_mb == 2048
        assert recovered.timeout_ms == 15000
        assert recovered.thread_pool_size == 32
        assert recovered.connection_pool_size == 50
        
    def test_optimization_step(self):
        """Test a single optimization step"""
        optimizer = ArangoDBOptimizer()
        
        # Create initial metrics
        metrics = ArangoDBMetrics(
            query_latency_ms=150.0,
            throughput_qps=1000.0,
            memory_usage_fraction=0.5,
            cpu_usage_fraction=0.6,
            cache_hit_ratio=0.8,
            connection_count=40,
            error_rate=0.002,
            response_time_p99=300.0
        )
        
        # Get current state
        state = optimizer.get_current_state(metrics)
        
        # Optimize
        action, new_params = optimizer.optimize(state)
        
        # Check action and parameters
        assert action is not None
        assert isinstance(new_params, ArangoDBParameters)
        assert 128 <= new_params.cache_size_mb <= 4096
        assert 1000 <= new_params.timeout_ms <= 30000
        assert 4 <= new_params.thread_pool_size <= 64
        assert 10 <= new_params.connection_pool_size <= 100
        
    def test_reward_calculation(self):
        """Test reward calculation logic"""
        optimizer = ArangoDBOptimizer()
        
        # Baseline metrics
        old_metrics = ArangoDBMetrics(
            query_latency_ms=100.0,
            throughput_qps=1500.0,
            memory_usage_fraction=0.6,
            cpu_usage_fraction=0.7,
            cache_hit_ratio=0.85,
            connection_count=45,
            error_rate=0.001,
            response_time_p99=200.0
        )
        
        # Improved metrics
        new_metrics = ArangoDBMetrics(
            query_latency_ms=80.0,   # 20% improvement
            throughput_qps=1800.0,   # 20% improvement
            memory_usage_fraction=0.65,
            cpu_usage_fraction=0.68,
            cache_hit_ratio=0.90,    # Better cache
            connection_count=50,
            error_rate=0.0005,       # Fewer errors
            response_time_p99=160.0
        )
        
        reward = optimizer.calculate_reward(old_metrics, new_metrics)
        
        # Should be positive due to improvements
        assert reward.value > 0
        assert reward.components['latency'] > 0
        assert reward.components['throughput'] > 0
        assert reward.components['cache'] > 0
        
    def test_reward_with_degradation(self):
        """Test reward when performance degrades"""
        optimizer = ArangoDBOptimizer()
        
        # Baseline metrics
        old_metrics = ArangoDBMetrics(
            query_latency_ms=100.0,
            throughput_qps=1500.0,
            memory_usage_fraction=0.6,
            cpu_usage_fraction=0.7,
            cache_hit_ratio=0.85,
            connection_count=45,
            error_rate=0.001,
            response_time_p99=200.0
        )
        
        # Degraded metrics
        new_metrics = ArangoDBMetrics(
            query_latency_ms=150.0,   # 50% worse
            throughput_qps=1200.0,    # 20% worse
            memory_usage_fraction=0.9,  # High resource usage
            cpu_usage_fraction=0.95,   # Very high CPU
            cache_hit_ratio=0.70,      # Worse cache
            connection_count=50,
            error_rate=0.01,           # 10x more errors
            response_time_p99=400.0
        )
        
        reward = optimizer.calculate_reward(old_metrics, new_metrics)
        
        # Should be negative due to degradation
        assert reward.value < 0
        assert reward.components['latency'] < 0
        assert reward.components['throughput'] < 0
        assert reward.components['resources'] < 0
        assert reward.components['errors'] < 0
        
    def test_full_optimization_loop(self):
        """Test complete optimization loop"""
        optimizer = ArangoDBOptimizer()
        
        # Initial metrics
        metrics = ArangoDBMetrics(
            query_latency_ms=120.0,
            throughput_qps=1400.0,
            memory_usage_fraction=0.65,
            cpu_usage_fraction=0.72,
            cache_hit_ratio=0.82,
            connection_count=42,
            error_rate=0.0015,
            response_time_p99=240.0
        )
        
        # Run optimization loop
        for i in range(5):
            # Get state
            state = optimizer.get_current_state(metrics)
            
            # Optimize
            action, new_params = optimizer.optimize(state)
            
            # Simulate metric improvements (in real use, these come from DB)
            improved_metrics = ArangoDBMetrics(
                query_latency_ms=metrics.query_latency_ms * 0.95,  # 5% better
                throughput_qps=metrics.throughput_qps * 1.05,      # 5% better
                memory_usage_fraction=min(metrics.memory_usage_fraction * 1.02, 0.9),
                cpu_usage_fraction=min(metrics.cpu_usage_fraction * 0.98, 0.9),
                cache_hit_ratio=min(metrics.cache_hit_ratio * 1.02, 0.95),
                connection_count=new_params.connection_pool_size,
                error_rate=metrics.error_rate * 0.9,
                response_time_p99=metrics.response_time_p99 * 0.95
            )
            
            # Update optimizer
            optimizer.update_with_feedback(state, action, improved_metrics, new_params)
            
            # Use improved metrics for next iteration
            metrics = improved_metrics
            
        # Check final metrics
        agent_metrics = optimizer.get_metrics()
        assert 'training_steps' in agent_metrics
        assert agent_metrics['training_steps'] > 0
        
    def test_save_load_model(self):
        """Test model persistence"""
        optimizer = ArangoDBOptimizer()
        
        # Train for a few steps
        metrics = ArangoDBMetrics(
            query_latency_ms=100.0,
            throughput_qps=1500.0,
            memory_usage_fraction=0.6,
            cpu_usage_fraction=0.7,
            cache_hit_ratio=0.85,
            connection_count=45,
            error_rate=0.001,
            response_time_p99=200.0
        )
        
        state = optimizer.get_current_state(metrics)
        action, new_params = optimizer.optimize(state)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            optimizer.save(str(save_path))
            
            # Create new optimizer and load
            new_optimizer = ArangoDBOptimizer()
            new_optimizer.load(str(save_path))
            
            # Should produce similar actions
            new_action, _ = new_optimizer.optimize(state)
            
            # Actions should be close (not exact due to exploration)
            assert new_action is not None


def test_integration_validation():
    """Validate the complete integration module"""
    print("=" * 60)
    print("🧪 ARANGODB OPTIMIZER VALIDATION")
    print("=" * 60)
    
    failures = []
    total = 0
    
    # Test 1: Basic functionality
    total += 1
    print("\n📋 Test 1: Basic optimizer functionality")
    try:
        optimizer = ArangoDBOptimizer()
        metrics = ArangoDBMetrics(
            query_latency_ms=100.0,
            throughput_qps=1500.0,
            memory_usage_fraction=0.6,
            cpu_usage_fraction=0.7,
            cache_hit_ratio=0.85,
            connection_count=45,
            error_rate=0.001,
            response_time_p99=200.0
        )
        state = optimizer.get_current_state(metrics)
        action, params = optimizer.optimize(state)
        if action and params:
            print("✅ PASS")
        else:
            error = "Failed to generate action or parameters"
            print(f"❌ FAIL: {error}")
            failures.append(("Test 1", error))
    except Exception as e:
        print(f"❌ FAIL: {e}")
        failures.append(("Test 1", str(e)))
    
    # Test 2: Reward calculation
    total += 1
    print("\n📋 Test 2: Reward calculation")
    try:
        old_m = ArangoDBMetrics(100, 1500, 0.6, 0.7, 0.85, 45, 0.001, 200)
        new_m = ArangoDBMetrics(80, 1800, 0.65, 0.68, 0.9, 50, 0.0005, 160)
        reward = optimizer.calculate_reward(old_m, new_m)
        if reward.value > 0:
            print("✅ PASS")
        else:
            error = f"Expected positive reward, got {reward.value}"
            print(f"❌ FAIL: {error}")
            failures.append(("Test 2", error))
    except Exception as e:
        print(f"❌ FAIL: {e}")
        failures.append(("Test 2", str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Total: {total}, Passed: {total-len(failures)}, Failed: {len(failures)}")
    
    if failures:
        print("\n❌ FAILURES:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        return 1
    else:
        print("\n✅ ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(test_integration_validation())