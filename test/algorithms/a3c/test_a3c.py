"""
Test A3C implementation
"""

import pytest
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import tempfile
import time

from graham_rl_commons.algorithms.a3c import A3CAgent, A3CConfig, ActorCriticNetwork
from graham_rl_commons.core.base import RLState, RLAction


class TestA3CComponents:
    """Test A3C components"""
    
    def test_actor_critic_network_discrete(self):
        """Test ActorCriticNetwork for discrete actions"""
        network = ActorCriticNetwork(
            state_dim=4,
            action_dim=2,
            hidden_dims=[64, 32],
            continuous=False
        )
        
        # Test forward pass
        state = torch.randn(10, 4)
        action_params, values = network.forward(state)
        
        assert action_params.shape == (10, 2)  # Logits
        assert values.shape == (10, 1)
        
        # Test action selection
        action, log_prob, value = network.get_action(state[0:1])
        assert action.shape == (1,)
        assert log_prob.shape == (1, 1)
        assert value.shape == (1, 1)
        
    def test_actor_critic_network_continuous(self):
        """Test ActorCriticNetwork for continuous actions"""
        network = ActorCriticNetwork(
            state_dim=3,
            action_dim=2,
            hidden_dims=[64, 32],
            continuous=True,
            action_bounds=(-1.0, 1.0)
        )
        
        # Test forward pass
        state = torch.randn(5, 3)
        action_params, values = network.forward(state)
        
        mean, log_std = action_params
        assert mean.shape == (5, 2)
        assert log_std.shape == (5, 2)
        assert values.shape == (5, 1)
        
        # Check bounds
        assert torch.all(mean >= -1.0) and torch.all(mean <= 1.0)
        
    def test_a3c_config(self):
        """Test A3C configuration"""
        config = A3CConfig(
            learning_rate=1e-3,
            num_workers=8,
            num_steps=32
        )
        
        assert config.learning_rate == 1e-3
        assert config.num_workers == 8
        assert config.num_steps == 32
        assert config.discount_factor == 0.99  # Default


class TestA3CAgent:
    """Test A3C agent functionality"""
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = A3CAgent(
            name="test_agent",
            state_dim=4,
            action_dim=2,
            continuous=False
        )
        
        assert agent.name == "test_agent"
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert not agent.continuous
        
    def test_action_selection_discrete(self):
        """Test discrete action selection"""
        agent = A3CAgent(
            name="discrete_agent",
            state_dim=4,
            action_dim=3,
            continuous=False
        )
        
        state = RLState(features=np.array([0.1, -0.2, 0.3, -0.4]))
        
        # Test stochastic action
        action = agent.select_action(state, deterministic=False)
        assert action.action_type == "discrete"
        assert 0 <= action.action_id < 3
        
        # Test deterministic action
        action_det = agent.select_action(state, deterministic=True)
        assert action_det.action_type == "discrete"
        assert 0 <= action_det.action_id < 3
        
    def test_action_selection_continuous(self):
        """Test continuous action selection"""
        agent = A3CAgent(
            name="continuous_agent",
            state_dim=3,
            action_dim=2,
            continuous=True,
            action_bounds=(-2.0, 2.0)
        )
        
        state = RLState(features=np.array([0.5, -0.3, 0.8]))
        
        # Test action selection
        action = agent.select_action(state)
        assert action.action_type == "continuous"
        assert "values" in action.parameters
        assert len(action.parameters["values"]) == 2
        
        # Check bounds
        values = action.parameters["values"]
        assert all(-2.0 <= v <= 2.0 for v in values)
        
    def test_save_load(self):
        """Test save and load functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and configure agent
            agent1 = A3CAgent(
                name="save_test",
                state_dim=4,
                action_dim=2,
                config=A3CConfig(learning_rate=5e-4)
            )
            
            # Modify network weights
            for param in agent1.global_network.parameters():
                param.data.fill_(0.5)
                
            # Save
            save_path = Path(tmpdir) / "model"
            agent1.save(save_path)
            
            # Create new agent and load
            agent2 = A3CAgent(
                name="save_test",
                state_dim=4,
                action_dim=2
            )
            agent2.load(save_path)
            
            # Compare network weights
            for p1, p2 in zip(agent1.global_network.parameters(),
                            agent2.global_network.parameters()):
                assert torch.allclose(p1, p2)
                
    def test_metrics(self):
        """Test metrics tracking"""
        agent = A3CAgent(
            name="metrics_test",
            state_dim=4,
            action_dim=2,
            config=A3CConfig(num_workers=4)
        )
        
        metrics = agent.get_metrics()
        assert 'episodes' in metrics
        assert 'steps' in metrics
        assert 'avg_episode_reward' in metrics
        assert 'avg_episode_length' in metrics
        assert metrics['num_workers'] == 4


def test_a3c_integration():
    """Integration test with simple environment"""
    print("=" * 60)
    print("🧪 A3C INTEGRATION TEST")
    print("=" * 60)
    
    failures = []
    total = 0
    
    # Test 1: Basic CartPole training
    total += 1
    print("\n📋 Test 1: CartPole training with A3C")
    try:
        def make_env():
            return gym.make('CartPole-v1')
            
        agent = A3CAgent(
            name="cartpole_a3c",
            state_dim=4,
            action_dim=2,
            continuous=False,
            config=A3CConfig(
                num_workers=2,
                num_steps=5,
                learning_rate=1e-3
            )
        )
        
        # Train for a few episodes
        start_time = time.time()
        metrics = agent.train(make_env, episodes=20)
        duration = time.time() - start_time
        
        if metrics['episodes'] >= 20:
            print(f"✅ PASS - Trained {metrics['episodes']} episodes in {duration:.1f}s")
        else:
            error = f"Only trained {metrics['episodes']} episodes"
            print(f"❌ FAIL: {error}")
            failures.append(("Test 1", error))
    except Exception as e:
        print(f"❌ FAIL: {e}")
        failures.append(("Test 1", str(e)))
        
    # Test 2: Continuous action space
    total += 1
    print("\n📋 Test 2: Continuous action space (Pendulum)")
    try:
        def make_pendulum():
            return gym.make('Pendulum-v1')
            
        # Get env info
        env = make_pendulum()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bounds = (env.action_space.low[0], env.action_space.high[0])
        env.close()
        
        agent = A3CAgent(
            name="pendulum_a3c",
            state_dim=state_dim,
            action_dim=action_dim,
            continuous=True,
            action_bounds=action_bounds,
            config=A3CConfig(
                num_workers=2,
                num_steps=10,
                entropy_coef=0.01
            )
        )
        
        # Test action selection
        state = RLState(features=np.random.randn(state_dim))
        action = agent.select_action(state)
        
        if action.action_type == "continuous" and len(action.parameters["values"]) == action_dim:
            print("✅ PASS - Continuous action selection works")
        else:
            error = f"Invalid action: {action}"
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
    sys.exit(test_a3c_integration())