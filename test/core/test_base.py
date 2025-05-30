"""Test core base classes"""

import pytest
import numpy as np
from graham_rl_commons.core import RLState, RLAction, RLReward, ReplayBuffer


def test_rl_state_creation(sample_state):
    """Test RLState creation and methods"""
    assert len(sample_state) == 5
    assert sample_state.context["request_id"] == "test123"
    
    # Test serialization
    state_dict = sample_state.to_dict()
    assert "features" in state_dict
    assert "context" in state_dict
    
    # Test deserialization
    restored = RLState.from_dict(state_dict)
    assert np.array_equal(restored.features, sample_state.features)
    assert restored.context == sample_state.context


def test_rl_action_creation(sample_action):
    """Test RLAction creation and methods"""
    assert sample_action.action_type == "select_provider"
    assert sample_action.action_id == 1
    assert sample_action.parameters["provider"] == "claude-3"
    
    # Test serialization
    action_dict = sample_action.to_dict()
    restored = RLAction.from_dict(action_dict)
    assert restored.action_type == sample_action.action_type
    assert restored.action_id == sample_action.action_id


def test_rl_reward_creation(sample_reward):
    """Test RLReward creation and methods"""
    assert sample_reward.value == 0.85
    assert len(sample_reward.components) == 3
    assert sample_reward.components["quality"] == 0.9
    
    # Test serialization
    reward_dict = sample_reward.to_dict()
    restored = RLReward.from_dict(reward_dict)
    assert restored.value == sample_reward.value
    assert restored.components == sample_reward.components


def test_replay_buffer():
    """Test replay buffer functionality"""
    buffer = ReplayBuffer(capacity=100)
    
    # Add some experiences
    for i in range(10):
        state = RLState(features=np.random.rand(5))
        action = RLAction("test", i % 3)
        reward = RLReward(np.random.rand())
        next_state = RLState(features=np.random.rand(5))
        done = i == 9
        
        buffer.push(state, action, reward, next_state, done)
    
    assert len(buffer) == 10
    assert buffer.is_ready(5)
    
    # Test sampling
    batch = buffer.sample(5)
    assert len(batch) == 5
    
    # Test stats
    stats = buffer.get_stats()
    assert stats["size"] == 10
    assert stats["capacity"] == 100
    assert "avg_reward" in stats
