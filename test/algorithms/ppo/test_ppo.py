"""Test PPO implementation"""

import numpy as np
import sys
sys.path.append('/home/graham/workspace/experiments/rl_commons/src')

from graham_rl_commons.algorithms.ppo import PPOAgent
from graham_rl_commons.core.base import RLState, RLAction, RLReward


def test_ppo_continuous():
    """Test PPO with continuous action space"""
    print("Testing PPO with continuous actions...")
    
    # Create agent
    agent = PPOAgent(
        name="test_ppo_continuous",
        state_dim=4,
        action_dim=2,
        continuous=True,
        action_bounds=(-1.0, 1.0),
        learning_rate=3e-4,
        buffer_size=128,
        batch_size=32,
        n_epochs=5
    )
    
    # Test basic functionality
    for episode in range(5):
        state = RLState(features=np.random.randn(4))
        episode_reward = 0
        
        for step in range(20):
            # Select action
            action = agent.select_action(state)
            print(f"Episode {episode}, Step {step}: Action = {action.parameters['values']}")
            
            # Simulate environment
            reward = RLReward(value=np.random.randn())
            next_state = RLState(features=np.random.randn(4))
            done = step == 19
            
            # Update agent
            losses = agent.update(state, action, reward, next_state, done)
            if losses:
                print(f"Training losses: {losses}")
            
            episode_reward += reward.value
            state = next_state
            
            if done:
                agent.reset()
                break
        
        print(f"Episode {episode} total reward: {episode_reward}")
    
    # Get metrics
    metrics = agent.get_metrics()
    print(f"\nFinal metrics: {metrics}")
    
    # Test save/load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        agent.save(tmpdir)
        agent.load(tmpdir)
        print("Save/load test passed!")
    
    print("\nPPO continuous test completed successfully!")


def test_ppo_discrete():
    """Test PPO with discrete action space"""
    print("\nTesting PPO with discrete actions...")
    
    # Create agent
    agent = PPOAgent(
        name="test_ppo_discrete",
        state_dim=4,
        action_dim=4,  # 4 discrete actions
        continuous=False,
        learning_rate=3e-4,
        buffer_size=128,
        batch_size=32,
        n_epochs=5
    )
    
    # Test one episode
    state = RLState(features=np.random.randn(4))
    
    for step in range(10):
        action = agent.select_action(state)
        print(f"Step {step}: Action ID = {action.action_id}")
        
        reward = RLReward(value=np.random.randn())
        next_state = RLState(features=np.random.randn(4))
        done = step == 9
        
        agent.update(state, action, reward, next_state, done)
        state = next_state
    
    print("PPO discrete test completed successfully!")


if __name__ == "__main__":
    test_ppo_continuous()
    test_ppo_discrete()
