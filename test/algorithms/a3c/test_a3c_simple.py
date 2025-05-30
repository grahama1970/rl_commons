#!/usr/bin/env python3
"""
Simple A3C test without multiprocessing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
from graham_rl_commons.algorithms.a3c import A3CAgent, A3CConfig, ActorCriticNetwork
from graham_rl_commons.core.base import RLState


def test_a3c_components():
    """Test A3C components"""
    print("=" * 60)
    print("🧪 A3C COMPONENT TESTS")
    print("=" * 60)
    
    failures = []
    total = 0
    
    # Test 1: Actor-Critic Network (Discrete)
    total += 1
    print("\n📋 Test 1: Actor-Critic Network (Discrete Actions)")
    try:
        network = ActorCriticNetwork(
            state_dim=4,
            action_dim=2,
            hidden_dims=[64, 32],
            continuous=False
        )
        
        # Test forward pass
        state = torch.randn(5, 4)
        action_params, values = network.forward(state)
        
        if action_params.shape == (5, 2) and values.shape == (5, 1):
            print("✅ PASS - Network output shapes correct")
        else:
            error = f"Wrong shapes: action_params={action_params.shape}, values={values.shape}"
            print(f"❌ FAIL: {error}")
            failures.append(("Test 1", error))
    except Exception as e:
        print(f"❌ FAIL: {e}")
        failures.append(("Test 1", str(e)))
    
    # Test 2: Actor-Critic Network (Continuous)
    total += 1
    print("\n📋 Test 2: Actor-Critic Network (Continuous Actions)")
    try:
        network = ActorCriticNetwork(
            state_dim=3,
            action_dim=2,
            hidden_dims=[64, 32],
            continuous=True,
            action_bounds=(-1.0, 1.0)
        )
        
        # Test forward pass
        state = torch.randn(3, 3)
        action_params, values = network.forward(state)
        
        mean, log_std = action_params
        if mean.shape == (3, 2) and log_std.shape == (3, 2) and values.shape == (3, 1):
            print("✅ PASS - Continuous network output shapes correct")
        else:
            error = f"Wrong shapes: mean={mean.shape}, log_std={log_std.shape}, values={values.shape}"
            print(f"❌ FAIL: {error}")
            failures.append(("Test 2", error))
    except Exception as e:
        print(f"❌ FAIL: {e}")
        failures.append(("Test 2", str(e)))
    
    # Test 3: A3C Agent Creation
    total += 1
    print("\n📋 Test 3: A3C Agent Creation")
    try:
        agent = A3CAgent(
            name="test_agent",
            state_dim=4,
            action_dim=2,
            continuous=False,
            config=A3CConfig(num_workers=2)
        )
        
        if agent.name == "test_agent" and agent.state_dim == 4:
            print("✅ PASS - Agent created successfully")
        else:
            error = "Agent properties incorrect"
            print(f"❌ FAIL: {error}")
            failures.append(("Test 3", error))
    except Exception as e:
        print(f"❌ FAIL: {e}")
        failures.append(("Test 3", str(e)))
    
    # Test 4: Action Selection (Discrete)
    total += 1
    print("\n📋 Test 4: Discrete Action Selection")
    try:
        agent = A3CAgent(
            name="discrete_agent",
            state_dim=4,
            action_dim=3,
            continuous=False
        )
        
        state = RLState(features=np.array([0.1, -0.2, 0.3, -0.4]))
        action = agent.select_action(state)
        
        if action.action_type == "discrete" and 0 <= action.action_id < 3:
            print(f"✅ PASS - Selected discrete action: {action.action_id}")
        else:
            error = f"Invalid action: {action}"
            print(f"❌ FAIL: {error}")
            failures.append(("Test 4", error))
    except Exception as e:
        print(f"❌ FAIL: {e}")
        failures.append(("Test 4", str(e)))
    
    # Test 5: Action Selection (Continuous)
    total += 1
    print("\n📋 Test 5: Continuous Action Selection")
    try:
        agent = A3CAgent(
            name="continuous_agent",
            state_dim=3,
            action_dim=2,
            continuous=True,
            action_bounds=(-2.0, 2.0)
        )
        
        state = RLState(features=np.array([0.5, -0.3, 0.8]))
        action = agent.select_action(state)
        
        if action.action_type == "continuous" and "values" in action.parameters:
            values = action.parameters["values"]
            if isinstance(values, list) and len(values) == 2:
                print(f"✅ PASS - Selected continuous action: {values}")
            else:
                error = f"Wrong action values: {values}"
                print(f"❌ FAIL: {error}")
                failures.append(("Test 5", error))
        else:
            error = f"Invalid action: {action}"
            print(f"❌ FAIL: {error}")
            failures.append(("Test 5", error))
    except Exception as e:
        print(f"❌ FAIL: {e}")
        failures.append(("Test 5", str(e)))
    
    # Test 6: Save/Load
    total += 1
    print("\n📋 Test 6: Save/Load Functionality")
    try:
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create agent
            agent1 = A3CAgent(
                name="save_test",
                state_dim=4,
                action_dim=2
            )
            
            # Modify some weights
            for param in agent1.global_network.parameters():
                param.data.fill_(0.5)
            
            # Save
            save_path = Path(tmpdir) / "model"
            agent1.save(save_path)
            
            # Load into new agent
            agent2 = A3CAgent(
                name="save_test",
                state_dim=4,
                action_dim=2
            )
            agent2.load(save_path)
            
            # Compare weights
            weights_match = True
            for p1, p2 in zip(agent1.global_network.parameters(),
                            agent2.global_network.parameters()):
                if not torch.allclose(p1, p2):
                    weights_match = False
                    break
                    
            if weights_match:
                print("✅ PASS - Save/load works correctly")
            else:
                error = "Weights don't match after load"
                print(f"❌ FAIL: {error}")
                failures.append(("Test 6", error))
    except Exception as e:
        print(f"❌ FAIL: {e}")
        failures.append(("Test 6", str(e)))
    
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
        print("\n📝 Note: Full multiprocessing training test skipped in simple mode")
        return 0


if __name__ == "__main__":
    sys.exit(test_a3c_components())