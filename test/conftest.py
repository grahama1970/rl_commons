"""Pytest configuration for RL Commons tests"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_state():
    """Create a sample RL state for testing"""
    from graham_rl_commons.core import RLState
    return RLState(
        features=np.array([0.5, 0.3, 0.8, 0.2, 0.9]),
        context={"request_id": "test123"}
    )


@pytest.fixture
def sample_action():
    """Create a sample RL action for testing"""
    from graham_rl_commons.core import RLAction
    return RLAction(
        action_type="select_provider",
        action_id=1,
        parameters={"provider": "claude-3"}
    )


@pytest.fixture
def sample_reward():
    """Create a sample RL reward for testing"""
    from graham_rl_commons.core import RLReward
    return RLReward(
        value=0.85,
        components={
            "quality": 0.9,
            "speed": 0.8,
            "cost": 0.85
        }
    )
