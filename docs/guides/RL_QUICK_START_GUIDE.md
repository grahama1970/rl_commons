# Quick Start Guide: RL Integration

## Step 1: Create rl_commons (Task #001)

```bash
# Create the shared module
cd /home/graham/workspace/experiments
mkdir -p rl_commons/src/rl_commons/{core,algorithms,monitoring}
cd rl_commons

# Create pyproject.toml
cat > pyproject.toml << 'TOML'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "rl_commons"
version = "0.1.0"
dependencies = [
    "numpy>=1.20.0",
    "torch>=2.0.0",
    "pydantic>=2.0.0",
]
TOML

# Create base agent
cat > src/rl_commons/core/base.py << 'PYTHON'
from abc import ABC, abstractmethod
import numpy as np

class RLAgent(ABC):
    @abstractmethod
    def select_action(self, state):
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state):
        pass
PYTHON
```

## Step 2: First Integration - claude_max_proxy (Task #005)

```python
# In claude_max_proxy/src/llm_call/rl_integration.py
from rl_commons import ContextualBandit
import numpy as np

class ProviderSelector:
    def __init__(self, providers=["gpt-4", "claude-3", "llama"]):
        self.providers = providers
        self.bandit = ContextualBandit(
            n_arms=len(providers),
            n_features=5
        )
    
    def select_provider(self, request):
        # Extract features
        features = self._extract_features(request)
        
        # Get RL decision
        provider_idx = self.bandit.select_action(features)
        return self.providers[provider_idx]
    
    def _extract_features(self, request):
        return np.array([
            len(request.get("prompt", "")) / 1000,  # Normalized length
            float(request.get("priority") == "speed"),
            float(request.get("priority") == "quality"),
            # Add more features
        ])
    
    def update_from_response(self, request, provider, metrics):
        features = self._extract_features(request)
        provider_idx = self.providers.index(provider)
        
        # Calculate reward
        reward = (
            0.5 * metrics["quality"] +
            0.3 * (1 - metrics["cost"]) +
            0.2 * (1 - metrics["latency"])
        )
        
        self.bandit.update(features, provider_idx, reward)

# Integration with existing code
class LLMClient:
    def __init__(self):
        self.rl_selector = ProviderSelector()
        
    async def call(self, prompt, **kwargs):
        request = {"prompt": prompt, **kwargs}
        
        # RL selects provider
        provider = self.rl_selector.select_provider(request)
        
        # Make API call
        response = await self._call_provider(provider, request)
        
        # Update RL model
        metrics = self._calculate_metrics(response)
        self.rl_selector.update_from_response(request, provider, metrics)
        
        return response
```

## Step 3: Test the Integration

```python
# tests/test_rl_integration.py
import pytest
from llm_call.rl_integration import ProviderSelector

def test_provider_selection_adapts():
    selector = ProviderSelector()
    
    # Simulate requests preferring speed
    speed_requests = [{"prompt": "Quick test", "priority": "speed"} for _ in range(20)]
    
    providers_used = []
    for request in speed_requests:
        provider = selector.select_provider(request)
        providers_used.append(provider)
        
        # Simulate llama being fastest
        if provider == "llama":
            metrics = {"quality": 0.7, "cost": 0.0, "latency": 0.1}
        else:
            metrics = {"quality": 0.9, "cost": 0.5, "latency": 0.5}
            
        selector.update_from_response(request, provider, metrics)
    
    # Check that selector learned to prefer llama for speed
    last_10_selections = providers_used[-10:]
    llama_percentage = last_10_selections.count("llama") / 10
    assert llama_percentage > 0.6  # Should mostly select llama
```

## Step 4: Monitor Performance

```python
# Enable monitoring
from rl_commons.monitoring import RLTracker

tracker = RLTracker("claude_max_proxy")

# In your integration
class ProviderSelector:
    def __init__(self):
        self.tracker = RLTracker("provider_selection")
        
    def select_provider(self, request):
        provider = self._select(request)
        self.tracker.log_decision("provider", provider)
        return provider
    
    def update_from_response(self, request, provider, metrics):
        self.tracker.log_reward(metrics["reward"])
        self.tracker.log_metrics({
            "cost_saved": 1 - metrics["cost"],
            "latency": metrics["latency"]
        })
```

## Step 5: Gradual Rollout

```python
# Safe deployment with fallback
class SafeProviderSelector:
    def __init__(self, exploration_rate=0.1):
        self.rl_selector = ProviderSelector()
        self.baseline = ExistingProviderLogic()
        self.exploration_rate = exploration_rate
    
    def select_provider(self, request):
        if random.random() < self.exploration_rate:
            # Use RL (10% of traffic initially)
            return self.rl_selector.select_provider(request)
        else:
            # Use existing logic (90% of traffic)
            return self.baseline.select_provider(request)
```

## Expected Timeline

- **Day 1-2**: Create rl_commons base
- **Day 3-4**: Implement contextual bandit
- **Day 5**: Integrate with claude_max_proxy
- **Day 6-7**: Test and monitor
- **Week 2**: Expand to other modules

## Common Pitfalls to Avoid

1. **Don't start with complex algorithms** - Contextual bandit is enough for provider selection
2. **Don't skip monitoring** - You need to track if RL is actually improving
3. **Don't deploy at 100%** - Start with 1-10% of traffic
4. **Don't forget fallback** - Always have a way to disable RL
5. **Don't optimize prematurely** - Get basic version working first

## Success Checklist

- [ ] rl_commons package created and installable
- [ ] Basic contextual bandit implemented
- [ ] First integration (claude_max_proxy) working
- [ ] Tests showing adaptation/learning
- [ ] Monitoring dashboard accessible
- [ ] Fallback strategy implemented
- [ ] 10% traffic rollout successful
- [ ] Cost reduction measured
- [ ] No quality degradation

Start with these steps and expand gradually!
