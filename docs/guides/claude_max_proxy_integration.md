# Integrating RL into claude_max_proxy

## Installation

1. **Install rl_commons dependency**:
   ```bash
   cd /home/graham/workspace/experiments/claude_max_proxy
   pip install -e /home/graham/workspace/experiments/rl_commons
   ```

   Or add to `pyproject.toml`:
   ```toml
   dependencies = [
       # ... existing dependencies ...
       "rl-commons @ git+file:///home/graham/workspace/experiments/rl_commons",
   ]
   ```

2. **Import RL components**:
   ```python
   from llm_call.rl_integration import RLProviderSelector
   ```

## Basic Integration

```python
# In your existing LLMClient class:

class LLMClient:
    def __init__(self, config):
        # ... existing init ...
        
        # Add RL selector
        self.rl_selector = RLProviderSelector(
            providers=list(self.providers.keys()),
            exploration_rate=0.1
        )
    
    async def call(self, prompt, **kwargs):
        # Select provider using RL
        provider, metadata = self.rl_selector.select_provider({
            "prompt": prompt,
            **kwargs
        })
        
        # Make API call with selected provider
        result = await self._call_provider(provider, prompt, **kwargs)
        
        # Update RL model with observed performance
        self.rl_selector.update_from_result(
            request={"prompt": prompt, **kwargs},
            provider=provider,
            result=result,
            latency=result["latency"],
            cost=result["cost"]
        )
        
        return result
```

## Gradual Rollout

Start with 10% of traffic:

```python
import random

class LLMClient:
    def __init__(self, config, use_rl_percentage=0.1):
        self.use_rl_percentage = use_rl_percentage
        # ... rest of init ...
    
    async def call(self, prompt, **kwargs):
        # Randomly decide whether to use RL
        if random.random() < self.use_rl_percentage:
            # Use RL selection
            provider = self._select_provider_rl(prompt, kwargs)
        else:
            # Use existing logic
            provider = self._select_provider_baseline(prompt, kwargs)
        
        # ... rest of method ...
```

## Monitoring

Track RL performance:

```python
# Get RL statistics
stats = self.rl_selector.get_provider_stats()
print(stats)

# Get recommendations
report = self.rl_selector.get_recommendation_report()
print(report)
```

## Production Checklist

- [ ] Install rl_commons dependency
- [ ] Add RLProviderSelector to LLMClient
- [ ] Implement gradual rollout (start at 10%)
- [ ] Set up monitoring dashboard
- [ ] Configure model persistence
- [ ] Add error handling and fallbacks
- [ ] Run A/B tests for 1 week
- [ ] Analyze results and adjust
- [ ] Gradually increase RL traffic
- [ ] Document learned patterns
