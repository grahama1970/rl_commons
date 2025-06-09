# Module Integration Guide for RL Commons

## Overview

This guide explains how `claude-module-communicator` and other modules can leverage RL Commons to optimize their interactions and decision-making processes.

## Quick Start for Module Developers

### 1. Identify Your Use Case

First, determine what type of optimization your module needs:

- **Selection Problems**: Choosing between options (providers, strategies, methods)
- **Sequential Decisions**: Multi-step processes with dependencies
- **Coordination**: Multiple modules working together
- **Parameter Tuning**: Optimizing continuous values
- **Learning from Examples**: Mimicking expert behavior

### 2. Describe Your Module

Create a module specification that RL Commons can understand:

```python
module_spec = {
    'name': 'your_module_name',
    'type': 'processor',  # processor, provider, storage, etc.
    
    # What decisions does your module make?
    'action_space': {
        'type': 'discrete',  # or 'continuous'
        'size': 5,  # number of choices
        'actions': ['method1', 'method2', 'method3', 'method4', 'method5']
    },
    
    # What information is available for decisions?
    'state_space': {
        'features': [
            'input_size',
            'complexity_score', 
            'available_memory',
            'time_constraint'
        ],
        'continuous': True
    },
    
    # What are you optimizing for?
    'objectives': {
        'primary': 'quality',
        'secondary': ['speed', 'cost'],
        'constraints': {
            'max_latency_ms': 1000,
            'min_quality_score': 0.8
        }
    },
    
    # Module connections
    'connections': {
        'inputs_from': ['module_a', 'module_b'],
        'outputs_to': ['module_c', 'module_d']
    }
}
```

### 3. Let RL Commons Choose the Algorithm

```python
from rl_commons import RLCommonsIntegration

# Initialize RL Commons integration
rl = RLCommonsIntegration()

# Let it analyze and select the best algorithm
algorithm = rl.select_algorithm(module_spec)
print(f"Selected algorithm: {algorithm.name}")
# Output might be: "Selected algorithm: DQN" or "Selected algorithm: MARL"
```

### 4. Integrate with Your Module

```python
class YourModule:
    def __init__(self):
        self.rl = RLCommonsIntegration()
        self.algorithm = None
        
    def initialize_rl(self):
        """Call this once when module starts"""
        module_spec = self.get_module_spec()
        self.algorithm = self.rl.select_algorithm(module_spec)
        
    def make_decision(self, context):
        """Use RL to make optimal decisions"""
        # Convert your context to state
        state = self.extract_state(context)
        
        # Get action from RL algorithm
        action = self.algorithm.select_action(state)
        
        # Execute the action
        result = self.execute_action(action, context)
        
        # Calculate reward based on outcome
        reward = self.calculate_reward(result)
        
        # Update the algorithm
        next_state = self.extract_state(result.context)
        self.algorithm.update(state, action, reward, next_state)
        
        return result
```

## Integration Patterns by Module Type

### 1. Provider Selection Modules (e.g., claude_max_proxy)

```python
class ProviderSelector:
    def __init__(self):
        self.rl = RLCommonsIntegration()
        # Contextual Bandits are ideal for provider selection
        self.bandit = self.rl.create_contextual_bandit(
            arms=['gpt-4', 'claude-3', 'gemini', 'llama-local']
        )
    
    def select_provider(self, request):
        # Extract context features
        context = np.array([
            request.prompt_length,
            request.complexity,
            request.priority,
            self.get_current_load()
        ])
        
        # Select provider
        provider = self.bandit.select_arm(context)
        
        # Make request
        response = self.call_provider(provider, request)
        
        # Calculate multi-objective reward
        reward = self.calculate_reward(
            quality=response.quality,
            latency=response.latency,
            cost=response.cost
        )
        
        # Update bandit
        self.bandit.update(context, provider, reward)
        
        return response
```

### 2. Processing Pipeline Modules (e.g., marker)

```python
class DocumentProcessor:
    def __init__(self):
        self.rl = RLCommonsIntegration()
        # DQN for sequential processing decisions
        self.dqn = self.rl.create_dqn(
            state_dim=10,
            action_dim=4  # different processing methods
        )
    
    def process_document(self, document):
        results = []
        state = self.extract_document_features(document)
        
        for page in document.pages:
            # Decide processing method
            action = self.dqn.select_action(state)
            
            # Process page
            result = self.process_page(page, method=action)
            results.append(result)
            
            # Update state and learn
            next_state = self.extract_features(result)
            reward = self.calculate_quality_reward(result)
            self.dqn.update(state, action, reward, next_state)
            
            state = next_state
            
        return results
```

### 3. Orchestration Modules (e.g., claude-module-communicator)

```python
class ModuleOrchestrator:
    def __init__(self):
        self.rl = RLCommonsIntegration()
        # Hierarchical RL for complex orchestration
        self.hierarchical_rl = self.rl.create_hierarchical_rl()
        # MARL for multi-module coordination
        self.marl = self.rl.create_marl()
        
    def orchestrate_workflow(self, task):
        # High-level: decide workflow strategy
        workflow_option = self.hierarchical_rl.select_option(task)
        
        # Low-level: coordinate modules
        module_assignments = self.marl.coordinate_agents(
            task=task,
            available_modules=self.get_active_modules(),
            workflow=workflow_option
        )
        
        # Execute with monitoring
        results = self.execute_workflow(module_assignments)
        
        # Learn from outcome
        self.update_all_algorithms(task, results)
        
        return results
```

### 4. Storage/Database Modules (e.g., arangodb)

```python
class DatabaseOptimizer:
    def __init__(self):
        self.rl = RLCommonsIntegration()
        # PPO for continuous parameter optimization
        self.ppo = self.rl.create_ppo(
            state_dim=15,  # DB metrics
            action_dim=5,  # tunable parameters
            continuous=True
        )
    
    def optimize_query(self, query_context):
        # Get current DB state
        state = self.get_database_metrics()
        
        # Get optimization parameters
        params = self.ppo.select_action(state)
        
        # Apply parameters
        self.apply_parameters(params)
        
        # Execute query
        result = self.execute_query(query_context)
        
        # Calculate reward
        reward = self.calculate_performance_reward(result)
        
        # Update PPO
        next_state = self.get_database_metrics()
        self.ppo.update(state, params, reward, next_state)
        
        return result
```

## Advanced Integration Scenarios

### 1. Multi-Module Coordination with MARL

```python
# For modules that need to coordinate without central control
class DistributedProcessor:
    def __init__(self, module_id):
        self.module_id = module_id
        self.rl = RLCommonsIntegration()
        self.agent = self.rl.create_marl_agent(module_id)
        
    def process_with_coordination(self, task):
        # Share intentions with other modules
        intention = self.agent.broadcast_intention(task)
        
        # Receive other modules' intentions
        other_intentions = self.agent.receive_intentions()
        
        # Make coordinated decision
        action = self.agent.select_coordinated_action(
            task, 
            other_intentions
        )
        
        # Execute and share results
        result = self.execute(action)
        self.agent.broadcast_result(result)
        
        # Learn from collective outcome
        collective_reward = self.calculate_collective_reward()
        self.agent.update(collective_reward)
        
        return result
```

### 2. Learning from Demonstrations with IRL

```python
# For modules that should mimic expert behavior
class ExpertMimicModule:
    def __init__(self):
        self.rl = RLCommonsIntegration()
        self.irl = self.rl.create_irl()
        
    def learn_from_expert(self, expert_traces):
        """Learn reward function from expert demonstrations"""
        self.irl.add_demonstrations(expert_traces)
        learned_reward = self.irl.learn_reward_function()
        
        # Create policy based on learned reward
        self.policy = self.rl.create_policy_from_reward(learned_reward)
        
    def execute_like_expert(self, situation):
        """Execute actions similar to expert"""
        state = self.extract_state(situation)
        action = self.policy.select_action(state)
        return self.execute(action)
```

### 3. Rapid Adaptation with Meta-Learning

```python
# For modules that encounter new types of tasks
class AdaptiveModule:
    def __init__(self):
        self.rl = RLCommonsIntegration()
        self.maml = self.rl.create_maml()
        
    def adapt_to_new_task(self, new_task_examples):
        """Quickly adapt to new task type"""
        # Fine-tune on few examples
        adapted_policy = self.maml.adapt(
            new_task_examples,
            adaptation_steps=5
        )
        
        # Use adapted policy
        state = self.extract_state(new_task_examples[-1])
        action = adapted_policy.select_action(state)
        return self.execute(action)
```

## Best Practices

### 1. Start Simple
- Begin with basic algorithms (Bandits, DQN)
- Add complexity only when simple approaches fail
- Measure improvement over baseline

### 2. Define Clear Rewards
```python
def calculate_reward(self, outcome):
    """Example of clear, interpretable reward"""
    # Normalize components to [0, 1]
    quality = outcome.quality_score  # 0-1
    speed = 1.0 / (1.0 + outcome.latency_seconds)  # 0-1
    cost = 1.0 - (outcome.cost / self.max_cost)  # 0-1
    
    # Weighted combination
    weights = {'quality': 0.5, 'speed': 0.3, 'cost': 0.2}
    reward = sum(weights[k] * v for k, v in 
                 {'quality': quality, 'speed': speed, 'cost': cost}.items())
    
    return reward
```

### 3. Monitor Performance
```python
# Always track algorithm performance
self.rl.monitor.track_episode(
    algorithm='DQN',
    module='marker',
    reward=reward,
    metadata={
        'document_type': 'pdf',
        'pages': 42,
        'processing_time': 3.2
    }
)
```

### 4. Implement Safeguards
```python
# Always have fallback behavior
if not self.rl.is_confident(state):
    # Fall back to default behavior
    return self.default_action()
else:
    # Use RL recommendation
    return self.rl.select_action(state)
```

## Troubleshooting

### Algorithm Not Learning
1. Check reward scale (should be normalized)
2. Verify state features are informative
3. Ensure sufficient exploration
4. Consider using demonstrations (IRL)

### Unstable Performance
1. Reduce learning rate
2. Increase batch size
3. Use more conservative algorithms (PPO vs A3C)
4. Implement reward clipping

### Slow Decision Making
1. Use simpler algorithms for real-time decisions
2. Implement action caching
3. Consider async execution
4. Pre-compute features when possible

## Getting Help

- Check the [Algorithm Selection Guide](guides/algorithm_selection_guide.md)
- Review [example integrations](examples/)
- Open an issue with your module spec
- Join the discussion forum

Remember: RL Commons is designed to handle the complexity of RL so you can focus on your module's core functionality!