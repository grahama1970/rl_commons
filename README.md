# RL Commons 🤖

Shared Reinforcement Learning components for optimizing decisions across Graham's project ecosystem.

## 🚀 Features

### Core Algorithms
- **Contextual Bandits**: For provider/strategy selection (claude_max_proxy)
- **Deep Q-Networks (DQN)**: For sequential decision making (marker)
- **Hierarchical RL**: For complex orchestration (claude-module-communicator)
- **PPO/A3C**: For continuous optimization (arangodb, sparta)

### Advanced Techniques ✅
- **Multi-Agent RL (MARL)**: Decentralized module coordination with communication
- **Graph Neural Networks (GNN)**: Topology-aware decision making for module networks
- **Meta-Learning (MAML)**: Rapid adaptation to new modules (5-10 gradient steps)
- **Inverse RL (IRL)**: Learning from expert demonstrations and human feedback
- **Multi-Objective RL (MORL)**: Balancing latency, throughput, and cost
- **Curriculum Learning**: Progressive task complexity for robust training

### Infrastructure
- **Automatic Algorithm Selection**: Intelligently choose the best RL approach
- **Unified Monitoring**: Track performance across all projects
- **Transfer Learning**: Share knowledge between similar tasks
- **Safe Deployment**: Gradual rollout with fallback strategies

## 📦 Installation

```bash
# Install from local development
cd /home/graham/workspace/experiments/rl_commons
pip install -e .

# Install from git (once published)
pip install git+https://github.com/grahama1970/rl-commons.git
```

## 🎯 Quick Start

### Automatic Algorithm Selection
```python
from rl_commons.core.algorithm_selector import AlgorithmSelector, TaskProperties

# Let RL Commons choose the best algorithm
selector = AlgorithmSelector()
task_props = TaskProperties(
    action_space="discrete",
    state_dim=10,
    action_dim=4,
    requires_stability=True
)
agent = selector.select_algorithm(task_props)
```

### Module Orchestration with MORL
```python
from rl_commons.integrations import ModuleCommunicatorIntegration

# Setup module orchestration
integration = ModuleCommunicatorIntegration()
integration.register_module("marker", "pdf_processor", {"file_types": ["pdf"]})
integration.register_module("arangodb", "database", {"operations": ["store", "query"]})

# Create multi-objective orchestrator
orchestrator = integration.create_orchestrator(
    modules=["marker", "arangodb"],
    use_multi_objective=True  # Balance latency, throughput, reliability
)
```

### Provider Selection (Contextual Bandit)
```python
from rl_commons import ContextualBandit

# Initialize bandit for provider selection
bandit = ContextualBandit(
    arms=["gpt-4", "claude-3", "llama-local"],
    context_dim=5
)

# Select provider based on context
context = extract_features(request)
provider = bandit.select_arm(context)

# Update with observed reward
reward = calculate_reward(response_quality, latency, cost)
bandit.update(context, provider, reward)
```

### Document Processing (DQN)
```python
from rl_commons import DQNAgent

# Initialize DQN for processing decisions
agent = DQNAgent(
    state_dim=10,
    action_dim=4,  # [method1, method2, method3, method4]
    learning_rate=0.001
)

# Make decision
state = extract_document_features(document)
action = agent.select_action(state)
process_document(action)

# Update with experience
next_state = extract_document_features(processed_doc)
reward = calculate_processing_reward(quality, time)
agent.update(state, action, reward, next_state)
```

### Progressive Module Learning (Curriculum Learning)
```python
from rl_commons.algorithms.curriculum import ProgressiveCurriculum, Task
from rl_commons.algorithms import PPOAgent

# Define tasks with increasing complexity
tasks = [
    Task("single_marker", difficulty=0.2, context={"modules": ["marker"]}),
    Task("marker_to_db", difficulty=0.5, context={"modules": ["marker", "arangodb"]},
         prerequisites=["single_marker"]),
    Task("full_pipeline", difficulty=0.8, 
         context={"modules": ["marker", "arangodb", "chat"]},
         prerequisites=["marker_to_db"])
]

# Create curriculum
curriculum = ProgressiveCurriculum(tasks, progression_rate=0.05)

# Train with progressive difficulty
agent = PPOAgent(state_dim=10, action_dim=4)
for episode in range(100):
    task = curriculum.select_next_task()
    performance = train_on_task(agent, task)
    curriculum.update_performance(task.task_id, performance, success=performance > 0.7)
```

## 🔗 Integration Examples

- [claude_max_proxy Integration](docs/examples/claude_max_proxy_integration.md)
- [marker Integration](docs/examples/marker_integration.md)
- [Module Orchestration](docs/examples/module_orchestration.md)

## 🧠 Algorithm Selection

RL Commons includes an intelligent algorithm selection system that automatically chooses the best RL approach based on your task characteristics. See the [Algorithm Selection Guide](docs/guides/algorithm_selection_guide.md) for details.

```python
from rl_commons.core.algorithm_selector import AlgorithmSelector

# Automatic selection based on task characteristics
selector = AlgorithmSelector()
algorithm = selector.select_for_task({
    'modules': ['marker', 'claude_max_proxy', 'arangodb'],
    'objectives': ['quality', 'speed', 'cost'],
    'constraints': {'max_latency': 500}
})
```

## 📊 Monitoring

Access the unified dashboard:
```bash
rl-monitor --port 8501
```

Or use the CLI:
```bash
rl-commons status
rl-commons benchmark
rl-commons rollback
```

## 🏗️ Architecture

```
rl_commons/
├── core/              # Base classes and interfaces
├── algorithms/        # RL algorithm implementations
│   ├── bandits/      # Contextual bandits for selection
│   ├── dqn/          # Deep Q-Networks
│   └── hierarchical/ # Hierarchical RL
├── monitoring/        # Performance tracking
├── safety/           # Fallback and rollback
└── utils/            # Feature extraction, rewards
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rl_commons

# Run specific test category
pytest test/unit/
pytest test/integration/
```

## 📚 Documentation

- [API Reference](docs/api/)
- [Architecture Guide](docs/architecture/)
- [Algorithm Selection Guide](docs/guides/algorithm_selection_guide.md)
- [Module Integration Guide](docs/MODULE_INTEGRATION_GUIDE.md) - **Start Here for claude-module-communicator**
- [Curriculum Learning Implementation](docs/CURRICULUM_LEARNING_IMPLEMENTATION.md) - **Progressive task complexity**
- [Integration Guide](docs/guides/integration.md)
- [Best Practices](docs/guides/best_practices.md)
- [Advanced RL Implementation Tasks](docs/tasks/034_Advanced_RL_Techniques_Implementation.md)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🎯 Project Status

Currently integrating with:
- ✅ claude_max_proxy (provider selection)
- 🚧 marker (processing strategy)
- 🚧 claude-module-communicator (orchestration)
- 📋 arangodb (query optimization)
- 📋 sparta (pipeline optimization)

## 🔗 Related Projects

- [claude_max_proxy](https://github.com/grahama1970/claude_max_proxy)
- [marker](https://github.com/grahama1970/marker)
- [claude-module-communicator](https://github.com/grahama1970/claude-module-communicator)
