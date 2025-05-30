# Graham RL Commons 🤖

Shared Reinforcement Learning components for optimizing decisions across Graham's project ecosystem.

## 🚀 Features

- **Contextual Bandits**: For provider/strategy selection (claude_max_proxy)
- **Deep Q-Networks (DQN)**: For sequential decision making (marker)
- **Hierarchical RL**: For complex orchestration (claude-module-communicator)
- **PPO/A3C**: For continuous optimization (arangodb, sparta)
- **Unified Monitoring**: Track performance across all projects
- **Transfer Learning**: Share knowledge between similar tasks
- **Safe Deployment**: Gradual rollout with fallback strategies

## 📦 Installation

```bash
# Install from local development
cd /home/graham/workspace/experiments/rl_commons
pip install -e .

# Install from git (once published)
pip install git+https://github.com/grahama1970/graham-rl-commons.git
```

## 🎯 Quick Start

### Provider Selection (Contextual Bandit)
```python
from graham_rl_commons import ContextualBandit

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
from graham_rl_commons import DQNAgent

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

## 🔗 Integration Examples

- [claude_max_proxy Integration](docs/examples/claude_max_proxy_integration.md)
- [marker Integration](docs/examples/marker_integration.md)
- [Module Orchestration](docs/examples/module_orchestration.md)

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
graham_rl_commons/
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
pytest --cov=graham_rl_commons

# Run specific test category
pytest test/unit/
pytest test/integration/
```

## 📚 Documentation

- [API Reference](docs/api/)
- [Architecture Guide](docs/architecture/)
- [Integration Guide](docs/guides/integration.md)
- [Best Practices](docs/guides/best_practices.md)

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
