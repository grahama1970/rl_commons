# RL COMMONS CONTEXT — CLAUDE.md

> **Inherits standards from global and workspace CLAUDE.md files with overrides below.**

## Project Context
**Purpose:** Shared reinforcement learning components for ecosystem optimization  
**Type:** Intelligence Core (Core Infrastructure)  
**Status:** Development  
**Dependencies:** Hub, Test Reporting

## Project-Specific Overrides

### Special Dependencies
```toml
# RL Commons specific packages
gymnasium = "^0.29.0"
stable-baselines3 = "^2.0.0"
torch = "^2.0.0"
numpy = "^1.24.0"
scikit-learn = "^1.3.0"
```

### Environment Variables  
```bash
# .env additions for RL Commons
RL_DEVICE=cuda  # or cpu
RL_LOGGING_LEVEL=INFO
MODEL_SAVE_PATH=/home/graham/workspace/data/rl_models
EXPLORATION_RATE=0.1
LEARNING_RATE=0.001
```

### Special Considerations
- **GPU Acceleration:** CUDA support for training algorithms
- **Model Persistence:** Saves trained models for cross-module use
- **Multi-Agent Coordination:** Manages RL agents across different modules
- **Safe Deployment:** Gradual rollout patterns for production RL

---

## License

MIT License — see [LICENSE](LICENSE) for details.