# Detailed RL Integration Plan for Remaining Projects

## 📋 Overview

This document provides a comprehensive plan for integrating RL into:
1. **marker** - Document processing strategy selection
2. **claude-module-communicator** - Module routing optimization

## 🎯 Integration 1: Marker - Processing Strategy Selection

### Purpose
Optimize document processing strategy selection based on document characteristics to balance speed and accuracy.

### RL Algorithm: DQN (Deep Q-Network)
- **Why DQN**: Sequential decision making with discrete actions
- **State Space**: Document features (10 dimensions)
- **Action Space**: 4 processing strategies
- **Reward**: Weighted combination of accuracy and speed

### Implementation Plan

#### Phase 1: Feature Engineering (Day 1)
Create document feature extractor:
```python
def extract_document_features(document_path: Path) -> np.ndarray:
    """Extract features for RL state representation"""
    features = []
    
    # 1. File size (MB, log-normalized)
    file_size_mb = document_path.stat().st_size / (1024 * 1024)
    features.append(np.log1p(file_size_mb))
    
    # 2. Page count estimate
    # 3. Text density (text/total content ratio)
    # 4. Image count
    # 5. Table count
    # 6. Language complexity score
    # 7. Contains code blocks (binary)
    # 8. Contains mathematical equations (binary)
    # 9. Document type (PDF=1, DOCX=2, etc.)
    # 10. Previous processing history encoding
    
    return np.array(features)
```

#### Phase 2: Action Mapping (Day 1)
Define processing strategies:
```python
PROCESSING_STRATEGIES = {
    0: {
        "name": "fast_parse",
        "method": "pypdf_basic",
        "expected_time": 1.0,  # seconds per page
        "expected_accuracy": 0.7
    },
    1: {
        "name": "standard_ocr",
        "method": "tesseract_standard",
        "expected_time": 3.0,
        "expected_accuracy": 0.85
    },
    2: {
        "name": "advanced_ocr",
        "method": "tesseract_advanced",
        "expected_time": 5.0,
        "expected_accuracy": 0.95
    },
    3: {
        "name": "hybrid_smart",
        "method": "marker_ml_enhanced",
        "expected_time": 4.0,
        "expected_accuracy": 0.92
    }
}
```

#### Phase 3: Integration Structure (Day 2)
Create `/home/graham/workspace/experiments/marker/src/marker/rl_integration/`:

```
marker/src/marker/rl_integration/
├── __init__.py
├── strategy_selector.py      # Main RL integration
├── feature_extractor.py      # Document feature extraction
├── reward_calculator.py      # Multi-objective reward
└── performance_tracker.py    # Track processing metrics
```

#### Phase 4: Core Integration (Day 2-3)
```python
# strategy_selector.py
from rl_commons import DQNAgent, RLState, RLTracker

class ProcessingStrategySelector:
    def __init__(self, model_path: Optional[Path] = None):
        self.feature_extractor = DocumentFeatureExtractor()
        self.agent = DQNAgent(
            name="marker_strategy_selector",
            state_dim=10,
            action_dim=4,
            learning_rate=1e-3,
            epsilon_start=0.3,  # Less exploration needed
            batch_size=32,
            buffer_size=5000
        )
        
        self.tracker = RLTracker("marker_rl")
        self.processing_history = []
        
        if model_path and model_path.exists():
            self.agent.load(model_path)
    
    def select_strategy(self, document_path: Path, 
                       quality_requirement: float = 0.8) -> Dict[str, Any]:
        """Select optimal processing strategy"""
        # Extract features
        features = self.feature_extractor.extract(document_path)
        features = np.append(features, quality_requirement)  # Add requirement
        
        state = RLState(features=features, context={"path": str(document_path)})
        
        # Get RL decision
        action = self.agent.select_action(state)
        strategy = PROCESSING_STRATEGIES[action.action_id]
        
        return {
            "strategy": strategy,
            "action_id": action.action_id,
            "expected_time": strategy["expected_time"],
            "expected_accuracy": strategy["expected_accuracy"]
        }
    
    def update_from_result(self, document_path: Path, strategy_id: int,
                          actual_time: float, actual_accuracy: float,
                          quality_requirement: float = 0.8):
        """Update RL model based on processing results"""
        # Calculate reward
        time_score = 1.0 - min(1.0, actual_time / 10.0)  # Normalize to [0,1]
        accuracy_score = actual_accuracy
        requirement_met = float(actual_accuracy >= quality_requirement)
        
        # Multi-objective reward
        reward_value = (
            0.4 * accuracy_score +      # 40% weight on accuracy
            0.3 * time_score +          # 30% weight on speed
            0.3 * requirement_met       # 30% weight on meeting requirements
        )
        
        # Update agent
        features = self.feature_extractor.extract(document_path)
        features = np.append(features, quality_requirement)
        
        state = RLState(features=features)
        action = RLAction("select_strategy", strategy_id)
        reward = RLReward(
            value=reward_value,
            components={
                "accuracy": accuracy_score,
                "speed": time_score,
                "requirement_met": requirement_met
            }
        )
        
        # For DQN, we need next_state (use same state for terminal)
        self.agent.update(state, action, reward, state, done=True)
```

#### Phase 5: Testing & Validation (Day 4)
Create test scenarios:
```python
# test_rl_strategy_selection.py
async def test_strategy_adaptation():
    selector = ProcessingStrategySelector()
    
    test_documents = [
        ("simple_text.pdf", 0.7),      # Low quality OK
        ("complex_scan.pdf", 0.95),    # High quality needed
        ("mixed_content.pdf", 0.85),   # Medium quality
        ("code_heavy.pdf", 0.9),       # Code extraction
    ]
    
    # Process documents and learn
    for doc_path, quality_req in test_documents:
        # Select strategy
        result = selector.select_strategy(doc_path, quality_req)
        
        # Process document (mock)
        actual_time, actual_accuracy = process_document(doc_path, result["strategy"])
        
        # Update RL
        selector.update_from_result(
            doc_path, result["action_id"], 
            actual_time, actual_accuracy, quality_req
        )
    
    # Check if learning occurred
    assert selector.agent.epsilon < 0.3  # Exploration decreased
    print(selector.get_performance_report())
```

## 🎯 Integration 2: Claude-Module-Communicator - Route Optimization

### Purpose
Optimize module routing decisions to minimize latency and resource usage while maximizing task completion quality.

### RL Algorithm: Hierarchical RL (Options Framework)
- **Why Hierarchical**: Complex multi-level decisions
- **High-Level**: Select routing pattern
- **Low-Level**: Select specific modules and parameters

### Implementation Plan

#### Phase 1: State Representation (Day 1)
```python
def extract_task_features(request: Dict[str, Any]) -> np.ndarray:
    """Extract features for routing decisions"""
    features = []
    
    # 1. Task type encoding (one-hot or embedding)
    # 2. Input size/complexity
    # 3. Priority level
    # 4. Available modules bitmap
    # 5. Current system load
    # 6. Previous task results encoding
    # 7. Time constraints
    # 8. Quality requirements
    # 9. Cost budget
    # 10-20. Task-specific features
    
    return np.array(features)
```

#### Phase 2: Action Space Design (Day 2)
```python
# High-level actions (routing patterns)
ROUTING_PATTERNS = {
    0: "direct_single",      # Single module, no preprocessing
    1: "sequential_basic",   # Linear pipeline
    2: "parallel_split",     # Parallel processing
    3: "hierarchical_deep",  # Multi-stage with feedback
    4: "adaptive_dynamic"    # Runtime adaptation
}

# Low-level actions (module selection per pattern)
MODULE_ACTIONS = {
    "direct_single": ["marker", "sparta", "claude_proxy"],
    "sequential_basic": ["marker->sparta->claude", "sparta->claude", ...],
    "parallel_split": ["marker||screenshot", "sparta||arangodb", ...],
    # etc.
}
```

#### Phase 3: Hierarchical Agent (Day 3-4)
First, implement hierarchical RL in rl_commons:
```python
# rl_commons/algorithms/hierarchical/options_framework.py
class HierarchicalRLAgent(RLAgent):
    def __init__(self, name: str, 
                 high_level_state_dim: int,
                 high_level_action_dim: int,
                 low_level_configs: Dict[int, Dict]):
        """
        Initialize hierarchical agent
        
        Args:
            high_level_state_dim: State dimension for meta-controller
            high_level_action_dim: Number of high-level options
            low_level_configs: Config for each low-level policy
        """
        super().__init__(name)
        
        # High-level meta-controller (selects options)
        self.meta_controller = DQNAgent(
            name=f"{name}_meta",
            state_dim=high_level_state_dim,
            action_dim=high_level_action_dim
        )
        
        # Low-level controllers (one per option)
        self.option_controllers = {}
        for option_id, config in low_level_configs.items():
            self.option_controllers[option_id] = DQNAgent(
                name=f"{name}_option_{option_id}",
                **config
            )
        
        self.current_option = None
        self.option_start_state = None
```

#### Phase 4: Module Orchestrator Integration (Day 5-6)
```python
# claude-module-communicator/src/rl_orchestration/module_router.py
from rl_commons import HierarchicalRLAgent, RLState

class RLModuleOrchestrator:
    def __init__(self):
        # Define low-level configs for each routing pattern
        low_level_configs = {
            0: {"state_dim": 20, "action_dim": 3},   # direct_single
            1: {"state_dim": 25, "action_dim": 10},  # sequential_basic
            2: {"state_dim": 30, "action_dim": 15},  # parallel_split
            # etc.
        }
        
        self.agent = HierarchicalRLAgent(
            name="module_orchestrator",
            high_level_state_dim=20,
            high_level_action_dim=5,
            low_level_configs=low_level_configs
        )
        
        self.module_registry = ModuleRegistry()
        self.execution_engine = ExecutionEngine()
    
    def route_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request through optimal module pipeline"""
        # Extract features
        features = extract_task_features(request)
        state = RLState(features=features, context=request)
        
        # High-level decision: routing pattern
        pattern_action = self.agent.select_high_level_action(state)
        pattern = ROUTING_PATTERNS[pattern_action.action_id]
        
        # Low-level decision: specific modules
        module_action = self.agent.select_low_level_action(state, pattern_action.action_id)
        modules = self._decode_module_selection(pattern, module_action)
        
        # Execute plan
        execution_plan = {
            "pattern": pattern,
            "modules": modules,
            "parallel": "parallel" in pattern
        }
        
        return self.execution_engine.execute(execution_plan, request)
    
    def update_from_execution(self, request, execution_plan, results, metrics):
        """Update RL model based on execution results"""
        # Calculate hierarchical rewards
        high_level_reward = self._calculate_pattern_reward(metrics)
        low_level_reward = self._calculate_module_reward(results, metrics)
        
        # Update both levels
        self.agent.update_hierarchical(
            high_level_reward=high_level_reward,
            low_level_reward=low_level_reward,
            metrics=metrics
        )
```

#### Phase 5: Advanced Features (Day 7-8)
1. **Dynamic Module Discovery**
   - Automatically detect new modules
   - Update action space dynamically

2. **Transfer Learning**
   - Share learned patterns between similar tasks
   - Pre-train on historical data

3. **Safety Constraints**
   - Maximum latency bounds
   - Cost limits
   - Fallback to known-good patterns

## 📊 Integration Timeline

### Week 1: Foundation
- **Day 1-2**: DQN implementation ✅
- **Day 3-4**: Marker integration setup
- **Day 5**: Marker testing

### Week 2: Advanced
- **Day 1-2**: Hierarchical RL implementation
- **Day 3-4**: Module communicator integration
- **Day 5**: Integration testing

### Week 3: Production
- **Day 1-2**: Performance optimization
- **Day 3-4**: Monitoring dashboard
- **Day 5**: Documentation and deployment

## 🔧 Common Integration Components

### 1. Gradual Rollout Strategy
```python
class GradualRLRollout:
    def __init__(self, rl_agent, baseline, initial_percentage=0.1):
        self.rl_agent = rl_agent
        self.baseline = baseline
        self.percentage = initial_percentage
        self.performance_history = []
    
    def should_use_rl(self):
        if random.random() < self.percentage:
            return True
        return False
    
    def auto_adjust_percentage(self):
        """Automatically increase/decrease based on performance"""
        if len(self.performance_history) > 100:
            rl_performance = np.mean([p for p in self.performance_history if p["method"] == "rl"])
            baseline_performance = np.mean([p for p in self.performance_history if p["method"] == "baseline"])
            
            if rl_performance > baseline_performance * 1.1:  # 10% better
                self.percentage = min(1.0, self.percentage * 1.2)
            elif rl_performance < baseline_performance * 0.9:  # 10% worse
                self.percentage = max(0.05, self.percentage * 0.8)
```

### 2. Unified Monitoring
```python
# rl_commons/monitoring/unified_dashboard.py
class UnifiedRLDashboard:
    def __init__(self, projects: List[str]):
        self.projects = projects
        self.trackers = {p: RLTracker(p) for p in projects}
    
    def generate_comparison_report(self):
        """Compare RL performance across projects"""
        report = {}
        for project, tracker in self.trackers.items():
            stats = tracker.get_current_stats()
            report[project] = {
                "avg_reward": stats["avg_reward_100"],
                "improvement_over_baseline": stats.get("improvement", 0),
                "adoption_rate": stats.get("rl_usage_percentage", 0),
                "total_decisions": stats["total_steps"]
            }
        return report
```

### 3. Transfer Learning
```python
class TransferLearning:
    def __init__(self, source_agent: RLAgent, target_agent: RLAgent):
        self.source = source_agent
        self.target = target_agent
    
    def transfer_knowledge(self, transfer_ratio: float = 0.5):
        """Transfer learned patterns from source to target"""
        if hasattr(self.source, "q_network") and hasattr(self.target, "q_network"):
            # Transfer network weights
            source_state = self.source.q_network.state_dict()
            target_state = self.target.q_network.state_dict()
            
            for key in target_state:
                if key in source_state and source_state[key].shape == target_state[key].shape:
                    # Blend weights
                    target_state[key] = (
                        transfer_ratio * source_state[key] + 
                        (1 - transfer_ratio) * target_state[key]
                    )
            
            self.target.q_network.load_state_dict(target_state)
```

## 🎯 Success Criteria

### For Marker Integration
- [ ] 30% reduction in average processing time
- [ ] Maintain 95% of baseline accuracy
- [ ] Adapt to new document types within 50 examples
- [ ] Stable performance after 1000 documents

### For Module Communicator Integration
- [ ] 40% reduction in average task completion time
- [ ] 25% reduction in resource usage
- [ ] Successful routing for 95% of requests
- [ ] Learn optimal patterns for common task types

## 🚀 Next Steps

1. **Immediate**: Complete marker integration using DQN
2. **This Week**: Test marker with real documents
3. **Next Week**: Implement hierarchical RL
4. **Following Week**: Complete module communicator integration

The modular design of `rl_commons` makes it easy to add new algorithms and integrate with additional projects as needed.
