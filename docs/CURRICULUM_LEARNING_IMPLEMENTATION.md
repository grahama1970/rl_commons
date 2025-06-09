# Curriculum Learning Implementation

## Overview

The curriculum learning module provides a comprehensive framework for training RL agents with progressively challenging tasks. This implementation supports automatic curriculum generation, performance-based adaptation, and integration with meta-learning (MAML) for fast adaptation to new domains.

## Architecture

### Core Components

1. **Base Classes** (`base.py`)
   - `Task`: Represents a learning task with difficulty, prerequisites, and context
   - `TaskDifficulty`: Predefined difficulty levels (TRIVIAL to IMPOSSIBLE)
   - `PerformanceTracker`: Tracks agent performance across tasks
   - `CurriculumScheduler`: Base class for curriculum scheduling strategies
   - `TaskGenerator`: Abstract base for automatic task generation

2. **Curriculum Types**

   - **Automatic Curriculum** (`automatic.py`)
     - Generates tasks based on performance history
     - Uses performance-based task generation
     - Adapts difficulty based on learning progress
     - Supports exploration vs exploitation trade-offs

   - **Progressive Curriculum** (`progressive.py`)
     - Smooth difficulty transitions
     - Zone of Proximal Development (ZPD) calculation
     - Mastery tracking and review recommendations
     - Handles regression and progression rates

   - **Adaptive Curriculum** (`adaptive.py`)
     - Learns individual learner profiles
     - Uses Gaussian Process for performance prediction
     - Adapts to learning styles (exploration, mastery, balanced)
     - Tracks strengths and weaknesses by task type

   - **Meta-Curriculum** (`meta_curriculum.py`)
     - Integrates with MAML for fast adaptation
     - Supports domain-specific task generation
     - Meta-updates for improving adaptation
     - Progressive complexity scheduling

## Key Features

### 1. Task Management
```python
# Create tasks with prerequisites
task = Task(
    task_id="advanced_task",
    difficulty=0.7,
    context={"env": "complex", "steps": 100},
    prerequisites=["basic_task_1", "basic_task_2"]
)
```

### 2. Performance Tracking
```python
# Track performance across attempts
tracker = PerformanceTracker()
tracker.record_attempt("task_1", reward=0.8, success=True)
readiness = tracker.get_readiness_score(difficulty=0.5)
```

### 3. Automatic Task Generation
```python
# Generate tasks based on performance
curriculum = AutomaticCurriculum(
    difficulty_bins=10,
    performance_window=20,
    exploration_rate=0.1
)
next_task = curriculum.select_next_task()
```

### 4. Learner Adaptation
```python
# Adapt to individual learning patterns
curriculum = AdaptiveCurriculum(
    tasks,
    learning_style="balanced",
    use_gp_model=True
)
insights = curriculum.get_learner_insights()
```

### 5. Meta-Learning Integration
```python
# Fast adaptation to new domains
meta_curriculum = MetaCurriculum(
    maml_agent=agent,
    meta_batch_size=4,
    adaptation_steps=5
)
adapted_agent = meta_curriculum.adapt_to_new_domain(domain_tasks)
```

## Module Orchestration Example

The curriculum learning system is particularly well-suited for training agents to orchestrate multiple modules, as in the claude-module-communicator use case:

```python
# Define progressive module complexity
tasks = [
    # Level 1: Single modules
    Task("single_marker", difficulty=0.2, context={"modules": ["marker"]}),
    
    # Level 2: Two-module integration
    Task("marker_to_arangodb", difficulty=0.5, 
         context={"modules": ["marker", "arangodb"]},
         prerequisites=["single_marker", "single_arangodb"]),
    
    # Level 3: Full pipeline
    Task("full_pipeline", difficulty=0.8,
         context={"modules": ["marker", "arangodb", "chat"]},
         prerequisites=["marker_to_arangodb", "arangodb_to_chat"])
]
```

## Usage Patterns

### Basic Usage
```python
from rl_commons.algorithms.curriculum import ProgressiveCurriculum, Task

# Create tasks
tasks = [Task(f"task_{i}", difficulty=i*0.1) for i in range(10)]

# Initialize curriculum
curriculum = ProgressiveCurriculum(tasks, progression_rate=0.05)

# Training loop
for episode in range(100):
    task = curriculum.select_next_task()
    if task:
        # Train on task
        performance = train_on_task(agent, task)
        
        # Update curriculum
        curriculum.update_performance(
            task.task_id, 
            performance, 
            success=(performance > 0.7)
        )
```

### Advanced Adaptive Usage
```python
from rl_commons.algorithms.curriculum import AdaptiveCurriculum

# Create curriculum that adapts to learner
curriculum = AdaptiveCurriculum(
    tasks,
    learning_style="exploration",  # or "mastery", "balanced"
    use_gp_model=True  # Use Gaussian Process for prediction
)

# The curriculum will:
# - Learn individual strengths/weaknesses
# - Predict performance on new tasks
# - Adapt difficulty and task selection
# - Track preferred task types
```

### Meta-Learning Integration
```python
from rl_commons.algorithms.curriculum import MetaCurriculum
from rl_commons.algorithms.meta import MAMLAgent

# Create MAML agent
maml_agent = MAMLAgent(state_dim=10, action_dim=4)

# Create meta-curriculum
curriculum = MetaCurriculum(
    maml_agent=maml_agent,
    meta_batch_size=4,
    adaptation_steps=5
)

# Generate curriculum tasks
curriculum.generate_curriculum_tasks(num_tasks=20)

# Train with meta-updates
for episode in range(100):
    task = curriculum.select_next_task()
    performance = train_on_task(maml_agent, task)
    curriculum.update_performance(task.task_id, performance)
    
    # Meta-updates happen automatically based on frequency

# Adapt to new domain
new_domain_tasks = generate_domain_tasks("robotics")
adapted_agent = curriculum.adapt_to_new_domain(new_domain_tasks)
```

## Implementation Details

### Performance Metrics
- **Success Rate**: Percentage of successful task completions
- **Average Reward**: Mean reward across attempts
- **Improvement Rate**: Rate of performance improvement over time
- **Readiness Score**: Calculated based on current performance and difficulty gap

### Difficulty Progression
- **Zone of Proximal Development (ZPD)**: Tasks selected within learner's capability range
- **Smooth Transitions**: Difficulty changes are smoothed to prevent large jumps
- **Adaptive Pacing**: Progression rate adapts to learner's consistency

### Task Generation
- **Performance-Based**: New tasks generated based on performance gaps
- **Context Modifiers**: Task difficulty modified through context parameters
- **Domain Diversity**: Ensures exposure to different task types

## Testing

Comprehensive tests are provided in `test/algorithms/curriculum/test_curriculum.py`:

```bash
# Run all curriculum tests
pytest test/algorithms/curriculum/test_curriculum.py -v

# Run specific test class
pytest test/algorithms/curriculum/test_curriculum.py::TestProgressiveCurriculum -v
```

## Future Enhancements

1. **Multi-Agent Curriculum**: Support for training multiple agents with shared curriculum
2. **Hierarchical Tasks**: Support for task hierarchies and sub-tasks
3. **Transfer Learning**: Better support for transferring knowledge between domains
4. **Curriculum Visualization**: Tools for visualizing progression and performance
5. **Benchmark Tasks**: Standard benchmark tasks for curriculum learning evaluation

## References

- Bengio et al. (2009) - "Curriculum Learning"
- Graves et al. (2017) - "Automated Curriculum Learning for Neural Networks"
- Finn et al. (2017) - "Model-Agnostic Meta-Learning"
- Matiisen et al. (2019) - "Teacher-Student Curriculum Learning"