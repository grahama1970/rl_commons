"""
Module: task_distribution.py
Purpose: Task distribution and sampling for meta-learning

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.meta import TaskDistribution, TaskSampler
>>> dist = TaskDistribution(task_type='classification', num_classes=5)
>>> sampler = TaskSampler(dist)
>>> task = sampler.sample_task()
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Task:
    """Container for a single meta-learning task."""
    task_id: str
    support_data: Dict[str, torch.Tensor]
    query_data: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]
    
    def split_support_query(self, support_ratio: float = 0.5) -> Tuple[Dict, Dict]:
        """Split data into support and query sets."""
        # Already split
        return self.support_data, self.query_data


class TaskDistribution(ABC):
    """Abstract base class for task distributions."""
    
    @abstractmethod
    def sample_task(self) -> Task:
        """Sample a task from the distribution."""
        pass
    
    @abstractmethod
    def get_task_by_id(self, task_id: str) -> Task:
        """Get a specific task by ID."""
        pass


class SyntheticTaskDistribution(TaskDistribution):
    """Synthetic task distribution for testing."""
    
    def __init__(
        self,
        task_type: str = 'regression',
        input_dim: int = 10,
        output_dim: int = 1,
        num_classes: Optional[int] = None,
        task_complexity: str = 'medium'
    ):
        """Initialize synthetic task distribution.
        
        Args:
            task_type: Type of task ('regression', 'classification')
            input_dim: Input dimension
            output_dim: Output dimension (for regression)
            num_classes: Number of classes (for classification)
            task_complexity: Task complexity ('easy', 'medium', 'hard')
        """
        self.task_type = task_type
        self.input_dim = input_dim
        self.output_dim = output_dim if task_type == 'regression' else (num_classes or 2)
        self.num_classes = num_classes
        self.task_complexity = task_complexity
        
        # Task generation parameters
        self.complexity_params = {
            'easy': {'noise': 0.1, 'nonlinearity': 0.2},
            'medium': {'noise': 0.3, 'nonlinearity': 0.5},
            'hard': {'noise': 0.5, 'nonlinearity': 0.8}
        }
        
        self._task_counter = 0
        
    def sample_task(self, num_samples: int = 100) -> Task:
        """Sample a synthetic task.
        
        Args:
            num_samples: Total number of samples for the task
            
        Returns:
            Sampled task
        """
        self._task_counter += 1
        task_id = f"synthetic_{self.task_type}_{self._task_counter}"
        
        # Generate task parameters
        if self.task_type == 'regression':
            task_data = self._generate_regression_task(num_samples)
        else:
            task_data = self._generate_classification_task(num_samples)
        
        # Split into support and query
        split_idx = num_samples // 2
        support_data = {
            'states': task_data['states'][:split_idx],
            'targets': task_data['targets'][:split_idx]
        }
        query_data = {
            'states': task_data['states'][split_idx:],
            'targets': task_data['targets'][split_idx:]
        }
        
        metadata = {
            'task_type': self.task_type,
            'complexity': self.task_complexity,
            'num_samples': num_samples
        }
        
        return Task(task_id, support_data, query_data, metadata)
    
    def _generate_regression_task(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Generate a regression task."""
        params = self.complexity_params[self.task_complexity]
        
        # Random linear transformation
        W = torch.randn(self.input_dim, self.output_dim) * 2
        b = torch.randn(self.output_dim)
        
        # Generate input data
        X = torch.randn(num_samples, self.input_dim)
        
        # Compute outputs with optional nonlinearity
        Y = X @ W + b
        
        # Add nonlinearity based on complexity
        if params['nonlinearity'] > 0.3:
            Y = torch.sin(Y) * params['nonlinearity'] + Y * (1 - params['nonlinearity'])
        
        # Add noise
        Y += torch.randn_like(Y) * params['noise']
        
        return {'states': X, 'targets': Y.squeeze()}
    
    def _generate_classification_task(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Generate a classification task."""
        params = self.complexity_params[self.task_complexity]
        num_classes = self.num_classes or 2
        
        # Generate cluster centers
        centers = torch.randn(num_classes, self.input_dim) * 3
        
        # Generate samples
        samples_per_class = num_samples // num_classes
        X = []
        Y = []
        
        for i in range(num_classes):
            # Generate samples around center
            class_samples = centers[i] + torch.randn(samples_per_class, self.input_dim) * (1 + params['noise'])
            X.append(class_samples)
            Y.extend([i] * samples_per_class)
        
        # Handle remainder
        remainder = num_samples - (samples_per_class * num_classes)
        if remainder > 0:
            extra_class = np.random.randint(num_classes)
            extra_samples = centers[extra_class] + torch.randn(remainder, self.input_dim) * (1 + params['noise'])
            X.append(extra_samples)
            Y.extend([extra_class] * remainder)
        
        X = torch.cat(X, dim=0)
        Y = torch.tensor(Y, dtype=torch.long)
        
        # Shuffle
        perm = torch.randperm(len(Y))
        X = X[perm]
        Y = Y[perm]
        
        return {'states': X, 'targets': Y}
    
    def get_task_by_id(self, task_id: str) -> Task:
        """Get task by ID (generates new task for synthetic distribution)."""
        # For synthetic tasks, we generate a new one
        return self.sample_task()


class ModuleTaskDistribution(TaskDistribution):
    """Task distribution for module routing/optimization tasks."""
    
    def __init__(
        self,
        module_configs: List[Dict[str, Any]],
        scenario_generator: Optional[Callable] = None
    ):
        """Initialize module task distribution.
        
        Args:
            module_configs: List of module configurations
            scenario_generator: Function to generate task scenarios
        """
        self.module_configs = module_configs
        self.scenario_generator = scenario_generator or self._default_scenario_generator
        self._task_cache = {}
        self._task_counter = 0
        
    def sample_task(self) -> Task:
        """Sample a module optimization task."""
        self._task_counter += 1
        task_id = f"module_task_{self._task_counter}"
        
        # Generate scenario
        scenario = self.scenario_generator(self.module_configs)
        
        # Create task data
        support_data = self._create_task_data(scenario, num_episodes=50)
        query_data = self._create_task_data(scenario, num_episodes=50)
        
        metadata = {
            'num_modules': len(self.module_configs),
            'scenario': scenario['name'],
            'objective': scenario['objective']
        }
        
        task = Task(task_id, support_data, query_data, metadata)
        self._task_cache[task_id] = task
        
        return task
    
    def _default_scenario_generator(self, module_configs: List[Dict]) -> Dict[str, Any]:
        """Generate a default module routing scenario."""
        scenarios = [
            {
                'name': 'load_balancing',
                'objective': 'minimize_max_load',
                'load_pattern': 'uniform'
            },
            {
                'name': 'latency_optimization',
                'objective': 'minimize_latency',
                'load_pattern': 'bursty'
            },
            {
                'name': 'cost_optimization',
                'objective': 'minimize_cost',
                'load_pattern': 'periodic'
            }
        ]
        
        return np.random.choice(scenarios)
    
    def _create_task_data(self, scenario: Dict[str, Any], num_episodes: int) -> Dict[str, torch.Tensor]:
        """Create task data for module scenario."""
        states = []
        actions = []
        rewards = []
        
        for _ in range(num_episodes):
            # Simulate module states
            module_states = torch.randn(len(self.module_configs), 10)  # 10 features per module
            
            # Flatten for state representation
            state = module_states.flatten()
            states.append(state)
            
            # Random action (which module to route to)
            action = np.random.randint(len(self.module_configs))
            actions.append(action)
            
            # Compute reward based on scenario objective
            if scenario['objective'] == 'minimize_max_load':
                # Simulate load distribution
                loads = torch.randn(len(self.module_configs)).abs()
                loads[action] += 0.5  # Increase load on selected module
                reward = -loads.max().item()  # Negative max load
            elif scenario['objective'] == 'minimize_latency':
                # Simulate latency
                latencies = torch.randn(len(self.module_configs)).abs() * 100
                reward = -latencies[action].item()
            else:  # minimize_cost
                # Simulate cost
                costs = torch.randn(len(self.module_configs)).abs() * 10
                reward = -costs[action].item()
            
            rewards.append(reward)
        
        return {
            'states': torch.stack(states),
            'actions': torch.tensor(actions, dtype=torch.long),
            'rewards': torch.tensor(rewards, dtype=torch.float),
            'targets': torch.tensor(actions, dtype=torch.long)  # For compatibility
        }
    
    def get_task_by_id(self, task_id: str) -> Task:
        """Get cached task by ID."""
        if task_id in self._task_cache:
            return self._task_cache[task_id]
        else:
            raise ValueError(f"Task {task_id} not found in cache")


class TaskSampler:
    """Sampler for meta-learning tasks."""
    
    def __init__(
        self,
        task_distribution: TaskDistribution,
        batch_size: int = 4
    ):
        """Initialize task sampler.
        
        Args:
            task_distribution: Distribution to sample from
            batch_size: Number of tasks per batch
        """
        self.task_distribution = task_distribution
        self.batch_size = batch_size
        
    def sample_batch(self) -> List[Task]:
        """Sample a batch of tasks."""
        return [self.task_distribution.sample_task() for _ in range(self.batch_size)]
    
    def sample_task(self) -> Task:
        """Sample a single task."""
        return self.task_distribution.sample_task()


if __name__ == "__main__":
    # Validation test
    # Test synthetic distribution
    dist = SyntheticTaskDistribution(
        task_type='classification',
        input_dim=8,
        num_classes=3,
        task_complexity='medium'
    )
    
    task = dist.sample_task(num_samples=100)
    assert task.support_data['states'].shape == (50, 8), "Wrong support data shape"
    assert task.query_data['states'].shape == (50, 8), "Wrong query data shape"
    assert task.metadata['task_type'] == 'classification', "Wrong task type"
    
    # Test module distribution
    module_configs = [
        {'name': f'module_{i}', 'capacity': 100} for i in range(5)
    ]
    
    module_dist = ModuleTaskDistribution(module_configs)
    module_task = module_dist.sample_task()
    
    assert 'states' in module_task.support_data, "Missing states in support data"
    assert 'actions' in module_task.support_data, "Missing actions in support data"
    assert module_task.metadata['num_modules'] == 5, "Wrong number of modules"
    
    # Test task sampler
    sampler = TaskSampler(dist, batch_size=3)
    batch = sampler.sample_batch()
    
    assert len(batch) == 3, f"Wrong batch size: {len(batch)}"
    assert all(isinstance(task, Task) for task in batch), "Invalid task type in batch"
    
    print(" Task distribution validation passed")