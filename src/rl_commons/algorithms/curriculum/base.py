"""
Module: base.py
Purpose: Base classes for curriculum learning

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- dataclasses: Built-in Python module

Example Usage:
>>> from rl_commons.algorithms.curriculum import Task, CurriculumScheduler
>>> task = Task("easy_task", difficulty=0.2, context={"env": "simple"})
>>> scheduler = CurriculumScheduler()
>>> next_task = scheduler.select_next_task(performance=0.8)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
from enum import Enum
import logging
from datetime import datetime
from collections import deque

from ...core.base import RLState, RLAction, RLReward

logger = logging.getLogger(__name__)


class TaskDifficulty(Enum):
    """Predefined difficulty levels"""
    TRIVIAL = 0.1
    EASY = 0.3
    MEDIUM = 0.5
    HARD = 0.7
    EXPERT = 0.9
    IMPOSSIBLE = 1.0


@dataclass
class Task:
    """Represents a task in the curriculum"""
    task_id: str
    difficulty: float  # 0-1 scale
    context: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate difficulty"""
        if not 0 <= self.difficulty <= 1:
            raise ValueError(f"Difficulty must be in [0, 1], got {self.difficulty}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "context": self.context,
            "prerequisites": self.prerequisites,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TaskPerformance:
    """Track performance on a specific task"""
    task_id: str
    attempts: int = 0
    successes: int = 0
    average_reward: float = 0.0
    best_reward: float = float('-inf')
    worst_reward: float = float('inf')
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=10))
    last_attempt: Optional[datetime] = None
    
    def add_attempt(self, reward: float, success: bool = False) -> None:
        """Record a new attempt"""
        self.attempts += 1
        if success:
            self.successes += 1
        
        self.recent_rewards.append(reward)
        self.average_reward = (
            (self.average_reward * (self.attempts - 1) + reward) / self.attempts
        )
        self.best_reward = max(self.best_reward, reward)
        self.worst_reward = min(self.worst_reward, reward)
        self.last_attempt = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successes / max(1, self.attempts)
    
    @property
    def recent_average(self) -> float:
        """Average of recent attempts"""
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)
    
    @property
    def improvement_rate(self) -> float:
        """Rate of improvement in recent attempts"""
        if len(self.recent_rewards) < 2:
            return 0.0
        
        # Linear regression on recent rewards
        x = np.arange(len(self.recent_rewards))
        y = np.array(self.recent_rewards)
        
        if np.std(y) == 0:
            return 0.0
        
        slope = np.polyfit(x, y, 1)[0]
        return slope / (np.abs(np.mean(y)) + 1e-8)


class PerformanceTracker:
    """Track overall performance across tasks"""
    
    def __init__(self):
        self.task_performances: Dict[str, TaskPerformance] = {}
        self.global_performance = deque(maxlen=100)
        self.current_difficulty = 0.1
        
    def record_attempt(
        self, 
        task_id: str, 
        reward: float, 
        success: bool = False
    ) -> None:
        """Record performance on a task"""
        if task_id not in self.task_performances:
            self.task_performances[task_id] = TaskPerformance(task_id)
        
        self.task_performances[task_id].add_attempt(reward, success)
        self.global_performance.append(reward)
        
        logger.info(f"Task {task_id}: reward={reward:.3f}, success={success}")
    
    def get_task_performance(self, task_id: str) -> Optional[TaskPerformance]:
        """Get performance for a specific task"""
        return self.task_performances.get(task_id)
    
    def get_global_performance(self) -> float:
        """Get overall performance metric"""
        if not self.global_performance:
            return 0.0
        return np.mean(self.global_performance)
    
    def get_readiness_score(self, difficulty: float) -> float:
        """Calculate readiness for a given difficulty level"""
        # Consider recent performance and current difficulty
        recent_performance = self.get_global_performance()
        difficulty_gap = difficulty - self.current_difficulty
        
        # Readiness decreases with larger difficulty gaps
        readiness = recent_performance * np.exp(-2 * max(0, difficulty_gap))
        
        return np.clip(readiness, 0, 1)
    
    def update_current_difficulty(self, new_difficulty: float) -> None:
        """Update current difficulty level"""
        self.current_difficulty = new_difficulty


class CurriculumScheduler(ABC):
    """Base class for curriculum scheduling"""
    
    def __init__(
        self,
        tasks: List[Task],
        performance_threshold: float = 0.7,
        min_attempts: int = 5
    ):
        """Initialize scheduler
        
        Args:
            tasks: List of available tasks
            performance_threshold: Performance needed to progress
            min_attempts: Minimum attempts before progressing
        """
        self.tasks = sorted(tasks, key=lambda t: t.difficulty)
        self.task_dict = {t.task_id: t for t in tasks}
        self.performance_threshold = performance_threshold
        self.min_attempts = min_attempts
        
        self.performance_tracker = PerformanceTracker()
        self.current_task_idx = 0
        self.completed_tasks = set()
        
        logger.info(f"Initialized curriculum with {len(tasks)} tasks")
    
    @abstractmethod
    def select_next_task(self, performance: Optional[float] = None) -> Optional[Task]:
        """Select the next task based on performance
        
        Args:
            performance: Current performance metric
            
        Returns:
            Next task or None if curriculum complete
        """
        pass
    
    def is_task_ready(self, task: Task) -> bool:
        """Check if prerequisites are met for a task"""
        return all(prereq in self.completed_tasks for prereq in task.prerequisites)
    
    def mark_task_completed(self, task_id: str) -> None:
        """Mark a task as completed"""
        self.completed_tasks.add(task_id)
        logger.info(f"Task {task_id} marked as completed")
    
    def get_available_tasks(self) -> List[Task]:
        """Get all tasks with met prerequisites"""
        return [
            task for task in self.tasks 
            if self.is_task_ready(task) and task.task_id not in self.completed_tasks
        ]
    
    def get_progress(self) -> Dict[str, Any]:
        """Get curriculum progress"""
        return {
            "total_tasks": len(self.tasks),
            "completed_tasks": len(self.completed_tasks),
            "completion_rate": len(self.completed_tasks) / len(self.tasks),
            "current_difficulty": self.performance_tracker.current_difficulty,
            "global_performance": self.performance_tracker.get_global_performance()
        }
    
    def reset(self) -> None:
        """Reset curriculum progress"""
        self.current_task_idx = 0
        self.completed_tasks.clear()
        self.performance_tracker = PerformanceTracker()
        logger.info("Curriculum reset")


class TaskGenerator(ABC):
    """Base class for automatic task generation"""
    
    @abstractmethod
    def generate_task(
        self, 
        difficulty: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Generate a task with specified difficulty
        
        Args:
            difficulty: Target difficulty [0, 1]
            context: Optional context for task generation
            
        Returns:
            Generated task
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        num_tasks: int,
        difficulty_range: Tuple[float, float],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Task]:
        """Generate a batch of tasks
        
        Args:
            num_tasks: Number of tasks to generate
            difficulty_range: (min, max) difficulty
            context: Optional context
            
        Returns:
            List of generated tasks
        """
        pass


if __name__ == "__main__":
    # Validation test
    # Create sample tasks
    tasks = [
        Task("task_1", difficulty=0.1, context={"type": "basic"}),
        Task("task_2", difficulty=0.3, prerequisites=["task_1"]),
        Task("task_3", difficulty=0.5, prerequisites=["task_2"]),
        Task("task_4", difficulty=0.7, prerequisites=["task_2", "task_3"]),
        Task("task_5", difficulty=0.9, prerequisites=["task_4"])
    ]
    
    # Test task creation and validation
    assert tasks[0].difficulty == 0.1, "Task difficulty mismatch"
    assert len(tasks[2].prerequisites) == 1, "Prerequisites not set correctly"
    
    # Test performance tracking
    tracker = PerformanceTracker()
    tracker.record_attempt("task_1", reward=0.5, success=False)
    tracker.record_attempt("task_1", reward=0.8, success=True)
    
    perf = tracker.get_task_performance("task_1")
    assert perf.attempts == 2, f"Expected 2 attempts, got {perf.attempts}"
    assert perf.success_rate == 0.5, f"Expected 50% success rate, got {perf.success_rate}"
    assert abs(perf.average_reward - 0.65) < 0.01, "Average reward calculation error"
    
    # Test readiness score
    readiness = tracker.get_readiness_score(0.3)
    assert 0 <= readiness <= 1, f"Readiness score out of bounds: {readiness}"
    
    print("✅ Curriculum base classes validation passed")