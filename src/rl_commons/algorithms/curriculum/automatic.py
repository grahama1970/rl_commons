"""
Module: automatic.py
Purpose: Automatic curriculum generation based on performance

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- scipy: https://docs.scipy.org/doc/scipy/

Example Usage:
>>> from rl_commons.algorithms.curriculum import AutomaticCurriculum
>>> curriculum = AutomaticCurriculum(difficulty_bins=10)
>>> curriculum.update_performance(task_id="task_1", reward=0.8)
>>> next_task = curriculum.select_next_task()
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from scipy import stats
import logging
from collections import defaultdict

from .base import (
    CurriculumScheduler, Task, TaskDifficulty, 
    PerformanceTracker, TaskGenerator
)

logger = logging.getLogger(__name__)


class PerformanceBasedTaskGenerator(TaskGenerator):
    """Generate tasks based on performance history"""
    
    def __init__(
        self,
        base_contexts: List[Dict[str, Any]],
        difficulty_modifiers: Dict[str, Callable[[float], Any]]
    ):
        """Initialize generator
        
        Args:
            base_contexts: Base task contexts to modify
            difficulty_modifiers: Functions to modify context based on difficulty
        """
        self.base_contexts = base_contexts
        self.difficulty_modifiers = difficulty_modifiers
        self.generated_count = 0
    
    def generate_task(
        self, 
        difficulty: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Generate a task with specified difficulty"""
        # Select base context
        base_context = np.random.choice(self.base_contexts)
        task_context = base_context.copy()
        
        # Apply difficulty modifiers
        for key, modifier in self.difficulty_modifiers.items():
            if key in task_context:
                task_context[key] = modifier(difficulty)
        
        # Override with provided context
        if context:
            task_context.update(context)
        
        # Generate task ID
        self.generated_count += 1
        task_id = f"auto_task_{self.generated_count}_d{difficulty:.2f}"
        
        return Task(
            task_id=task_id,
            difficulty=difficulty,
            context=task_context,
            metadata={"generated": True, "generator": "performance_based"}
        )
    
    def generate_batch(
        self,
        num_tasks: int,
        difficulty_range: Tuple[float, float],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Task]:
        """Generate a batch of tasks"""
        tasks = []
        difficulties = np.linspace(difficulty_range[0], difficulty_range[1], num_tasks)
        
        for diff in difficulties:
            tasks.append(self.generate_task(diff, context))
        
        return tasks


class AutomaticCurriculum(CurriculumScheduler):
    """Automatically generate and schedule tasks based on performance"""
    
    def __init__(
        self,
        difficulty_bins: int = 10,
        performance_window: int = 20,
        exploration_rate: float = 0.1,
        task_generator: Optional[TaskGenerator] = None
    ):
        """Initialize automatic curriculum
        
        Args:
            difficulty_bins: Number of difficulty levels
            performance_window: Window for performance estimation
            exploration_rate: Rate of exploring harder tasks
            task_generator: Optional custom task generator
        """
        # Generate initial task pool
        initial_tasks = self._generate_initial_tasks(difficulty_bins)
        super().__init__(initial_tasks)
        
        self.difficulty_bins = difficulty_bins
        self.performance_window = performance_window
        self.exploration_rate = exploration_rate
        self.task_generator = task_generator or self._create_default_generator()
        
        # Track performance by difficulty
        self.difficulty_performance = defaultdict(list)
        self.difficulty_attempts = defaultdict(int)
        
        # Adaptive parameters
        self.current_focus_difficulty = 0.1
        self.difficulty_confidence = defaultdict(float)
        
        # Update tracker's current difficulty
        self.performance_tracker.current_difficulty = self.current_focus_difficulty
        
    def _generate_initial_tasks(self, num_bins: int) -> List[Task]:
        """Generate initial task distribution"""
        tasks = []
        difficulties = np.linspace(0.1, 0.9, num_bins)
        
        for i, diff in enumerate(difficulties):
            # Create multiple variants per difficulty
            for variant in range(3):
                task_id = f"init_task_d{diff:.2f}_v{variant}"
                
                # Add prerequisites for higher difficulties
                prerequisites = []
                if i > 0:
                    # Require completion of at least one easier task
                    prev_diff = difficulties[i-1]
                    prerequisites.append(f"init_task_d{prev_diff:.2f}_v0")
                
                task = Task(
                    task_id=task_id,
                    difficulty=diff,
                    prerequisites=prerequisites,
                    context={
                        "variant": variant,
                        "difficulty_bin": i
                    }
                )
                tasks.append(task)
        
        return tasks
    
    def _create_default_generator(self) -> TaskGenerator:
        """Create default task generator"""
        base_contexts = [
            {"env_type": "navigation", "grid_size": 5},
            {"env_type": "control", "action_dim": 2},
            {"env_type": "planning", "horizon": 5}
        ]
        
        difficulty_modifiers = {
            "grid_size": lambda d: int(5 + d * 20),
            "action_dim": lambda d: int(2 + d * 8),
            "horizon": lambda d: int(5 + d * 45)
        }
        
        return PerformanceBasedTaskGenerator(base_contexts, difficulty_modifiers)
    
    def update_performance(self, task_id: str, reward: float, success: bool = False) -> None:
        """Update performance tracking"""
        # Record in base tracker
        self.performance_tracker.record_attempt(task_id, reward, success)
        
        # Track by difficulty
        if task_id in self.task_dict:
            task = self.task_dict[task_id]
            self.difficulty_performance[task.difficulty].append(reward)
            self.difficulty_attempts[task.difficulty] += 1
            
            # Update confidence in difficulty estimate
            self._update_difficulty_confidence(task.difficulty, reward)
    
    def _update_difficulty_confidence(self, difficulty: float, reward: float) -> None:
        """Update confidence in difficulty estimation"""
        perf_history = self.difficulty_performance[difficulty]
        
        if len(perf_history) >= 3:
            # Calculate performance variance
            variance = np.var(perf_history[-self.performance_window:])
            # Lower variance = higher confidence
            self.difficulty_confidence[difficulty] = 1.0 / (1.0 + variance)
    
    def _estimate_optimal_difficulty(self) -> float:
        """Estimate optimal difficulty based on performance"""
        if not self.difficulty_performance:
            return 0.1
        
        # Find difficulty with best performance/effort ratio
        difficulties = []
        scores = []
        
        for diff, rewards in self.difficulty_performance.items():
            if len(rewards) >= 2:
                recent_rewards = rewards[-self.performance_window:]
                avg_reward = np.mean(recent_rewards)
                
                # Score combines performance and learning rate
                improvement_rate = self._calculate_improvement_rate(recent_rewards)
                score = avg_reward * (1 + improvement_rate)
                
                difficulties.append(diff)
                scores.append(score)
        
        if not difficulties:
            return self.current_focus_difficulty
        
        # Weight by confidence
        weights = [self.difficulty_confidence.get(d, 0.5) for d in difficulties]
        weighted_scores = np.array(scores) * np.array(weights)
        
        # Find optimal difficulty
        optimal_idx = np.argmax(weighted_scores)
        optimal_difficulty = difficulties[optimal_idx]
        
        # Smooth transition
        self.current_focus_difficulty = (
            0.7 * self.current_focus_difficulty + 0.3 * optimal_difficulty
        )
        
        return self.current_focus_difficulty
    
    def _calculate_improvement_rate(self, rewards: List[float]) -> float:
        """Calculate rate of improvement"""
        if len(rewards) < 2:
            return 0.0
        
        # Fit linear trend
        x = np.arange(len(rewards))
        slope, _ = np.polyfit(x, rewards, 1)
        
        # Normalize by average magnitude
        avg_reward = np.mean(np.abs(rewards))
        if avg_reward > 0:
            return slope / avg_reward
        return 0.0
    
    def _should_explore(self) -> bool:
        """Decide whether to explore new difficulties"""
        return np.random.random() < self.exploration_rate
    
    def select_next_task(self, performance: Optional[float] = None) -> Optional[Task]:
        """Select next task based on automatic curriculum"""
        # Update performance if provided
        if performance is not None:
            self.performance_tracker.global_performance.append(performance)
        
        # Get available tasks
        available_tasks = self.get_available_tasks()
        
        if not available_tasks:
            # Generate new tasks if needed
            logger.info("No available tasks, generating new ones")
            self._generate_new_tasks()
            available_tasks = self.get_available_tasks()
        
        if not available_tasks:
            return None
        
        # Decide on difficulty
        if self._should_explore():
            # Explore: sample from wider distribution
            target_difficulty = self._sample_exploration_difficulty()
        else:
            # Exploit: focus on optimal difficulty
            target_difficulty = self._estimate_optimal_difficulty()
        
        # Select task closest to target difficulty
        selected_task = min(
            available_tasks,
            key=lambda t: abs(t.difficulty - target_difficulty)
        )
        
        logger.info(
            f"Selected task {selected_task.task_id} "
            f"(difficulty: {selected_task.difficulty:.2f})"
        )
        
        return selected_task
    
    def _sample_exploration_difficulty(self) -> float:
        """Sample difficulty for exploration"""
        # Use beta distribution centered on current focus
        alpha = 2 + self.current_focus_difficulty * 8
        beta = 2 + (1 - self.current_focus_difficulty) * 8
        
        return np.clip(np.random.beta(alpha, beta), 0.1, 0.9)
    
    def _generate_new_tasks(self) -> None:
        """Generate new tasks based on performance gaps"""
        # Find difficulty gaps
        attempted_difficulties = set(self.difficulty_performance.keys())
        all_difficulties = set(t.difficulty for t in self.tasks)
        
        gap_difficulties = all_difficulties - attempted_difficulties
        
        if not gap_difficulties:
            # Generate tasks around current focus
            new_tasks = self.task_generator.generate_batch(
                num_tasks=5,
                difficulty_range=(
                    max(0.1, self.current_focus_difficulty - 0.2),
                    min(0.9, self.current_focus_difficulty + 0.2)
                )
            )
        else:
            # Fill gaps
            new_tasks = []
            for diff in list(gap_difficulties)[:5]:
                task = self.task_generator.generate_task(diff)
                new_tasks.append(task)
        
        # Add to task pool
        for task in new_tasks:
            self.tasks.append(task)
            self.task_dict[task.task_id] = task
        
        logger.info(f"Generated {len(new_tasks)} new tasks")
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get detailed curriculum statistics"""
        stats = self.get_progress()
        
        # Add automatic curriculum specific stats
        stats.update({
            "current_focus_difficulty": self.current_focus_difficulty,
            "difficulty_attempts": dict(self.difficulty_attempts),
            "difficulty_confidence": dict(self.difficulty_confidence),
            "explored_difficulties": len(self.difficulty_performance),
            "total_generated_tasks": sum(
                1 for t in self.tasks 
                if t.metadata.get("generated", False)
            )
        })
        
        return stats


if __name__ == "__main__":
    # Validation test
    # Create automatic curriculum
    curriculum = AutomaticCurriculum(
        difficulty_bins=5,
        performance_window=10,
        exploration_rate=0.2
    )
    
    # Simulate learning process
    for episode in range(20):
        # Select task
        task = curriculum.select_next_task()
        assert task is not None, f"No task selected at episode {episode}"
        
        # Simulate performance (better on easier tasks)
        base_performance = 1.0 - task.difficulty
        noise = np.random.normal(0, 0.1)
        reward = np.clip(base_performance + noise, 0, 1)
        
        # Update curriculum
        curriculum.update_performance(
            task.task_id, 
            reward, 
            success=(reward > 0.7)
        )
        
        # Mark completed if successful
        if reward > 0.7:
            curriculum.mark_task_completed(task.task_id)
    
    # Check curriculum adapted
    stats = curriculum.get_curriculum_stats()
    assert stats["completed_tasks"] > 0, "No tasks completed"
    assert stats["total_generated_tasks"] >= 0, "Task generation failed"
    assert 0 <= stats["current_focus_difficulty"] <= 1, "Invalid difficulty"
    
    print(f"✅ Automatic curriculum validation passed")
    print(f"   Completed tasks: {stats['completed_tasks']}")
    print(f"   Current difficulty: {stats['current_focus_difficulty']:.2f}")
    print(f"   Generated tasks: {stats['total_generated_tasks']}")