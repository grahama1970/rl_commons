"""
Module: progressive.py
Purpose: Progressive curriculum learning with smooth difficulty transitions

External Dependencies:
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.curriculum import ProgressiveCurriculum
>>> tasks = [Task(f"task_{i}", difficulty=i/10) for i in range(1, 11)]
>>> curriculum = ProgressiveCurriculum(tasks, progression_rate=0.1)
>>> next_task = curriculum.select_next_task(performance=0.85)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import deque

from .base import (
    CurriculumScheduler, Task, TaskDifficulty,
    PerformanceTracker
)

logger = logging.getLogger(__name__)


class ProgressiveCurriculum(CurriculumScheduler):
    """Progressive curriculum with smooth difficulty transitions"""
    
    def __init__(
        self,
        tasks: List[Task],
        progression_rate: float = 0.05,
        regression_rate: float = 0.1,
        performance_threshold: float = 0.7,
        min_attempts: int = 5,
        smoothing_window: int = 10
    ):
        """Initialize progressive curriculum
        
        Args:
            tasks: List of tasks ordered by difficulty
            progression_rate: Rate of difficulty increase
            regression_rate: Rate of difficulty decrease on failure
            performance_threshold: Performance needed to progress
            min_attempts: Minimum attempts before progressing
            smoothing_window: Window for performance smoothing
        """
        super().__init__(tasks, performance_threshold, min_attempts)
        
        self.progression_rate = progression_rate
        self.regression_rate = regression_rate
        self.smoothing_window = smoothing_window
        
        # Progressive tracking
        self.current_difficulty = 0.1
        self.difficulty_history = deque(maxlen=100)
        self.performance_by_difficulty = {}
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        
        # Zone of proximal development
        self.zpd_lower = 0.05
        self.zpd_upper = 0.15
        
    def _calculate_zpd(self) -> Tuple[float, float]:
        """Calculate zone of proximal development"""
        # Base ZPD on current performance
        global_perf = self.performance_tracker.get_global_performance()
        
        # Adjust ZPD based on performance
        if global_perf > 0.8:
            # Performing well, widen ZPD upward
            zpd_lower = self.current_difficulty
            zpd_upper = self.current_difficulty + 0.2
        elif global_perf < 0.4:
            # Struggling, narrow ZPD
            zpd_lower = max(0.1, self.current_difficulty - 0.1)
            zpd_upper = self.current_difficulty + 0.05
        else:
            # Normal ZPD
            zpd_lower = self.current_difficulty - 0.05
            zpd_upper = self.current_difficulty + 0.15
        
        return (
            np.clip(zpd_lower, 0.1, 0.9),
            np.clip(zpd_upper, 0.2, 1.0)
        )
    
    def _smooth_difficulty_transition(self, target_difficulty: float) -> float:
        """Smooth difficulty transitions"""
        # Apply exponential smoothing
        alpha = 0.3  # Smoothing factor
        smoothed = (
            alpha * target_difficulty + 
            (1 - alpha) * self.current_difficulty
        )
        
        # Limit rate of change
        max_change = 0.1
        change = smoothed - self.current_difficulty
        if abs(change) > max_change:
            smoothed = self.current_difficulty + np.sign(change) * max_change
        
        return np.clip(smoothed, 0.1, 0.9)
    
    def update_performance(self, task_id: str, reward: float, success: bool = False) -> None:
        """Update performance and adjust difficulty"""
        # Record performance
        self.performance_tracker.record_attempt(task_id, reward, success)
        
        # Track consecutive results
        if success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
        
        # Get task difficulty
        if task_id in self.task_dict:
            task_difficulty = self.task_dict[task_id].difficulty
            
            # Record performance by difficulty
            if task_difficulty not in self.performance_by_difficulty:
                self.performance_by_difficulty[task_difficulty] = []
            self.performance_by_difficulty[task_difficulty].append(reward)
        
        # Update difficulty based on performance
        self._update_difficulty(reward, success)
    
    def _update_difficulty(self, reward: float, success: bool) -> None:
        """Update current difficulty based on performance"""
        # Calculate performance trend
        recent_performance = list(self.performance_tracker.global_performance)[-self.smoothing_window:]
        
        if len(recent_performance) >= 3:
            avg_performance = np.mean(recent_performance)
            performance_trend = np.polyfit(
                range(len(recent_performance)), 
                recent_performance, 
                1
            )[0]
            
            # Adjust difficulty based on performance and trend
            if avg_performance > self.performance_threshold and performance_trend >= 0:
                # Progress: increase difficulty
                target_difficulty = self.current_difficulty + self.progression_rate
            elif avg_performance < 0.5 or performance_trend < -0.05:
                # Regress: decrease difficulty
                target_difficulty = self.current_difficulty - self.regression_rate
            else:
                # Maintain current difficulty
                target_difficulty = self.current_difficulty
            
            # Apply consecutive success/failure bonuses
            if self.consecutive_successes >= 3:
                target_difficulty += 0.02 * self.consecutive_successes
            elif self.consecutive_failures >= 3:
                target_difficulty -= 0.02 * self.consecutive_failures
            
            # Smooth transition
            self.current_difficulty = self._smooth_difficulty_transition(target_difficulty)
            self.difficulty_history.append(self.current_difficulty)
    
    def select_next_task(self, performance: Optional[float] = None) -> Optional[Task]:
        """Select next task with progressive difficulty"""
        # Update global performance if provided
        if performance is not None:
            self.performance_tracker.global_performance.append(performance)
        
        # Calculate ZPD
        zpd_lower, zpd_upper = self._calculate_zpd()
        
        # Get tasks within ZPD
        available_tasks = self.get_available_tasks()
        zpd_tasks = [
            task for task in available_tasks
            if zpd_lower <= task.difficulty <= zpd_upper
        ]
        
        if not zpd_tasks:
            # No tasks in ZPD, find closest
            if available_tasks:
                zpd_tasks = [min(
                    available_tasks,
                    key=lambda t: abs(t.difficulty - self.current_difficulty)
                )]
            else:
                return None
        
        # Select task based on exploration vs exploitation
        if np.random.random() < 0.2:  # 20% exploration
            # Explore: random task from ZPD
            selected_task = np.random.choice(zpd_tasks)
        else:
            # Exploit: task closest to current difficulty
            selected_task = min(
                zpd_tasks,
                key=lambda t: abs(t.difficulty - self.current_difficulty)
            )
        
        logger.info(
            f"Selected task {selected_task.task_id} "
            f"(difficulty: {selected_task.difficulty:.2f}, "
            f"current: {self.current_difficulty:.2f}, "
            f"ZPD: [{zpd_lower:.2f}, {zpd_upper:.2f}])"
        )
        
        return selected_task
    
    def get_progression_curve(self) -> Dict[str, Any]:
        """Get difficulty progression statistics"""
        if not self.difficulty_history:
            return {
                "current_difficulty": self.current_difficulty,
                "progression_rate": 0.0,
                "stability": 0.0
            }
        
        difficulties = list(self.difficulty_history)
        
        # Calculate progression rate
        if len(difficulties) >= 2:
            progression_rate = np.polyfit(
                range(len(difficulties)), 
                difficulties, 
                1
            )[0]
        else:
            progression_rate = 0.0
        
        # Calculate stability (inverse of variance)
        if len(difficulties) >= 5:
            recent_difficulties = difficulties[-10:]
            stability = 1.0 / (1.0 + np.var(recent_difficulties))
        else:
            stability = 0.5
        
        return {
            "current_difficulty": self.current_difficulty,
            "progression_rate": progression_rate,
            "stability": stability,
            "difficulty_history": difficulties[-20:],  # Last 20 values
            "zpd": self._calculate_zpd()
        }
    
    def get_mastery_levels(self) -> Dict[float, float]:
        """Get mastery level for each difficulty"""
        mastery = {}
        
        for difficulty, performances in self.performance_by_difficulty.items():
            if len(performances) >= self.min_attempts:
                # Calculate mastery as average of recent performances
                recent_perfs = performances[-self.smoothing_window:]
                mastery[difficulty] = np.mean(recent_perfs)
        
        return mastery
    
    def recommend_review(self) -> List[Task]:
        """Recommend tasks for review based on forgetting curve"""
        review_tasks = []
        mastery_levels = self.get_mastery_levels()
        
        for task in self.completed_tasks:
            if task not in self.task_dict:
                continue
                
            task_obj = self.task_dict[task]
            difficulty = task_obj.difficulty
            
            # Check if performance has degraded
            if difficulty in mastery_levels:
                current_mastery = mastery_levels[difficulty]
                
                # Recommend review if mastery dropped below threshold
                if current_mastery < self.performance_threshold * 0.9:
                    review_tasks.append(task_obj)
        
        return review_tasks


if __name__ == "__main__":
    # Validation test
    # Create progressive curriculum
    tasks = []
    for i in range(20):
        difficulty = 0.05 + (i * 0.045)  # 0.05 to 0.95
        task = Task(
            f"prog_task_{i}",
            difficulty=difficulty,
            context={"level": i}
        )
        if i > 0:
            task.prerequisites = [f"prog_task_{i-1}"]
        tasks.append(task)
    
    curriculum = ProgressiveCurriculum(
        tasks,
        progression_rate=0.05,
        regression_rate=0.1
    )
    
    # Simulate progressive learning
    for episode in range(30):
        task = curriculum.select_next_task()
        
        if task is None:
            break
        
        # Simulate performance (with learning curve)
        skill_level = 0.3 + (episode * 0.02)  # Gradually improving
        performance = skill_level * (1.0 - task.difficulty * 0.5)
        performance += np.random.normal(0, 0.1)
        performance = np.clip(performance, 0, 1)
        
        success = performance > curriculum.performance_threshold
        
        # Update curriculum
        curriculum.update_performance(task.task_id, performance, success)
        
        if success:
            curriculum.mark_task_completed(task.task_id)
    
    # Check progression
    progression = curriculum.get_progression_curve()
    assert "current_difficulty" in progression, "Missing difficulty info"
    assert 0.1 <= progression["current_difficulty"] <= 0.9, "Difficulty out of bounds"
    
    # Check mastery tracking
    mastery = curriculum.get_mastery_levels()
    assert len(mastery) > 0, "No mastery levels recorded"
    
    print(" Progressive curriculum validation passed")
    print(f"   Final difficulty: {progression['current_difficulty']:.2f}")
    print(f"   Progression rate: {progression['progression_rate']:.4f}")
    print(f"   Stability: {progression['stability']:.2f}")
    print(f"   Mastery levels: {len(mastery)}")