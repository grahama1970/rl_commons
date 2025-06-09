"""
Module: adaptive.py
Purpose: Adaptive curriculum that adjusts to individual learning patterns

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- scikit-learn: https://scikit-learn.org/stable/

Example Usage:
>>> from rl_commons.algorithms.curriculum import AdaptiveCurriculum
>>> curriculum = AdaptiveCurriculum(tasks, learning_style="balanced")
>>> curriculum.adapt_to_learner(performance_history)
>>> next_task = curriculum.select_next_task()
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Deque
import logging
from collections import defaultdict, deque
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .base import (
    CurriculumScheduler, Task, TaskDifficulty,
    PerformanceTracker
)

logger = logging.getLogger(__name__)


class LearnerProfile:
    """Profile of learner characteristics"""
    
    def __init__(self):
        self.learning_rate: float = 0.5
        self.exploration_preference: float = 0.5
        self.difficulty_tolerance: float = 0.5
        self.consistency: float = 0.5
        self.peak_performance_time: Optional[int] = None
        self.preferred_task_types: Dict[str, float] = {}
        self.strength_areas: Dict[str, float] = {}
        self.weakness_areas: Dict[str, float] = {}
        
        # Learning patterns
        self.performance_by_time: Deque[Tuple[int, float]] = deque(maxlen=100)
        self.performance_by_difficulty: Dict[float, List[float]] = defaultdict(list)
        self.task_type_performance: Dict[str, List[float]] = defaultdict(list)
        
    def update_from_performance(
        self, 
        task: Task, 
        performance: float,
        time_step: int
    ) -> None:
        """Update profile based on task performance"""
        # Track performance patterns
        self.performance_by_time.append((time_step, performance))
        self.performance_by_difficulty[task.difficulty].append(performance)
        
        # Track task type performance
        task_type = task.context.get("type", "default")
        self.task_type_performance[task_type].append(performance)
        
        # Update preferences
        self._update_preferences()
        
    def _update_preferences(self) -> None:
        """Update learner preferences based on history"""
        # Calculate learning rate from improvement
        if len(self.performance_by_time) >= 10:
            times, perfs = zip(*list(self.performance_by_time)[-20:])
            if np.std(times) > 0:
                slope = np.polyfit(times, perfs, 1)[0]
                self.learning_rate = np.clip(0.5 + slope * 10, 0.1, 0.9)
        
        # Calculate consistency
        if len(self.performance_by_time) >= 5:
            _, recent_perfs = zip(*list(self.performance_by_time)[-10:])
            self.consistency = 1.0 / (1.0 + np.std(recent_perfs))
        
        # Update task type preferences
        for task_type, perfs in self.task_type_performance.items():
            if len(perfs) >= 3:
                avg_perf = np.mean(perfs[-10:])
                self.preferred_task_types[task_type] = avg_perf
        
        # Identify strengths and weaknesses by difficulty
        for diff, perfs in self.performance_by_difficulty.items():
            if len(perfs) >= 3:
                avg_perf = np.mean(perfs[-5:])
                if avg_perf > 0.7:
                    self.strength_areas[diff] = avg_perf
                elif avg_perf < 0.4:
                    self.weakness_areas[diff] = avg_perf


class AdaptiveCurriculum(CurriculumScheduler):
    """Curriculum that adapts to individual learning patterns"""
    
    def __init__(
        self,
        tasks: List[Task],
        learning_style: str = "balanced",
        adaptation_rate: float = 0.1,
        use_gp_model: bool = True
    ):
        """Initialize adaptive curriculum
        
        Args:
            tasks: List of available tasks
            learning_style: Initial learning style (balanced, exploration, mastery)
            adaptation_rate: Rate of adaptation to learner
            use_gp_model: Whether to use Gaussian Process for prediction
        """
        super().__init__(tasks)
        
        self.learning_style = learning_style
        self.adaptation_rate = adaptation_rate
        self.use_gp_model = use_gp_model
        
        # Learner modeling
        self.learner_profile = LearnerProfile()
        self.time_step = 0
        
        # Performance prediction model
        if use_gp_model:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True
            )
            self.scaler = StandardScaler()
            self.training_data = []
        
        # Adaptive parameters
        self.difficulty_adjustment = 0.0
        self.exploration_bonus = 0.1
        self.mastery_threshold = 0.8
        
        # Style-specific parameters
        self._initialize_style_parameters()
        
    def _initialize_style_parameters(self) -> None:
        """Initialize parameters based on learning style"""
        if self.learning_style == "exploration":
            self.exploration_bonus = 0.3
            self.mastery_threshold = 0.7
        elif self.learning_style == "mastery":
            self.exploration_bonus = 0.05
            self.mastery_threshold = 0.9
        else:  # balanced
            self.exploration_bonus = 0.15
            self.mastery_threshold = 0.8
    
    def _extract_features(self, task: Task) -> np.ndarray:
        """Extract features for performance prediction"""
        features = [
            task.difficulty,
            self.learner_profile.learning_rate,
            self.learner_profile.consistency,
            len(self.learner_profile.performance_by_time) / 100,  # Experience
            self.time_step / 1000,  # Normalized time
        ]
        
        # Add task type features
        task_type = task.context.get("type", "default")
        type_perf = self.learner_profile.preferred_task_types.get(task_type, 0.5)
        features.append(type_perf)
        
        return np.array(features)
    
    def _predict_performance(self, task: Task) -> Tuple[float, float]:
        """Predict expected performance on a task
        
        Returns:
            (mean_performance, uncertainty)
        """
        if not self.use_gp_model or len(self.training_data) < 5:
            # Fallback to simple prediction
            base_perf = 1.0 - task.difficulty
            adjustment = self.learner_profile.learning_rate * 0.2
            return base_perf + adjustment, 0.3
        
        # Use GP model
        features = self._extract_features(task).reshape(1, -1)
        
        # Check if scaler is fitted
        try:
            features_scaled = self.scaler.transform(features)
            mean, std = self.gp_model.predict(features_scaled, return_std=True)
            return float(mean[0]), float(std[0])
        except Exception:
            # Fallback if model not fitted
            base_perf = 1.0 - task.difficulty
            adjustment = self.learner_profile.learning_rate * 0.2
            return base_perf + adjustment, 0.3
    
    def update_performance(self, task_id: str, reward: float, success: bool = False) -> None:
        """Update performance and adapt curriculum"""
        # Base update
        self.performance_tracker.record_attempt(task_id, reward, success)
        
        # Update learner profile
        if task_id in self.task_dict:
            task = self.task_dict[task_id]
            self.learner_profile.update_from_performance(
                task, reward, self.time_step
            )
            
            # Update GP model training data
            if self.use_gp_model:
                features = self._extract_features(task)
                self.training_data.append((features, reward))
                
                # Retrain GP model periodically
                if len(self.training_data) >= 10 and len(self.training_data) % 5 == 0:
                    self._retrain_gp_model()
        
        self.time_step += 1
        
        # Adapt curriculum parameters
        self._adapt_curriculum()
    
    def _retrain_gp_model(self) -> None:
        """Retrain the Gaussian Process model"""
        if len(self.training_data) < 5:
            return
        
        # Prepare training data
        X, y = zip(*self.training_data[-100:])  # Use last 100 samples
        X = np.array(X)
        y = np.array(y)
        
        # Fit scaler and model
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        try:
            self.gp_model.fit(X_scaled, y)
            logger.info("GP model retrained successfully")
        except Exception as e:
            logger.warning(f"GP model training failed: {e}")
    
    def _adapt_curriculum(self) -> None:
        """Adapt curriculum parameters to learner"""
        profile = self.learner_profile
        
        # Adjust difficulty based on consistency
        if profile.consistency > 0.8:
            # High consistency, can handle larger jumps
            self.difficulty_adjustment = 0.05
        elif profile.consistency < 0.3:
            # Low consistency, smaller steps
            self.difficulty_adjustment = -0.05
        else:
            self.difficulty_adjustment = 0.0
        
        # Adjust exploration based on learning rate
        if profile.learning_rate > 0.7:
            # Fast learner, increase exploration
            self.exploration_bonus = min(0.5, self.exploration_bonus + 0.02)
        elif profile.learning_rate < 0.3:
            # Slow learner, reduce exploration
            self.exploration_bonus = max(0.05, self.exploration_bonus - 0.02)
        
        # Adjust mastery threshold based on performance patterns
        global_perf = self.performance_tracker.get_global_performance()
        if global_perf > 0.85:
            self.mastery_threshold = min(0.95, self.mastery_threshold + 0.01)
        elif global_perf < 0.5:
            self.mastery_threshold = max(0.6, self.mastery_threshold - 0.01)
    
    def select_next_task(self, performance: Optional[float] = None) -> Optional[Task]:
        """Select next task adapted to learner"""
        if performance is not None:
            self.performance_tracker.global_performance.append(performance)
        
        available_tasks = self.get_available_tasks()
        if not available_tasks:
            return None
        
        # Score each task
        task_scores = []
        for task in available_tasks:
            score = self._score_task(task)
            task_scores.append((task, score))
        
        # Sort by score
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select based on exploration
        if np.random.random() < self.exploration_bonus:
            # Explore: sample from top tasks
            weights = np.array([score for _, score in task_scores[:5]])
            weights = weights / weights.sum()
            idx = np.random.choice(len(weights), p=weights)
            selected_task = task_scores[idx][0]
        else:
            # Exploit: select best task
            selected_task = task_scores[0][0]
        
        logger.info(
            f"Selected task {selected_task.task_id} "
            f"(difficulty: {selected_task.difficulty:.2f}, "
            f"score: {task_scores[0][1]:.3f})"
        )
        
        return selected_task
    
    def _score_task(self, task: Task) -> float:
        """Score a task for the current learner"""
        # Predict performance
        expected_perf, uncertainty = self._predict_performance(task)
        
        # Base score on expected performance
        score = expected_perf
        
        # Adjust for optimal difficulty
        profile = self.learner_profile
        optimal_difficulty = 0.5 + profile.learning_rate * 0.3
        difficulty_match = 1.0 - abs(task.difficulty - optimal_difficulty)
        score *= (0.7 + 0.3 * difficulty_match)
        
        # Bonus for preferred task types
        task_type = task.context.get("type", "default")
        if task_type in profile.preferred_task_types:
            type_preference = profile.preferred_task_types[task_type]
            score *= (0.8 + 0.2 * type_preference)
        
        # Exploration bonus for high uncertainty
        if uncertainty > 0.3:
            score += self.exploration_bonus * uncertainty
        
        # Penalty for weakness areas
        if task.difficulty in profile.weakness_areas:
            # But don't avoid completely - opportunity to improve
            score *= 0.7
        
        # Bonus for strength areas if consolidating
        if self.learning_style == "mastery" and task.difficulty in profile.strength_areas:
            score *= 1.2
        
        return score
    
    def get_learner_insights(self) -> Dict[str, Any]:
        """Get insights about the learner"""
        profile = self.learner_profile
        
        insights = {
            "learning_rate": profile.learning_rate,
            "consistency": profile.consistency,
            "exploration_preference": self.exploration_bonus,
            "preferred_task_types": dict(profile.preferred_task_types),
            "strength_difficulties": list(profile.strength_areas.keys()),
            "weakness_difficulties": list(profile.weakness_areas.keys()),
            "current_adaptation": {
                "difficulty_adjustment": self.difficulty_adjustment,
                "mastery_threshold": self.mastery_threshold,
                "exploration_bonus": self.exploration_bonus
            }
        }
        
        # Add performance predictions if available
        if self.use_gp_model and len(self.training_data) >= 10:
            # Predict on a range of difficulties
            predictions = {}
            for diff in np.linspace(0.1, 0.9, 9):
                dummy_task = Task(f"pred_{diff}", difficulty=diff)
                mean, std = self._predict_performance(dummy_task)
                predictions[f"{diff:.1f}"] = {"mean": mean, "std": std}
            insights["performance_predictions"] = predictions
        
        return insights


if __name__ == "__main__":
    # Validation test
    # Create varied tasks
    tasks = []
    task_types = ["navigation", "control", "planning"]
    
    for i in range(30):
        difficulty = 0.1 + (i % 10) * 0.08
        task_type = task_types[i % 3]
        
        task = Task(
            f"adaptive_task_{i}",
            difficulty=difficulty,
            context={"type": task_type, "variant": i // 10}
        )
        tasks.append(task)
    
    # Create adaptive curriculum
    curriculum = AdaptiveCurriculum(
        tasks,
        learning_style="balanced",
        use_gp_model=True
    )
    
    # Simulate learning with individual patterns
    # Simulate a learner who is good at navigation, struggles with control
    for episode in range(40):
        task = curriculum.select_next_task()
        
        if task is None:
            break
        
        # Performance based on task type and learner characteristics
        base_perf = 0.6
        if task.context.get("type") == "navigation":
            base_perf = 0.8  # Good at navigation
        elif task.context.get("type") == "control":
            base_perf = 0.4  # Struggles with control
        
        # Add learning curve
        learning_bonus = min(0.3, episode * 0.01)
        
        # Calculate performance
        performance = base_perf + learning_bonus - task.difficulty * 0.3
        performance += np.random.normal(0, 0.1)
        performance = np.clip(performance, 0, 1)
        
        success = performance > curriculum.mastery_threshold
        
        # Update curriculum
        curriculum.update_performance(task.task_id, performance, success)
        
        if success:
            curriculum.mark_task_completed(task.task_id)
    
    # Check adaptation
    insights = curriculum.get_learner_insights()
    
    assert "learning_rate" in insights, "Missing learning rate"
    assert "preferred_task_types" in insights, "Missing task preferences"
    assert len(insights["preferred_task_types"]) > 0, "No task preferences learned"
    
    print(" Adaptive curriculum validation passed")
    print(f"   Learning rate: {insights['learning_rate']:.2f}")
    print(f"   Consistency: {insights['consistency']:.2f}")
    print(f"   Task preferences: {insights['preferred_task_types']}")
    print(f"   Strengths: {insights['strength_difficulties']}")
    print(f"   Weaknesses: {insights['weakness_difficulties']}")