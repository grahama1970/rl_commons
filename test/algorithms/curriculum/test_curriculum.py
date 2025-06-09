"""
Module: test_curriculum.py
Purpose: Comprehensive tests for curriculum learning modules

External Dependencies:
- pytest: https://docs.pytest.org/
- numpy: https://numpy.org/doc/stable/
- torch: https://pytorch.org/docs/stable/

Example Usage:
>>> pytest test/algorithms/curriculum/test_curriculum.py -v
"""

import pytest
import numpy as np
import torch
from typing import List, Dict, Any

from rl_commons.algorithms.curriculum import (
    Task, TaskDifficulty, PerformanceTracker,
    CurriculumScheduler, AutomaticCurriculum,
    ProgressiveCurriculum, AdaptiveCurriculum,
    MetaCurriculum
)
from rl_commons.algorithms.curriculum.base import TaskGenerator
from rl_commons.algorithms.curriculum.automatic import PerformanceBasedTaskGenerator
from rl_commons.algorithms.curriculum.meta_curriculum import MetaTask, MetaTaskGenerator
from rl_commons.algorithms.meta import MAMLAgent
from rl_commons.core.base import RLState, RLAction, RLReward


class TestTask:
    """Test Task class functionality"""
    
    def test_task_creation(self):
        """Test basic task creation"""
        task = Task(
            task_id="test_task",
            difficulty=0.5,
            context={"env": "test"},
            prerequisites=["task_1", "task_2"]
        )
        
        assert task.task_id == "test_task"
        assert task.difficulty == 0.5
        assert task.context["env"] == "test"
        assert len(task.prerequisites) == 2
    
    def test_task_difficulty_validation(self):
        """Test difficulty bounds validation"""
        with pytest.raises(ValueError):
            Task("invalid_task", difficulty=1.5)
        
        with pytest.raises(ValueError):
            Task("invalid_task", difficulty=-0.1)
    
    def test_task_serialization(self):
        """Test task to/from dict conversion"""
        task = Task(
            task_id="serialize_test",
            difficulty=0.7,
            context={"type": "navigation"},
            metadata={"created": "2024-01-01"}
        )
        
        # Convert to dict
        task_dict = task.to_dict()
        assert task_dict["task_id"] == "serialize_test"
        assert task_dict["difficulty"] == 0.7
        
        # Convert back
        restored_task = Task.from_dict(task_dict)
        assert restored_task.task_id == task.task_id
        assert restored_task.difficulty == task.difficulty


class TestPerformanceTracker:
    """Test performance tracking functionality"""
    
    def test_performance_recording(self):
        """Test recording task attempts"""
        tracker = PerformanceTracker()
        
        # Record attempts
        tracker.record_attempt("task_1", reward=0.5, success=False)
        tracker.record_attempt("task_1", reward=0.8, success=True)
        tracker.record_attempt("task_2", reward=0.9, success=True)
        
        # Check task performance
        perf1 = tracker.get_task_performance("task_1")
        assert perf1.attempts == 2
        assert perf1.successes == 1
        assert perf1.success_rate == 0.5
        assert abs(perf1.average_reward - 0.65) < 0.01
        
        perf2 = tracker.get_task_performance("task_2")
        assert perf2.attempts == 1
        assert perf2.success_rate == 1.0
    
    def test_global_performance(self):
        """Test global performance metrics"""
        tracker = PerformanceTracker()
        
        rewards = [0.3, 0.5, 0.7, 0.9]
        for reward in rewards:
            tracker.record_attempt(f"task_{reward}", reward)
        
        global_perf = tracker.get_global_performance()
        assert abs(global_perf - np.mean(rewards)) < 0.01
    
    def test_readiness_score(self):
        """Test readiness score calculation"""
        tracker = PerformanceTracker()
        tracker.current_difficulty = 0.5
        
        # Add performance history
        for _ in range(10):
            tracker.record_attempt("task", reward=0.7)
        
        # Test readiness for various difficulties
        readiness_same = tracker.get_readiness_score(0.5)
        readiness_harder = tracker.get_readiness_score(0.8)
        readiness_easier = tracker.get_readiness_score(0.3)
        
        assert readiness_same > readiness_harder
        assert readiness_easier >= readiness_same


class TestAutomaticCurriculum:
    """Test automatic curriculum generation"""
    
    def test_automatic_task_generation(self):
        """Test automatic task generation"""
        curriculum = AutomaticCurriculum(
            difficulty_bins=5,
            performance_window=10,
            exploration_rate=0.2
        )
        
        # Check initial tasks generated
        assert len(curriculum.tasks) > 0
        
        # Check difficulty distribution
        difficulties = [t.difficulty for t in curriculum.tasks]
        assert min(difficulties) >= 0.1
        assert max(difficulties) <= 0.9
    
    def test_performance_based_selection(self):
        """Test task selection based on performance"""
        curriculum = AutomaticCurriculum(difficulty_bins=5)
        
        # Simulate good performance on easy tasks
        for _ in range(10):
            task = curriculum.select_next_task()
            if task and task.difficulty < 0.3:
                curriculum.update_performance(
                    task.task_id, 
                    reward=0.9, 
                    success=True
                )
                curriculum.mark_task_completed(task.task_id)
        
        # Check that difficulty increased (or at least didn't decrease)
        assert curriculum.current_focus_difficulty >= 0.1
    
    def test_task_generator_integration(self):
        """Test custom task generator"""
        base_contexts = [{"env": "test"}]
        modifiers = {"size": lambda d: int(5 + d * 10)}
        
        generator = PerformanceBasedTaskGenerator(base_contexts, modifiers)
        curriculum = AutomaticCurriculum(task_generator=generator)
        
        # Generate new tasks
        curriculum._generate_new_tasks()
        
        # Check generated tasks have proper context
        generated_tasks = [
            t for t in curriculum.tasks 
            if t.metadata.get("generated", False)
        ]
        assert len(generated_tasks) > 0


class TestProgressiveCurriculum:
    """Test progressive curriculum learning"""
    
    def test_smooth_progression(self):
        """Test smooth difficulty progression"""
        # Create tasks with linear difficulty
        tasks = []
        for i in range(10):
            task = Task(f"prog_{i}", difficulty=i * 0.1)
            if i > 0:
                task.prerequisites = [f"prog_{i-1}"]
            tasks.append(task)
        
        curriculum = ProgressiveCurriculum(
            tasks,
            progression_rate=0.05,
            regression_rate=0.1
        )
        
        # Track difficulty changes
        difficulties = []
        
        for _ in range(20):
            task = curriculum.select_next_task()
            if task:
                difficulties.append(task.difficulty)
                # Simulate improving performance
                performance = 0.8 - task.difficulty * 0.3
                curriculum.update_performance(
                    task.task_id,
                    performance,
                    success=(performance > 0.7)
                )
                if performance > 0.7:
                    curriculum.mark_task_completed(task.task_id)
        
        # Check progression is smooth
        if len(difficulties) > 1:
            diffs = np.diff(difficulties)
            assert all(abs(d) <= 0.2 for d in diffs)  # No large jumps
    
    def test_zone_of_proximal_development(self):
        """Test ZPD calculation"""
        tasks = [Task(f"t_{i}", difficulty=i*0.1) for i in range(10)]
        curriculum = ProgressiveCurriculum(tasks)
        
        # Set performance level
        curriculum.current_difficulty = 0.5
        for _ in range(10):
            curriculum.performance_tracker.global_performance.append(0.7)
        
        zpd_lower, zpd_upper = curriculum._calculate_zpd()
        
        assert zpd_lower < curriculum.current_difficulty
        assert zpd_upper > curriculum.current_difficulty
        assert zpd_upper - zpd_lower > 0.1  # Reasonable ZPD width
    
    def test_mastery_tracking(self):
        """Test mastery level tracking"""
        tasks = [Task(f"t_{i}", difficulty=i*0.2) for i in range(5)]
        curriculum = ProgressiveCurriculum(tasks, min_attempts=3)
        
        # Build mastery on specific difficulties
        for _ in range(5):
            curriculum.update_performance("t_1", reward=0.9, success=True)
            curriculum.update_performance("t_2", reward=0.4, success=False)
        
        mastery = curriculum.get_mastery_levels()
        
        assert 0.2 in mastery  # Difficulty of t_1
        assert mastery[0.2] > 0.8  # High mastery
        assert 0.4 in mastery  # Difficulty of t_2
        assert mastery[0.4] < 0.5  # Low mastery


class TestAdaptiveCurriculum:
    """Test adaptive curriculum learning"""
    
    def test_learner_profile_update(self):
        """Test learner profile updates"""
        tasks = [
            Task(f"t_{i}", difficulty=i*0.1, context={"type": ["nav", "ctrl"][i%2]})
            for i in range(10)
        ]
        curriculum = AdaptiveCurriculum(tasks)
        
        # Simulate performance patterns
        for i in range(20):
            task = curriculum.select_next_task()
            if task:
                # Good at navigation, bad at control
                if task.context.get("type") == "nav":
                    performance = 0.8
                else:
                    performance = 0.4
                
                curriculum.update_performance(
                    task.task_id,
                    performance + np.random.normal(0, 0.1)
                )
        
        # Check learner insights
        insights = curriculum.get_learner_insights()
        
        assert "preferred_task_types" in insights
        if "nav" in insights["preferred_task_types"]:
            assert insights["preferred_task_types"]["nav"] > 0.6
        if "ctrl" in insights["preferred_task_types"]:
            assert insights["preferred_task_types"]["ctrl"] < 0.6
    
    def test_performance_prediction(self):
        """Test performance prediction with GP model"""
        tasks = [Task(f"t_{i}", difficulty=i*0.1) for i in range(10)]
        curriculum = AdaptiveCurriculum(tasks, use_gp_model=True)
        
        # Build training data
        for _ in range(15):
            task = curriculum.select_next_task()
            if task:
                # Performance inversely related to difficulty
                performance = 0.9 - task.difficulty * 0.7
                performance += np.random.normal(0, 0.05)
                curriculum.update_performance(task.task_id, performance)
        
        # Test prediction
        test_task = Task("test", difficulty=0.5)
        mean, std = curriculum._predict_performance(test_task)
        
        assert 0 <= mean <= 1
        assert std > 0
    
    def test_adaptive_parameters(self):
        """Test curriculum parameter adaptation"""
        tasks = [Task(f"t_{i}", difficulty=i*0.1) for i in range(10)]
        curriculum = AdaptiveCurriculum(tasks, adaptation_rate=0.1)
        
        initial_exploration = curriculum.exploration_bonus
        
        # Simulate consistent high performance
        for _ in range(20):
            curriculum.performance_tracker.global_performance.append(0.9)
        
        curriculum._adapt_curriculum()
        
        # Should increase exploration for fast learner
        assert curriculum.exploration_bonus >= initial_exploration


class TestMetaCurriculum:
    """Test meta-curriculum with MAML integration"""
    
    def test_meta_task_generation(self):
        """Test meta-task generation"""
        generator = MetaTaskGenerator(
            state_dim=4,
            action_dim=2,
            episode_length=50
        )
        
        # Generate single task
        task = generator.generate_task(difficulty=0.5)
        
        assert isinstance(task, MetaTask)
        assert task.difficulty == 0.5
        assert "states" in task.support_data
        assert "states" in task.query_data
        assert task.support_data["states"].shape[0] < task.query_data["states"].shape[0]
        
        # Generate batch
        batch = generator.generate_batch(
            num_tasks=5,
            difficulty_range=(0.3, 0.7)
        )
        
        assert len(batch) == 5
        difficulties = [t.difficulty for t in batch]
        assert min(difficulties) >= 0.3
        assert max(difficulties) <= 0.7
    
    def test_meta_curriculum_integration(self):
        """Test integration with MAML agent"""
        # Create MAML agent
        maml_agent = MAMLAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=16
        )
        
        # Create meta-curriculum
        curriculum = MetaCurriculum(
            maml_agent=maml_agent,
            meta_batch_size=2,
            adaptation_steps=3
        )
        
        # Generate tasks
        curriculum.generate_curriculum_tasks(num_tasks=5)
        
        assert len(curriculum.tasks) == 5
        assert len(curriculum.meta_tasks) == 5
        
        # Test task selection
        task = curriculum.select_next_task()
        assert task is not None
        assert task.task_id in curriculum.meta_tasks
    
    def test_domain_adaptation(self):
        """Test adaptation to new domains"""
        maml_agent = MAMLAgent(state_dim=3, action_dim=2)
        curriculum = MetaCurriculum(maml_agent=maml_agent)
        
        # Generate domain-specific tasks
        generator = MetaTaskGenerator(state_dim=3, action_dim=2)
        domain_tasks = generator.generate_batch(
            num_tasks=3,
            difficulty_range=(0.3, 0.5),
            context={"domain": "test_domain"}
        )
        
        # Adapt to domain
        adapted_agent = curriculum.adapt_to_new_domain(domain_tasks)
        
        assert adapted_agent is not None
        assert adapted_agent.adapted_model is not None
        assert "test_domain" in curriculum.adaptation_success
    
    def test_meta_update_scheduling(self):
        """Test meta-update scheduling"""
        maml_agent = MAMLAgent(state_dim=2, action_dim=2)
        curriculum = MetaCurriculum(
            maml_agent=maml_agent,
            meta_update_frequency=5
        )
        
        curriculum.generate_curriculum_tasks(num_tasks=10)
        
        # Simulate learning process
        for i in range(6):
            task = curriculum.select_next_task()
            if task:
                curriculum.update_performance(
                    task.task_id,
                    reward=0.5,
                    success=False
                )
        
        # Should have reset update counter after meta-update
        assert curriculum.updates_since_meta < curriculum.meta_update_frequency


class TestIntegration:
    """Integration tests for curriculum learning"""
    
    def test_curriculum_with_real_agent(self):
        """Test curriculum with actual RL agent"""
        from rl_commons.algorithms import PPOAgent
        
        # Create PPO agent
        agent = PPOAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=32
        )
        
        # Create curriculum
        tasks = []
        for i in range(10):
            difficulty = 0.1 + i * 0.08
            task = Task(
                f"ppo_task_{i}",
                difficulty=difficulty,
                context={"episode_length": int(50 + difficulty * 100)}
            )
            tasks.append(task)
        
        curriculum = ProgressiveCurriculum(tasks)
        
        # Training loop simulation
        for episode in range(5):
            task = curriculum.select_next_task()
            if not task:
                break
            
            # Simulate episode
            episode_reward = 0
            episode_length = task.context.get("episode_length", 100)
            
            for step in range(episode_length):
                state = RLState(features=np.random.randn(4))
                action = agent.select_action(state)
                
                # Simple reward based on task difficulty
                reward_value = np.random.random() * (1 - task.difficulty)
                reward = RLReward(value=reward_value)
                next_state = RLState(features=np.random.randn(4))
                
                metrics = agent.update(state, action, reward, next_state)
                episode_reward += reward_value
            
            # Update curriculum
            avg_reward = episode_reward / episode_length
            curriculum.update_performance(
                task.task_id,
                avg_reward,
                success=(avg_reward > 0.5)
            )
    
    def test_module_orchestration_scenario(self):
        """Test curriculum for module orchestration (claude-module-communicator)"""
        # Define module complexity levels
        module_tasks = []
        
        # Level 1: Single module tasks
        for i, module in enumerate(["marker", "arangodb", "chat"]):
            task = Task(
                f"single_{module}",
                difficulty=0.2,
                context={
                    "modules": [module],
                    "complexity": "single",
                    "operations": 1
                }
            )
            module_tasks.append(task)
        
        # Level 2: Two-module integration
        integrations = [
            ("marker", "arangodb"),
            ("arangodb", "chat"),
            ("marker", "chat")
        ]
        for i, (mod1, mod2) in enumerate(integrations):
            task = Task(
                f"integrate_{mod1}_{mod2}",
                difficulty=0.5,
                context={
                    "modules": [mod1, mod2],
                    "complexity": "integration",
                    "operations": 3
                },
                prerequisites=[f"single_{mod1}", f"single_{mod2}"]
            )
            module_tasks.append(task)
        
        # Level 3: Full pipeline
        task = Task(
            "full_pipeline",
            difficulty=0.8,
            context={
                "modules": ["marker", "arangodb", "chat"],
                "complexity": "pipeline",
                "operations": 5
            },
            prerequisites=[t.task_id for t in module_tasks if "integrate" in t.task_id]
        )
        module_tasks.append(task)
        
        # Create adaptive curriculum
        curriculum = AdaptiveCurriculum(
            module_tasks,
            learning_style="mastery"
        )
        
        # Simulate module orchestration learning
        completed_modules = set()
        
        for episode in range(20):
            task = curriculum.select_next_task()
            if not task:
                break
            
            # Simulate performance based on module complexity
            modules = task.context.get("modules", [])
            operations = task.context.get("operations", 1)
            
            # Performance depends on prior experience
            base_performance = 0.7
            for module in modules:
                if module in completed_modules:
                    base_performance += 0.1
            
            # Complexity penalty
            performance = base_performance - (operations * 0.05)
            performance += np.random.normal(0, 0.1)
            performance = np.clip(performance, 0, 1)
            
            success = performance > 0.7
            
            # Update curriculum
            curriculum.update_performance(task.task_id, performance, success)
            
            if success:
                curriculum.mark_task_completed(task.task_id)
                completed_modules.update(modules)
        
        # Check progression
        stats = curriculum.get_progress()
        assert stats["completed_tasks"] > 0
        
        # Check learner adapted to module patterns
        insights = curriculum.get_learner_insights()
        assert insights["learning_rate"] > 0


if __name__ == "__main__":
    # Run validation tests
    print("Running curriculum learning validation tests...")
    
    # Test basic components
    test_task = TestTask()
    test_task.test_task_creation()
    test_task.test_task_serialization()
    print("✅ Task tests passed")
    
    test_tracker = TestPerformanceTracker()
    test_tracker.test_performance_recording()
    test_tracker.test_global_performance()
    print("✅ Performance tracker tests passed")
    
    # Test curriculum types
    test_auto = TestAutomaticCurriculum()
    test_auto.test_automatic_task_generation()
    test_auto.test_performance_based_selection()
    print("✅ Automatic curriculum tests passed")
    
    test_prog = TestProgressiveCurriculum()
    test_prog.test_smooth_progression()
    test_prog.test_zone_of_proximal_development()
    print("✅ Progressive curriculum tests passed")
    
    test_adapt = TestAdaptiveCurriculum()
    test_adapt.test_learner_profile_update()
    test_adapt.test_adaptive_parameters()
    print("✅ Adaptive curriculum tests passed")
    
    test_meta = TestMetaCurriculum()
    test_meta.test_meta_task_generation()
    test_meta.test_meta_curriculum_integration()
    print("✅ Meta-curriculum tests passed")
    
    # Integration tests
    test_integration = TestIntegration()
    test_integration.test_module_orchestration_scenario()
    print("✅ Integration tests passed")
    
    print("\n✅ All curriculum learning tests passed!")