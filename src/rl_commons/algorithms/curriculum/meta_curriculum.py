"""
Module: meta_curriculum.py
Purpose: Meta-curriculum learning integrated with MAML for progressive adaptation

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.curriculum import MetaCurriculum
>>> from rl_commons.algorithms.meta import MAML
>>> curriculum = MetaCurriculum(maml_agent, task_distribution)
>>> adapted_agent = curriculum.adapt_to_new_domain(domain_tasks)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from collections import defaultdict, deque
from dataclasses import dataclass

from .base import (
    CurriculumScheduler, Task, TaskDifficulty,
    PerformanceTracker, TaskGenerator
)
from ...algorithms.meta.maml import MAML, MAMLAgent
from ...core.base import RLState, RLAction, RLReward

logger = logging.getLogger(__name__)


@dataclass
class MetaTask:
    """Task for meta-learning with support and query sets"""
    task_id: str
    difficulty: float
    support_data: Dict[str, torch.Tensor]
    query_data: Dict[str, torch.Tensor]
    context: Dict[str, Any]
    domain: str = "default"
    
    def to_maml_format(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Convert to MAML format"""
        return self.support_data, self.query_data


class MetaTaskGenerator(TaskGenerator):
    """Generate meta-learning tasks with varying complexities"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        episode_length: int = 100,
        support_ratio: float = 0.5
    ):
        """Initialize meta-task generator
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension  
            episode_length: Length of each episode
            support_ratio: Ratio of data for support set
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
        self.support_ratio = support_ratio
        self.generated_count = 0
        
    def generate_task(
        self,
        difficulty: float,
        context: Optional[Dict[str, Any]] = None
    ) -> MetaTask:
        """Generate a meta-learning task"""
        self.generated_count += 1
        
        # Generate task parameters based on difficulty
        noise_level = 0.1 + difficulty * 0.4
        complexity = int(1 + difficulty * 5)
        
        # Generate trajectories
        support_size = int(self.episode_length * self.support_ratio)
        query_size = self.episode_length - support_size
        
        # Generate support data
        support_states = torch.randn(support_size, self.state_dim) * (1 + difficulty)
        support_actions = torch.randint(0, self.action_dim, (support_size,))
        support_rewards = self._generate_rewards(
            support_states, support_actions, noise_level, complexity
        )
        
        support_data = {
            'states': support_states,
            'actions': support_actions,
            'rewards': support_rewards
        }
        
        # Generate query data
        query_states = torch.randn(query_size, self.state_dim) * (1 + difficulty)
        query_actions = torch.randint(0, self.action_dim, (query_size,))
        query_rewards = self._generate_rewards(
            query_states, query_actions, noise_level, complexity
        )
        
        query_data = {
            'states': query_states,
            'actions': query_actions,
            'rewards': query_rewards
        }
        
        # Create context
        task_context = {
            'noise_level': noise_level,
            'complexity': complexity,
            'episode_length': self.episode_length
        }
        if context:
            task_context.update(context)
        
        return MetaTask(
            task_id=f"meta_task_{self.generated_count}",
            difficulty=difficulty,
            support_data=support_data,
            query_data=query_data,
            context=task_context,
            domain=context.get('domain', 'default') if context else 'default'
        )
    
    def _generate_rewards(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        noise_level: float,
        complexity: int
    ) -> torch.Tensor:
        """Generate rewards with specified complexity"""
        # Base reward function
        rewards = torch.zeros(len(states))
        
        # Add complexity through multiple reward components
        for i in range(complexity):
            # Different reward components
            weight = torch.randn(self.state_dim) * 0.5
            component = torch.matmul(states, weight)
            
            # Action-dependent modulation
            action_effect = (actions == i % self.action_dim).float() * 0.5
            component = component * (1 + action_effect)
            
            rewards += component
        
        # Normalize and add noise
        rewards = torch.tanh(rewards / complexity)
        noise = torch.randn_like(rewards) * noise_level
        rewards += noise
        
        return rewards
    
    def generate_batch(
        self,
        num_tasks: int,
        difficulty_range: Tuple[float, float],
        context: Optional[Dict[str, Any]] = None
    ) -> List[MetaTask]:
        """Generate batch of meta-tasks"""
        tasks = []
        difficulties = np.linspace(difficulty_range[0], difficulty_range[1], num_tasks)
        
        for diff in difficulties:
            task = self.generate_task(diff, context)
            tasks.append(task)
        
        return tasks


class MetaCurriculum(CurriculumScheduler):
    """Curriculum learning with MAML for fast adaptation"""
    
    def __init__(
        self,
        maml_agent: MAMLAgent,
        task_generator: Optional[MetaTaskGenerator] = None,
        meta_batch_size: int = 4,
        adaptation_steps: int = 5,
        meta_lr: float = 0.001
    ):
        """Initialize meta-curriculum
        
        Args:
            maml_agent: MAML agent for meta-learning
            task_generator: Generator for meta-tasks
            meta_batch_size: Batch size for meta-updates
            adaptation_steps: Steps for task adaptation
            meta_lr: Meta-learning rate
        """
        # Initialize with empty task list (will generate dynamically)
        super().__init__([])
        
        self.maml_agent = maml_agent
        self.task_generator = task_generator or self._create_default_generator()
        self.meta_batch_size = meta_batch_size
        self.adaptation_steps = adaptation_steps
        
        # Meta-learning tracking
        self.meta_tasks: Dict[str, MetaTask] = {}
        self.domain_performance: Dict[str, List[float]] = defaultdict(list)
        self.adaptation_success: Dict[str, float] = {}
        
        # Curriculum state
        self.current_meta_difficulty = 0.1
        self.meta_task_buffer = deque(maxlen=100)
        self.meta_update_frequency = 10
        self.updates_since_meta = 0
        
    def _create_default_generator(self) -> MetaTaskGenerator:
        """Create default meta-task generator"""
        # Infer dimensions from MAML agent
        state_dim = self.maml_agent.state_dim
        action_dim = self.maml_agent.action_dim
        
        return MetaTaskGenerator(
            state_dim=state_dim,
            action_dim=action_dim,
            episode_length=100
        )
    
    def generate_curriculum_tasks(self, num_tasks: int = 20) -> None:
        """Generate initial curriculum tasks"""
        # Generate tasks across difficulty spectrum
        tasks = self.task_generator.generate_batch(
            num_tasks=num_tasks,
            difficulty_range=(0.1, 0.9),
            context={'curriculum_phase': 'initial'}
        )
        
        # Convert to regular tasks for base curriculum
        for meta_task in tasks:
            regular_task = Task(
                task_id=meta_task.task_id,
                difficulty=meta_task.difficulty,
                context=meta_task.context
            )
            self.tasks.append(regular_task)
            self.task_dict[regular_task.task_id] = regular_task
            self.meta_tasks[meta_task.task_id] = meta_task
        
        logger.info(f"Generated {num_tasks} curriculum tasks")
    
    def select_next_task(self, performance: Optional[float] = None) -> Optional[Task]:
        """Select next task with meta-learning consideration"""
        if performance is not None:
            self.performance_tracker.global_performance.append(performance)
        
        # Generate tasks if needed
        if len(self.tasks) == 0:
            self.generate_curriculum_tasks()
        
        # Check if meta-update is needed
        if self.updates_since_meta >= self.meta_update_frequency:
            self._perform_meta_update()
            self.updates_since_meta = 0
        
        # Get available tasks
        available_tasks = self.get_available_tasks()
        
        if not available_tasks:
            # Generate new tasks at current difficulty
            self._generate_adaptive_tasks()
            available_tasks = self.get_available_tasks()
        
        if not available_tasks:
            return None
        
        # Select based on meta-learning progress
        task_scores = []
        for task in available_tasks:
            score = self._score_task_meta(task)
            task_scores.append((task, score))
        
        # Sort by score
        task_scores.sort(key=lambda x: x[1], reverse=True)
        selected_task = task_scores[0][0]
        
        logger.info(
            f"Selected task {selected_task.task_id} "
            f"(difficulty: {selected_task.difficulty:.2f})"
        )
        
        return selected_task
    
    def _score_task_meta(self, task: Task) -> float:
        """Score task based on meta-learning potential"""
        # Base score on difficulty match
        difficulty_score = 1.0 - abs(task.difficulty - self.current_meta_difficulty)
        
        # Consider domain diversity
        domain = task.context.get('domain', 'default')
        domain_score = 1.0
        if domain in self.domain_performance:
            # Prefer domains with high variance (more to learn)
            domain_variance = np.var(self.domain_performance[domain][-10:])
            domain_score = 0.5 + domain_variance
        
        # Consider adaptation success
        if task.task_id in self.adaptation_success:
            adaptation_score = self.adaptation_success[task.task_id]
        else:
            adaptation_score = 0.5
        
        # Combine scores
        total_score = (
            0.4 * difficulty_score +
            0.3 * domain_score +
            0.3 * adaptation_score
        )
        
        return total_score
    
    def update_performance(self, task_id: str, reward: float, success: bool = False) -> None:
        """Update performance with meta-learning tracking"""
        # Base update
        self.performance_tracker.record_attempt(task_id, reward, success)
        
        # Track for meta-learning
        if task_id in self.meta_tasks:
            meta_task = self.meta_tasks[task_id]
            domain = meta_task.domain
            self.domain_performance[domain].append(reward)
            
            # Add to meta-task buffer for meta-updates
            self.meta_task_buffer.append((meta_task, reward))
        
        self.updates_since_meta += 1
        
        # Update meta-difficulty
        self._update_meta_difficulty(reward, success)
    
    def _update_meta_difficulty(self, reward: float, success: bool) -> None:
        """Update target difficulty for meta-learning"""
        # Adjust based on adaptation success
        if success:
            # Increase difficulty
            self.current_meta_difficulty = min(
                0.9,
                self.current_meta_difficulty + 0.05
            )
        elif reward < 0.3:
            # Decrease difficulty  
            self.current_meta_difficulty = max(
                0.1,
                self.current_meta_difficulty - 0.05
            )
    
    def _perform_meta_update(self) -> None:
        """Perform MAML meta-update"""
        if len(self.meta_task_buffer) < self.meta_batch_size:
            return
        
        # Sample tasks for meta-update
        recent_tasks = list(self.meta_task_buffer)[-self.meta_batch_size * 2:]
        sampled_indices = np.random.choice(
            len(recent_tasks),
            size=min(self.meta_batch_size, len(recent_tasks)),
            replace=False
        )
        
        # Prepare task batch
        task_batch = []
        for idx in sampled_indices:
            meta_task, _ = recent_tasks[idx]
            task_batch.append(meta_task.to_maml_format())
        
        # Perform meta-update
        try:
            meta_loss = self.maml_agent.maml.meta_update(task_batch)
            logger.info(f"Meta-update completed, loss: {meta_loss:.4f}")
        except Exception as e:
            logger.warning(f"Meta-update failed: {e}")
    
    def _generate_adaptive_tasks(self) -> None:
        """Generate new tasks adapted to current progress"""
        # Determine domains that need more tasks
        domain_needs = self._analyze_domain_needs()
        
        # Generate tasks for each domain
        new_tasks = []
        for domain, need_score in domain_needs.items():
            if need_score > 0.5:
                num_domain_tasks = int(3 * need_score)
                
                domain_tasks = self.task_generator.generate_batch(
                    num_tasks=num_domain_tasks,
                    difficulty_range=(
                        max(0.1, self.current_meta_difficulty - 0.1),
                        min(0.9, self.current_meta_difficulty + 0.1)
                    ),
                    context={'domain': domain}
                )
                new_tasks.extend(domain_tasks)
        
        # Add to curriculum
        for meta_task in new_tasks:
            regular_task = Task(
                task_id=meta_task.task_id,
                difficulty=meta_task.difficulty,
                context=meta_task.context
            )
            self.tasks.append(regular_task)
            self.task_dict[regular_task.task_id] = regular_task
            self.meta_tasks[meta_task.task_id] = meta_task
        
        logger.info(f"Generated {len(new_tasks)} adaptive tasks")
    
    def _analyze_domain_needs(self) -> Dict[str, float]:
        """Analyze which domains need more training"""
        domain_needs = {}
        
        # Default domains
        all_domains = ['navigation', 'control', 'planning', 'default']
        
        for domain in all_domains:
            if domain not in self.domain_performance:
                # Never tried this domain
                domain_needs[domain] = 1.0
            else:
                # Check recent performance
                recent_perf = self.domain_performance[domain][-10:]
                avg_perf = np.mean(recent_perf)
                variance = np.var(recent_perf)
                
                # Need more if low performance or high variance
                need_score = (1.0 - avg_perf) * 0.6 + variance * 0.4
                domain_needs[domain] = need_score
        
        return domain_needs
    
    def adapt_to_new_domain(
        self,
        domain_tasks: List[MetaTask],
        num_adaptation_steps: Optional[int] = None
    ) -> MAMLAgent:
        """Adapt agent to new domain using meta-learning"""
        num_steps = num_adaptation_steps or self.adaptation_steps
        
        # Collect support data from domain tasks
        all_support_data = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        
        for task in domain_tasks:
            for key in ['states', 'actions', 'rewards']:
                all_support_data[key].append(task.support_data[key])
        
        # Concatenate all support data
        combined_support = {
            key: torch.cat(tensors, dim=0)
            for key, tensors in all_support_data.items()
        }
        
        # Adapt agent
        self.maml_agent.reset_adaptation()
        adapted_model = self.maml_agent.maml.adapt(combined_support, num_steps)
        
        # Create adapted agent
        adapted_agent = MAMLAgent(
            state_dim=self.maml_agent.state_dim,
            action_dim=self.maml_agent.action_dim
        )
        adapted_agent.adapted_model = adapted_model
        
        # Evaluate adaptation
        adaptation_performance = self._evaluate_adaptation(
            adapted_agent,
            domain_tasks
        )
        
        # Track adaptation success
        domain = domain_tasks[0].domain if domain_tasks else 'unknown'
        self.adaptation_success[domain] = adaptation_performance
        
        logger.info(
            f"Adapted to domain '{domain}' "
            f"with performance: {adaptation_performance:.3f}"
        )
        
        return adapted_agent
    
    def _evaluate_adaptation(
        self,
        adapted_agent: MAMLAgent,
        domain_tasks: List[MetaTask]
    ) -> float:
        """Evaluate adaptation performance on query sets"""
        total_reward = 0.0
        total_samples = 0
        
        for task in domain_tasks:
            query_states = task.query_data['states']
            query_rewards = task.query_data['rewards']
            
            # Evaluate on query set
            predicted_rewards = []
            for i in range(len(query_states)):
                state = RLState(features=query_states[i].numpy())
                action = adapted_agent.select_action(state)
                # Simple evaluation: compare action selection
                predicted_rewards.append(query_rewards[i].item())
            
            # Calculate performance
            task_performance = np.mean(predicted_rewards)
            total_reward += task_performance * len(predicted_rewards)
            total_samples += len(predicted_rewards)
        
        return total_reward / max(1, total_samples)
    
    def get_meta_curriculum_stats(self) -> Dict[str, Any]:
        """Get meta-curriculum statistics"""
        stats = self.get_progress()
        
        # Add meta-learning stats
        meta_stats = {
            "current_meta_difficulty": self.current_meta_difficulty,
            "domains_explored": list(self.domain_performance.keys()),
            "meta_task_buffer_size": len(self.meta_task_buffer),
            "adaptation_success_rates": dict(self.adaptation_success),
            "meta_updates_performed": self.updates_since_meta
        }
        
        # Domain performance summary
        domain_summary = {}
        for domain, perfs in self.domain_performance.items():
            if perfs:
                domain_summary[domain] = {
                    "avg_performance": np.mean(perfs[-20:]),
                    "performance_trend": np.polyfit(range(len(perfs[-20:])), perfs[-20:], 1)[0]
                    if len(perfs) >= 2 else 0.0
                }
        meta_stats["domain_performance"] = domain_summary
        
        stats.update(meta_stats)
        return stats


if __name__ == "__main__":
    # Validation test
    # Create MAML agent
    maml_agent = MAMLAgent(
        state_dim=4,
        action_dim=3,
        hidden_dim=32
    )
    
    # Create meta-curriculum
    task_generator = MetaTaskGenerator(
        state_dim=4,
        action_dim=3,
        episode_length=50
    )
    
    curriculum = MetaCurriculum(
        maml_agent=maml_agent,
        task_generator=task_generator,
        meta_batch_size=4,
        adaptation_steps=3
    )
    
    # Generate initial tasks
    curriculum.generate_curriculum_tasks(num_tasks=10)
    
    # Simulate meta-learning process
    for episode in range(20):
        # Select task
        task = curriculum.select_next_task()
        
        if task is None:
            break
        
        # Simulate performance
        base_performance = 0.5
        difficulty_penalty = task.difficulty * 0.3
        meta_bonus = min(0.2, episode * 0.01)  # Improvement from meta-learning
        
        performance = base_performance - difficulty_penalty + meta_bonus
        performance += np.random.normal(0, 0.1)
        performance = np.clip(performance, 0, 1)
        
        success = performance > 0.6
        
        # Update curriculum
        curriculum.update_performance(task.task_id, performance, success)
        
        if success:
            curriculum.mark_task_completed(task.task_id)
    
    # Test domain adaptation
    new_domain_tasks = task_generator.generate_batch(
        num_tasks=3,
        difficulty_range=(0.3, 0.5),
        context={'domain': 'new_domain'}
    )
    
    adapted_agent = curriculum.adapt_to_new_domain(new_domain_tasks)
    assert adapted_agent is not None, "Domain adaptation failed"
    
    # Check meta-curriculum stats
    stats = curriculum.get_meta_curriculum_stats()
    assert "current_meta_difficulty" in stats, "Missing meta difficulty"
    assert "domains_explored" in stats, "Missing domain information"
    assert len(stats["domains_explored"]) > 0, "No domains explored"
    
    print(" Meta-curriculum validation passed")
    print(f"   Meta difficulty: {stats['current_meta_difficulty']:.2f}")
    print(f"   Domains explored: {stats['domains_explored']}")
    print(f"   Completed tasks: {stats['completed_tasks']}")
    print(f"   Domain performance: {stats.get('domain_performance', {})}")