"""
Module: algorithm_benchmarks.py
Purpose: Benchmarks for RL algorithms in various scenarios

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- gym: https://gymnasium.farama.org/

Example Usage:
>>> from rl_commons.benchmarks import AlgorithmBenchmark
>>> benchmark = AlgorithmBenchmark('ppo', 'CartPole-v1')
>>> result = benchmark.execute()
"""

import time
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, List, Tuple
import logging
from collections import deque

from .benchmark_suite import Benchmark, BenchmarkResult
from ..core.base import RLState, RLAction, RLReward
from ..core.algorithm_selector import AlgorithmSelector, TaskProperties, TaskType, ActionSpace
from ..algorithms import PPOAgent, DQNAgent, A3CAgent
from ..algorithms.marl import IndependentQLearning
from ..algorithms.morl import WeightedSumMORL
from ..algorithms.bandits import ContextualBandit

logger = logging.getLogger(__name__)


class AlgorithmBenchmark(Benchmark):
    """Benchmark for RL algorithms"""
    
    def __init__(self, 
                 algorithm_name: str,
                 env_name: str,
                 num_episodes: int = 100,
                 max_steps: int = 500):
        super().__init__(
            name=f"{algorithm_name}_{env_name}",
            description=f"Benchmark {algorithm_name} on {env_name}"
        )
        self.algorithm_name = algorithm_name
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Initialize components
        self.env = None
        self.agent = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_times = []
    
    def setup(self) -> None:
        """Setup environment and agent"""
        # Create environment
        self.env = gym.make(self.env_name)
        
        # Get environment properties
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            action_space_type = ActionSpace.DISCRETE
        else:
            action_dim = self.env.action_space.shape[0]
            action_space_type = ActionSpace.CONTINUOUS
        
        # Create agent based on algorithm name
        if self.algorithm_name == "ppo":
            self.agent = PPOAgent(
                name=f"{self.algorithm_name}_benchmark",
                state_dim=state_dim,
                action_dim=action_dim,
                continuous=True,
                learning_rate=0.0003
            )
        elif self.algorithm_name == "dqn":
            self.agent = DQNAgent(
                name=f"{self.algorithm_name}_benchmark",
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=0.001
            )
        elif self.algorithm_name == "a3c":
            self.agent = A3CAgent(
                name=f"{self.algorithm_name}_benchmark",
                state_dim=state_dim,
                action_dim=action_dim,
                continuous=True,
                num_workers=4
            )
        elif self.algorithm_name == "contextual_bandit":
            self.agent = ContextualBandit(
                arms=list(range(action_dim)),
                context_dim=state_dim
            )
        else:
            # Use algorithm selector
            task_props = TaskProperties(
                task_type=TaskType.EPISODIC,
                action_space=action_space_type,
                action_dim=action_dim,
                state_dim=state_dim
            )
            selector = AlgorithmSelector()
            self.agent = selector.select_algorithm(task_props)
    
    def run(self) -> Dict[str, float]:
        """Run benchmark"""
        total_start_time = time.time()
        
        for episode in range(self.num_episodes):
            episode_start_time = time.time()
            
            # Run episode
            episode_reward, episode_length = self._run_episode()
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.training_times.append(time.time() - episode_start_time)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.debug(f"Episode {episode + 1}/{self.num_episodes}, "
                           f"Avg Reward: {avg_reward:.2f}")
        
        total_time = time.time() - total_start_time
        
        # Calculate metrics
        metrics = {
            'average_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'final_reward': np.mean(self.episode_rewards[-10:]),
            'average_length': np.mean(self.episode_lengths),
            'convergence_episodes': self._calculate_convergence(),
            'throughput': self.num_episodes / total_time,
            'training_efficiency': np.mean(self.episode_rewards) / total_time
        }
        
        return metrics
    
    def teardown(self) -> None:
        """Cleanup"""
        if self.env:
            self.env.close()
        self.env = None
        self.agent = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_times = []
    
    def _run_episode(self) -> Tuple[float, int]:
        """Run a single episode"""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle new gym API
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.max_steps):
            # Select action
            rl_state = RLState(features=state)
            action = self.agent.select_action(rl_state)
            
            # Take action
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                env_action = action.action_id
            else:
                env_action = action.continuous_values
            
            next_state, reward, done, truncated, info = self.env.step(env_action)
            
            # Update agent
            next_rl_state = RLState(features=next_state)
            rl_reward = RLReward(value=reward)
            
            self.agent.update_policy(
                rl_state, action, rl_reward, next_rl_state, 
                done or truncated
            )
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done or truncated:
                break
        
        return episode_reward, episode_length
    
    def _calculate_convergence(self) -> int:
        """Calculate episodes to convergence"""
        if len(self.episode_rewards) < 20:
            return len(self.episode_rewards)
        
        # Use rolling average to detect convergence
        window_size = 10
        threshold = 0.05  # 5% variation
        
        for i in range(window_size, len(self.episode_rewards) - window_size):
            window = self.episode_rewards[i:i+window_size]
            mean = np.mean(window)
            std = np.std(window)
            
            if mean > 0 and std / mean < threshold:
                return i
        
        return len(self.episode_rewards)


class MultiObjectiveBenchmark(Benchmark):
    """Benchmark for multi-objective RL"""
    
    def __init__(self, num_objectives: int = 3, num_episodes: int = 100):
        super().__init__(
            name=f"morl_{num_objectives}obj",
            description=f"Multi-objective RL with {num_objectives} objectives"
        )
        self.num_objectives = num_objectives
        self.num_episodes = num_episodes
        self.agent = None
        self.pareto_solutions = []
    
    def setup(self) -> None:
        """Setup MORL agent"""
        self.agent = WeightedSumMORL(
            num_objectives=self.num_objectives,
            state_dim=10,
            action_dim=4
        )
    
    def run(self) -> Dict[str, float]:
        """Run MORL benchmark"""
        # Test different preference vectors
        preference_vectors = self._generate_preference_vectors()
        
        all_rewards = []
        for pref in preference_vectors:
            episode_rewards = self._run_with_preference(pref)
            all_rewards.append(episode_rewards)
        
        # Get Pareto front
        pareto_front = self.agent.pareto_front
        
        metrics = {
            'pareto_solutions': len(pareto_front.solutions),
            'hypervolume': pareto_front.calculate_hypervolume(),
            'spread': self._calculate_spread(pareto_front.solutions),
            'convergence': self._calculate_convergence_metric(all_rewards)
        }
        
        return metrics
    
    def teardown(self) -> None:
        """Cleanup"""
        self.agent = None
        self.pareto_solutions = []
    
    def _generate_preference_vectors(self) -> List[np.ndarray]:
        """Generate diverse preference vectors"""
        if self.num_objectives == 2:
            # Simple linear interpolation
            weights = np.linspace(0, 1, 11)
            return [np.array([w, 1-w]) for w in weights]
        else:
            # Random sampling for higher dimensions
            preferences = []
            for _ in range(20):
                pref = np.random.dirichlet(np.ones(self.num_objectives))
                preferences.append(pref)
            return preferences
    
    def _run_with_preference(self, preference: np.ndarray) -> List[float]:
        """Run episodes with specific preference"""
        rewards = []
        
        for _ in range(self.num_episodes // 10):  # Fewer episodes per preference
            # Simulate multi-objective environment
            state = np.random.randn(10)
            total_rewards = np.zeros(self.num_objectives)
            
            for _ in range(100):  # Steps per episode
                rl_state = RLState(features=state)
                action = self.agent.select_action(rl_state, preference)
                
                # Simulate multi-objective rewards
                obj_rewards = np.random.randn(self.num_objectives) * 0.1
                obj_rewards[action.action_id % self.num_objectives] += 0.5
                
                total_rewards += obj_rewards
                
                # Update agent
                from ..algorithms.morl import MultiObjectiveReward
                mo_reward = MultiObjectiveReward(obj_rewards)
                next_state = state + np.random.randn(10) * 0.1
                next_rl_state = RLState(features=next_state)
                
                self.agent.update(rl_state, action, mo_reward, next_rl_state, False)
                state = next_state
            
            scalarized_reward = np.dot(preference, total_rewards)
            rewards.append(scalarized_reward)
        
        return rewards
    
    def _calculate_spread(self, solutions: List[np.ndarray]) -> float:
        """Calculate spread of Pareto solutions"""
        if len(solutions) < 2:
            return 0.0
        
        # Calculate average distance between consecutive solutions
        distances = []
        for i in range(len(solutions) - 1):
            dist = np.linalg.norm(solutions[i] - solutions[i+1])
            distances.append(dist)
        
        return np.mean(distances)
    
    def _calculate_convergence_metric(self, all_rewards: List[List[float]]) -> float:
        """Calculate convergence metric for MORL"""
        # Average improvement over preferences
        improvements = []
        for rewards in all_rewards:
            if len(rewards) > 1:
                improvement = (rewards[-1] - rewards[0]) / max(abs(rewards[0]), 1e-6)
                improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0


class ScalabilityBenchmark(Benchmark):
    """Benchmark for testing scalability"""
    
    def __init__(self, 
                 algorithm_name: str,
                 state_dims: List[int] = [10, 50, 100, 500],
                 action_dims: List[int] = [2, 5, 10, 20]):
        super().__init__(
            name=f"scalability_{algorithm_name}",
            description=f"Scalability test for {algorithm_name}"
        )
        self.algorithm_name = algorithm_name
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.results = {}
    
    def setup(self) -> None:
        """Setup benchmark"""
        self.results = {}
    
    def run(self) -> Dict[str, float]:
        """Run scalability tests"""
        for state_dim in self.state_dims:
            for action_dim in self.action_dims:
                # Create agent
                agent = self._create_agent(state_dim, action_dim)
                
                # Measure performance
                start_time = time.time()
                memory_before = self.process.memory_info().rss / 1024 / 1024
                
                # Run training steps
                for _ in range(100):
                    state = np.random.randn(state_dim)
                    rl_state = RLState(features=state)
                    action = agent.select_action(rl_state)
                    
                    # Simulate update
                    reward = RLReward(value=np.random.randn())
                    next_state = RLState(features=np.random.randn(state_dim))
                    agent.update_policy(rl_state, action, reward, next_state, False)
                
                # Measure results
                duration = time.time() - start_time
                memory_after = self.process.memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before
                
                key = f"s{state_dim}_a{action_dim}"
                self.results[key] = {
                    'duration': duration,
                    'memory': memory_used,
                    'throughput': 100 / duration
                }
        
        # Calculate scalability metrics
        metrics = self._calculate_scalability_metrics()
        return metrics
    
    def teardown(self) -> None:
        """Cleanup"""
        self.results = {}
    
    def _create_agent(self, state_dim: int, action_dim: int):
        """Create agent for testing"""
        if self.algorithm_name == "ppo":
            return PPOAgent(state_dim=state_dim, action_dim=action_dim)
        elif self.algorithm_name == "dqn":
            return DQNAgent(state_dim=state_dim, action_dim=action_dim)
        else:
            # Default to PPO
            return PPOAgent(state_dim=state_dim, action_dim=action_dim)
    
    def _calculate_scalability_metrics(self) -> Dict[str, float]:
        """Calculate scalability metrics"""
        # Extract dimensions and performance
        state_performance = {}
        action_performance = {}
        
        for key, perf in self.results.items():
            parts = key.split('_')
            state_dim = int(parts[0][1:])
            action_dim = int(parts[1][1:])
            
            if action_dim == self.action_dims[0]:  # Fix action dim
                state_performance[state_dim] = perf['throughput']
            
            if state_dim == self.state_dims[0]:  # Fix state dim
                action_performance[action_dim] = perf['throughput']
        
        # Calculate scaling factors
        state_scaling = self._calculate_scaling_factor(state_performance)
        action_scaling = self._calculate_scaling_factor(action_performance)
        
        # Average memory usage
        avg_memory = np.mean([r['memory'] for r in self.results.values()])
        max_memory = np.max([r['memory'] for r in self.results.values()])
        
        return {
            'state_scaling_factor': state_scaling,
            'action_scaling_factor': action_scaling,
            'average_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'total_configurations': len(self.results)
        }
    
    def _calculate_scaling_factor(self, performance: Dict[int, float]) -> float:
        """Calculate scaling factor (closer to 1.0 is better)"""
        if len(performance) < 2:
            return 1.0
        
        dims = sorted(performance.keys())
        throughputs = [performance[d] for d in dims]
        
        # Calculate how throughput scales with dimension
        # Ideal is constant throughput (scaling factor = 1.0)
        # Worse scaling gives lower values
        scaling_factors = []
        for i in range(1, len(dims)):
            dim_ratio = dims[i] / dims[i-1]
            perf_ratio = throughputs[i] / throughputs[i-1]
            scaling = perf_ratio  # Should be close to 1.0 for good scaling
            scaling_factors.append(scaling)
        
        return np.mean(scaling_factors)


# Validation
if __name__ == "__main__":
    print("Testing Algorithm Benchmarks...")
    
    # Test basic algorithm benchmark
    benchmark = AlgorithmBenchmark("ppo", "CartPole-v1", num_episodes=10)
    result = benchmark.execute()
    
    assert result.benchmark_name == "ppo_CartPole-v1"
    assert 'average_reward' in result.metrics
    assert result.duration > 0
    
    # Test multi-objective benchmark
    mo_benchmark = MultiObjectiveBenchmark(num_objectives=2, num_episodes=10)
    mo_result = mo_benchmark.execute()
    
    assert 'pareto_solutions' in mo_result.metrics
    
    # Test scalability benchmark
    scale_benchmark = ScalabilityBenchmark("ppo", state_dims=[10, 20], action_dims=[2, 4])
    scale_result = scale_benchmark.execute()
    
    assert 'state_scaling_factor' in scale_result.metrics
    
    print(" Algorithm benchmarks validation passed")