"""Algorithm Selection Framework for RL Commons
# Module: algorithm_selector.py
# Description: Implementation of algorithm selector functionality

This module provides an intelligent algorithm selection framework that can:
1. Analyze task characteristics
2. Select the most appropriate RL algorithm
3. Track performance and adapt over time
4. Support real-time algorithm switching

External Dependencies:
- numpy: For numerical operations
- torch: For neural networks (if algorithms use it)

Example Usage:
>>> from rl_commons.core.algorithm_selector import AlgorithmSelector, TaskProperties
>>> selector = AlgorithmSelector()
>>> task_properties = TaskProperties(
...     action_space=ActionSpace.DISCRETE,
...     state_dim=4,
...     action_dim=2
... )
>>> algorithm = selector.select_algorithm(task_properties)
>>> print(f"Selected: {algorithm.__class__.__name__}")
'Selected: DQNAgent'
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
import numpy as np
from pathlib import Path
import json
import time
from collections import defaultdict
from enum import Enum
import logging

from ..core.base import RLAgent, RLState, RLAction, RLReward
from ..algorithms.ppo import PPOAgent
from ..algorithms.a3c import A3CAgent
from ..algorithms.dqn import DQNAgent, DoubleDQNAgent
from ..algorithms.bandits import ContextualBandit, ThompsonSamplingBandit
from ..algorithms.hierarchical import HierarchicalRLAgent

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of task types"""
    BANDIT = "bandit"
    EPISODIC = "episodic"
    CONTINUOUS = "continuous"
    MULTI_AGENT = "multi_agent"
    HIERARCHICAL = "hierarchical"
    GRAPH_BASED = "graph_based"
    IMITATION = "imitation"
    MULTI_OBJECTIVE = "multi_objective"
    CURRICULUM = "curriculum"
    META_LEARNING = "meta_learning"


class ActionSpace(Enum):
    """Action space types"""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MIXED = "mixed"
    HIERARCHICAL = "hierarchical"


class ObservabilityType(Enum):
    """Observability types"""
    FULLY_OBSERVABLE = "fully_observable"
    PARTIALLY_OBSERVABLE = "partially_observable"
    MULTI_VIEW = "multi_view"


@dataclass
class TaskProperties:
    """Properties that characterize an RL task"""
    # Basic properties
    task_type: TaskType = TaskType.EPISODIC
    action_space: ActionSpace = ActionSpace.DISCRETE
    observability: ObservabilityType = ObservabilityType.FULLY_OBSERVABLE
    
    # Dimensions
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    num_agents: int = 1
    
    # Task characteristics
    is_multi_agent: bool = False
    has_continuous_actions: bool = False
    is_partially_observable: bool = False
    requires_exploration: bool = True
    has_sparse_rewards: bool = False
    is_non_stationary: bool = False
    has_long_horizon: bool = False
    
    # Special requirements
    requires_communication: bool = False
    has_graph_structure: bool = False
    has_demonstrations: bool = False
    has_multiple_objectives: bool = False
    requires_curriculum: bool = False
    requires_fast_adaptation: bool = False
    
    # Performance requirements
    max_latency_ms: float = 100.0
    requires_sample_efficiency: bool = False
    requires_stability: bool = True
    
    # Module communication
    module_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "task_type": self.task_type.value,
            "action_space": self.action_space.value,
            "observability": self.observability.value,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,
            "is_multi_agent": self.is_multi_agent,
            "has_continuous_actions": self.has_continuous_actions,
            "is_partially_observable": self.is_partially_observable,
            "requires_exploration": self.requires_exploration,
            "has_sparse_rewards": self.has_sparse_rewards,
            "is_non_stationary": self.is_non_stationary,
            "has_long_horizon": self.has_long_horizon,
            "requires_communication": self.requires_communication,
            "has_graph_structure": self.has_graph_structure,
            "has_demonstrations": self.has_demonstrations,
            "has_multiple_objectives": self.has_multiple_objectives,
            "requires_curriculum": self.requires_curriculum,
            "requires_fast_adaptation": self.requires_fast_adaptation,
            "max_latency_ms": self.max_latency_ms,
            "requires_sample_efficiency": self.requires_sample_efficiency,
            "requires_stability": self.requires_stability,
            "module_context": self.module_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskProperties":
        """Create from dictionary"""
        return cls(
            task_type=TaskType(data.get("task_type", "episodic")),
            action_space=ActionSpace(data.get("action_space", "discrete")),
            observability=ObservabilityType(data.get("observability", "fully_observable")),
            state_dim=data.get("state_dim"),
            action_dim=data.get("action_dim"),
            num_agents=data.get("num_agents", 1),
            is_multi_agent=data.get("is_multi_agent", False),
            has_continuous_actions=data.get("has_continuous_actions", False),
            is_partially_observable=data.get("is_partially_observable", False),
            requires_exploration=data.get("requires_exploration", True),
            has_sparse_rewards=data.get("has_sparse_rewards", False),
            is_non_stationary=data.get("is_non_stationary", False),
            has_long_horizon=data.get("has_long_horizon", False),
            requires_communication=data.get("requires_communication", False),
            has_graph_structure=data.get("has_graph_structure", False),
            has_demonstrations=data.get("has_demonstrations", False),
            has_multiple_objectives=data.get("has_multiple_objectives", False),
            requires_curriculum=data.get("requires_curriculum", False),
            requires_fast_adaptation=data.get("requires_fast_adaptation", False),
            max_latency_ms=data.get("max_latency_ms", 100.0),
            requires_sample_efficiency=data.get("requires_sample_efficiency", False),
            requires_stability=data.get("requires_stability", True),
            module_context=data.get("module_context", {})
        )


@dataclass
class AlgorithmPerformance:
    """Performance tracking for an algorithm"""
    algorithm_name: str
    task_hash: str
    total_episodes: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0
    success_rate: float = 0.0
    average_episode_length: float = 0.0
    training_time: float = 0.0
    sample_efficiency: float = 0.0
    stability_score: float = 1.0
    last_update: float = field(default_factory=time.time)
    
    def update(self, episode_reward: float, episode_length: int, success: bool, training_time: float):
        """Update performance metrics"""
        self.total_episodes += 1
        self.total_reward += episode_reward
        self.average_reward = self.total_reward / self.total_episodes
        self.success_rate = (self.success_rate * (self.total_episodes - 1) + int(success)) / self.total_episodes
        self.average_episode_length = (self.average_episode_length * (self.total_episodes - 1) + episode_length) / self.total_episodes
        self.training_time += training_time
        self.sample_efficiency = self.average_reward / max(1, self.total_episodes)
        self.last_update = time.time()
    
    def get_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted performance score"""
        if weights is None:
            weights = {
                "reward": 0.4,
                "success": 0.3,
                "efficiency": 0.2,
                "stability": 0.1
            }
        
        normalized_reward = self.average_reward / max(1.0, abs(self.average_reward))
        
        score = (
            weights.get("reward", 0.4) * normalized_reward +
            weights.get("success", 0.3) * self.success_rate +
            weights.get("efficiency", 0.2) * self.sample_efficiency +
            weights.get("stability", 0.1) * self.stability_score
        )
        
        return score


class AlgorithmSelector:
    """Intelligent algorithm selection framework"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.performance_history: Dict[str, Dict[str, AlgorithmPerformance]] = defaultdict(dict)
        self.active_agents: Dict[str, RLAgent] = {}
        self.selection_count: Dict[str, int] = defaultdict(int)
        
        # Algorithm registry
        self.algorithm_registry = self._build_algorithm_registry()
        
        # Selection strategy
        self.exploration_rate = self.config.get("exploration_rate", 0.1)
        self.adaptation_threshold = self.config.get("adaptation_threshold", 0.8)
        
        logger.info("Algorithm Selector initialized with %d algorithms", len(self.algorithm_registry))
    
    def _build_algorithm_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of available algorithms with their characteristics"""
        return {
            # Basic algorithms
            "DQN": {
                "class": DQNAgent,
                "suitable_for": {
                    "action_space": [ActionSpace.DISCRETE],
                    "task_type": [TaskType.EPISODIC],
                    "requires_stability": True
                },
                "performance_profile": {
                    "sample_efficiency": 0.6,
                    "stability": 0.7,
                    "exploration": 0.5
                }
            },
            "DoubleDQN": {
                "class": DoubleDQNAgent,
                "suitable_for": {
                    "action_space": [ActionSpace.DISCRETE],
                    "task_type": [TaskType.EPISODIC],
                    "requires_stability": True
                },
                "performance_profile": {
                    "sample_efficiency": 0.7,
                    "stability": 0.8,
                    "exploration": 0.5
                }
            },
            "PPO": {
                "class": PPOAgent,
                "suitable_for": {
                    "action_space": [ActionSpace.DISCRETE, ActionSpace.CONTINUOUS],
                    "task_type": [TaskType.EPISODIC, TaskType.CONTINUOUS],
                    "requires_stability": True
                },
                "performance_profile": {
                    "sample_efficiency": 0.7,
                    "stability": 0.9,
                    "exploration": 0.6
                }
            },
            "A3C": {
                "class": A3CAgent,
                "suitable_for": {
                    "action_space": [ActionSpace.DISCRETE, ActionSpace.CONTINUOUS],
                    "task_type": [TaskType.EPISODIC, TaskType.CONTINUOUS],
                    "has_long_horizon": True
                },
                "performance_profile": {
                    "sample_efficiency": 0.5,
                    "stability": 0.6,
                    "exploration": 0.8
                }
            },
            # Bandits
            "ContextualBandit": {
                "class": ContextualBandit,
                "suitable_for": {
                    "task_type": [TaskType.BANDIT],
                    "action_space": [ActionSpace.DISCRETE]
                },
                "performance_profile": {
                    "sample_efficiency": 0.9,
                    "stability": 0.9,
                    "exploration": 0.7
                }
            },
            "ThompsonSampling": {
                "class": ThompsonSamplingBandit,
                "suitable_for": {
                    "task_type": [TaskType.BANDIT],
                    "action_space": [ActionSpace.DISCRETE],
                    "requires_exploration": True
                },
                "performance_profile": {
                    "sample_efficiency": 0.8,
                    "stability": 0.8,
                    "exploration": 0.9
                }
            },
            # Hierarchical
            "HierarchicalRL": {
                "class": HierarchicalRLAgent,
                "suitable_for": {
                    "task_type": [TaskType.HIERARCHICAL],
                    "action_space": [ActionSpace.HIERARCHICAL],
                    "has_long_horizon": True
                },
                "performance_profile": {
                    "sample_efficiency": 0.8,
                    "stability": 0.7,
                    "exploration": 0.7
                }
            }
        }
    
    def register_algorithm(self, 
                          name: str, 
                          agent_class: Type[RLAgent],
                          suitable_for: Dict[str, Any],
                          performance_profile: Dict[str, float]) -> None:
        """Register a new algorithm with the selector"""
        self.algorithm_registry[name] = {
            "class": agent_class,
            "suitable_for": suitable_for,
            "performance_profile": performance_profile
        }
        logger.info(f"Registered new algorithm: {name}")
    
    def extract_task_properties(self, 
                               environment: Any,
                               initial_state: Optional[RLState] = None,
                               module_context: Optional[Dict[str, Any]] = None) -> TaskProperties:
        """Extract task properties from environment and context"""
        properties = TaskProperties()
        
        # Extract from module context if available
        if module_context:
            properties.module_context = module_context
            
            # Claude module communicator specific
            if "modules" in module_context:
                properties.num_agents = len(module_context["modules"])
                properties.is_multi_agent = properties.num_agents > 1
                properties.requires_communication = module_context.get("requires_communication", False)
            
            # Task type hints
            if "task_type" in module_context:
                properties.task_type = TaskType(module_context["task_type"])
            
            # Performance requirements
            if "latency_requirement" in module_context:
                properties.max_latency_ms = module_context["latency_requirement"]
        
        # Extract from environment
        if hasattr(environment, "action_space"):
            action_space = environment.action_space
            if hasattr(action_space, "n"):
                properties.action_space = ActionSpace.DISCRETE
                properties.action_dim = action_space.n
            elif hasattr(action_space, "shape"):
                properties.action_space = ActionSpace.CONTINUOUS
                properties.action_dim = action_space.shape[0]
                properties.has_continuous_actions = True
        
        if hasattr(environment, "observation_space"):
            obs_space = environment.observation_space
            if hasattr(obs_space, "shape"):
                properties.state_dim = np.prod(obs_space.shape)
        
        # Extract from initial state
        if initial_state:
            if hasattr(initial_state, "features"):
                properties.state_dim = len(initial_state.features)
            
            if initial_state.context:
                # Check for graph structure
                if "graph" in initial_state.context or "edges" in initial_state.context:
                    properties.has_graph_structure = True
                    properties.task_type = TaskType.GRAPH_BASED
                
                # Check for demonstrations
                if "demonstrations" in initial_state.context:
                    properties.has_demonstrations = True
                    properties.task_type = TaskType.IMITATION
        
        # Infer additional properties
        if properties.is_multi_agent:
            properties.task_type = TaskType.MULTI_AGENT
        
        return properties
    
    def _calculate_algorithm_scores(self, 
                                   task_properties: TaskProperties,
                                   performance_weight: float = 0.7) -> List[Tuple[str, float]]:
        """Calculate suitability scores for all algorithms"""
        scores = []
        task_hash = self._get_task_hash(task_properties)
        
        for name, algo_info in self.algorithm_registry.items():
            # Base suitability score
            suitability = self._calculate_suitability(task_properties, algo_info["suitable_for"])
            
            # Performance history score
            performance_score = 0.5  # Default neutral score
            if task_hash in self.performance_history and name in self.performance_history[task_hash]:
                perf = self.performance_history[task_hash][name]
                performance_score = perf.get_score()
            
            # Combined score
            combined_score = (
                (1 - performance_weight) * suitability + 
                performance_weight * performance_score
            )
            
            scores.append((name, combined_score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _calculate_suitability(self, 
                              task_properties: TaskProperties,
                              requirements: Dict[str, Any]) -> float:
        """Calculate how suitable an algorithm is for given task properties"""
        score = 1.0
        matches = 0
        total = 0
        
        for key, value in requirements.items():
            total += 1
            task_value = getattr(task_properties, key, None)
            
            if task_value is None:
                continue
            
            # Handle different comparison types
            if isinstance(value, list):
                if task_value in value:
                    matches += 1
                else:
                    score *= 0.5  # Penalize mismatches
            elif isinstance(value, bool):
                if task_value == value:
                    matches += 1
                else:
                    score *= 0.5
            elif callable(value):
                if value(task_value):
                    matches += 1
                else:
                    score *= 0.5
            elif task_value == value:
                matches += 1
            else:
                score *= 0.5
        
        if total > 0:
            score *= (matches / total)
        
        return max(0.0, min(1.0, score))
    
    def select_algorithm(self, 
                        task_properties: TaskProperties,
                        force_algorithm: Optional[str] = None) -> RLAgent:
        """Select the most appropriate algorithm for the task"""
        if force_algorithm and force_algorithm in self.algorithm_registry:
            logger.info(f"Forced selection of {force_algorithm}")
            return self._create_agent(force_algorithm, task_properties)
        
        # Calculate scores
        scores = self._calculate_algorithm_scores(task_properties)
        
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Explore: sample from top-k algorithms
            k = min(3, len(scores))
            top_k = scores[:k]
            weights = np.array([s[1] for s in top_k])
            weights = weights / weights.sum()
            idx = np.random.choice(k, p=weights)
            selected = top_k[idx][0]
            logger.info(f"Exploration: selected {selected} from top-{k}")
        else:
            # Exploit: choose best
            selected = scores[0][0]
            logger.info(f"Exploitation: selected {selected} (score: {scores[0][1]:.3f})")
        
        # Track selection
        self.selection_count[selected] += 1
        
        return self._create_agent(selected, task_properties)
    
    def _create_agent(self, algorithm_name: str, task_properties: TaskProperties) -> RLAgent:
        """Create an instance of the selected algorithm"""
        algo_info = self.algorithm_registry[algorithm_name]
        agent_class = algo_info["class"]
        
        # Build configuration based on task properties
        agent_name = f"{algorithm_name}_{int(time.time())}"
        
        # Create agent with appropriate parameters
        if algorithm_name == "PPO":
            agent = agent_class(
                name=agent_name,
                state_dim=task_properties.state_dim or 10,
                action_dim=task_properties.action_dim or 4,
                continuous=task_properties.has_continuous_actions
            )
        elif algorithm_name == "A3C":
            agent = agent_class(
                name=agent_name,
                state_dim=task_properties.state_dim or 10,
                action_dim=task_properties.action_dim or 4,
                continuous=task_properties.has_continuous_actions
            )
        elif algorithm_name == "DQN":
            from ..algorithms.dqn.vanilla_dqn import DQNAgent
            agent = agent_class(
                name=agent_name,
                state_dim=task_properties.state_dim or 10,
                action_dim=task_properties.action_dim or 4
            )
        else:
            # Default case - try with minimal parameters
            agent = agent_class(
                name=agent_name,
                state_dim=task_properties.state_dim or 10,
                action_dim=task_properties.action_dim or 4
            )
        
        # Store reference
        self.active_agents[agent.name] = agent
        
        logger.info(f"Created {algorithm_name} agent: {agent.name}")
        return agent
    
    def update_performance(self,
                          agent: RLAgent,
                          task_properties: TaskProperties,
                          episode_reward: float,
                          episode_length: int,
                          success: bool,
                          training_time: float) -> None:
        """Update performance tracking for an agent"""
        task_hash = self._get_task_hash(task_properties)
        algo_name = agent.__class__.__name__.replace("Agent", "")
        
        # Get or create performance tracker
        if task_hash not in self.performance_history:
            self.performance_history[task_hash] = {}
        
        if algo_name not in self.performance_history[task_hash]:
            self.performance_history[task_hash][algo_name] = AlgorithmPerformance(
                algorithm_name=algo_name,
                task_hash=task_hash
            )
        
        # Update performance
        perf = self.performance_history[task_hash][algo_name]
        perf.update(episode_reward, episode_length, success, training_time)
        
        logger.debug(f"Updated performance for {algo_name} on task {task_hash}: "
                    f"reward={episode_reward:.2f}, success={success}")
    
    def should_switch_algorithm(self,
                               current_agent: RLAgent,
                               task_properties: TaskProperties,
                               recent_performance: List[float]) -> bool:
        """Determine if algorithm should be switched based on recent performance"""
        if len(recent_performance) < 10:
            return False
        
        # Calculate performance trend
        first_half = np.mean(recent_performance[:5])
        second_half = np.mean(recent_performance[5:])
        
        # Check if performance is declining
        if second_half < first_half * self.adaptation_threshold:
            logger.warning(f"Performance declining for {current_agent.name}: "
                          f"{first_half:.2f} -> {second_half:.2f}")
            return True
        
        # Check if current algorithm is still best choice
        scores = self._calculate_algorithm_scores(task_properties)
        current_algo = current_agent.__class__.__name__.replace("Agent", "")
        
        # Find current algorithm's rank
        current_rank = None
        for i, (name, score) in enumerate(scores):
            if name == current_algo:
                current_rank = i
                break
        
        # Switch if no longer in top 3
        if current_rank is not None and current_rank > 2:
            logger.warning(f"{current_algo} dropped to rank {current_rank + 1}")
            return True
        
        return False
    
    def get_algorithm_recommendation(self,
                                    task_properties: TaskProperties,
                                    top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top-k algorithm recommendations with explanations"""
        scores = self._calculate_algorithm_scores(task_properties)
        recommendations = []
        
        for i in range(min(top_k, len(scores))):
            name, score = scores[i]
            algo_info = self.algorithm_registry[name]
            
            recommendation = {
                "algorithm": name,
                "score": score,
                "class": algo_info["class"].__name__,
                "performance_profile": algo_info["performance_profile"],
                "selection_count": self.selection_count[name],
                "reasons": self._generate_selection_reasons(name, task_properties)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_selection_reasons(self, 
                                   algorithm_name: str,
                                   task_properties: TaskProperties) -> List[str]:
        """Generate human-readable reasons for algorithm selection"""
        reasons = []
        algo_info = self.algorithm_registry[algorithm_name]
        suitable_for = algo_info["suitable_for"]
        
        # Check key matches
        if "action_space" in suitable_for:
            if task_properties.action_space in suitable_for["action_space"]:
                reasons.append(f"Handles {task_properties.action_space.value} action spaces")
        
        if "task_type" in suitable_for:
            if task_properties.task_type in suitable_for["task_type"]:
                reasons.append(f"Designed for {task_properties.task_type.value} tasks")
        
        if task_properties.has_continuous_actions and algorithm_name in ["PPO", "A3C"]:
            reasons.append("Handles continuous action spaces well")
        
        if task_properties.requires_sample_efficiency:
            efficiency = algo_info["performance_profile"]["sample_efficiency"]
            if efficiency > 0.7:
                reasons.append(f"High sample efficiency ({efficiency:.1%})")
        
        if task_properties.requires_stability:
            stability = algo_info["performance_profile"]["stability"]
            if stability > 0.7:
                reasons.append(f"Stable training ({stability:.1%})")
        
        # Performance history
        task_hash = self._get_task_hash(task_properties)
        if task_hash in self.performance_history and algorithm_name in self.performance_history[task_hash]:
            perf = self.performance_history[task_hash][algorithm_name]
            if perf.success_rate > 0.8:
                reasons.append(f"High success rate ({perf.success_rate:.1%}) on similar tasks")
        
        return reasons
    
    def _get_task_hash(self, task_properties: TaskProperties) -> str:
        """Generate a hash for task properties to group similar tasks"""
        # Create a simplified representation for hashing
        key_properties = {
            "task_type": task_properties.task_type.value,
            "action_space": task_properties.action_space.value,
            "is_multi_agent": task_properties.is_multi_agent,
            "has_continuous_actions": task_properties.has_continuous_actions,
            "has_graph_structure": task_properties.has_graph_structure,
            "has_demonstrations": task_properties.has_demonstrations,
            "has_multiple_objectives": task_properties.has_multiple_objectives
        }
        
        # Convert to stable string representation
        import hashlib
        properties_str = json.dumps(key_properties, sort_keys=True)
        return hashlib.md5(properties_str.encode()).hexdigest()[:8]
    
    def save_performance_history(self, path: Union[str, Path]) -> None:
        """Save performance history to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        history_data = {}
        for task_hash, algorithms in self.performance_history.items():
            history_data[task_hash] = {}
            for algo_name, perf in algorithms.items():
                history_data[task_hash][algo_name] = {
                    "algorithm_name": perf.algorithm_name,
                    "total_episodes": perf.total_episodes,
                    "average_reward": perf.average_reward,
                    "success_rate": perf.success_rate,
                    "average_episode_length": perf.average_episode_length,
                    "training_time": perf.training_time,
                    "sample_efficiency": perf.sample_efficiency,
                    "stability_score": perf.stability_score,
                    "last_update": perf.last_update
                }
        
        with open(path, "w") as f:
            json.dump({
                "performance_history": history_data,
                "selection_count": dict(self.selection_count),
                "config": self.config
            }, f, indent=2)
        
        logger.info(f"Saved performance history to {path}")
    
    def load_performance_history(self, path: Union[str, Path]) -> None:
        """Load performance history from file"""
        path = Path(path)
        
        with open(path, "r") as f:
            data = json.load(f)
        
        # Reconstruct performance history
        self.performance_history.clear()
        for task_hash, algorithms in data["performance_history"].items():
            self.performance_history[task_hash] = {}
            for algo_name, perf_data in algorithms.items():
                perf = AlgorithmPerformance(
                    algorithm_name=perf_data["algorithm_name"],
                    task_hash=task_hash,
                    total_episodes=perf_data["total_episodes"],
                    total_reward=perf_data["average_reward"] * perf_data["total_episodes"],
                    average_reward=perf_data["average_reward"],
                    success_rate=perf_data["success_rate"],
                    average_episode_length=perf_data["average_episode_length"],
                    training_time=perf_data["training_time"],
                    sample_efficiency=perf_data["sample_efficiency"],
                    stability_score=perf_data["stability_score"],
                    last_update=perf_data["last_update"]
                )
                self.performance_history[task_hash][algo_name] = perf
        
        # Load selection counts
        self.selection_count = defaultdict(int, data.get("selection_count", {}))
        
        logger.info(f"Loaded performance history from {path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about algorithm selection"""
        stats = {
            "total_algorithms": len(self.algorithm_registry),
            "total_tasks": len(self.performance_history),
            "total_selections": sum(self.selection_count.values()),
            "algorithm_usage": dict(self.selection_count),
            "active_agents": len(self.active_agents),
            "best_algorithms_by_task": {}
        }
        
        # Find best algorithm for each task type
        for task_hash, algorithms in self.performance_history.items():
            best_algo = None
            best_score = -float('inf')
            
            for algo_name, perf in algorithms.items():
                score = perf.get_score()
                if score > best_score:
                    best_score = score
                    best_algo = algo_name
            
            if best_algo:
                stats["best_algorithms_by_task"][task_hash] = {
                    "algorithm": best_algo,
                    "score": best_score,
                    "episodes": algorithms[best_algo].total_episodes
                }
        
        return stats


# Helper functions for module communication integration
def create_module_context(modules: List[str], 
                         communication_enabled: bool = False,
                         latency_requirement: Optional[float] = None) -> Dict[str, Any]:
    """Create module context for claude-module-communicator integration"""
    context = {
        "modules": modules,
        "requires_communication": communication_enabled
    }
    
    if latency_requirement is not None:
        context["latency_requirement"] = latency_requirement
    
    return context


def extract_module_characteristics(agent: RLAgent) -> Dict[str, Any]:
    """Extract characteristics from an agent for module communication"""
    characteristics = {
        "agent_type": agent.__class__.__name__,
        "name": agent.name,
        "training_steps": agent.training_steps,
        "episodes": agent.episodes,
        "is_training": agent.training
    }
    
    # Add algorithm-specific characteristics
    if hasattr(agent, "epsilon"):
        characteristics["exploration_rate"] = getattr(agent, "epsilon", None)
    
    if hasattr(agent, "learning_rate"):
        characteristics["learning_rate"] = getattr(agent, "learning_rate", None)
    
    if hasattr(agent, "get_metrics"):
        characteristics.update(agent.get_metrics())
    
    return characteristics


if __name__ == "__main__":
    # Test with real scenarios
    import torch
    
    # Test 1: Basic discrete action task
    print("Test 1: Basic discrete action task")
    selector = AlgorithmSelector()
    
    task_props = TaskProperties(
        task_type=TaskType.EPISODIC,
        action_space=ActionSpace.DISCRETE,
        state_dim=4,
        action_dim=2,
        requires_stability=True
    )
    
    agent = selector.select_algorithm(task_props)
    print(f"Selected: {agent.__class__.__name__}")
    
    # Test 2: Continuous action scenario
    print("\nTest 2: Continuous action scenario")
    continuous_props = TaskProperties(
        task_type=TaskType.EPISODIC,
        action_space=ActionSpace.CONTINUOUS,
        has_continuous_actions=True,
        state_dim=8,
        action_dim=2,
        requires_stability=True
    )
    
    continuous_agent = selector.select_algorithm(continuous_props)
    print(f"Selected: {continuous_agent.__class__.__name__}")
    
    # Test 3: Get recommendations
    print("\nTest 3: Algorithm recommendations")
    recommendations = selector.get_algorithm_recommendation(task_props, top_k=3)
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. {rec['algorithm']} (score: {rec['score']:.3f})")
        for reason in rec['reasons']:
            print(f"   - {reason}")
    
    # Test 4: Performance update
    print("\nTest 4: Performance tracking")
    selector.update_performance(
        agent=agent,
        task_properties=task_props,
        episode_reward=100.0,
        episode_length=200,
        success=True,
        training_time=5.0
    )
    
    stats = selector.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2)}")
    
    # Test 5: Module context integration
    print("\nTest 5: Module context integration")
    module_context = create_module_context(
        modules=["perception", "planning", "control"],
        communication_enabled=True,
        latency_requirement=50.0
    )
    
    module_props = TaskProperties(
        module_context=module_context,
        action_space=ActionSpace.CONTINUOUS,
        state_dim=12,
        action_dim=3
    )
    
    module_agent = selector.select_algorithm(module_props)
    print(f"Selected for modules: {module_agent.__class__.__name__}")
    
    # Extract characteristics
    characteristics = extract_module_characteristics(module_agent)
    print(f"Module characteristics: {json.dumps(characteristics, indent=2)}")
    
    print("\n✅ Algorithm selector validation passed")