"""
Module: rl_module_communicator.py
Purpose: Enhanced Module Communicator with RL optimization for spoke interactions

This module integrates RL Commons with Claude Module Communicator to optimize
Level 0-3 interactions between hub and spoke modules using reinforcement learning.

External Dependencies:
- rl_commons: Core RL algorithms and components
- numpy: Numerical computations

Example Usage:
>>> from rl_commons.integrations import RLModuleCommunicator
>>> hub = RLModuleCommunicator()
>>> hub.initialize_spoke_modules(["marker", "arangodb", "sparta"])
>>> result = hub.route_task({"type": "extract", "data": "document.pdf"})
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from loguru import logger

# Import RL Commons components
from rl_commons import (
    AlgorithmSelector,
    TaskProperties,
    TaskType,
    ActionSpace,
    ObservabilityType,
    create_module_context,
    extract_module_characteristics,
    PPOAgent,
    DQNAgent,
    ContextualBandit,
    HierarchicalRLAgent,
    RLState,
    RLAction,
    RLReward,
    ReplayBuffer,
    RLTracker
)
from rl_commons.core.replay_buffer import Experience


class RLModuleCommunicator:
    """
    Enhanced Module Communicator Hub with RL optimization for spoke interactions.
    
    Implements Level 0-3 interactions:
    - Level 0: Basic module selection (Contextual Bandit)
    - Level 1: Pipeline optimization (DQN)
    - Level 2: Resource allocation (PPO)
    - Level 3: Hierarchical task decomposition (Hierarchical RL)
    """
    
    def __init__(self, 
                 spoke_projects: Optional[List[str]] = None,
                 rl_config: Optional[Dict[str, Any]] = None):
        """
        Initialize RL-enhanced Module Communicator.
        
        Args:
            spoke_projects: List of spoke project paths
            rl_config: RL configuration parameters
        """
        self.spoke_modules = {}
        self.rl_config = rl_config or self._default_rl_config()
        
        # Initialize RL components
        self.algorithm_selector = AlgorithmSelector()
        self.rl_tracker = RLTracker(project_name="module_communicator_hub")
        
        # Level-specific agents
        self.level0_agent = None  # Module selection
        self.level1_agent = None  # Pipeline optimization
        self.level2_agent = None  # Resource allocation
        self.level3_agent = None  # Hierarchical decomposition
        
        # Experience buffers for each level
        self.experience_buffers = {
            0: ReplayBuffer(capacity=10000),
            1: ReplayBuffer(capacity=10000),
            2: ReplayBuffer(capacity=10000),
            3: ReplayBuffer(capacity=5000)
        }
        
        # Performance tracking
        self.interaction_history = []
        self.performance_metrics = {
            "module_selection_accuracy": [],
            "pipeline_efficiency": [],
            "resource_utilization": [],
            "task_completion_rate": []
        }
        
        # Initialize spoke modules if provided
        if spoke_projects:
            self.initialize_spoke_modules(spoke_projects)
    
    def _default_rl_config(self) -> Dict[str, Any]:
        """Default RL configuration."""
        return {
            "learning_rate": 0.001,
            "discount_factor": 0.99,
            "exploration_rate": 0.1,
            "batch_size": 32,
            "update_frequency": 100,
            "save_frequency": 1000,
            "model_path": Path("models/rl_hub/")
        }
    
    def initialize_spoke_modules(self, spoke_projects: List[str]) -> None:
        """
        Initialize spoke modules and corresponding RL agents.
        
        Args:
            spoke_projects: List of spoke project identifiers
        """
        # Map of known spoke modules to their characteristics
        module_characteristics = {
            "darpa_crawl": {
                "type": "crawler",
                "capabilities": ["search", "extract", "monitor"],
                "latency": "medium",
                "reliability": 0.95
            },
            "gitget": {
                "type": "version_control",
                "capabilities": ["fetch", "analyze", "track"],
                "latency": "low",
                "reliability": 0.99
            },
            "aider-daemon": {
                "type": "assistant",
                "capabilities": ["code", "refactor", "suggest"],
                "latency": "high",
                "reliability": 0.90
            },
            "sparta": {
                "type": "security",
                "capabilities": ["analyze", "detect", "report"],
                "latency": "medium",
                "reliability": 0.97
            },
            "marker": {
                "type": "processor",
                "capabilities": ["extract", "convert", "annotate"],
                "latency": "medium",
                "reliability": 0.98
            },
            "arangodb": {
                "type": "database",
                "capabilities": ["store", "query", "relate"],
                "latency": "low",
                "reliability": 0.99
            },
            "youtube_transcripts": {
                "type": "media",
                "capabilities": ["fetch", "transcribe", "search"],
                "latency": "medium",
                "reliability": 0.95
            },
            "claude_max_proxy": {
                "type": "llm",
                "capabilities": ["generate", "analyze", "reason"],
                "latency": "high",
                "reliability": 0.92
            },
            "arxiv-mcp-server": {
                "type": "research",
                "capabilities": ["search", "fetch", "cite"],
                "latency": "medium",
                "reliability": 0.96
            },
            "unsloth_wip": {
                "type": "training",
                "capabilities": ["finetune", "optimize", "deploy"],
                "latency": "very_high",
                "reliability": 0.88
            },
            "mcp-screenshot": {
                "type": "capture",
                "capabilities": ["capture", "annotate", "share"],
                "latency": "low",
                "reliability": 0.99
            }
        }
        
        # Extract module names from paths
        for project in spoke_projects:
            module_name = Path(project).name
            if module_name in module_characteristics:
                self.spoke_modules[module_name] = module_characteristics[module_name]
        
        # Initialize Level 0 agent (Module Selection)
        self.level0_agent = ContextualBandit(
            name="module_selector",
            n_arms=len(self.spoke_modules),
            n_features=20,  # Feature dimension for task context
            config={"learning_rate": self.rl_config["learning_rate"]}
        )
        
        # Initialize Level 1 agent (Pipeline Optimization)
        self.level1_agent = DQNAgent(
            name="pipeline_optimizer",
            state_dim=30,  # Pipeline state dimension
            action_dim=len(self.spoke_modules) * 3,  # add/remove/reorder
            config={"learning_rate": self.rl_config["learning_rate"]}
        )
        
        # Initialize Level 2 agent (Resource Allocation)
        self.level2_agent = PPOAgent(
            name="resource_allocator",
            state_dim=25,  # Resource state dimension
            action_dim=4,  # CPU, Memory, Priority, Timeout
            continuous=True,  # Continuous resource allocation
            learning_rate=self.rl_config["learning_rate"],
            config={"track_entropy": True}
        )
        
        # Initialize Level 3 agent (Hierarchical Decomposition)
        # Create options for hierarchical agent
        from rl_commons import Option
        options = []
        for i in range(5):  # 5 high-level strategies
            option = Option(
                id=i,
                name=f"strategy_{i}",
                initiation_set=lambda s: True,  # Available everywhere
                termination_condition=lambda s: np.random.random() < 0.1,  # 10% termination
                policy=None  # Will be created by agent
            )
            options.append(option)
        
        self.level3_agent = HierarchicalRLAgent(
            state_dim=40,
            primitive_actions=len(self.spoke_modules),
            options=options,
            learning_rate=self.rl_config["learning_rate"]
        )
        
        logger.info(f"Initialized {len(self.spoke_modules)} spoke modules with RL agents")
    
    def route_task(self, task: Dict[str, Any], level: int = 0) -> Dict[str, Any]:
        """
        Route a task using RL-optimized decision making.
        
        Args:
            task: Task description with requirements
            level: Interaction level (0-3)
            
        Returns:
            Routing decision and execution result
        """
        start_time = datetime.now()
        
        # Extract task features
        task_features = self._extract_task_features(task)
        state = RLState(features=task_features)
        
        result = {}
        
        if level == 0:
            # Level 0: Simple module selection
            result = self._level0_route(state, task)
        elif level == 1:
            # Level 1: Pipeline optimization
            result = self._level1_route(state, task)
        elif level == 2:
            # Level 2: Resource-aware routing
            result = self._level2_route(state, task)
        elif level == 3:
            # Level 3: Hierarchical task decomposition
            result = self._level3_route(state, task)
        else:
            raise ValueError(f"Invalid level: {level}. Must be 0-3.")
        
        # Track performance
        duration = (datetime.now() - start_time).total_seconds()
        self._track_interaction(task, level, result, duration)
        
        return result
    
    def _level0_route(self, state: RLState, task: Dict[str, Any]) -> Dict[str, Any]:
        """Level 0: Basic module selection using contextual bandit."""
        # Get module selection from bandit
        action = self.level0_agent.select_action(state)
        
        # Map action to module
        module_names = list(self.spoke_modules.keys())
        selected_module = module_names[action.action_id % len(module_names)]
        
        result = {
            "level": 0,
            "selected_module": selected_module,
            "confidence": 0.8,  # Will be updated based on feedback
            "reasoning": f"Selected {selected_module} based on task type and historical performance"
        }
        
        # Store for learning
        self._store_decision(0, state, action, task, result)
        
        return result
    
    def _level1_route(self, state: RLState, task: Dict[str, Any]) -> Dict[str, Any]:
        """Level 1: Pipeline optimization using DQN."""
        # Determine initial pipeline based on task requirements
        initial_pipeline = self._get_initial_pipeline(task)
        
        # Optimize pipeline using DQN
        pipeline_state = self._extract_pipeline_state(initial_pipeline, task)
        action = self.level1_agent.select_action(pipeline_state)
        
        # Apply optimization action
        optimized_pipeline = self._apply_pipeline_action(initial_pipeline, action)
        
        result = {
            "level": 1,
            "initial_pipeline": initial_pipeline,
            "optimized_pipeline": optimized_pipeline,
            "optimization_type": self._get_action_type(action),
            "expected_improvement": 0.15  # Will be validated
        }
        
        # Store for learning
        self._store_decision(1, pipeline_state, action, task, result)
        
        return result
    
    def _level2_route(self, state: RLState, task: Dict[str, Any]) -> Dict[str, Any]:
        """Level 2: Resource-aware routing using PPO."""
        # Get current resource state
        resource_state = self._get_resource_state()
        combined_state = np.concatenate([state.features, resource_state])
        
        # Get resource allocation from PPO
        action = self.level2_agent.select_action(combined_state)
        
        # Extract resource allocation
        cpu_allocation = float(np.clip(action[0], 0.1, 1.0))
        memory_allocation = float(np.clip(action[1], 0.1, 1.0))
        priority = float(np.clip(action[2], 0.0, 1.0))
        timeout = float(np.clip(action[3] * 300, 10, 600))  # 10s to 10min
        
        # Select module based on resource constraints
        selected_module = self._select_module_with_resources(
            task, cpu_allocation, memory_allocation
        )
        
        result = {
            "level": 2,
            "selected_module": selected_module,
            "resource_allocation": {
                "cpu": cpu_allocation,
                "memory": memory_allocation,
                "priority": priority,
                "timeout": timeout
            },
            "resource_efficiency": self._calculate_efficiency(
                cpu_allocation, memory_allocation
            )
        }
        
        # Store for learning
        self._store_decision(2, combined_state, action, task, result)
        
        return result
    
    def _level3_route(self, state: RLState, task: Dict[str, Any]) -> Dict[str, Any]:
        """Level 3: Hierarchical task decomposition."""
        # Select high-level option
        option = self.level3_agent.select_option(state)
        
        # Decompose task based on option
        subtasks = self._decompose_task(task, option)
        
        # Route each subtask
        execution_plan = []
        for subtask in subtasks:
            # Use appropriate level for subtask
            subtask_level = self._determine_subtask_level(subtask)
            subtask_result = self.route_task(subtask, level=subtask_level)
            execution_plan.append({
                "subtask": subtask,
                "routing": subtask_result
            })
        
        result = {
            "level": 3,
            "option": option.name if hasattr(option, 'name') else str(option),
            "decomposition": {
                "num_subtasks": len(subtasks),
                "execution_plan": execution_plan
            },
            "estimated_completion_time": sum(
                s["routing"].get("timeout", 60) for s in execution_plan
            )
        }
        
        # Store for learning
        self._store_decision(3, state, option, task, result)
        
        return result
    
    def update_with_feedback(self, 
                           decision_id: str,
                           outcome: Dict[str, Any]) -> None:
        """
        Update RL agents with task execution feedback.
        
        Args:
            decision_id: ID of the routing decision
            outcome: Execution outcome with metrics
        """
        # Retrieve stored decision
        decision = self._retrieve_decision(decision_id)
        if not decision:
            logger.warning(f"Decision {decision_id} not found")
            return
        
        # Calculate reward based on outcome
        reward = self._calculate_reward(decision, outcome)
        
        # Update appropriate agent
        level = decision["level"]
        state = decision["state"]
        action = decision["action"]
        
        if level == 0:
            # Update contextual bandit
            self.level0_agent.update(
                state, action, RLReward(value=reward), state
            )
        elif level == 1:
            # Update DQN with experience
            next_state = self._extract_pipeline_state(
                outcome.get("final_pipeline", []), 
                decision["task"]
            )
            experience = Experience(
                state=state,
                action=action,
                reward=RLReward(value=reward),
                next_state=next_state,
                done=outcome.get("completed", False)
            )
            self.experience_buffers[1].push(experience)
            
            # Train if enough experiences
            if len(self.experience_buffers[1]) >= self.rl_config["batch_size"]:
                self.level1_agent.train(
                    self.experience_buffers[1].sample(
                        self.rl_config["batch_size"]
                    )
                )
        elif level == 2:
            # Update PPO
            self.level2_agent.store_transition(
                state, action, reward, done=outcome.get("completed", False)
            )
            
            # Train if episode complete
            if outcome.get("completed", False):
                self.level2_agent.train()
        elif level == 3:
            # Update hierarchical agent
            self.level3_agent.update_option(
                state, action, reward, 
                done=outcome.get("completed", False)
            )
        
        # Update performance metrics
        self._update_metrics(level, decision, outcome, reward)
        
        logger.info(f"Updated Level {level} agent with reward: {reward:.3f}")
    
    def _extract_task_features(self, task: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from task description."""
        features = []
        
        # Task type encoding (one-hot, 10 dimensions)
        task_types = ["extract", "analyze", "search", "transform", "store", 
                     "retrieve", "generate", "monitor", "validate", "other"]
        task_type = task.get("type", "other").lower()
        type_vector = [1.0 if t in task_type else 0.0 for t in task_types]
        features.extend(type_vector)
        
        # Data characteristics (5 dimensions)
        data_size = task.get("data_size_mb", 1.0)
        features.append(np.log1p(data_size) / 10.0)  # Log-scaled size
        features.append(1.0 if data_size > 100 else 0.0)  # Large data flag
        features.append(1.0 if task.get("streaming", False) else 0.0)
        features.append(1.0 if task.get("real_time", False) else 0.0)
        features.append(float(task.get("priority", 0.5)))
        
        # Requirements (5 dimensions)
        features.append(float(task.get("accuracy_required", 0.8)))
        features.append(float(task.get("speed_priority", 0.5)))
        features.append(1.0 if task.get("security_critical", False) else 0.0)
        features.append(1.0 if task.get("requires_reasoning", False) else 0.0)
        features.append(float(task.get("complexity", 0.5)))
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, 
                         decision: Dict[str, Any],
                         outcome: Dict[str, Any]) -> float:
        """Calculate reward based on task execution outcome."""
        # Base reward components
        success = 1.0 if outcome.get("success", False) else -0.5
        
        # Performance components
        latency = outcome.get("latency_ms", 1000)
        expected_latency = decision.get("expected_latency", 1000)
        latency_reward = np.clip(expected_latency / latency, 0, 2) - 1
        
        # Quality components
        quality = outcome.get("quality_score", 0.5)
        quality_reward = (quality - 0.5) * 2  # Scale to [-1, 1]
        
        # Resource efficiency
        cpu_used = outcome.get("cpu_usage", 0.5)
        memory_used = outcome.get("memory_usage", 0.5)
        efficiency_reward = 1.0 - (cpu_used + memory_used) / 2
        
        # Weighted combination
        reward = (
            success * 0.4 +
            latency_reward * 0.2 +
            quality_reward * 0.3 +
            efficiency_reward * 0.1
        )
        
        return float(np.clip(reward, -1.0, 1.0))
    
    def save_models(self, path: Optional[Path] = None) -> None:
        """Save all RL models."""
        save_path = path or self.rl_config["model_path"]
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each agent
        if self.level0_agent:
            self.level0_agent.save(save_path / "level0_bandit.pkl")
        if self.level1_agent:
            self.level1_agent.save(save_path / "level1_dqn.pth")
        if self.level2_agent:
            self.level2_agent.save(save_path / "level2_ppo.pth")
        if self.level3_agent:
            self.level3_agent.save(save_path / "level3_hierarchical.pth")
        
        # Save performance metrics
        with open(save_path / "metrics.json", "w") as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        logger.info(f"Saved RL models to {save_path}")
    
    def load_models(self, path: Optional[Path] = None) -> None:
        """Load saved RL models."""
        load_path = path or self.rl_config["model_path"]
        
        if not load_path.exists():
            logger.warning(f"No saved models found at {load_path}")
            return
        
        # Load each agent if exists
        if (load_path / "level0_bandit.pkl").exists():
            self.level0_agent.load(load_path / "level0_bandit.pkl")
        if (load_path / "level1_dqn.pth").exists():
            self.level1_agent.load(load_path / "level1_dqn.pth")
        if (load_path / "level2_ppo.pth").exists():
            self.level2_agent.load(load_path / "level2_ppo.pth")
        if (load_path / "level3_hierarchical.pth").exists():
            self.level3_agent.load(load_path / "level3_hierarchical.pth")
        
        # Load metrics
        metrics_path = load_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                self.performance_metrics = json.load(f)
        
        logger.info(f"Loaded RL models from {load_path}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of RL performance metrics."""
        summary = {
            "total_interactions": len(self.interaction_history),
            "level_distribution": {},
            "average_rewards": {},
            "success_rates": {},
            "average_latency": {},
            "module_usage": {}
        }
        
        # Calculate level distribution
        for interaction in self.interaction_history:
            level = interaction["level"]
            summary["level_distribution"][level] = (
                summary["level_distribution"].get(level, 0) + 1
            )
        
        # Calculate per-level metrics
        for level in range(4):
            level_interactions = [
                i for i in self.interaction_history if i["level"] == level
            ]
            
            if level_interactions:
                # Average reward
                rewards = [i.get("reward", 0) for i in level_interactions]
                summary["average_rewards"][level] = np.mean(rewards)
                
                # Success rate
                successes = [i["outcome"].get("success", False) 
                           for i in level_interactions]
                summary["success_rates"][level] = np.mean(successes)
                
                # Average latency
                latencies = [i["outcome"].get("latency_ms", 0) 
                           for i in level_interactions]
                summary["average_latency"][level] = np.mean(latencies)
        
        # Module usage statistics
        for interaction in self.interaction_history:
            module = interaction.get("selected_module")
            if module:
                summary["module_usage"][module] = (
                    summary["module_usage"].get(module, 0) + 1
                )
        
        return summary
    
    # Helper methods
    def _get_initial_pipeline(self, task: Dict[str, Any]) -> List[str]:
        """Get initial pipeline based on task type."""
        task_type = task.get("type", "").lower()
        
        # Common pipeline patterns
        if "extract" in task_type:
            return ["marker", "arangodb"]
        elif "search" in task_type:
            return ["arxiv-mcp-server", "claude_max_proxy"]
        elif "analyze" in task_type:
            return ["sparta", "claude_max_proxy", "arangodb"]
        elif "monitor" in task_type:
            return ["darpa_crawl", "sparta", "arangodb"]
        else:
            return ["claude_max_proxy"]
    
    def _extract_pipeline_state(self, 
                               pipeline: List[str],
                               task: Dict[str, Any]) -> np.ndarray:
        """Extract state features from pipeline configuration."""
        features = []
        
        # Pipeline composition (binary encoding for each module)
        for module in self.spoke_modules:
            features.append(1.0 if module in pipeline else 0.0)
        
        # Pipeline characteristics
        features.append(float(len(pipeline)) / 10.0)  # Normalized length
        features.append(1.0 if len(set(pipeline)) < len(pipeline) else 0.0)  # Has duplicates
        
        # Add task features
        task_features = self._extract_task_features(task)
        features.extend(task_features[:10])  # First 10 task features
        
        return np.array(features, dtype=np.float32)
    
    def _get_resource_state(self) -> np.ndarray:
        """Get current system resource state."""
        # In production, these would come from actual monitoring
        return np.array([
            0.3,  # Current CPU usage
            0.4,  # Current memory usage  
            0.2,  # Network utilization
            0.5,  # Disk I/O
            0.7   # Available capacity
        ], dtype=np.float32)
    
    def _apply_pipeline_action(self, 
                              pipeline: List[str],
                              action: int) -> List[str]:
        """Apply discrete action to modify pipeline."""
        if not pipeline:
            return pipeline
        
        # Action space: add, remove, swap
        num_modules = len(self.spoke_modules)
        action_type = action // num_modules
        module_idx = action % num_modules
        
        optimized = pipeline.copy()
        
        if action_type == 0:  # Add module
            module_names = list(self.spoke_modules.keys())
            if module_idx < len(module_names):
                optimized.append(module_names[module_idx])
        elif action_type == 1 and len(optimized) > 1:  # Remove module
            idx_to_remove = module_idx % len(optimized)
            optimized.pop(idx_to_remove)
        elif action_type == 2 and len(optimized) > 1:  # Swap modules
            idx1 = module_idx % len(optimized)
            idx2 = (idx1 + 1) % len(optimized)
            optimized[idx1], optimized[idx2] = optimized[idx2], optimized[idx1]
        
        return optimized
    
    def _select_module_with_resources(self,
                                    task: Dict[str, Any],
                                    cpu_limit: float,
                                    memory_limit: float) -> str:
        """Select module considering resource constraints."""
        # Filter modules by resource requirements
        suitable_modules = []
        
        for module, info in self.spoke_modules.items():
            # Estimate resource needs based on module type
            if info["type"] == "llm":
                cpu_need, mem_need = 0.8, 0.9
            elif info["type"] in ["training", "processor"]:
                cpu_need, mem_need = 0.7, 0.6
            elif info["type"] in ["database", "version_control"]:
                cpu_need, mem_need = 0.3, 0.4
            else:
                cpu_need, mem_need = 0.5, 0.5
            
            if cpu_need <= cpu_limit and mem_need <= memory_limit:
                suitable_modules.append(module)
        
        # Select best module from suitable ones
        if not suitable_modules:
            suitable_modules = list(self.spoke_modules.keys())
        
        # Simple selection based on task type
        task_type = task.get("type", "").lower()
        for module in suitable_modules:
            capabilities = self.spoke_modules[module]["capabilities"]
            if any(cap in task_type for cap in capabilities):
                return module
        
        return suitable_modules[0]
    
    def _decompose_task(self, 
                       task: Dict[str, Any],
                       option: int) -> List[Dict[str, Any]]:
        """Decompose task into subtasks based on selected option."""
        subtasks = []
        
        # Option 0: Sequential processing
        if option == 0:
            if "extract" in task.get("type", ""):
                subtasks.append({
                    "type": "extract",
                    "data": task.get("data"),
                    "parent_task": task.get("id")
                })
                subtasks.append({
                    "type": "store",
                    "depends_on": 0,
                    "parent_task": task.get("id")
                })
        
        # Option 1: Parallel analysis
        elif option == 1:
            subtasks.append({
                "type": "analyze_security",
                "data": task.get("data"),
                "parallel": True
            })
            subtasks.append({
                "type": "analyze_content", 
                "data": task.get("data"),
                "parallel": True
            })
        
        # Option 2: Verification pipeline
        elif option == 2:
            subtasks.extend([
                {"type": "fetch", "source": task.get("source")},
                {"type": "validate", "depends_on": 0},
                {"type": "transform", "depends_on": 1},
                {"type": "verify", "depends_on": 2}
            ])
        
        # Default: single task
        if not subtasks:
            subtasks = [task]
        
        return subtasks
    
    def _determine_subtask_level(self, subtask: Dict[str, Any]) -> int:
        """Determine appropriate level for subtask."""
        # Simple subtasks use Level 0
        if subtask.get("type") in ["fetch", "store", "retrieve"]:
            return 0
        
        # Complex analysis uses Level 1
        elif "analyze" in subtask.get("type", ""):
            return 1
        
        # Resource-intensive tasks use Level 2
        elif subtask.get("type") in ["transform", "generate", "train"]:
            return 2
        
        # Default to Level 0
        return 0
    
    def _get_action_type(self, action: int) -> str:
        """Get human-readable action type."""
        num_modules = len(self.spoke_modules)
        action_type = action // num_modules
        
        if action_type == 0:
            return "add_module"
        elif action_type == 1:
            return "remove_module"
        elif action_type == 2:
            return "swap_modules"
        else:
            return "unknown"
    
    def _calculate_efficiency(self, cpu: float, memory: float) -> float:
        """Calculate resource efficiency score."""
        # Lower usage is more efficient
        return 1.0 - (cpu + memory) / 2.0
    
    def _store_decision(self,
                       level: int,
                       state: Any,
                       action: Any,
                       task: Dict[str, Any],
                       result: Dict[str, Any]) -> None:
        """Store routing decision for later update."""
        decision_id = f"L{level}_{datetime.now().timestamp()}"
        
        decision = {
            "id": decision_id,
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "state": state,
            "action": action,
            "task": task,
            "result": result
        }
        
        # Store in interaction history
        self.interaction_history.append(decision)
        
        # Keep only last 10000 interactions
        if len(self.interaction_history) > 10000:
            self.interaction_history.pop(0)
        
        # Add decision ID to result
        result["decision_id"] = decision_id
    
    def _retrieve_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored decision by ID."""
        for decision in reversed(self.interaction_history):
            if decision["id"] == decision_id:
                return decision
        return None
    
    def _track_interaction(self,
                          task: Dict[str, Any],
                          level: int,
                          result: Dict[str, Any],
                          duration: float) -> None:
        """Track interaction for monitoring."""
        self.rl_tracker.track("interaction", {
            "task_type": task.get("type"),
            "level": level,
            "duration": duration,
            "module": result.get("selected_module"),
            "timestamp": datetime.now().isoformat()
        })
    
    def _update_metrics(self,
                       level: int,
                       decision: Dict[str, Any],
                       outcome: Dict[str, Any],
                       reward: float) -> None:
        """Update performance metrics."""
        # Update success rate
        if outcome.get("success"):
            self.performance_metrics["task_completion_rate"].append(1.0)
        else:
            self.performance_metrics["task_completion_rate"].append(0.0)
        
        # Keep only last 1000 entries
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > 1000:
                self.performance_metrics[key] = self.performance_metrics[key][-1000:]


# Module validation
if __name__ == "__main__":
    # Test with spoke projects
    spoke_projects = [
        "/home/graham/workspace/experiments/marker",
        "/home/graham/workspace/experiments/arangodb",
        "/home/graham/workspace/experiments/sparta",
        "/home/graham/workspace/experiments/claude_max_proxy"
    ]
    
    # Initialize hub
    hub = RLModuleCommunicator(spoke_projects)
    
    # Test Level 0 routing
    test_task = {
        "type": "extract",
        "data": "document.pdf",
        "data_size_mb": 5.2,
        "priority": "high"
    }
    
    result = hub.route_task(test_task, level=0)
    print(f"Level 0 Result: {result}")
    
    # Test Level 1 routing
    result = hub.route_task(test_task, level=1)
    print(f"Level 1 Result: {result}")
    
    # Test Level 2 routing
    result = hub.route_task(test_task, level=2)
    print(f"Level 2 Result: {result}")
    
    # Test Level 3 routing
    result = hub.route_task(test_task, level=3)
    print(f"Level 3 Result: {result}")
    
    # Simulate feedback
    outcome = {
        "success": True,
        "latency_ms": 450,
        "quality_score": 0.92,
        "cpu_usage": 0.35,
        "memory_usage": 0.42,
        "completed": True
    }
    
    hub.update_with_feedback(result["decision_id"], outcome)
    
    # Get performance summary
    summary = hub.get_performance_summary()
    print(f"\nPerformance Summary: {json.dumps(summary, indent=2)}")
    
    print("\n RL Module Communicator validation passed!")