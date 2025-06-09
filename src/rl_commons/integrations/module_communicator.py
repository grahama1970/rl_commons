"""
Module: module_communicator.py
Purpose: Integration layer between RL Commons and claude-module-communicator

This module provides:
1. Real-time module state extraction
2. Automatic algorithm selection based on module characteristics
3. Orchestration policy learning
4. Performance monitoring and adaptation

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- asyncio: For async module communication

Example Usage:
>>> from rl_commons.integrations import ModuleCommunicatorIntegration
>>> integrator = ModuleCommunicatorIntegration()
>>> orchestrator = integrator.create_orchestrator(modules=['marker', 'arangodb', 'chat'])
>>> action = orchestrator.select_routing_action(request)
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import logging

from ..core.base import RLState, RLAction, RLReward, RLAgent
from ..core.algorithm_selector import (
    AlgorithmSelector, TaskProperties, TaskType, ActionSpace,
    create_module_context, extract_module_characteristics
)
from ..algorithms.marl import MARLAgent, MARLEnvironment
from ..algorithms.gnn import GNNIntegration
from ..algorithms.morl import MultiObjectiveReward, WeightedSumMORL
from ..algorithms.curriculum import ProgressiveCurriculum, Task

logger = logging.getLogger(__name__)


@dataclass
class ModuleState:
    """State of a module in the system"""
    module_id: str
    module_type: str  # e.g., 'marker', 'arangodb', 'llm', 'chat'
    status: str = 'idle'  # idle, processing, error, offline
    load: float = 0.0  # 0.0 to 1.0
    queue_length: int = 0
    average_latency: float = 0.0  # milliseconds
    error_rate: float = 0.0  # 0.0 to 1.0
    capabilities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)
    
    def to_features(self) -> np.ndarray:
        """Convert state to feature vector for RL"""
        features = [
            float(self.status == 'idle'),
            float(self.status == 'processing'),
            float(self.status == 'error'),
            self.load,
            min(self.queue_length / 100.0, 1.0),  # Normalize queue length
            min(self.average_latency / 1000.0, 1.0),  # Normalize latency
            self.error_rate
        ]
        return np.array(features, dtype=np.float32)
    
    def is_available(self) -> bool:
        """Check if module is available for processing"""
        return self.status != 'offline' and self.error_rate < 0.5


@dataclass
class ModuleRequest:
    """Request to be routed to modules"""
    request_id: str
    request_type: str  # e.g., 'pdf_processing', 'llm_query', 'data_storage'
    payload: Dict[str, Any]
    requirements: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def matches_capabilities(self, capabilities: Dict[str, Any]) -> bool:
        """Check if request matches module capabilities"""
        for req_key, req_value in self.requirements.items():
            if req_key not in capabilities:
                return False
            if isinstance(req_value, (list, set)):
                if capabilities[req_key] not in req_value:
                    return False
            elif capabilities[req_key] != req_value:
                return False
        return True


class OrchestrationPolicy(ABC):
    """Abstract base class for orchestration policies"""
    
    @abstractmethod
    def select_module(self, 
                     request: ModuleRequest,
                     module_states: Dict[str, ModuleState]) -> Optional[str]:
        """Select module for request"""
        pass
    
    @abstractmethod
    def update(self, 
              request: ModuleRequest,
              selected_module: str,
              outcome: Dict[str, Any]) -> None:
        """Update policy based on outcome"""
        pass


class RLOrchestrationPolicy(OrchestrationPolicy):
    """RL-based orchestration policy"""
    
    def __init__(self, agent: RLAgent, module_ids: List[str]):
        self.agent = agent
        self.module_ids = module_ids
        self.module_to_action = {mid: i for i, mid in enumerate(module_ids)}
        self.action_to_module = {i: mid for i, mid in enumerate(module_ids)}
        self.request_history = deque(maxlen=1000)
    
    def select_module(self,
                     request: ModuleRequest,
                     module_states: Dict[str, ModuleState]) -> Optional[str]:
        """Select module using RL agent"""
        # Create state from request and module states
        state_features = self._create_state_features(request, module_states)
        state = RLState(features=state_features, context={
            'request': request,
            'module_states': module_states
        })
        
        # Get action from agent
        action = self.agent.select_action(state)
        
        # Convert action to module ID
        if action.action_id < len(self.action_to_module):
            selected_module = self.action_to_module[action.action_id]
            
            # Verify module is available
            if selected_module in module_states and module_states[selected_module].is_available():
                return selected_module
        
        # Fallback to least loaded available module
        return self._fallback_selection(request, module_states)
    
    def update(self,
              request: ModuleRequest,
              selected_module: str,
              outcome: Dict[str, Any]) -> None:
        """Update RL agent based on outcome"""
        # Store in history
        self.request_history.append({
            'request': request,
            'module': selected_module,
            'outcome': outcome,
            'timestamp': time.time()
        })
        
        # Create reward based on outcome
        reward_value = self._calculate_reward(outcome)
        reward = RLReward(value=reward_value, context=outcome)
        
        # Update agent if we have previous state
        if hasattr(self, '_last_state') and hasattr(self, '_last_action'):
            # Create next state
            next_state_features = self._create_state_features(request, outcome.get('module_states', {}))
            next_state = RLState(features=next_state_features)
            
            # Update policy
            self.agent.update_policy(
                self._last_state,
                self._last_action,
                reward,
                next_state,
                outcome.get('done', False)
            )
    
    def _create_state_features(self,
                              request: ModuleRequest,
                              module_states: Dict[str, ModuleState]) -> np.ndarray:
        """Create state features from request and module states"""
        features = []
        
        # Request features
        request_type_encoding = hash(request.request_type) % 10 / 10.0
        features.extend([
            request_type_encoding,
            request.priority / 10.0,
            len(request.requirements) / 10.0
        ])
        
        # Module features (fixed order)
        for module_id in self.module_ids:
            if module_id in module_states:
                features.extend(module_states[module_id].to_features())
            else:
                features.extend([0.0] * 7)  # Default features
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, outcome: Dict[str, Any]) -> float:
        """Calculate reward from outcome"""
        # Base reward
        reward = 0.0
        
        # Success/failure
        if outcome.get('success', False):
            reward += 1.0
        else:
            reward -= 1.0
        
        # Latency penalty (normalized)
        latency = outcome.get('latency', 0)
        reward -= min(latency / 1000.0, 1.0) * 0.5
        
        # Error penalty
        if outcome.get('error', False):
            reward -= 0.5
        
        return reward
    
    def _fallback_selection(self,
                           request: ModuleRequest,
                           module_states: Dict[str, ModuleState]) -> Optional[str]:
        """Fallback selection when RL fails"""
        available_modules = [
            (mid, state) for mid, state in module_states.items()
            if state.is_available() and request.matches_capabilities(state.capabilities)
        ]
        
        if not available_modules:
            return None
        
        # Select least loaded
        return min(available_modules, key=lambda x: x[1].load)[0]


class ModuleOrchestrator:
    """Orchestrates requests across multiple modules using RL"""
    
    def __init__(self,
                 module_ids: List[str],
                 algorithm_selector: AlgorithmSelector,
                 use_multi_objective: bool = False):
        self.module_ids = module_ids
        self.algorithm_selector = algorithm_selector
        self.use_multi_objective = use_multi_objective
        
        # Module states
        self.module_states: Dict[str, ModuleState] = {}
        
        # Initialize RL components
        self._initialize_rl_components()
        
        # Performance tracking
        self.performance_tracker = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'total_latency': 0,
            'errors': 0
        })
        
        logger.info(f"Module Orchestrator initialized with {len(module_ids)} modules")
    
    def _initialize_rl_components(self):
        """Initialize RL agent and policy"""
        # Determine task properties
        task_props = TaskProperties(
            task_type=TaskType.EPISODIC,
            action_space=ActionSpace.DISCRETE,
            action_dim=len(self.module_ids),
            state_dim=3 + 7 * len(self.module_ids),  # Request features + module features
            is_multi_agent=len(self.module_ids) > 1,
            has_multiple_objectives=self.use_multi_objective,
            module_context=create_module_context(
                modules=self.module_ids,
                communication_enabled=True
            )
        )
        
        # Select algorithm
        if self.use_multi_objective:
            # Use MORL for balancing latency, throughput, and reliability
            self.agent = WeightedSumMORL(
                num_objectives=3,  # latency, throughput, reliability
                state_dim=task_props.state_dim,
                action_dim=task_props.action_dim
            )
        else:
            # Let selector choose
            self.agent = self.algorithm_selector.select_algorithm(task_props)
        
        # Create orchestration policy
        self.policy = RLOrchestrationPolicy(self.agent, self.module_ids)
    
    def update_module_state(self, module_id: str, state: ModuleState):
        """Update state of a module"""
        self.module_states[module_id] = state
        state.last_update = time.time()
    
    def route_request(self, request: ModuleRequest) -> Optional[str]:
        """Route request to appropriate module"""
        # Select module using policy
        selected_module = self.policy.select_module(request, self.module_states)
        
        if selected_module:
            logger.debug(f"Routing request {request.request_id} to module {selected_module}")
        else:
            logger.warning(f"No suitable module found for request {request.request_id}")
        
        return selected_module
    
    def report_outcome(self,
                      request: ModuleRequest,
                      module_id: str,
                      success: bool,
                      latency: float,
                      error: Optional[str] = None):
        """Report outcome of request processing"""
        outcome = {
            'success': success,
            'latency': latency,
            'error': error is not None,
            'error_message': error,
            'module_states': self.module_states.copy()
        }
        
        # Update policy
        self.policy.update(request, module_id, outcome)
        
        # Update performance tracking
        perf = self.performance_tracker[module_id]
        perf['requests'] += 1
        perf['successes'] += int(success)
        perf['total_latency'] += latency
        perf['errors'] += int(error is not None)
        
        # Update algorithm selector if needed
        if hasattr(self.agent, 'episodes') and self.agent.episodes % 100 == 0:
            avg_reward = self._calculate_average_performance()
            self.algorithm_selector.update_performance(
                agent=self.agent,
                task_properties=self._get_current_task_properties(),
                episode_reward=avg_reward,
                episode_length=100,
                success=avg_reward > 0,
                training_time=time.time() - self.agent.start_time
            )
    
    def should_rebalance(self) -> bool:
        """Check if module allocation should be rebalanced"""
        if not self.module_states:
            return False
        
        # Check load variance
        loads = [state.load for state in self.module_states.values()]
        if np.std(loads) > 0.3:  # High variance
            return True
        
        # Check error rates
        error_rates = [state.error_rate for state in self.module_states.values()]
        if max(error_rates) > 0.2:  # High error rate
            return True
        
        return False
    
    def get_module_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for module configuration"""
        recommendations = {
            'rebalance_needed': self.should_rebalance(),
            'module_health': {},
            'routing_suggestions': {}
        }
        
        for module_id, state in self.module_states.items():
            health_score = (1 - state.error_rate) * (1 - state.load)
            recommendations['module_health'][module_id] = {
                'score': health_score,
                'status': 'healthy' if health_score > 0.7 else 'degraded',
                'load': state.load,
                'error_rate': state.error_rate
            }
        
        return recommendations
    
    def _calculate_average_performance(self) -> float:
        """Calculate average performance across modules"""
        total_score = 0
        total_requests = 0
        
        for module_id, perf in self.performance_tracker.items():
            if perf['requests'] > 0:
                success_rate = perf['successes'] / perf['requests']
                avg_latency = perf['total_latency'] / perf['requests']
                
                # Composite score
                score = success_rate - (avg_latency / 1000.0) * 0.5
                total_score += score * perf['requests']
                total_requests += perf['requests']
        
        return total_score / max(1, total_requests)
    
    def _get_current_task_properties(self) -> TaskProperties:
        """Get current task properties for algorithm selection"""
        return TaskProperties(
            task_type=TaskType.CONTINUOUS,
            action_space=ActionSpace.DISCRETE,
            action_dim=len(self.module_ids),
            state_dim=3 + 7 * len(self.module_ids),
            is_multi_agent=len(self.module_ids) > 1,
            has_multiple_objectives=self.use_multi_objective,
            is_non_stationary=True,  # Module characteristics change
            module_context={
                'modules': self.module_ids,
                'current_load': np.mean([s.load for s in self.module_states.values()])
            }
        )


class ModuleCommunicatorIntegration:
    """Main integration class for claude-module-communicator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.algorithm_selector = AlgorithmSelector(self.config.get('selector_config', {}))
        self.orchestrators: Dict[str, ModuleOrchestrator] = {}
        self.module_registry: Dict[str, Dict[str, Any]] = {}
        
        # Curriculum learning for progressive complexity
        self.curriculum = self._create_curriculum()
        
        logger.info("Module Communicator Integration initialized")
    
    def register_module(self,
                       module_id: str,
                       module_type: str,
                       capabilities: Dict[str, Any],
                       metadata: Optional[Dict[str, Any]] = None):
        """Register a module with the system"""
        self.module_registry[module_id] = {
            'type': module_type,
            'capabilities': capabilities,
            'metadata': metadata or {},
            'registered_at': time.time()
        }
        logger.info(f"Registered module {module_id} of type {module_type}")
    
    def create_orchestrator(self,
                           modules: List[str],
                           use_multi_objective: bool = False) -> ModuleOrchestrator:
        """Create an orchestrator for a set of modules"""
        orchestrator_id = "_".join(sorted(modules))
        
        if orchestrator_id not in self.orchestrators:
            orchestrator = ModuleOrchestrator(
                module_ids=modules,
                algorithm_selector=self.algorithm_selector,
                use_multi_objective=use_multi_objective
            )
            self.orchestrators[orchestrator_id] = orchestrator
            
            # Initialize module states
            for module_id in modules:
                if module_id in self.module_registry:
                    reg = self.module_registry[module_id]
                    state = ModuleState(
                        module_id=module_id,
                        module_type=reg['type'],
                        capabilities=reg['capabilities'],
                        metadata=reg['metadata']
                    )
                    orchestrator.update_module_state(module_id, state)
        
        return self.orchestrators[orchestrator_id]
    
    def create_multi_agent_setup(self, modules: List[str]) -> MARLEnvironment:
        """Create multi-agent setup for distributed learning"""
        # Create MARL environment
        env = MARLEnvironment(num_agents=len(modules))
        
        # Create agents for each module
        agents = {}
        for i, module_id in enumerate(modules):
            task_props = TaskProperties(
                task_type=TaskType.MULTI_AGENT,
                action_space=ActionSpace.DISCRETE,
                action_dim=4,  # Basic routing actions
                state_dim=10,  # Module state features
                is_multi_agent=True,
                requires_communication=True
            )
            
            agent = self.algorithm_selector.select_algorithm(task_props)
            agents[i] = agent
        
        return env, agents
    
    def create_gnn_orchestrator(self, modules: List[str]) -> GNNIntegration:
        """Create GNN-based orchestrator for graph-structured modules"""
        # Build module graph
        num_modules = len(modules)
        node_features = 10  # Features per module
        edge_features = 3   # Connection features
        
        # Create GNN integration
        gnn = GNNIntegration(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=64
        )
        
        # Create GNN-based policy
        gnn_actor = gnn.create_gnn_actor(
            node_features=node_features,
            edge_features=edge_features,
            action_dim=num_modules  # Route to any module
        )
        
        return gnn
    
    def _create_curriculum(self) -> ProgressiveCurriculum:
        """Create curriculum for progressive task complexity"""
        tasks = [
            # Level 1: Single module routing
            Task(
                task_id="single_module",
                difficulty=0.2,
                context={
                    'name': 'Single Module Routing',
                    'modules': ['marker'],
                    'request_types': ['pdf_processing']
                }
            ),
            # Level 2: Two module pipeline
            Task(
                task_id="two_module_pipeline",
                difficulty=0.4,
                prerequisites=["single_module"],
                context={
                    'name': 'Two Module Pipeline',
                    'modules': ['marker', 'arangodb'],
                    'request_types': ['pdf_processing', 'data_storage']
                }
            ),
            # Level 3: Multi-module with load balancing
            Task(
                task_id="load_balancing",
                difficulty=0.6,
                prerequisites=["two_module_pipeline"],
                context={
                    'name': 'Load Balancing',
                    'modules': ['marker', 'marker_backup', 'arangodb'],
                    'request_types': ['pdf_processing', 'data_storage'],
                    'requirements': ['load_balancing']
                }
            ),
            # Level 4: Full orchestration
            Task(
                task_id="full_orchestration",
                difficulty=0.8,
                prerequisites=["load_balancing"],
                context={
                    'name': 'Full Module Orchestration',
                    'modules': ['marker', 'arangodb', 'llm', 'chat'],
                    'request_types': ['pdf_processing', 'data_storage', 'llm_query', 'chat_interaction'],
                    'requirements': ['load_balancing', 'error_handling', 'priority_routing']
                }
            ),
            # Level 5: Adaptive orchestration
            Task(
                task_id="adaptive_orchestration",
                difficulty=1.0,
                prerequisites=["full_orchestration"],
                context={
                    'name': 'Adaptive Orchestration',
                    'modules': ['marker', 'arangodb', 'llm', 'chat', 'vector_db'],
                    'request_types': ['pdf_processing', 'data_storage', 'llm_query', 'chat_interaction', 'similarity_search'],
                    'requirements': ['load_balancing', 'error_handling', 'priority_routing', 'adaptive_routing']
                }
            )
        ]
        
        return ProgressiveCurriculum(
            tasks=tasks,
            performance_threshold=0.7,
            progression_rate=0.05
        )
    
    async def handle_request_async(self,
                                  request: ModuleRequest,
                                  orchestrator: ModuleOrchestrator) -> Dict[str, Any]:
        """Handle request asynchronously"""
        start_time = time.time()
        
        # Route request
        selected_module = orchestrator.route_request(request)
        
        if not selected_module:
            return {
                'success': False,
                'error': 'No suitable module available',
                'latency': (time.time() - start_time) * 1000
            }
        
        try:
            # Simulate async module communication
            # In real implementation, this would call actual module APIs
            await asyncio.sleep(0.1)  # Simulate processing
            
            # Simulate outcome
            success = np.random.random() > 0.1  # 90% success rate
            latency = (time.time() - start_time) * 1000
            
            # Report outcome
            orchestrator.report_outcome(
                request=request,
                module_id=selected_module,
                success=success,
                latency=latency,
                error=None if success else "Processing failed"
            )
            
            return {
                'success': success,
                'module': selected_module,
                'latency': latency,
                'error': None if success else "Processing failed"
            }
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            orchestrator.report_outcome(
                request=request,
                module_id=selected_module,
                success=False,
                latency=latency,
                error=str(e)
            )
            
            return {
                'success': False,
                'module': selected_module,
                'latency': latency,
                'error': str(e)
            }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        stats = {
            'registered_modules': len(self.module_registry),
            'active_orchestrators': len(self.orchestrators),
            'orchestrator_stats': {},
            'algorithm_stats': self.algorithm_selector.get_statistics(),
            'curriculum_progress': {
                'current_task': self.curriculum.current_task.task_id if self.curriculum.current_task else None,
                'completed_tasks': len([t for t, p in self.curriculum.performance.items() if p.success_rate > 0.7])
            }
        }
        
        for orch_id, orchestrator in self.orchestrators.items():
            stats['orchestrator_stats'][orch_id] = {
                'modules': orchestrator.module_ids,
                'total_requests': sum(p['requests'] for p in orchestrator.performance_tracker.values()),
                'average_success_rate': orchestrator._calculate_average_performance(),
                'recommendations': orchestrator.get_module_recommendations()
            }
        
        return stats


# Validation
if __name__ == "__main__":
    import asyncio
    
    # Test integration
    print("Testing Module Communicator Integration...")
    
    # Create integration
    integration = ModuleCommunicatorIntegration()
    
    # Register modules
    integration.register_module(
        module_id="marker",
        module_type="pdf_processor",
        capabilities={"file_types": ["pdf", "docx"], "max_size_mb": 100}
    )
    
    integration.register_module(
        module_id="arangodb",
        module_type="database",
        capabilities={"operations": ["store", "query", "graph"], "max_connections": 100}
    )
    
    integration.register_module(
        module_id="llm",
        module_type="language_model",
        capabilities={"models": ["gpt-4", "claude"], "max_tokens": 4096}
    )
    
    # Create orchestrator
    orchestrator = integration.create_orchestrator(
        modules=["marker", "arangodb", "llm"],
        use_multi_objective=True
    )
    
    # Update module states
    orchestrator.update_module_state("marker", ModuleState(
        module_id="marker",
        module_type="pdf_processor",
        status="idle",
        load=0.3,
        queue_length=2,
        average_latency=150.0,
        error_rate=0.02
    ))
    
    orchestrator.update_module_state("arangodb", ModuleState(
        module_id="arangodb",
        module_type="database",
        status="processing",
        load=0.7,
        queue_length=10,
        average_latency=50.0,
        error_rate=0.01
    ))
    
    orchestrator.update_module_state("llm", ModuleState(
        module_id="llm",
        module_type="language_model",
        status="idle",
        load=0.5,
        queue_length=5,
        average_latency=500.0,
        error_rate=0.05
    ))
    
    # Test routing
    test_request = ModuleRequest(
        request_id="test_001",
        request_type="pdf_processing",
        payload={"file": "test.pdf"},
        requirements={"file_types": ["pdf"]},
        priority=1
    )
    
    selected = orchestrator.route_request(test_request)
    print(f"Selected module: {selected}")
    
    # Simulate outcome
    orchestrator.report_outcome(
        request=test_request,
        module_id=selected,
        success=True,
        latency=123.4
    )
    
    # Test async handling
    async def test_async():
        result = await integration.handle_request_async(test_request, orchestrator)
        print(f"Async result: {result}")
    
    asyncio.run(test_async())
    
    # Get stats
    stats = integration.get_integration_stats()
    print(f"\nIntegration statistics:")
    print(json.dumps(stats, indent=2))
    
    # Test multi-agent setup
    marl_env, agents = integration.create_multi_agent_setup(["marker", "arangodb", "llm"])
    print(f"\nCreated MARL environment with {len(agents)} agents")
    
    # Test GNN orchestrator
    gnn = integration.create_gnn_orchestrator(["marker", "arangodb", "llm"])
    print(f"Created GNN orchestrator")
    
    print("\n Module Communicator Integration validation passed")