"""
Module: integration_benchmarks.py
Purpose: Benchmarks for module integrations and real-world scenarios

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- asyncio: For async operations

Example Usage:
>>> from rl_commons.benchmarks import IntegrationBenchmark
>>> benchmark = IntegrationBenchmark('module_orchestration')
>>> result = benchmark.execute()
"""

import time
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict
import random

from .benchmark_suite import Benchmark, BenchmarkResult
from ..integrations.module_communicator import (
    ModuleCommunicatorIntegration,
    ModuleState,
    ModuleRequest,
    ModuleOrchestrator
)
from ..integrations.arangodb_optimizer import ArangoDBOptimizer
from ..core.base import RLState, RLAction, RLReward

logger = logging.getLogger(__name__)


class IntegrationBenchmark(Benchmark):
    """Base class for integration benchmarks"""
    
    def __init__(self, integration_name: str, description: str):
        super().__init__(
            name=f"integration_{integration_name}",
            description=description
        )
        self.integration_name = integration_name


class ModuleOrchestrationBenchmark(IntegrationBenchmark):
    """Benchmark for module orchestration performance"""
    
    def __init__(self, 
                 num_modules: int = 5,
                 num_requests: int = 1000,
                 use_multi_objective: bool = False):
        super().__init__(
            "module_orchestration",
            f"Module orchestration with {num_modules} modules"
        )
        self.num_modules = num_modules
        self.num_requests = num_requests
        self.use_multi_objective = use_multi_objective
        
        self.integration = None
        self.orchestrator = None
        self.request_results = []
    
    def setup(self) -> None:
        """Setup module integration"""
        self.integration = ModuleCommunicatorIntegration()
        
        # Register modules
        module_types = ["processor", "database", "llm", "cache", "api"]
        for i in range(self.num_modules):
            module_id = f"module_{i}"
            module_type = module_types[i % len(module_types)]
            
            self.integration.register_module(
                module_id=module_id,
                module_type=module_type,
                capabilities={
                    "operations": [f"op_{j}" for j in range(3)],
                    "max_load": 100,
                    "latency_ms": random.randint(10, 100)
                }
            )
        
        # Create orchestrator
        module_ids = [f"module_{i}" for i in range(self.num_modules)]
        self.orchestrator = self.integration.create_orchestrator(
            modules=module_ids,
            use_multi_objective=self.use_multi_objective
        )
        
        # Initialize module states
        for module_id in module_ids:
            state = ModuleState(
                module_id=module_id,
                module_type="test",
                status="idle",
                load=random.uniform(0.1, 0.5),
                queue_length=random.randint(0, 10),
                average_latency=random.uniform(10, 50),
                error_rate=random.uniform(0, 0.05)
            )
            self.orchestrator.update_module_state(module_id, state)
    
    def run(self) -> Dict[str, float]:
        """Run orchestration benchmark"""
        start_time = time.time()
        
        # Track metrics
        routing_times = []
        successful_routes = 0
        load_balance_scores = []
        rebalance_count = 0
        
        for i in range(self.num_requests):
            # Create request
            request = self._generate_request(i)
            
            # Time routing decision
            route_start = time.time()
            selected_module = self.orchestrator.route_request(request)
            routing_time = time.time() - route_start
            routing_times.append(routing_time)
            
            if selected_module:
                successful_routes += 1
                
                # Simulate processing
                success = random.random() > 0.1  # 90% success rate
                latency = random.uniform(10, 100)
                
                # Report outcome
                self.orchestrator.report_outcome(
                    request=request,
                    module_id=selected_module,
                    success=success,
                    latency=latency
                )
                
                # Update module state (simulate load changes)
                self._update_module_loads(selected_module)
            
            # Check for rebalancing
            if self.orchestrator.should_rebalance():
                rebalance_count += 1
                self._perform_rebalancing()
            
            # Calculate load balance
            if i % 100 == 0:
                load_balance = self._calculate_load_balance()
                load_balance_scores.append(load_balance)
        
        duration = time.time() - start_time
        
        # Calculate final metrics
        metrics = {
            'total_requests': self.num_requests,
            'successful_routes': successful_routes,
            'routing_success_rate': successful_routes / self.num_requests,
            'average_routing_time_ms': np.mean(routing_times) * 1000,
            'throughput_rps': self.num_requests / duration,
            'load_balance_score': np.mean(load_balance_scores),
            'rebalance_operations': rebalance_count,
            'orchestrator_efficiency': self._calculate_efficiency()
        }
        
        return metrics
    
    def teardown(self) -> None:
        """Cleanup"""
        self.integration = None
        self.orchestrator = None
        self.request_results = []
    
    def _generate_request(self, index: int) -> ModuleRequest:
        """Generate a test request"""
        request_types = ["process", "query", "compute", "store", "fetch"]
        
        return ModuleRequest(
            request_id=f"req_{index}",
            request_type=random.choice(request_types),
            payload={"data": f"test_data_{index}"},
            requirements={"operations": [f"op_{random.randint(0, 2)}"]},
            priority=random.randint(0, 5)
        )
    
    def _update_module_loads(self, selected_module: str) -> None:
        """Update module loads after routing"""
        # Increase load for selected module
        current_state = self.orchestrator.module_states.get(selected_module)
        if current_state:
            new_load = min(current_state.load + 0.1, 0.95)
            current_state.load = new_load
            self.orchestrator.update_module_state(selected_module, current_state)
        
        # Randomly decrease load for other modules
        for module_id, state in self.orchestrator.module_states.items():
            if module_id != selected_module and random.random() > 0.7:
                state.load = max(state.load - 0.05, 0.1)
                self.orchestrator.update_module_state(module_id, state)
    
    def _perform_rebalancing(self) -> None:
        """Simulate rebalancing operation"""
        # Reset high-load modules
        for module_id, state in self.orchestrator.module_states.items():
            if state.load > 0.8:
                state.load = 0.5
                state.queue_length = max(0, state.queue_length - 5)
                self.orchestrator.update_module_state(module_id, state)
    
    def _calculate_load_balance(self) -> float:
        """Calculate load balance score (1.0 = perfect balance)"""
        loads = [state.load for state in self.orchestrator.module_states.values()]
        if not loads:
            return 0.0
        
        # Lower std deviation = better balance
        std_dev = np.std(loads)
        return 1.0 / (1.0 + std_dev)
    
    def _calculate_efficiency(self) -> float:
        """Calculate orchestrator efficiency"""
        perf_tracker = self.orchestrator.performance_tracker
        
        total_requests = sum(p['requests'] for p in perf_tracker.values())
        total_successes = sum(p['successes'] for p in perf_tracker.values())
        
        if total_requests == 0:
            return 0.0
        
        success_rate = total_successes / total_requests
        
        # Factor in load balance
        final_load_balance = self._calculate_load_balance()
        
        # Efficiency combines success rate and load balance
        return success_rate * 0.7 + final_load_balance * 0.3


class ClaudeMaxProxyBenchmark(IntegrationBenchmark):
    """Benchmark for Claude Max Proxy integration"""
    
    def __init__(self, num_providers: int = 4, num_requests: int = 500):
        super().__init__(
            "claude_max_proxy",
            f"LLM provider selection with {num_providers} providers"
        )
        self.num_providers = num_providers
        self.num_requests = num_requests
        self.provider_stats = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'total_latency': 0,
            'total_cost': 0
        })
    
    def setup(self) -> None:
        """Setup provider simulation"""
        from ..algorithms.morl import WeightedSumMORL
        
        # Setup MORL agent for provider selection
        self.agent = WeightedSumMORL(
            num_objectives=3,  # cost, quality, latency
            state_dim=10,
            action_dim=self.num_providers
        )
        
        # Provider characteristics
        self.providers = [
            {"name": "gpt-4", "cost": 0.03, "quality": 0.95, "base_latency": 500},
            {"name": "claude-3", "cost": 0.02, "quality": 0.93, "base_latency": 400},
            {"name": "llama-70b", "cost": 0.01, "quality": 0.85, "base_latency": 300},
            {"name": "mixtral", "cost": 0.005, "quality": 0.80, "base_latency": 250}
        ][:self.num_providers]
    
    def run(self) -> Dict[str, float]:
        """Run provider selection benchmark"""
        preference_schedules = [
            np.array([0.33, 0.34, 0.33]),  # Balanced
            np.array([0.7, 0.2, 0.1]),      # Cost-focused
            np.array([0.1, 0.8, 0.1]),      # Quality-focused
            np.array([0.2, 0.3, 0.5])       # Speed-focused
        ]
        
        selection_times = []
        total_cost = 0
        total_quality = 0
        total_latency = 0
        
        for i in range(self.num_requests):
            # Vary preferences
            preference = preference_schedules[i % len(preference_schedules)]
            
            # Create request context
            request_features = self._generate_request_features()
            state = RLState(features=request_features)
            
            # Time selection
            select_start = time.time()
            action = self.agent.select_action(state, preference)
            selection_time = time.time() - select_start
            selection_times.append(selection_time)
            
            # Get selected provider
            provider_idx = action.action_id
            provider = self.providers[provider_idx]
            
            # Simulate request execution
            success, cost, quality, latency = self._simulate_request(
                provider, request_features
            )
            
            # Update stats
            self.provider_stats[provider['name']]['requests'] += 1
            if success:
                self.provider_stats[provider['name']]['successes'] += 1
            self.provider_stats[provider['name']]['total_latency'] += latency
            self.provider_stats[provider['name']]['total_cost'] += cost
            
            total_cost += cost
            total_quality += quality
            total_latency += latency
            
            # Update agent
            from ..algorithms.morl import MultiObjectiveReward
            reward = MultiObjectiveReward([
                -cost,           # Minimize cost
                quality,         # Maximize quality
                -latency/1000   # Minimize latency (in seconds)
            ])
            
            next_state = RLState(features=self._generate_request_features())
            self.agent.update(state, action, reward, next_state, done=True)
        
        # Calculate metrics
        metrics = {
            'average_selection_time_ms': np.mean(selection_times) * 1000,
            'total_cost': total_cost,
            'average_quality': total_quality / self.num_requests,
            'average_latency_ms': total_latency / self.num_requests,
            'cost_per_request': total_cost / self.num_requests,
            'pareto_solutions': len(self.agent.pareto_front.solutions),
            'provider_distribution': self._calculate_provider_distribution()
        }
        
        return metrics
    
    def teardown(self) -> None:
        """Cleanup"""
        self.provider_stats.clear()
        self.agent = None
    
    def _generate_request_features(self) -> np.ndarray:
        """Generate request features"""
        return np.array([
            random.uniform(0, 1),      # Prompt complexity
            random.uniform(0, 1),      # Required quality
            random.uniform(0, 1),      # Time sensitivity
            random.uniform(0, 1),      # Cost sensitivity
            random.randint(0, 1),      # Requires JSON
            random.randint(0, 1),      # Requires streaming
            random.uniform(0, 1),      # Context length
            random.uniform(0, 1),      # User priority
            random.uniform(0, 1),      # Current load
            time.time() % 86400 / 86400  # Time of day
        ])
    
    def _simulate_request(self, provider: Dict, 
                         features: np.ndarray) -> Tuple[bool, float, float, float]:
        """Simulate request execution"""
        # Base values
        base_cost = provider['cost']
        base_quality = provider['quality']
        base_latency = provider['base_latency']
        
        # Add variability based on request
        complexity = features[0]
        cost = base_cost * (1 + complexity * 0.5)
        quality = base_quality * (0.9 + random.uniform(0, 0.1))
        latency = base_latency * (1 + complexity * 0.3) + random.uniform(-50, 50)
        
        # Success depends on quality and load
        success = random.random() < (0.95 * quality)
        
        return success, cost, quality, latency
    
    def _calculate_provider_distribution(self) -> Dict[str, float]:
        """Calculate provider usage distribution"""
        total_requests = sum(stats['requests'] for stats in self.provider_stats.values())
        
        distribution = {}
        for provider, stats in self.provider_stats.items():
            distribution[provider] = stats['requests'] / total_requests if total_requests > 0 else 0
        
        return distribution


class ArangoDBOptimizationBenchmark(IntegrationBenchmark):
    """Benchmark for ArangoDB query optimization"""
    
    def __init__(self, num_queries: int = 1000):
        super().__init__(
            "arangodb_optimization",
            "Query optimization for ArangoDB"
        )
        self.num_queries = num_queries
        self.optimizer = None
        self.query_types = [
            "simple_filter",
            "multi_collection_join",
            "graph_traversal",
            "aggregation",
            "geo_query"
        ]
    
    def setup(self) -> None:
        """Setup optimizer"""
        self.optimizer = ArangoDBOptimizer()
    
    def run(self) -> Dict[str, float]:
        """Run query optimization benchmark"""
        optimization_times = []
        baseline_times = []
        optimized_times = []
        improvement_ratios = []
        
        for i in range(self.num_queries):
            # Generate query
            query_info = self._generate_query_info()
            
            # Time optimization decision
            opt_start = time.time()
            strategy = self.optimizer.optimize_query(query_info)
            opt_time = time.time() - opt_start
            optimization_times.append(opt_time)
            
            # Simulate query execution
            baseline_time = self._simulate_baseline_execution(query_info)
            optimized_time = self._simulate_optimized_execution(query_info, strategy)
            
            baseline_times.append(baseline_time)
            optimized_times.append(optimized_time)
            
            # Calculate improvement
            if baseline_time > 0:
                improvement = (baseline_time - optimized_time) / baseline_time
                improvement_ratios.append(improvement)
            
            # Update optimizer
            self.optimizer.update_from_execution(
                query_info=query_info,
                strategy=strategy,
                execution_time=optimized_time,
                result_count=random.randint(10, 10000)
            )
        
        # Calculate metrics
        metrics = {
            'average_optimization_time_ms': np.mean(optimization_times) * 1000,
            'average_baseline_time_ms': np.mean(baseline_times),
            'average_optimized_time_ms': np.mean(optimized_times),
            'average_improvement': np.mean(improvement_ratios),
            'max_improvement': np.max(improvement_ratios) if improvement_ratios else 0,
            'optimization_overhead_ratio': np.mean(optimization_times) * 1000 / np.mean(baseline_times)
        }
        
        return metrics
    
    def teardown(self) -> None:
        """Cleanup"""
        self.optimizer = None
    
    def _generate_query_info(self) -> Dict[str, Any]:
        """Generate query information"""
        query_type = random.choice(self.query_types)
        
        base_info = {
            'query_type': query_type,
            'collection_size': random.randint(1000, 1000000),
            'filters': random.randint(0, 5),
            'join_count': 0,
            'uses_index': random.random() > 0.5,
            'result_size_estimate': random.randint(10, 10000),
            'query_complexity': random.randint(1, 10),
            'is_graph_query': False,
            'traversal_depth': 0
        }
        
        # Adjust based on query type
        if query_type == "multi_collection_join":
            base_info['join_count'] = random.randint(1, 4)
        elif query_type == "graph_traversal":
            base_info['is_graph_query'] = True
            base_info['traversal_depth'] = random.randint(1, 5)
        
        return base_info
    
    def _simulate_baseline_execution(self, query_info: Dict[str, Any]) -> float:
        """Simulate baseline query execution time (ms)"""
        # Base time depends on query type
        base_times = {
            "simple_filter": 10,
            "multi_collection_join": 50,
            "graph_traversal": 100,
            "aggregation": 30,
            "geo_query": 40
        }
        
        base_time = base_times.get(query_info['query_type'], 20)
        
        # Adjust for collection size
        size_factor = np.log10(query_info['collection_size']) / 3
        
        # Add complexity factor
        complexity_factor = query_info['query_complexity'] / 5
        
        # Calculate total time with some randomness
        total_time = base_time * size_factor * (1 + complexity_factor)
        total_time *= random.uniform(0.8, 1.2)  # Add variability
        
        return total_time
    
    def _simulate_optimized_execution(self, query_info: Dict[str, Any], 
                                    strategy: str) -> float:
        """Simulate optimized query execution time (ms)"""
        baseline = self._simulate_baseline_execution(query_info)
        
        # Optimization effectiveness depends on strategy
        optimization_factors = {
            "index_scan": 0.3,      # 70% improvement
            "hash_join": 0.5,       # 50% improvement
            "parallel_scan": 0.7,   # 30% improvement
            "cached_result": 0.1,   # 90% improvement
            "optimized_traversal": 0.6  # 40% improvement
        }
        
        factor = optimization_factors.get(strategy, 0.9)
        
        # Apply optimization
        optimized_time = baseline * factor
        
        # Add optimization overhead
        optimized_time += random.uniform(0.5, 2.0)  # Small overhead
        
        return optimized_time


# Validation
if __name__ == "__main__":
    print("Testing Integration Benchmarks...")
    
    # Test module orchestration
    orch_benchmark = ModuleOrchestrationBenchmark(
        num_modules=3,
        num_requests=100
    )
    orch_result = orch_benchmark.execute()
    
    assert 'routing_success_rate' in orch_result.metrics
    assert orch_result.metrics['total_requests'] == 100
    
    # Test Claude Max Proxy
    proxy_benchmark = ClaudeMaxProxyBenchmark(
        num_providers=3,
        num_requests=50
    )
    proxy_result = proxy_benchmark.execute()
    
    assert 'average_quality' in proxy_result.metrics
    assert 'provider_distribution' in proxy_result.metrics
    
    # Test ArangoDB optimization
    db_benchmark = ArangoDBOptimizationBenchmark(num_queries=50)
    db_result = db_benchmark.execute()
    
    assert 'average_improvement' in db_result.metrics
    assert db_result.metrics['average_optimized_time_ms'] > 0
    
    print(" Integration benchmarks validation passed")