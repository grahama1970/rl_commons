#!/usr/bin/env python
"""
Script to run RL Commons benchmarks.

Usage:
    python scripts/run_benchmarks.py --suite all
    python scripts/run_benchmarks.py --suite algorithms --algorithms ppo,dqn
    python scripts/run_benchmarks.py --suite integration --benchmarks module_orchestration
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_commons.benchmarks import (
    BenchmarkSuite,
    AlgorithmBenchmark,
    MultiObjectiveBenchmark,
    ScalabilityBenchmark,
    ModuleOrchestrationBenchmark,
    ClaudeMaxProxyBenchmark,
    ArangoDBOptimizationBenchmark,
    PerformanceProfiler
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_algorithm_benchmarks(suite: BenchmarkSuite, algorithms: list, environments: list):
    """Run algorithm benchmarks"""
    logger.info(f"Running algorithm benchmarks for {algorithms} on {environments}")
    
    for algo in algorithms:
        for env in environments:
            benchmark = AlgorithmBenchmark(
                algorithm_name=algo,
                env_name=env,
                num_episodes=100
            )
            suite.add_benchmark(benchmark)
    
    # Add multi-objective benchmark
    mo_benchmark = MultiObjectiveBenchmark(num_objectives=3, num_episodes=50)
    suite.add_benchmark(mo_benchmark)
    
    # Add scalability benchmarks
    for algo in algorithms:
        scale_benchmark = ScalabilityBenchmark(
            algorithm_name=algo,
            state_dims=[10, 50, 100],
            action_dims=[2, 5, 10]
        )
        suite.add_benchmark(scale_benchmark)


def run_integration_benchmarks(suite: BenchmarkSuite, benchmarks: list):
    """Run integration benchmarks"""
    logger.info(f"Running integration benchmarks: {benchmarks}")
    
    benchmark_map = {
        'module_orchestration': lambda: ModuleOrchestrationBenchmark(
            num_modules=5,
            num_requests=500,
            use_multi_objective=True
        ),
        'claude_max_proxy': lambda: ClaudeMaxProxyBenchmark(
            num_providers=4,
            num_requests=300
        ),
        'arangodb_optimization': lambda: ArangoDBOptimizationBenchmark(
            num_queries=500
        )
    }
    
    for benchmark_name in benchmarks:
        if benchmark_name in benchmark_map:
            benchmark = benchmark_map[benchmark_name]()
            suite.add_benchmark(benchmark)
        else:
            logger.warning(f"Unknown benchmark: {benchmark_name}")


def run_profiling(algorithms: list):
    """Run performance profiling"""
    logger.info(f"Running performance profiling for {algorithms}")
    
    profiler = PerformanceProfiler()
    
    # Define training function factory
    def train_fn_factory(algo_name):
        def train_fn(phase):
            if phase == "setup":
                # Create dummy environment and agent
                import gym
                env = gym.make('CartPole-v1')
                
                if algo_name == "ppo":
                    from rl_commons.algorithms import PPOAgent
                    agent = PPOAgent(
                        state_dim=env.observation_space.shape[0],
                        action_dim=env.action_space.n
                    )
                elif algo_name == "dqn":
                    from rl_commons.algorithms import DQNAgent
                    agent = DQNAgent(
                        state_dim=env.observation_space.shape[0],
                        action_dim=env.action_space.n
                    )
                else:
                    from rl_commons.algorithms import PPOAgent
                    agent = PPOAgent(
                        state_dim=env.observation_space.shape[0],
                        action_dim=env.action_space.n
                    )
                
                return agent, env
        return train_fn
    
    # Compare algorithms
    comparison_results = profiler.compare_algorithms(
        algorithms=algorithms,
        train_fn_factory=train_fn_factory,
        episodes=50
    )
    
    # Generate reports
    profiler.generate_report(format="markdown")
    profiler.generate_report(format="html")
    
    # Analyze bottlenecks
    bottlenecks = profiler.analyze_bottlenecks()
    logger.info("Performance bottlenecks:")
    for rec in bottlenecks['recommendations']:
        logger.info(f"  - {rec}")


def main():
    parser = argparse.ArgumentParser(description="Run RL Commons benchmarks")
    parser.add_argument(
        '--suite',
        choices=['all', 'algorithms', 'integration', 'profiling'],
        default='all',
        help='Which benchmark suite to run'
    )
    parser.add_argument(
        '--algorithms',
        type=str,
        default='ppo,dqn',
        help='Comma-separated list of algorithms to benchmark'
    )
    parser.add_argument(
        '--environments',
        type=str,
        default='CartPole-v1,MountainCar-v0',
        help='Comma-separated list of environments'
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        default='module_orchestration,claude_max_proxy,arangodb_optimization',
        help='Comma-separated list of integration benchmarks'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'json', 'both'],
        default='markdown',
        help='Report format'
    )
    
    args = parser.parse_args()
    
    # Parse lists
    algorithms = args.algorithms.split(',')
    environments = args.environments.split(',')
    benchmarks = args.benchmarks.split(',')
    
    # Create benchmark suite
    suite = BenchmarkSuite(output_dir=args.output_dir)
    
    # Run selected benchmarks
    if args.suite in ['all', 'algorithms']:
        run_algorithm_benchmarks(suite, algorithms, environments)
    
    if args.suite in ['all', 'integration']:
        run_integration_benchmarks(suite, benchmarks)
    
    if args.suite in ['all', 'profiling']:
        run_profiling(algorithms)
        return  # Profiling handles its own reporting
    
    # Run benchmarks
    logger.info("Starting benchmark execution...")
    
    def progress_callback(current, total, name):
        logger.info(f"Progress: {current+1}/{total} - Running {name}")
    
    results = suite.run_all_benchmarks(progress_callback=progress_callback)
    
    # Generate reports
    if args.format in ['markdown', 'both']:
        report_path = suite.generate_report(results, format="markdown")
        logger.info(f"Markdown report saved to: {report_path}")
    
    if args.format in ['json', 'both']:
        report_path = suite.generate_report(results, format="json")
        logger.info(f"JSON report saved to: {report_path}")
    
    logger.info("Benchmarking complete!")


if __name__ == "__main__":
    main()