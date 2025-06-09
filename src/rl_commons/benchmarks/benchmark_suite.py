"""
Module: benchmark_suite.py
Purpose: Core benchmarking framework for RL Commons

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- time: For performance timing
- psutil: https://psutil.readthedocs.io/

Example Usage:
>>> from rl_commons.benchmarks import BenchmarkSuite
>>> suite = BenchmarkSuite()
>>> results = suite.run_all_benchmarks()
>>> suite.generate_report(results)
"""

import time
import json
import psutil
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    benchmark_name: str
    algorithm: str
    metrics: Dict[str, float]
    duration: float  # seconds
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'benchmark_name': self.benchmark_name,
            'algorithm': self.algorithm,
            'metrics': self.metrics,
            'duration': self.duration,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class Benchmark(ABC):
    """Abstract base class for benchmarks"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.process = psutil.Process()
    
    @abstractmethod
    def setup(self) -> None:
        """Setup benchmark environment"""
        pass
    
    @abstractmethod
    def run(self) -> Dict[str, float]:
        """Run benchmark and return metrics"""
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """Cleanup after benchmark"""
        pass
    
    def execute(self) -> BenchmarkResult:
        """Execute complete benchmark with resource monitoring"""
        # Setup
        self.setup()
        
        # Get initial resource usage
        self.process.cpu_percent()  # First call to initialize
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Run benchmark
        start_time = time.time()
        try:
            metrics = self.run()
        except Exception as e:
            logger.error(f"Benchmark {self.name} failed: {e}")
            metrics = {'error': str(e)}
        duration = time.time() - start_time
        
        # Get resource usage
        cpu_usage = self.process.cpu_percent()
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_usage = final_memory - initial_memory
        
        # Cleanup
        self.teardown()
        
        return BenchmarkResult(
            benchmark_name=self.name,
            algorithm=self.__class__.__name__,
            metrics=metrics,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            metadata={'description': self.description}
        )


class BenchmarkSuite:
    """Suite for running and managing benchmarks"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.benchmarks: List[Benchmark] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Comparison baselines
        self.baselines: Dict[str, BenchmarkResult] = {}
        
        logger.info(f"Benchmark suite initialized with output directory: {self.output_dir}")
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite"""
        self.benchmarks.append(benchmark)
        logger.info(f"Added benchmark: {benchmark.name}")
    
    def run_benchmark(self, benchmark_name: str) -> Optional[BenchmarkResult]:
        """Run a specific benchmark"""
        for benchmark in self.benchmarks:
            if benchmark.name == benchmark_name:
                logger.info(f"Running benchmark: {benchmark_name}")
                return benchmark.execute()
        
        logger.warning(f"Benchmark not found: {benchmark_name}")
        return None
    
    def run_all_benchmarks(self, 
                          progress_callback: Optional[Callable] = None) -> List[BenchmarkResult]:
        """Run all benchmarks in the suite"""
        results = []
        total = len(self.benchmarks)
        
        for i, benchmark in enumerate(self.benchmarks):
            logger.info(f"Running benchmark {i+1}/{total}: {benchmark.name}")
            
            if progress_callback:
                progress_callback(i, total, benchmark.name)
            
            result = benchmark.execute()
            results.append(result)
            
            # Save intermediate results
            self._save_result(result)
        
        # Generate summary report
        self.generate_report(results)
        
        return results
    
    def run_comparison(self, algorithms: List[str], 
                      benchmark_names: Optional[List[str]] = None) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmarks comparing multiple algorithms"""
        if benchmark_names is None:
            benchmark_names = [b.name for b in self.benchmarks]
        
        comparison_results = defaultdict(list)
        
        for algo in algorithms:
            logger.info(f"Running benchmarks for algorithm: {algo}")
            
            for benchmark_name in benchmark_names:
                # Create algorithm-specific benchmark
                benchmark = self._create_algorithm_benchmark(benchmark_name, algo)
                if benchmark:
                    result = benchmark.execute()
                    comparison_results[algo].append(result)
        
        return dict(comparison_results)
    
    def set_baseline(self, benchmark_name: str, result: BenchmarkResult) -> None:
        """Set a baseline for comparison"""
        self.baselines[benchmark_name] = result
        logger.info(f"Set baseline for {benchmark_name}")
    
    def compare_to_baseline(self, result: BenchmarkResult) -> Dict[str, float]:
        """Compare result to baseline"""
        if result.benchmark_name not in self.baselines:
            return {}
        
        baseline = self.baselines[result.benchmark_name]
        comparison = {}
        
        # Compare metrics
        for metric, value in result.metrics.items():
            if metric in baseline.metrics:
                baseline_value = baseline.metrics[metric]
                if baseline_value > 0:
                    improvement = (value - baseline_value) / baseline_value * 100
                    comparison[f"{metric}_improvement"] = improvement
        
        # Compare resources
        comparison['duration_improvement'] = \
            (baseline.duration - result.duration) / baseline.duration * 100
        comparison['memory_improvement'] = \
            (baseline.memory_usage - result.memory_usage) / baseline.memory_usage * 100
        
        return comparison
    
    def generate_report(self, results: List[BenchmarkResult], 
                       format: str = "markdown") -> str:
        """Generate benchmark report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "markdown":
            report = self._generate_markdown_report(results)
            report_path = self.output_dir / f"benchmark_report_{timestamp}.md"
        elif format == "json":
            report = self._generate_json_report(results)
            report_path = self.output_dir / f"benchmark_report_{timestamp}.json"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to: {report_path}")
        return str(report_path)
    
    def _save_result(self, result: BenchmarkResult) -> None:
        """Save individual result"""
        result_path = self.output_dir / f"{result.benchmark_name}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _generate_markdown_report(self, results: List[BenchmarkResult]) -> str:
        """Generate markdown format report"""
        report = f"# RL Commons Benchmark Report\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary table
        report += "## Summary\n\n"
        report += "| Benchmark | Algorithm | Duration (s) | Memory (MB) | CPU (%) | Key Metric |\n"
        report += "|-----------|-----------|--------------|-------------|---------|------------|\n"
        
        for result in results:
            key_metric = self._get_key_metric(result)
            report += f"| {result.benchmark_name} | {result.algorithm} | "
            report += f"{result.duration:.2f} | {result.memory_usage:.2f} | "
            report += f"{result.cpu_usage:.1f} | {key_metric} |\n"
        
        # Detailed results
        report += "\n## Detailed Results\n\n"
        
        for result in results:
            report += f"### {result.benchmark_name}\n"
            report += f"- **Algorithm**: {result.algorithm}\n"
            report += f"- **Duration**: {result.duration:.3f} seconds\n"
            report += f"- **Memory Usage**: {result.memory_usage:.2f} MB\n"
            report += f"- **CPU Usage**: {result.cpu_usage:.1f}%\n"
            report += f"- **Metrics**:\n"
            
            for metric, value in result.metrics.items():
                report += f"  - {metric}: {value:.4f}\n"
            
            # Baseline comparison if available
            if result.benchmark_name in self.baselines:
                comparison = self.compare_to_baseline(result)
                if comparison:
                    report += "- **Vs Baseline**:\n"
                    for metric, improvement in comparison.items():
                        report += f"  - {metric}: {improvement:+.1f}%\n"
            
            report += "\n"
        
        return report
    
    def _generate_json_report(self, results: List[BenchmarkResult]) -> str:
        """Generate JSON format report"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in results],
            'summary': self._generate_summary(results)
        }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            'total_benchmarks': len(results),
            'total_duration': sum(r.duration for r in results),
            'average_duration': np.mean([r.duration for r in results]),
            'total_memory': sum(r.memory_usage for r in results),
            'average_cpu': np.mean([r.cpu_usage for r in results])
        }
        
        # Algorithm performance summary
        algo_performance = defaultdict(list)
        for result in results:
            algo_performance[result.algorithm].append(result)
        
        summary['algorithm_summary'] = {}
        for algo, algo_results in algo_performance.items():
            summary['algorithm_summary'][algo] = {
                'benchmark_count': len(algo_results),
                'avg_duration': np.mean([r.duration for r in algo_results]),
                'avg_memory': np.mean([r.memory_usage for r in algo_results]),
                'avg_cpu': np.mean([r.cpu_usage for r in algo_results])
            }
        
        return summary
    
    def _get_key_metric(self, result: BenchmarkResult) -> str:
        """Extract key metric for summary"""
        if 'accuracy' in result.metrics:
            return f"Acc: {result.metrics['accuracy']:.2%}"
        elif 'reward' in result.metrics:
            return f"Reward: {result.metrics['reward']:.2f}"
        elif 'throughput' in result.metrics:
            return f"Throughput: {result.metrics['throughput']:.0f}/s"
        elif result.metrics:
            # Use first metric
            key, value = list(result.metrics.items())[0]
            return f"{key}: {value:.2f}"
        else:
            return "N/A"
    
    def _create_algorithm_benchmark(self, benchmark_name: str, algorithm: str) -> Optional[Benchmark]:
        """Create algorithm-specific benchmark instance"""
        # This would be implemented by specific benchmark types
        # For now, return None
        return None


# Validation
if __name__ == "__main__":
    print("Testing BenchmarkSuite...")
    
    # Create test benchmark
    class TestBenchmark(Benchmark):
        def setup(self):
            self.data = np.random.randn(1000, 10)
        
        def run(self):
            result = np.mean(self.data)
            return {'mean': result, 'std': np.std(self.data)}
        
        def teardown(self):
            self.data = None
    
    # Create suite
    suite = BenchmarkSuite()
    
    # Add benchmark
    test_bench = TestBenchmark("test_benchmark", "Test benchmark for validation")
    suite.add_benchmark(test_bench)
    
    # Run benchmark
    results = suite.run_all_benchmarks()
    
    # Check results
    assert len(results) == 1
    assert results[0].benchmark_name == "test_benchmark"
    assert 'mean' in results[0].metrics
    assert results[0].duration > 0
    
    print(" BenchmarkSuite validation passed")