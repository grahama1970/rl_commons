"""
Module: performance_profiler.py
Purpose: Detailed performance profiling for RL algorithms

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- cProfile: Python profiling
- memory_profiler: https://pypi.org/project/memory-profiler/

Example Usage:
>>> from rl_commons.benchmarks import PerformanceProfiler
>>> profiler = PerformanceProfiler()
>>> with profiler.profile("training_loop"):
>>>     agent.train(episodes=100)
>>> profiler.generate_report()
"""

import time
import cProfile
import pstats
import io
import tracemalloc
import numpy as np
from typing import Dict, List, Any, Optional, Callable, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import json
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ProfileSection:
    """Profile data for a code section"""
    name: str
    start_time: float
    end_time: float = 0.0
    memory_start: int = 0
    memory_peak: int = 0
    memory_end: int = 0
    cpu_stats: Optional[pstats.Stats] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Duration in seconds"""
        return self.end_time - self.start_time
    
    @property
    def memory_used(self) -> int:
        """Memory used in bytes"""
        return self.memory_end - self.memory_start
    
    @property
    def memory_peak_usage(self) -> int:
        """Peak memory usage in bytes"""
        return self.memory_peak - self.memory_start


class PerformanceProfiler:
    """Comprehensive performance profiler for RL algorithms"""
    
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.sections: Dict[str, List[ProfileSection]] = defaultdict(list)
        self.current_section: Optional[ProfileSection] = None
        self.memory_tracing = False
        
        # Initialize memory tracking
        tracemalloc.start()
        
        logger.info(f"Performance profiler initialized with output: {self.output_dir}")
    
    @contextmanager
    def profile(self, section_name: str, 
                enable_cpu_profiling: bool = True) -> ContextManager:
        """Profile a code section"""
        # Create section
        section = ProfileSection(
            name=section_name,
            start_time=time.time(),
            memory_start=tracemalloc.get_traced_memory()[0]
        )
        
        # Setup CPU profiling
        cpu_profiler = None
        if enable_cpu_profiling:
            cpu_profiler = cProfile.Profile()
            cpu_profiler.enable()
        
        try:
            self.current_section = section
            yield section
        finally:
            # Finalize section
            section.end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            section.memory_end = current
            section.memory_peak = peak
            
            # Stop CPU profiling
            if cpu_profiler:
                cpu_profiler.disable()
                
                # Create stats
                s = io.StringIO()
                stats = pstats.Stats(cpu_profiler, stream=s)
                section.cpu_stats = stats
            
            # Store section
            self.sections[section_name].append(section)
            self.current_section = None
            
            logger.debug(f"Profiled {section_name}: {section.duration:.3f}s, "
                        f"{section.memory_used / 1024 / 1024:.1f}MB")
    
    def add_metric(self, metric_name: str, value: Any) -> None:
        """Add custom metric to current section"""
        if self.current_section:
            self.current_section.custom_metrics[metric_name] = value
    
    def profile_algorithm(self, 
                         algorithm_name: str,
                         train_fn: Callable,
                         episodes: int = 100) -> Dict[str, Any]:
        """Profile an RL algorithm's training"""
        profile_results = {
            'algorithm': algorithm_name,
            'episodes': episodes,
            'sections': {}
        }
        
        # Overall profiling
        with self.profile("overall_training"):
            # Setup phase
            with self.profile("setup"):
                agent, env = train_fn("setup")
            
            # Training loop profiling
            episode_times = []
            step_times = []
            update_times = []
            
            for episode in range(episodes):
                with self.profile(f"episode_{episode}", enable_cpu_profiling=False):
                    # Episode execution
                    episode_start = time.time()
                    
                    with self.profile("episode_execution", enable_cpu_profiling=False):
                        state = env.reset()
                        done = False
                        episode_reward = 0
                        steps = 0
                        
                        while not done and steps < 1000:
                            # Action selection
                            step_start = time.time()
                            with self.profile("action_selection", enable_cpu_profiling=False):
                                action = agent.select_action(state)
                            
                            # Environment step
                            with self.profile("env_step", enable_cpu_profiling=False):
                                next_state, reward, done, _ = env.step(action)
                            
                            # Policy update
                            update_start = time.time()
                            with self.profile("policy_update", enable_cpu_profiling=False):
                                agent.update_policy(state, action, reward, next_state, done)
                            update_times.append(time.time() - update_start)
                            
                            state = next_state
                            episode_reward += reward
                            steps += 1
                            step_times.append(time.time() - step_start)
                    
                    episode_times.append(time.time() - episode_start)
                    self.add_metric("episode_reward", episode_reward)
                    self.add_metric("episode_steps", steps)
            
            # Cleanup
            with self.profile("cleanup"):
                env.close()
        
        # Aggregate results
        profile_results['timing'] = {
            'total_time': self.sections['overall_training'][0].duration,
            'average_episode_time': np.mean(episode_times),
            'average_step_time': np.mean(step_times),
            'average_update_time': np.mean(update_times)
        }
        
        profile_results['memory'] = {
            'peak_memory_mb': self.sections['overall_training'][0].memory_peak_usage / 1024 / 1024,
            'final_memory_mb': self.sections['overall_training'][0].memory_used / 1024 / 1024
        }
        
        return profile_results
    
    def compare_algorithms(self, 
                          algorithms: List[str],
                          train_fn_factory: Callable[[str], Callable],
                          episodes: int = 100) -> Dict[str, Dict[str, Any]]:
        """Compare performance of multiple algorithms"""
        comparison_results = {}
        
        for algo in algorithms:
            logger.info(f"Profiling algorithm: {algo}")
            train_fn = train_fn_factory(algo)
            results = self.profile_algorithm(algo, train_fn, episodes)
            comparison_results[algo] = results
        
        # Generate comparison report
        self._generate_comparison_report(comparison_results)
        
        return comparison_results
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        bottlenecks = {
            'slowest_sections': [],
            'memory_intensive_sections': [],
            'frequent_sections': [],
            'recommendations': []
        }
        
        # Find slowest sections
        all_sections = []
        for section_name, section_list in self.sections.items():
            for section in section_list:
                all_sections.append((section_name, section))
        
        # Sort by duration
        sorted_by_duration = sorted(all_sections, 
                                   key=lambda x: x[1].duration, 
                                   reverse=True)
        bottlenecks['slowest_sections'] = [
            {
                'name': name,
                'duration': section.duration,
                'percentage': section.duration / self._get_total_time() * 100
            }
            for name, section in sorted_by_duration[:10]
        ]
        
        # Sort by memory usage
        sorted_by_memory = sorted(all_sections,
                                 key=lambda x: x[1].memory_peak_usage,
                                 reverse=True)
        bottlenecks['memory_intensive_sections'] = [
            {
                'name': name,
                'memory_mb': section.memory_peak_usage / 1024 / 1024,
                'percentage': section.memory_peak_usage / self._get_peak_memory() * 100
            }
            for name, section in sorted_by_memory[:10]
        ]
        
        # Find most frequent sections
        section_counts = defaultdict(int)
        for section_name, section_list in self.sections.items():
            section_counts[section_name] = len(section_list)
        
        sorted_by_frequency = sorted(section_counts.items(),
                                    key=lambda x: x[1],
                                    reverse=True)
        bottlenecks['frequent_sections'] = [
            {'name': name, 'count': count}
            for name, count in sorted_by_frequency[:10]
        ]
        
        # Generate recommendations
        bottlenecks['recommendations'] = self._generate_recommendations(bottlenecks)
        
        return bottlenecks
    
    def generate_report(self, format: str = "markdown") -> str:
        """Generate profiling report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "markdown":
            report = self._generate_markdown_report()
            report_path = self.output_dir / f"profile_report_{timestamp}.md"
        elif format == "json":
            report = self._generate_json_report()
            report_path = self.output_dir / f"profile_report_{timestamp}.json"
        elif format == "html":
            report = self._generate_html_report()
            report_path = self.output_dir / f"profile_report_{timestamp}.html"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Profile report saved to: {report_path}")
        return str(report_path)
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown format report"""
        report = "# Performance Profile Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary
        report += "## Summary\n\n"
        total_time = self._get_total_time()
        peak_memory = self._get_peak_memory()
        
        report += f"- **Total Time**: {total_time:.2f} seconds\n"
        report += f"- **Peak Memory**: {peak_memory / 1024 / 1024:.2f} MB\n"
        report += f"- **Sections Profiled**: {sum(len(s) for s in self.sections.values())}\n\n"
        
        # Bottleneck analysis
        bottlenecks = self.analyze_bottlenecks()
        
        report += "## Performance Bottlenecks\n\n"
        report += "### Slowest Sections\n\n"
        report += "| Section | Duration (s) | % of Total |\n"
        report += "|---------|--------------|------------|\n"
        for section in bottlenecks['slowest_sections'][:5]:
            report += f"| {section['name']} | {section['duration']:.3f} | {section['percentage']:.1f}% |\n"
        
        report += "\n### Memory Intensive Sections\n\n"
        report += "| Section | Memory (MB) | % of Peak |\n"
        report += "|---------|-------------|-----------|\n"
        for section in bottlenecks['memory_intensive_sections'][:5]:
            report += f"| {section['name']} | {section['memory_mb']:.2f} | {section['percentage']:.1f}% |\n"
        
        # Recommendations
        report += "\n## Recommendations\n\n"
        for rec in bottlenecks['recommendations']:
            report += f"- {rec}\n"
        
        return report
    
    def _generate_json_report(self) -> str:
        """Generate JSON format report"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_time': self._get_total_time(),
                'peak_memory_mb': self._get_peak_memory() / 1024 / 1024,
                'sections_profiled': sum(len(s) for s in self.sections.values())
            },
            'sections': {},
            'bottlenecks': self.analyze_bottlenecks()
        }
        
        # Add section details
        for section_name, section_list in self.sections.items():
            report_data['sections'][section_name] = [
                {
                    'duration': section.duration,
                    'memory_used_mb': section.memory_used / 1024 / 1024,
                    'memory_peak_mb': section.memory_peak_usage / 1024 / 1024,
                    'custom_metrics': section.custom_metrics
                }
                for section in section_list
            ]
        
        return json.dumps(report_data, indent=2)
    
    def _generate_html_report(self) -> str:
        """Generate HTML format report"""
        # Simple HTML report
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Performance Profile Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric { font-weight: bold; color: #2196F3; }
    </style>
</head>
<body>
    <h1>Performance Profile Report</h1>
"""
        
        html += f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        
        # Summary
        html += "<h2>Summary</h2>"
        html += f"<p>Total Time: <span class='metric'>{self._get_total_time():.2f}s</span></p>"
        html += f"<p>Peak Memory: <span class='metric'>{self._get_peak_memory() / 1024 / 1024:.2f}MB</span></p>"
        
        # Bottlenecks table
        bottlenecks = self.analyze_bottlenecks()
        html += "<h2>Performance Bottlenecks</h2>"
        html += "<table>"
        html += "<tr><th>Section</th><th>Duration (s)</th><th>% of Total</th></tr>"
        
        for section in bottlenecks['slowest_sections'][:10]:
            html += f"<tr><td>{section['name']}</td>"
            html += f"<td>{section['duration']:.3f}</td>"
            html += f"<td>{section['percentage']:.1f}%</td></tr>"
        
        html += "</table></body></html>"
        
        return html
    
    def _generate_comparison_report(self, comparison_results: Dict[str, Dict[str, Any]]) -> None:
        """Generate algorithm comparison report"""
        report = "# Algorithm Comparison Report\n\n"
        
        # Summary table
        report += "## Performance Summary\n\n"
        report += "| Algorithm | Total Time (s) | Avg Episode (ms) | Peak Memory (MB) |\n"
        report += "|-----------|----------------|------------------|------------------|\n"
        
        for algo, results in comparison_results.items():
            report += f"| {algo} | "
            report += f"{results['timing']['total_time']:.2f} | "
            report += f"{results['timing']['average_episode_time'] * 1000:.1f} | "
            report += f"{results['memory']['peak_memory_mb']:.2f} |\n"
        
        # Save report
        report_path = self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
    
    def _get_total_time(self) -> float:
        """Get total profiled time"""
        if 'overall_training' in self.sections and self.sections['overall_training']:
            return self.sections['overall_training'][0].duration
        
        # Sum all sections
        total = 0
        for section_list in self.sections.values():
            for section in section_list:
                total += section.duration
        return total
    
    def _get_peak_memory(self) -> int:
        """Get peak memory usage"""
        if 'overall_training' in self.sections and self.sections['overall_training']:
            return self.sections['overall_training'][0].memory_peak_usage
        
        # Find max peak
        peak = 0
        for section_list in self.sections.values():
            for section in section_list:
                peak = max(peak, section.memory_peak_usage)
        return peak
    
    def _generate_recommendations(self, bottlenecks: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Check for slow sections
        if bottlenecks['slowest_sections']:
            slowest = bottlenecks['slowest_sections'][0]
            if slowest['percentage'] > 50:
                recommendations.append(
                    f"'{slowest['name']}' takes {slowest['percentage']:.1f}% of total time. "
                    "Consider optimizing this section."
                )
        
        # Check for memory issues
        if bottlenecks['memory_intensive_sections']:
            memory_heavy = bottlenecks['memory_intensive_sections'][0]
            if memory_heavy['memory_mb'] > 1000:
                recommendations.append(
                    f"'{memory_heavy['name']}' uses {memory_heavy['memory_mb']:.0f}MB. "
                    "Consider reducing memory usage."
                )
        
        # Check for frequent operations
        if bottlenecks['frequent_sections']:
            frequent = bottlenecks['frequent_sections'][0]
            if frequent['count'] > 1000:
                recommendations.append(
                    f"'{frequent['name']}' is called {frequent['count']} times. "
                    "Consider batching or caching."
                )
        
        if not recommendations:
            recommendations.append("Performance looks good. No major bottlenecks detected.")
        
        return recommendations


# Validation
if __name__ == "__main__":
    print("Testing Performance Profiler...")
    
    # Create profiler
    profiler = PerformanceProfiler()
    
    # Test basic profiling
    with profiler.profile("test_section"):
        # Simulate some work
        data = np.random.randn(1000, 100)
        result = np.mean(data, axis=0)
        profiler.add_metric("data_shape", data.shape)
    
    # Test nested profiling
    with profiler.profile("outer_section"):
        for i in range(3):
            with profiler.profile(f"inner_section_{i}"):
                time.sleep(0.01)
    
    # Generate report
    report_path = profiler.generate_report()
    assert Path(report_path).exists()
    
    # Analyze bottlenecks
    bottlenecks = profiler.analyze_bottlenecks()
    assert 'slowest_sections' in bottlenecks
    assert len(bottlenecks['recommendations']) > 0
    
    print("✅ Performance profiler validation passed")