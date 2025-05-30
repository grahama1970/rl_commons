#!/usr/bin/env python3
"""
Comprehensive test runner for RL Commons project
Validates all completed work with detailed reporting
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import importlib.util

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class TestReporter:
    """Simple test reporter using only stdlib"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.start_time = time.time()
        
    def add_result(self, test_name: str, status: str, duration: float, 
                   output: str = "", error: str = ""):
        """Add a test result"""
        self.results.append({
            "test": test_name,
            "status": status,
            "duration": round(duration, 3),
            "output": output,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        
    def generate_report(self) -> Path:
        """Generate HTML and JSON reports"""
        total_duration = time.time() - self.start_time
        
        # Calculate statistics
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        skipped = sum(1 for r in self.results if r["status"] == "SKIP")
        
        # JSON report
        json_path = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "duration": round(total_duration, 3),
                    "success_rate": f"{(passed/total*100):.1f}%" if total > 0 else "0%"
                },
                "results": self.results
            }, f, indent=2)
            
        # HTML report
        html_path = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RL Commons Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        .skip {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .error {{ background-color: #ffeeee; font-family: monospace; padding: 10px; }}
    </style>
</head>
<body>
    <h1>RL Commons Test Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Tests: {total}</p>
        <p class="pass">Passed: {passed}</p>
        <p class="fail">Failed: {failed}</p>
        <p class="skip">Skipped: {skipped}</p>
        <p>Duration: {total_duration:.1f}s</p>
        <p>Success Rate: {(passed/total*100):.1f}%</p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Status</th>
            <th>Duration</th>
            <th>Details</th>
        </tr>
"""
        
        for result in self.results:
            status_class = result['status'].lower()
            html_content += f"""
        <tr>
            <td>{result['test']}</td>
            <td class="{status_class}">{result['status']}</td>
            <td>{result['duration']}s</td>
            <td>
"""
            if result['error']:
                html_content += f'<div class="error">{result["error"]}</div>'
            elif result['output']:
                html_content += f'<pre>{result["output"][:500]}...</pre>'
            html_content += """
            </td>
        </tr>
"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        return html_path


def run_command(cmd: List[str], cwd: Optional[Path] = None, 
                timeout: int = 300) -> Tuple[int, str, str]:
    """Run a command and return (exit_code, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def test_imports(reporter: TestReporter):
    """Test that all modules can be imported"""
    print("\n🔍 Testing module imports...")
    
    modules = [
        "graham_rl_commons",
        "graham_rl_commons.algorithms.ppo",
        "graham_rl_commons.integrations.arangodb_optimizer",
        "graham_rl_commons.algorithms.bandits",
        "graham_rl_commons.algorithms.dqn",
        "graham_rl_commons.algorithms.hierarchical",
    ]
    
    for module in modules:
        start = time.time()
        try:
            importlib.import_module(module)
            reporter.add_result(f"Import {module}", "PASS", time.time() - start)
            print(f"✅ {module}")
        except Exception as e:
            reporter.add_result(f"Import {module}", "FAIL", time.time() - start, 
                              error=str(e))
            print(f"❌ {module}: {e}")


def test_ppo_implementation(reporter: TestReporter):
    """Test PPO implementation"""
    print("\n🧪 Testing PPO implementation...")
    
    start = time.time()
    exit_code, stdout, stderr = run_command(
        ["python", "test_ppo.py"],
        cwd=Path(__file__).parent
    )
    
    if exit_code == 0:
        reporter.add_result("PPO Implementation", "PASS", time.time() - start, output=stdout)
        print("✅ PPO implementation tests passed")
    else:
        reporter.add_result("PPO Implementation", "FAIL", time.time() - start, 
                          output=stdout, error=stderr)
        print(f"❌ PPO implementation tests failed: {stderr}")


def test_arangodb_integration(reporter: TestReporter):
    """Test ArangoDB integration"""
    print("\n🧪 Testing ArangoDB integration...")
    
    # Test the integration module
    start = time.time()
    exit_code, stdout, stderr = run_command(
        ["python", "-m", "graham_rl_commons.integrations.arangodb_optimizer"],
        cwd=Path(__file__).parent
    )
    
    if exit_code == 0 and "✅ ArangoDBOptimizer validation passed" in stdout:
        reporter.add_result("ArangoDB Module Validation", "PASS", time.time() - start, 
                          output=stdout)
        print("✅ ArangoDB module validation passed")
    else:
        reporter.add_result("ArangoDB Module Validation", "FAIL", time.time() - start, 
                          output=stdout, error=stderr)
        print(f"❌ ArangoDB module validation failed")
    
    # Test the integration tests
    start = time.time()
    exit_code, stdout, stderr = run_command(
        ["python", "test/integration/test_arangodb_optimizer.py"],
        cwd=Path(__file__).parent
    )
    
    if exit_code == 0 and "ALL TESTS PASSED" in stdout:
        reporter.add_result("ArangoDB Integration Tests", "PASS", time.time() - start, 
                          output=stdout)
        print("✅ ArangoDB integration tests passed")
    else:
        reporter.add_result("ArangoDB Integration Tests", "FAIL", time.time() - start, 
                          output=stdout, error=stderr)
        print(f"❌ ArangoDB integration tests failed")


def test_example_scripts(reporter: TestReporter):
    """Test example scripts"""
    print("\n🧪 Testing example scripts...")
    
    # Test ArangoDB example (limited iterations for speed)
    start = time.time()
    test_script = """
import sys
sys.path.insert(0, 'src')
from examples.arangodb_integration import ArangoDBOptimizer, ArangoDBClient, run_optimization_loop

optimizer = ArangoDBOptimizer(name="test_example")
client = ArangoDBClient()
try:
    run_optimization_loop(optimizer, client, num_iterations=5, evaluation_interval=1)
    print("✅ Example completed successfully")
except Exception as e:
    print(f"❌ Example failed: {e}")
    sys.exit(1)
"""
    
    exit_code, stdout, stderr = run_command(
        ["python", "-c", test_script],
        cwd=Path(__file__).parent,
        timeout=60
    )
    
    if exit_code == 0 and "✅ Example completed successfully" in stdout:
        reporter.add_result("ArangoDB Example", "PASS", time.time() - start, 
                          output=stdout[-1000:])  # Last 1000 chars
        print("✅ ArangoDB example passed")
    else:
        reporter.add_result("ArangoDB Example", "FAIL", time.time() - start, 
                          output=stdout, error=stderr)
        print(f"❌ ArangoDB example failed")


def test_marker_fixes(reporter: TestReporter):
    """Test Marker integration fixes"""
    print("\n🧪 Testing Marker RL integration fixes...")
    
    # Run the verification script
    start = time.time()
    exit_code, stdout, stderr = run_command(
        ["python", "verify_marker_fixes.py"],
        cwd=Path(__file__).parent
    )
    
    if exit_code == 0:
        reporter.add_result("Marker Test Fixes", "PASS", time.time() - start, 
                          output=stdout)
        print("✅ Marker test fixes verified")
    else:
        reporter.add_result("Marker Test Fixes", "FAIL", time.time() - start, 
                          output=stdout, error=stderr)
        print(f"❌ Marker test fix verification failed")


def test_pytest_suite(reporter: TestReporter):
    """Run pytest suite if available"""
    print("\n🧪 Running pytest suite...")
    
    start = time.time()
    exit_code, stdout, stderr = run_command(
        ["python", "-m", "pytest", "-v", "--tb=short"],
        cwd=Path(__file__).parent
    )
    
    if exit_code == 0:
        reporter.add_result("Pytest Suite", "PASS", time.time() - start, output=stdout)
        print("✅ Pytest suite passed")
    else:
        # Check if pytest is not installed
        if "No module named pytest" in stderr:
            reporter.add_result("Pytest Suite", "SKIP", time.time() - start, 
                              output="pytest not installed")
            print("⏭️  Pytest not installed, skipping")
        else:
            reporter.add_result("Pytest Suite", "FAIL", time.time() - start, 
                              output=stdout, error=stderr)
            print(f"❌ Pytest suite failed")


def main():
    """Run all tests and generate report"""
    print("=" * 60)
    print("🚀 RL Commons Comprehensive Test Suite")
    print("=" * 60)
    
    # Create reporter
    report_dir = Path(__file__).parent / "test_reports"
    reporter = TestReporter(report_dir)
    
    # Run all test suites
    test_imports(reporter)
    test_ppo_implementation(reporter)
    test_arangodb_integration(reporter)
    test_example_scripts(reporter)
    test_marker_fixes(reporter)
    test_pytest_suite(reporter)
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 Generating test report...")
    
    report_path = reporter.generate_report()
    
    # Print summary
    summary = {
        "total": len(reporter.results),
        "passed": sum(1 for r in reporter.results if r["status"] == "PASS"),
        "failed": sum(1 for r in reporter.results if r["status"] == "FAIL"),
        "skipped": sum(1 for r in reporter.results if r["status"] == "SKIP")
    }
    
    print(f"\n✅ Total: {summary['total']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⏭️  Skipped: {summary['skipped']}")
    print(f"\n📄 Report saved to: {report_path}")
    
    # Exit with failure if any tests failed
    sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == "__main__":
    main()