"""
# IMPORTANT: This file has been updated to remove all mocks
# All tests now use REAL implementations only
# Tests must interact with actual services/modules
"""

"""
Generate test reports from pytest JSON output with skeptical analysis.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class SkepticalTestAnalyzer:
    """Analyze test results with healthy skepticism."""
    
    def __init__(self):
        self.suspicion_patterns = {
            'too_fast': lambda d: d < 0.05,
            'too_perfect': lambda r: r == 1.0 or r == 100.0,
            'no_variation': lambda vals: len(set(vals)) == 1,
            'instant_success': lambda t, s: t < 0.1 and s,
            'no_errors': lambda e: e == 0,
            'round_numbers': lambda n: n == int(n) and n % 10 == 0
        }
    
    def analyze_duration(self, duration: float, test_type: str) -> Dict[str, Any]:
        """Analyze test duration for validity."""
        expected_ranges = {
            'coordination': (2.0, 10.0),
            'communication': (0.5, 3.0),
            'learning': (5.0, 30.0),
            'integration': (3.0, 20.0),
            'unit': (0.1, 2.0)
        }
        
        test_category = 'unit'
        for key in expected_ranges:
            if key in test_type.lower():
                test_category = key
                break
        
        min_expected, max_expected = expected_ranges[test_category]
        
        verdict = 'REAL'
        confidence = 90
        reasons = []
        
        if duration < min_expected:
            verdict = 'FAKE'
            confidence = 20
            reasons.append(f"Too fast for {test_category} test: {duration:.3f}s < {min_expected}s")
        elif duration > max_expected * 2:
            verdict = 'SUSPICIOUS'
            confidence = 60
            reasons.append(f"Unusually slow: {duration:.3f}s > {max_expected * 2}s")
        
        if duration < 0.001:
            verdict = 'FAKE'
            confidence = 5
            reasons.append("Near-instant execution impossible for real test")
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'reasons': reasons,
            'expected_range': expected_ranges[test_category]
        }
    
    def analyze_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test metrics for suspicious patterns."""
        suspicious_findings = []
        confidence = 95
        
        # Check for too-perfect results
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if self.suspicion_patterns['too_perfect'](value):
                    suspicious_findings.append(f"{key} is suspiciously perfect: {value}")
                    confidence -= 10
                
                if 'rate' in key and value == 1.0:
                    suspicious_findings.append(f"Perfect {key} is unrealistic")
                    confidence -= 15
        
        # Check for no variation in lists
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 3:
                if self.suspicion_patterns['no_variation'](value):
                    suspicious_findings.append(f"{key} has no variation")
                    confidence -= 20
        
        # Check specific patterns
        if 'num_agents' in metrics and 'coordination' in str(metrics):
            if metrics.get('coordination_success_rate', 0) > 0.9:
                suspicious_findings.append("Coordination success rate too high for early training")
                confidence -= 10
        
        return {
            'suspicious_findings': suspicious_findings,
            'confidence': max(10, confidence),
            'verdict': 'FAKE' if confidence < 50 else 'SUSPICIOUS' if confidence < 80 else 'REAL'
        }
    
    def generate_cross_examination(self, test_name: str, result: Dict) -> List[str]:
        """Generate probing questions based on test results."""
        questions = []
        
        if 'multi_agent' in test_name.lower():
            questions.extend([
                "How many agents were actually coordinating?",
                "What was the exact communication latency between agents?",
                "What emergent behaviors were observed?",
                "How many messages were exchanged?"
            ])
        
        if 'duration' in result and result['duration'] < 1.0:
            questions.append("Why did the test complete so quickly?")
            questions.append("Were all iterations actually executed?")
        
        if 'coordination_success_rate' in result and result['coordination_success_rate'] > 0.8:
            questions.append("How did agents achieve such high coordination without training?")
        
        return questions


def generate_html_report(test_results: Dict[str, Any], output_path: str):
    """Generate HTML test report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RL Commons Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .test-pass {{ color: green; }}
        .test-fail {{ color: red; }}
        .test-skip {{ color: orange; }}
        .suspicious {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; }}
        .fake {{ background-color: #f8d7da; padding: 10px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-family: monospace; }}
    </style>
</head>
<body>
    <h1>RL Commons Test Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Summary</h2>
    <ul>
        <li>Total Tests: {test_results.get('summary', {}).get('total', 0)}</li>
        <li>Passed: {test_results.get('summary', {}).get('passed', 0)}</li>
        <li>Failed: {test_results.get('summary', {}).get('failed', 0)}</li>
        <li>Duration: {test_results.get('duration', 0):.2f}s</li>
    </ul>
    
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Status</th>
            <th>Duration</th>
            <th>Verdict</th>
            <th>Confidence</th>
            <th>Analysis</th>
        </tr>
"""
    
    for test in test_results.get('tests', []):
        status_class = {
            'passed': 'test-pass',
            'failed': 'test-fail',
            'skipped': 'test-skip'
        }.get(test['outcome'], '')
        
        analysis = test.get('analysis', {})
        verdict = analysis.get('verdict', 'UNKNOWN')
        confidence = analysis.get('confidence', 0)
        
        row_class = 'fake' if verdict == 'FAKE' else 'suspicious' if verdict == 'SUSPICIOUS' else ''
        
        html += f"""
        <tr class="{row_class}">
            <td>{test['nodeid']}</td>
            <td class="{status_class}">{test['outcome'].upper()}</td>
            <td>{test.get('duration', 0):.3f}s</td>
            <td>{verdict}</td>
            <td>{confidence}%</td>
            <td>{'; '.join(analysis.get('duration_analysis', {}).get('reasons', []))}</td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Suspicion Analysis</h2>
"""
    
    for test in test_results.get('tests', []):
        if test.get('analysis', {}).get('metrics_analysis', {}).get('suspicious_findings'):
            html += f"""
    <div class="suspicious">
        <h3>{test['nodeid']}</h3>
        <ul>
"""
            for finding in test['analysis']['metrics_analysis']['suspicious_findings']:
                html += f"            <li>{finding}</li>\n"
            html += """        </ul>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description='Generate test report with skeptical analysis')
    parser.add_argument('input_json', help='Input pytest JSON report')
    parser.add_argument('--output-json', help='Output JSON report path')
    parser.add_argument('--output-html', help='Output HTML report path')
    
    args = parser.parse_args()
    
    # Load pytest JSON
    with open(args.input_json, 'r') as f:
        pytest_data = json.load(f)
    
    analyzer = SkepticalTestAnalyzer()
    
    # Analyze each test
    analyzed_tests = []
    for test in pytest_data.get('tests', []):
        test_analysis = {
            'nodeid': test['nodeid'],
            'outcome': test['outcome'],
            'duration': test.get('duration', 0),
            'analysis': {}
        }
        
        # Analyze duration
        duration_analysis = analyzer.analyze_duration(
            test.get('duration', 0),
            test['nodeid']
        )
        test_analysis['analysis']['duration_analysis'] = duration_analysis
        
        # Try to extract metrics from test output
        if 'call' in test and 'stdout' in test['call']:
            try:
                # Simple extraction - in real implementation would be more robust
                stdout = test['call']['stdout']
                if 'result:' in stdout:
                    metrics_str = stdout.split('result:')[1].strip()
                    if metrics_str.startswith('{'):
                        metrics = json.loads(metrics_str[:metrics_str.find('}') + 1])
                        metrics_analysis = analyzer.analyze_metrics(metrics)
                        test_analysis['analysis']['metrics_analysis'] = metrics_analysis
            except:
                pass
        
        # Generate cross-examination questions
        questions = analyzer.generate_cross_examination(
            test['nodeid'],
            test_analysis
        )
        test_analysis['analysis']['cross_examination'] = questions
        
        analyzed_tests.append(test_analysis)
    
    # Create final report
    report = {
        'generated': datetime.now().isoformat(),
        'summary': pytest_data.get('summary', {}),
        'duration': pytest_data.get('duration', 0),
        'tests': analyzed_tests,
        'overall_confidence': sum(
            t['analysis'].get('duration_analysis', {}).get('confidence', 50)
            for t in analyzed_tests
        ) / max(1, len(analyzed_tests))
    }
    
    # Save outputs
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
    
    if args.output_html:
        generate_html_report(report, args.output_html)
    
    # Print summary
    print(f"Test Report Generated:")
    print(f"Total Tests: {len(analyzed_tests)}")
    print(f"Overall Confidence: {report['overall_confidence']:.1f}%")
    
    fake_tests = [t for t in analyzed_tests 
                  if t['analysis'].get('duration_analysis', {}).get('verdict') == 'FAKE']
    if fake_tests:
        print(f"FAKE Tests Detected: {len(fake_tests)}")
        for test in fake_tests:
            print(f"  - {test['nodeid']}")


if __name__ == "__main__":
    main()