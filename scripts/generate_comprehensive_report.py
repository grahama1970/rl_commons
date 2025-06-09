#!/usr/bin/env python3
"""
Module: generate_comprehensive_report.py
Purpose: Generate comprehensive test report with confidence analysis

External Dependencies:
- None (uses stdlib only)

Example Usage:
>>> python scripts/generate_comprehensive_report.py
 Module validation passed
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple


def analyze_test_results(json_path: Path) -> Tuple[float, Dict[str, Any]]:
    """Analyze test results and calculate confidence"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    total = summary.get('total', 0)
    passed = summary.get('passed', 0)
    failed = summary.get('failed', 0)
    
    if total == 0:
        return 0.0, {"error": "No tests found"}
    
    # Calculate base confidence
    pass_rate = passed / total
    confidence = pass_rate * 100
    
    # Analyze failure patterns
    failure_types = {}
    for test in data.get('tests', []):
        if test.get('outcome') == 'failed':
            module = test['nodeid'].split('/')[1] if '/' in test['nodeid'] else 'unknown'
            failure_types[module] = failure_types.get(module, 0) + 1
    
    analysis = {
        'total_tests': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': pass_rate,
        'failure_types': failure_types,
        'duration': data.get('duration', 0)
    }
    
    return confidence, analysis


def generate_markdown_report(confidence: float, analysis: Dict[str, Any]) -> str:
    """Generate markdown report"""
    status = " HIGH" if confidence >= 90 else "⚠️ MODERATE" if confidence >= 70 else " LOW"
    
    report = f"""# RL Commons Test Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
**Confidence: {confidence:.1f}% - {status} CONFIDENCE**

- Total Tests: {analysis['total_tests']}
- Passed: {analysis['passed']} ({analysis['pass_rate']*100:.1f}%)
- Failed: {analysis['failed']}
- Duration: {analysis['duration']:.2f}s

## Failure Breakdown
"""
    
    if analysis['failure_types']:
        report += "\n| Module | Failures |\n|--------|----------|\n"
        for module, count in sorted(analysis['failure_types'].items()):
            report += f"| {module} | {count} |\n"
    else:
        report += "\nNo failures detected!\n"
    
    # Add recommendations
    report += "\n## Recommendations\n"
    if confidence >= 90:
        report += "-  Test suite is in excellent condition\n"
        report += "- Continue regular testing and maintenance\n"
    elif confidence >= 70:
        report += "- ⚠️ Address failing tests in priority order\n"
        report += "- Focus on modules with most failures first\n"
    else:
        report += "-  Immediate attention required\n"
        report += "- Fix critical initialization and integration issues\n"
    
    return report


def main():
    """Main execution"""
    # Check if test results exist
    json_path = Path("reports/test_report.json")
    
    if not json_path.exists():
        print("Test results not found. Please run tests first.")
        return 1
    
    # Analyze results
    confidence, analysis = analyze_test_results(json_path)
    
    # Generate report
    report = generate_markdown_report(confidence, analysis)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"reports/comprehensive_report_{timestamp}.md")
    report_path.write_text(report)
    
    # Display results
    print(f"\n{report}")
    print(f"\nReport saved to: {report_path}")
    
    if confidence >= 90:
        print("\n Test verification PASSED with high confidence!")
        return 0
    else:
        print(f"\n⚠️ Test confidence below 90% threshold: {confidence:.1f}%")
        return 1


if __name__ == "__main__":
    result = main()
    if result == 0:
        print(" Module validation passed")
    sys.exit(result)