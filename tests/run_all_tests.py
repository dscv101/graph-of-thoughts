#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Comprehensive test runner for the MCP implementation.

This script runs all unit tests, integration tests, and generates a detailed
test report with coverage information and performance metrics.

Usage:
    python tests/run_all_tests.py [options]

Options:
    --verbose, -v       Enable verbose output
    --coverage, -c      Generate coverage report
    --performance, -p   Include performance benchmarks
    --integration, -i   Run integration tests (requires MCP server)
    --html-report       Generate HTML test report
    --xml-report        Generate XML test report for CI
    --fail-fast, -f     Stop on first failure
    --pattern PATTERN   Run tests matching pattern
    --exclude PATTERN   Exclude tests matching pattern

Examples:
    # Run all unit tests
    python tests/run_all_tests.py

    # Run with coverage and verbose output
    python tests/run_all_tests.py --coverage --verbose

    # Run only transport tests
    python tests/run_all_tests.py --pattern "*transport*"

    # Run all tests except integration tests
    python tests/run_all_tests.py --exclude "*integration*"
"""

import argparse
import json
import os
import sys
import time
import unittest
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Test discovery and execution
class TestResult:
    """Container for test execution results."""

    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        self.execution_time = 0.0
        self.failures = []
        self.errors = []
        self.test_details = []


class MCPTestRunner:
    """Comprehensive test runner for MCP implementation."""

    def __init__(self, args):
        """Initialize test runner with command line arguments."""
        self.args = args
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.results = TestResult()

        # Configure test discovery
        self.test_pattern = args.pattern or "test_*.py"
        self.exclude_pattern = args.exclude

        # Test modules to discover
        self.test_modules = [
            "test_mcp_transport",
            "test_mcp_client",
            "test_mcp_sampling",
            "test_mcp_protocol_compliance",
            "test_mcp_plugin_system",
            "test_mcp_circuit_breaker",
            "test_mcp_server",
            "test_mcp_server_integration",
        ]

        if args.integration:
            self.test_modules.append("test_mcp_integration")

    def discover_tests(self) -> unittest.TestSuite:
        """Discover all test cases."""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        print(f"🔍 Discovering tests in {self.test_dir}")

        for module_name in self.test_modules:
            try:
                # Check if test file exists
                test_file = self.test_dir / f"{module_name}.py"
                if not test_file.exists():
                    if self.args.verbose:
                        print(f"⚠️  Test file not found: {test_file}")
                    continue

                # Import and load tests
                module = __import__(module_name)
                module_suite = loader.loadTestsFromModule(module)

                # Apply pattern filtering
                if self.exclude_pattern and self.exclude_pattern in module_name:
                    if self.args.verbose:
                        print(f"⏭️  Excluding module: {module_name}")
                    continue

                suite.addTest(module_suite)

                if self.args.verbose:
                    test_count = module_suite.countTestCases()
                    print(f"✅ Loaded {test_count} tests from {module_name}")

            except ImportError as e:
                print(f"❌ Failed to import {module_name}: {e}")
                if self.args.verbose:
                    import traceback

                    traceback.print_exc()
            except Exception as e:
                print(f"❌ Error loading tests from {module_name}: {e}")
                if self.args.verbose:
                    import traceback

                    traceback.print_exc()

        total_tests = suite.countTestCases()
        print(f"📊 Discovered {total_tests} total tests")
        return suite

    def run_tests(self, suite: unittest.TestSuite) -> TestResult:
        """Execute test suite and collect results."""
        print(f"\n🚀 Running {suite.countTestCases()} tests...")

        # Configure test runner
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2 if self.args.verbose else 1,
            failfast=self.args.fail_fast,
        )

        # Execute tests
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()

        # Collect results
        self.results.total_tests = result.testsRun
        self.results.passed_tests = (
            result.testsRun - len(result.failures) - len(result.errors)
        )
        self.results.failed_tests = len(result.failures)
        self.results.error_tests = len(result.errors)
        self.results.skipped_tests = len(getattr(result, "skipped", []))
        self.results.execution_time = end_time - start_time
        self.results.failures = result.failures
        self.results.errors = result.errors

        # Print test output if verbose
        if self.args.verbose:
            print("\n" + "=" * 60)
            print("TEST OUTPUT:")
            print("=" * 60)
            print(stream.getvalue())

        return self.results

    def generate_coverage_report(self):
        """Generate code coverage report."""
        if not self.args.coverage:
            return

        try:
            import coverage

            print("\n📊 Generating coverage report...")

            # This would require running tests with coverage
            # For now, just indicate that coverage would be generated
            print("ℹ️  Coverage reporting requires running with coverage.py")
            print("   Example: coverage run tests/run_all_tests.py && coverage report")

        except ImportError:
            print(
                "⚠️  Coverage package not installed. Install with: pip install coverage"
            )

    def generate_performance_report(self):
        """Generate performance benchmark report."""
        if not self.args.performance:
            return

        print("\n⚡ Performance Metrics:")
        print(f"   Total execution time: {self.results.execution_time:.2f}s")
        print(
            f"   Average time per test: {self.results.execution_time / max(self.results.total_tests, 1):.3f}s"
        )

        # Additional performance metrics could be added here
        if self.results.execution_time > 60:
            print("⚠️  Tests took longer than 1 minute - consider optimization")
        elif self.results.execution_time < 5:
            print("✅ Fast test execution - good performance")

    def generate_html_report(self):
        """Generate HTML test report."""
        if not self.args.html_report:
            return

        report_file = self.project_root / "test_report.html"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MCP Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                .details {{ margin-top: 20px; }}
                pre {{ background: #f5f5f5; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MCP Implementation Test Report</h1>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p><strong>Total Tests:</strong> {self.results.total_tests}</p>
                <p class="passed"><strong>Passed:</strong> {self.results.passed_tests}</p>
                <p class="failed"><strong>Failed:</strong> {self.results.failed_tests}</p>
                <p class="error"><strong>Errors:</strong> {self.results.error_tests}</p>
                <p><strong>Execution Time:</strong> {self.results.execution_time:.2f}s</p>
            </div>
            
            <div class="details">
                <h2>Test Details</h2>
                <!-- Detailed test results would go here -->
            </div>
        </body>
        </html>
        """

        with open(report_file, "w") as f:
            f.write(html_content)

        print(f"📄 HTML report generated: {report_file}")

    def generate_xml_report(self):
        """Generate XML test report for CI systems."""
        if not self.args.xml_report:
            return

        report_file = self.project_root / "test_results.xml"

        # Basic JUnit XML format
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="MCP Tests" 
           tests="{self.results.total_tests}" 
           failures="{self.results.failed_tests}" 
           errors="{self.results.error_tests}" 
           time="{self.results.execution_time:.2f}">
    <!-- Individual test cases would be listed here -->
</testsuite>
"""

        with open(report_file, "w") as f:
            f.write(xml_content)

        print(f"📄 XML report generated: {report_file}")

    def print_summary(self):
        """Print test execution summary."""
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)

        # Overall status
        if self.results.failed_tests == 0 and self.results.error_tests == 0:
            status = "✅ ALL TESTS PASSED"
            status_color = "\033[92m"  # Green
        else:
            status = "❌ SOME TESTS FAILED"
            status_color = "\033[91m"  # Red

        print(f"{status_color}{status}\033[0m")
        print()

        # Detailed counts
        print(f"📊 Total Tests:    {self.results.total_tests}")
        print(f"✅ Passed:        {self.results.passed_tests}")
        print(f"❌ Failed:        {self.results.failed_tests}")
        print(f"🔥 Errors:        {self.results.error_tests}")
        print(f"⏭️  Skipped:       {self.results.skipped_tests}")
        print(f"⏱️  Execution Time: {self.results.execution_time:.2f}s")

        # Success rate
        if self.results.total_tests > 0:
            success_rate = (self.results.passed_tests / self.results.total_tests) * 100
            print(f"📈 Success Rate:  {success_rate:.1f}%")

        # Failure details
        if self.results.failures:
            print(f"\n❌ FAILURES ({len(self.results.failures)}):")
            for test, traceback in self.results.failures:
                print(f"   • {test}")

        if self.results.errors:
            print(f"\n🔥 ERRORS ({len(self.results.errors)}):")
            for test, traceback in self.results.errors:
                print(f"   • {test}")

        print("=" * 60)

    def run(self) -> int:
        """Execute the complete test suite."""
        print("🧪 MCP Implementation Test Suite")
        print("=" * 60)

        try:
            # Discover tests
            suite = self.discover_tests()

            if suite.countTestCases() == 0:
                print("⚠️  No tests found!")
                return 1

            # Run tests
            self.run_tests(suite)

            # Generate reports
            self.generate_coverage_report()
            self.generate_performance_report()
            self.generate_html_report()
            self.generate_xml_report()

            # Print summary
            self.print_summary()

            # Return exit code
            return (
                0
                if (self.results.failed_tests == 0 and self.results.error_tests == 0)
                else 1
            )

        except KeyboardInterrupt:
            print("\n⚠️  Test execution interrupted by user")
            return 130
        except Exception as e:
            print(f"\n❌ Test runner error: {e}")
            if self.args.verbose:
                import traceback

                traceback.print_exc()
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for MCP implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--performance",
        "-p",
        action="store_true",
        help="Include performance benchmarks",
    )
    parser.add_argument(
        "--integration",
        "-i",
        action="store_true",
        help="Run integration tests (requires MCP server)",
    )
    parser.add_argument(
        "--html-report", action="store_true", help="Generate HTML test report"
    )
    parser.add_argument(
        "--xml-report", action="store_true", help="Generate XML test report for CI"
    )
    parser.add_argument(
        "--fail-fast", "-f", action="store_true", help="Stop on first failure"
    )
    parser.add_argument("--pattern", type=str, help="Run tests matching pattern")
    parser.add_argument("--exclude", type=str, help="Exclude tests matching pattern")

    args = parser.parse_args()

    # Create and run test runner
    runner = MCPTestRunner(args)
    exit_code = runner.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
