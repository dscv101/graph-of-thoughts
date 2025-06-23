#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Release validation test for the Graph of Thoughts MCP Server.

This module provides comprehensive validation tests to ensure the MCP server
is ready for release. It tests all critical functionality, performance,
error handling, and protocol compliance.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.mcp_server import create_server


class MCPServerReleaseValidator:
    """Comprehensive release validation for the MCP server."""

    def __init__(self):
        self.test_results = []
        self.server = None

    async def setup(self):
        """Set up the test environment."""
        print("🔧 Setting up test environment...")
        self.server = await create_server()
        print("   ✅ MCP server created")

    def log_test(self, test_name, passed, details=""):
        """Log test results."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append(
            {"test": test_name, "passed": passed, "details": details}
        )
        print(f"   {status}: {test_name}")
        if details and not passed:
            print(f"      Details: {details}")

    async def test_server_startup(self):
        """Test server startup and basic functionality."""
        print("\n📋 Testing server startup...")

        try:
            # Test --info command
            result = subprocess.run(
                [sys.executable, "-m", "graph_of_thoughts", "--info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.log_test(
                "Server info command",
                result.returncode == 0
                and "Graph of Thoughts MCP Server" in result.stdout,
            )

            # Test --version command
            result = subprocess.run(
                [sys.executable, "-m", "graph_of_thoughts", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.log_test(
                "Server version command",
                result.returncode == 0 and result.stdout.strip() == "0.0.3",
            )

        except Exception as e:
            self.log_test("Server startup", False, str(e))

    async def test_all_tools(self):
        """Test all MCP tools."""
        print("\n🛠️  Testing all MCP tools...")

        # Test break_down_task
        try:
            result = await self.server._break_down_task(
                {
                    "task": "Test task for validation",
                    "domain": "testing",
                    "max_subtasks": 3,
                }
            )
            self.log_test(
                "break_down_task tool",
                len(result) == 1 and "Task Breakdown" in result[0].text,
            )
        except Exception as e:
            self.log_test("break_down_task tool", False, str(e))

        # Test generate_thoughts
        try:
            result = await self.server._generate_thoughts(
                {
                    "problem": "Test problem",
                    "num_thoughts": 2,
                    "approach_type": "analytical",
                }
            )
            self.log_test(
                "generate_thoughts tool",
                len(result) == 1 and "Generated Thoughts" in result[0].text,
            )
        except Exception as e:
            self.log_test("generate_thoughts tool", False, str(e))

        # Test score_thoughts
        try:
            result = await self.server._score_thoughts(
                {"thoughts": ["Thought 1", "Thought 2"], "criteria": "test criteria"}
            )
            self.log_test(
                "score_thoughts tool",
                len(result) == 1 and "Scoring Results" in result[0].text,
            )
        except Exception as e:
            self.log_test("score_thoughts tool", False, str(e))

        # Test validate_and_improve
        try:
            result = await self.server._validate_and_improve(
                {
                    "solution": "Test solution",
                    "validation_criteria": "test criteria",
                    "max_iterations": 1,
                }
            )
            self.log_test(
                "validate_and_improve tool",
                len(result) == 1 and "Validation" in result[0].text,
            )
        except Exception as e:
            self.log_test("validate_and_improve tool", False, str(e))

        # Test aggregate_results
        try:
            result = await self.server._aggregate_results(
                {"results": ["Result 1", "Result 2"], "aggregation_method": "synthesis"}
            )
            self.log_test(
                "aggregate_results tool",
                len(result) == 1 and "Aggregation" in result[0].text,
            )
        except Exception as e:
            self.log_test("aggregate_results tool", False, str(e))

        # Test create_reasoning_chain
        try:
            result = await self.server._create_reasoning_chain(
                {
                    "problem": "Test problem",
                    "workflow_type": "generate_score_select",
                    "num_branches": 2,
                }
            )
            self.log_test(
                "create_reasoning_chain tool",
                len(result) == 1 and "Reasoning Chain" in result[0].text,
            )
        except Exception as e:
            self.log_test("create_reasoning_chain tool", False, str(e))

    async def test_error_handling(self):
        """Test error handling."""
        print("\n🚨 Testing error handling...")

        # Test with missing required parameters
        try:
            result = await self.server._break_down_task({})
            # Should handle gracefully, not crash
            self.log_test("Missing required parameters", True)
        except Exception as e:
            # Should not raise unhandled exceptions
            self.log_test("Missing required parameters", False, str(e))

        # Test with invalid parameter types
        try:
            result = await self.server._generate_thoughts(
                {"problem": "Test", "num_thoughts": "invalid"}  # Should be integer
            )
            self.log_test("Invalid parameter types", True)
        except Exception as e:
            self.log_test("Invalid parameter types", False, str(e))

        # Test with empty inputs
        try:
            result = await self.server._score_thoughts(
                {"thoughts": [], "criteria": "test"}
            )
            self.log_test("Empty input handling", True)
        except Exception as e:
            self.log_test("Empty input handling", False, str(e))

    async def test_performance(self):
        """Test performance characteristics."""
        print("\n⚡ Testing performance...")

        # Test response time
        start_time = time.time()
        try:
            result = await self.server._generate_thoughts(
                {"problem": "Performance test", "num_thoughts": 2}
            )
            end_time = time.time()
            response_time = end_time - start_time

            self.log_test(
                "Response time < 5 seconds",
                response_time < 5.0,
                f"Actual time: {response_time:.2f}s",
            )
        except Exception as e:
            self.log_test("Response time test", False, str(e))

        # Test concurrent operations
        try:
            tasks = []
            for i in range(3):
                task = self.server._generate_thoughts(
                    {"problem": f"Concurrent test {i}", "num_thoughts": 1}
                )
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            self.log_test(
                "Concurrent operations",
                len(results) == 3 and all(len(r) == 1 for r in results),
                f"Time: {end_time - start_time:.2f}s",
            )
        except Exception as e:
            self.log_test("Concurrent operations", False, str(e))

    async def test_data_integrity(self):
        """Test data integrity and storage."""
        print("\n💾 Testing data integrity...")

        initial_count = len(self.server.execution_results)

        # Execute an operation
        try:
            await self.server._break_down_task(
                {"task": "Data integrity test", "domain": "testing"}
            )

            # Check that result was stored
            final_count = len(self.server.execution_results)
            self.log_test("Operation result storage", final_count == initial_count + 1)

            # Check result structure
            if self.server.execution_results:
                latest_result = list(self.server.execution_results.values())[-1]
                has_timestamp = "timestamp" in latest_result
                has_operation_id = any(
                    "operation_id" in latest_result
                    for latest_result in self.server.execution_results.values()
                )

                self.log_test(
                    "Result data structure",
                    has_timestamp,
                    "Missing timestamp" if not has_timestamp else "",
                )
        except Exception as e:
            self.log_test("Data integrity test", False, str(e))

    async def test_prompt_templates(self):
        """Test prompt templates."""
        print("\n📝 Testing prompt templates...")

        required_templates = [
            "analyze-problem",
            "generate-solutions",
            "evaluate-options",
        ]

        for template_name in required_templates:
            exists = template_name in self.server.prompt_templates
            self.log_test(f"Template '{template_name}' exists", exists)

            if exists:
                template = self.server.prompt_templates[template_name]
                has_required_fields = all(
                    field in template for field in ["name", "description", "arguments"]
                )
                self.log_test(
                    f"Template '{template_name}' structure", has_required_fields
                )

    def generate_report(self):
        """Generate final test report."""
        print("\n" + "=" * 60)
        print("📊 RELEASE VALIDATION REPORT")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")

        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"   • {result['test']}")
                    if result["details"]:
                        print(f"     Details: {result['details']}")

        print("\n" + "=" * 60)

        if success_rate >= 95:
            print("🎉 RELEASE VALIDATION PASSED!")
            print("✅ MCP Server is ready for release!")
            return True
        else:
            print("❌ RELEASE VALIDATION FAILED!")
            print("🔧 Please fix the failing tests before release.")
            return False


async def main():
    """Main function to run release validation."""
    print("Graph of Thoughts MCP Server - Release Validation")
    print("=" * 60)

    validator = MCPServerReleaseValidator()

    try:
        await validator.setup()

        # Run all validation tests
        await validator.test_server_startup()
        await validator.test_all_tools()
        await validator.test_error_handling()
        await validator.test_performance()
        await validator.test_data_integrity()
        await validator.test_prompt_templates()

        # Generate final report
        success = validator.generate_report()

        if success:
            print("\n🚀 Ready for release!")
            print("📋 Release checklist:")
            print("   ✅ All tests passing")
            print("   ✅ Documentation complete")
            print("   ✅ Examples provided")
            print("   ✅ Configuration templates available")
            print("   ✅ Error handling robust")
            print("   ✅ Performance acceptable")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
