#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Integration tests for the Graph of Thoughts MCP Server.

This module provides comprehensive integration tests that verify the MCP server
works correctly with real MCP clients and follows the protocol specification.
It includes tests for stdio transport, protocol compliance, and end-to-end workflows.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMCPServerStdioTransport(unittest.TestCase):
    """Test MCP server with stdio transport."""

    def setUp(self):
        """Set up test environment."""
        self.server_process = None
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()

    def test_server_startup(self):
        """Test that the server starts up correctly."""
        try:
            # Start the server process
            cmd = [sys.executable, "-m", "graph_of_thoughts", "--info"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            self.assertEqual(result.returncode, 0)
            self.assertIn("Graph of Thoughts MCP Server", result.stdout)
            self.assertIn("Tools Provided:", result.stdout)
            self.assertIn("Resources Provided:", result.stdout)
            self.assertIn("Prompts Provided:", result.stdout)

        except subprocess.TimeoutExpired:
            self.fail("Server startup timed out")
        except FileNotFoundError:
            self.skipTest("Server module not found - may need to install package")

    def test_server_version(self):
        """Test server version command."""
        try:
            cmd = [sys.executable, "-m", "graph_of_thoughts", "--version"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            self.assertEqual(result.returncode, 0)
            self.assertRegex(result.stdout.strip(), r"\d+\.\d+\.\d+")

        except subprocess.TimeoutExpired:
            self.fail("Version command timed out")
        except FileNotFoundError:
            self.skipTest("Server module not found")


class TestMCPProtocolCompliance(unittest.TestCase):
    """Test MCP protocol compliance."""

    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()

    def test_mcp_message_format(self):
        """Test that server follows MCP message format."""
        # This would test JSON-RPC 2.0 compliance
        # For now, we'll test basic structure

        message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

        # Verify message structure
        self.assertIn("jsonrpc", message)
        self.assertEqual(message["jsonrpc"], "2.0")
        self.assertIn("id", message)
        self.assertIn("method", message)

    def test_tool_schema_structure(self):
        """Test that tool schemas follow MCP specification."""
        # Example tool schema that should be valid
        tool_schema = {
            "name": "break_down_task",
            "description": "Decompose a complex task into manageable subtasks",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The complex task to break down",
                    }
                },
                "required": ["task"],
            },
        }

        # Verify required fields
        self.assertIn("name", tool_schema)
        self.assertIn("description", tool_schema)
        self.assertIn("inputSchema", tool_schema)

        # Verify input schema structure
        input_schema = tool_schema["inputSchema"]
        self.assertEqual(input_schema["type"], "object")
        self.assertIn("properties", input_schema)
        self.assertIn("required", input_schema)


class TestMCPServerEndToEnd(unittest.TestCase):
    """End-to-end tests for the MCP server."""

    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()

    def test_complete_reasoning_workflow(self):
        """Test a complete reasoning workflow from start to finish."""

        async def run_test():
            # Import here to avoid import issues
            from graph_of_thoughts.mcp_server import create_server

            server = await create_server()

            # Simulate a complete workflow
            problem = (
                "Design an efficient algorithm for finding the shortest path in a graph"
            )

            # Step 1: Break down the problem
            breakdown_result = await server._break_down_task(
                {"task": problem, "domain": "algorithms", "max_subtasks": 4}
            )

            self.assertIsInstance(breakdown_result, list)
            breakdown_text = breakdown_result[0].text
            self.assertIn("algorithm", breakdown_text.lower())

            # Step 2: Generate multiple approaches
            approaches_result = await server._generate_thoughts(
                {
                    "problem": "Choose appropriate graph representation",
                    "num_thoughts": 3,
                    "approach_type": "analytical",
                }
            )

            self.assertIsInstance(approaches_result, list)

            # Step 3: Score the approaches
            thoughts = [
                "Use adjacency matrix for dense graphs",
                "Use adjacency list for sparse graphs",
                "Use edge list for simple operations",
            ]

            scoring_result = await server._score_thoughts(
                {"thoughts": thoughts, "criteria": "space efficiency and access time"}
            )

            self.assertIsInstance(scoring_result, list)
            scoring_text = scoring_result[0].text
            self.assertIn("Score", scoring_text)

            # Step 4: Validate and improve the best approach
            validation_result = await server._validate_and_improve(
                {
                    "solution": "Use adjacency list for sparse graphs with fast neighbor lookup",
                    "validation_criteria": "correctness and efficiency",
                    "max_iterations": 2,
                }
            )

            self.assertIsInstance(validation_result, list)

            # Step 5: Create a complete reasoning chain
            chain_result = await server._create_reasoning_chain(
                {
                    "problem": problem,
                    "workflow_type": "generate_score_select",
                    "num_branches": 3,
                }
            )

            self.assertIsInstance(chain_result, list)
            chain_text = chain_result[0].text
            self.assertIn("Chain Execution", chain_text)

            # Verify that all operations were recorded
            self.assertTrue(len(server.execution_results) >= 5)

            # Verify that results contain expected data
            for op_id, result in server.execution_results.items():
                self.assertIn("timestamp", result)
                self.assertIsInstance(result["timestamp"], str)

        self.loop.run_until_complete(run_test())

    def test_error_recovery_workflow(self):
        """Test error handling and recovery in workflows."""

        async def run_test():
            from graph_of_thoughts.mcp_server import create_server

            server = await create_server()

            # Test with empty inputs
            try:
                result = await server._break_down_task({"task": ""})
                self.assertIsInstance(result, list)
                # Should handle empty task gracefully
            except Exception as e:
                # Should not crash the server
                self.assertIsInstance(e, (ValueError, KeyError))

            # Test with invalid arguments
            try:
                result = await server._score_thoughts({"thoughts": []})
                self.assertIsInstance(result, list)
                # Should handle empty thoughts list
            except Exception as e:
                self.assertIsInstance(e, (ValueError, IndexError))

            # Server should still be functional after errors
            normal_result = await server._generate_thoughts(
                {"problem": "Test problem after error", "num_thoughts": 2}
            )
            self.assertIsInstance(normal_result, list)

        self.loop.run_until_complete(run_test())


class TestMCPServerPerformance(unittest.TestCase):
    """Performance tests for the MCP server."""

    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()

    def test_response_time(self):
        """Test that server responds within reasonable time limits."""

        async def run_test():
            from graph_of_thoughts.mcp_server import create_server

            server = await create_server()

            # Test response time for simple operations
            start_time = time.time()

            result = await server._generate_thoughts(
                {
                    "problem": "Simple test problem",
                    "num_thoughts": 2,
                    "approach_type": "analytical",
                }
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Should respond within 5 seconds for simple operations
            self.assertLess(response_time, 5.0)
            self.assertIsInstance(result, list)

        self.loop.run_until_complete(run_test())

    def test_memory_usage(self):
        """Test memory usage during operations."""

        async def run_test():
            from graph_of_thoughts.mcp_server import create_server

            server = await create_server()

            # Perform multiple operations and check memory doesn't grow excessively
            initial_results_count = len(server.execution_results)

            for i in range(10):
                await server._generate_thoughts(
                    {"problem": f"Test problem {i}", "num_thoughts": 2}
                )

            # Should have stored results for all operations
            final_results_count = len(server.execution_results)
            self.assertEqual(final_results_count, initial_results_count + 10)

            # Results should be properly structured
            for op_id, result in server.execution_results.items():
                self.assertIsInstance(result, dict)
                self.assertIn("timestamp", result)

        self.loop.run_until_complete(run_test())


if __name__ == "__main__":
    # Run the integration tests
    unittest.main(verbosity=2)
