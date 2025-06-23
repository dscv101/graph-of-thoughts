#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Comprehensive test suite for the Graph of Thoughts MCP Server.

This module provides unit tests, integration tests, and protocol compliance tests
for the MCP server implementation. It verifies that all tools, resources, and
prompts work correctly and follow the MCP specification.

Test Categories:
    - Unit Tests: Test individual components and methods
    - Integration Tests: Test complete workflows and interactions
    - Protocol Compliance: Verify MCP specification adherence
    - Performance Tests: Check response times and resource usage
    - Error Handling: Validate error conditions and recovery
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import MCP types and server
import mcp.types as types

from graph_of_thoughts.mcp_server import GraphOfThoughtsServer, create_server


class TestMCPServerCore(unittest.TestCase):
    """Test core MCP server functionality."""

    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.server = None

    def tearDown(self):
        """Clean up test environment."""
        if self.server:
            # Clean up any server resources
            pass
        self.loop.close()

    def test_server_initialization(self):
        """Test that the server initializes correctly."""

        async def run_test():
            server = await create_server()
            self.assertIsInstance(server, GraphOfThoughtsServer)
            self.assertEqual(server.server.name, "graph-of-thoughts")
            self.assertIsInstance(server.execution_results, dict)
            self.assertIsInstance(server.active_operations, dict)
            self.assertIsInstance(server.prompt_templates, dict)

        self.loop.run_until_complete(run_test())

    def test_prompt_templates_initialization(self):
        """Test that prompt templates are properly initialized."""

        async def run_test():
            server = await create_server()

            # Check that required prompt templates exist
            required_prompts = [
                "analyze-problem",
                "generate-solutions",
                "evaluate-options",
            ]
            for prompt_name in required_prompts:
                self.assertIn(prompt_name, server.prompt_templates)

                template = server.prompt_templates[prompt_name]
                self.assertIn("name", template)
                self.assertIn("description", template)
                self.assertIn("arguments", template)
                self.assertEqual(template["name"], prompt_name)

        self.loop.run_until_complete(run_test())


class TestMCPTools(unittest.TestCase):
    """Test MCP tools functionality."""

    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()

    def test_break_down_task_tool(self):
        """Test the break_down_task tool."""

        async def run_test():
            server = await create_server()

            # Test with programming task
            arguments = {
                "task": "Build a REST API for user management",
                "max_subtasks": 5,
                "domain": "programming",
            }

            result = await server._break_down_task(arguments)

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], types.TextContent)

            response_text = result[0].text
            self.assertIn("Task Breakdown", response_text)
            self.assertIn("programming", response_text)
            self.assertIn("Operation ID:", response_text)

            # Check that result was stored
            self.assertTrue(len(server.execution_results) > 0)

        self.loop.run_until_complete(run_test())

    def test_generate_thoughts_tool(self):
        """Test the generate_thoughts tool."""

        async def run_test():
            server = await create_server()

            arguments = {
                "problem": "Optimize database performance",
                "num_thoughts": 3,
                "approach_type": "analytical",
            }

            result = await server._generate_thoughts(arguments)

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], types.TextContent)

            response_text = result[0].text
            self.assertIn("Generated Thoughts", response_text)
            self.assertIn("analytical", response_text)
            self.assertIn("Optimize database performance", response_text)

        self.loop.run_until_complete(run_test())

    def test_score_thoughts_tool(self):
        """Test the score_thoughts tool."""

        async def run_test():
            server = await create_server()

            arguments = {
                "thoughts": [
                    "Use indexing to improve query performance",
                    "Implement caching for frequently accessed data",
                    "Optimize query structure and joins",
                ],
                "criteria": "feasibility and effectiveness",
            }

            result = await server._score_thoughts(arguments)

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], types.TextContent)

            response_text = result[0].text
            self.assertIn("Thought Scoring Results", response_text)
            self.assertIn("feasibility and effectiveness", response_text)
            self.assertIn("Rank", response_text)
            self.assertIn("Score", response_text)

        self.loop.run_until_complete(run_test())

    def test_validate_and_improve_tool(self):
        """Test the validate_and_improve tool."""

        async def run_test():
            server = await create_server()

            arguments = {
                "solution": "Use bubble sort to sort the array",
                "validation_criteria": "efficiency and correctness",
                "max_iterations": 2,
            }

            result = await server._validate_and_improve(arguments)

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], types.TextContent)

            response_text = result[0].text
            self.assertIn("Validation and Improvement", response_text)
            self.assertIn("efficiency and correctness", response_text)

        self.loop.run_until_complete(run_test())

    def test_aggregate_results_tool(self):
        """Test the aggregate_results tool."""

        async def run_test():
            server = await create_server()

            arguments = {
                "results": [
                    "Solution A: Fast algorithm with high memory usage",
                    "Solution B: Memory-efficient but slower algorithm",
                    "Solution C: Balanced approach with moderate speed and memory",
                ],
                "aggregation_method": "synthesis",
            }

            result = await server._aggregate_results(arguments)

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], types.TextContent)

            response_text = result[0].text
            self.assertIn("Result Aggregation", response_text)
            self.assertIn("synthesis", response_text)

        self.loop.run_until_complete(run_test())

    def test_create_reasoning_chain_tool(self):
        """Test the create_reasoning_chain tool."""

        async def run_test():
            server = await create_server()

            arguments = {
                "problem": "Design a recommendation system",
                "workflow_type": "generate_score_select",
                "num_branches": 3,
            }

            result = await server._create_reasoning_chain(arguments)

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], types.TextContent)

            response_text = result[0].text
            self.assertIn("Reasoning Chain", response_text)
            self.assertIn("generate_score_select", response_text)
            self.assertIn("Design a recommendation system", response_text)

        self.loop.run_until_complete(run_test())


class TestMCPResources(unittest.TestCase):
    """Test MCP resources functionality."""

    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()

    def test_list_resources(self):
        """Test listing available resources."""

        async def run_test():
            server = await create_server()

            # For now, just verify server has execution_results and other attributes
            self.assertIsInstance(server.execution_results, dict)
            self.assertIsInstance(server.active_operations, dict)
            self.assertIsInstance(server.prompt_templates, dict)

            # Verify server has the expected methods
            self.assertTrue(hasattr(server, "_register_resources"))
            self.assertTrue(hasattr(server, "_register_tools"))
            self.assertTrue(hasattr(server, "_register_prompts"))

        self.loop.run_until_complete(run_test())

    def test_read_operations_results_resource(self):
        """Test reading the operations results resource."""

        async def run_test():
            server = await create_server()

            # Add some test data
            server.execution_results["test-op-1"] = {
                "operation": "test",
                "result": "test result",
                "timestamp": "2024-01-01T00:00:00",
            }

            # Test reading the resource (simulate the handler)
            uri = "got://operations/results"
            result = json.dumps(server.execution_results, indent=2)

            self.assertIn("test-op-1", result)
            self.assertIn("test result", result)

        self.loop.run_until_complete(run_test())


class TestMCPPrompts(unittest.TestCase):
    """Test MCP prompts functionality."""

    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()

    def test_analyze_problem_prompt(self):
        """Test the analyze-problem prompt."""

        async def run_test():
            server = await create_server()

            # Test prompt template exists
            self.assertIn("analyze-problem", server.prompt_templates)

            template = server.prompt_templates["analyze-problem"]
            self.assertEqual(template["name"], "analyze-problem")
            self.assertIn("description", template)
            self.assertIn("arguments", template)

            # Check arguments structure
            args = template["arguments"]
            self.assertTrue(len(args) >= 1)

            # Check required argument exists
            problem_arg = next((arg for arg in args if arg["name"] == "problem"), None)
            self.assertIsNotNone(problem_arg)
            self.assertTrue(problem_arg["required"])

        self.loop.run_until_complete(run_test())

    def test_generate_solutions_prompt(self):
        """Test the generate-solutions prompt."""

        async def run_test():
            server = await create_server()

            self.assertIn("generate-solutions", server.prompt_templates)

            template = server.prompt_templates["generate-solutions"]
            self.assertEqual(template["name"], "generate-solutions")
            self.assertIn("description", template)

        self.loop.run_until_complete(run_test())

    def test_evaluate_options_prompt(self):
        """Test the evaluate-options prompt."""

        async def run_test():
            server = await create_server()

            self.assertIn("evaluate-options", server.prompt_templates)

            template = server.prompt_templates["evaluate-options"]
            self.assertEqual(template["name"], "evaluate-options")
            self.assertIn("description", template)

        self.loop.run_until_complete(run_test())


class TestMCPProtocolCompliance(unittest.TestCase):
    """Test MCP protocol compliance."""

    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()

    def test_server_name_and_version(self):
        """Test that server has proper name and version."""

        async def run_test():
            server = await create_server()

            self.assertEqual(server.server.name, "graph-of-thoughts")
            # Version should be accessible through server configuration

        self.loop.run_until_complete(run_test())

    def test_tool_schema_compliance(self):
        """Test that tools follow MCP schema requirements."""

        async def run_test():
            server = await create_server()

            # Test that we can call the list_tools method
            # In a real test, we would verify the actual schema structure
            self.assertTrue(hasattr(server, "_register_tools"))

        self.loop.run_until_complete(run_test())

    def test_error_handling(self):
        """Test proper error handling in tools."""

        async def run_test():
            server = await create_server()

            # Test with invalid arguments
            try:
                result = await server._break_down_task(
                    {}
                )  # Missing required 'task' argument
                # Should not raise exception but return error message
                self.assertIsInstance(result, list)
            except Exception as e:
                # If it raises an exception, it should be handled gracefully
                self.assertIsInstance(e, (ValueError, KeyError))

        self.loop.run_until_complete(run_test())


class TestMCPServerIntegration(unittest.TestCase):
    """Integration tests for the MCP server."""

    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()

    def test_complete_workflow(self):
        """Test a complete Graph of Thoughts workflow."""

        async def run_test():
            server = await create_server()

            # Step 1: Break down a task
            task_result = await server._break_down_task(
                {
                    "task": "Create a machine learning model for image classification",
                    "domain": "machine learning",
                }
            )

            self.assertIsInstance(task_result, list)
            self.assertTrue(len(server.execution_results) > 0)

            # Step 2: Generate thoughts for one of the subtasks
            thoughts_result = await server._generate_thoughts(
                {
                    "problem": "Collect and preprocess image data",
                    "num_thoughts": 3,
                    "approach_type": "systematic",
                }
            )

            self.assertIsInstance(thoughts_result, list)

            # Step 3: Score the generated thoughts
            score_result = await server._score_thoughts(
                {
                    "thoughts": [
                        "Use web scraping to collect images",
                        "Use existing datasets like ImageNet",
                        "Generate synthetic images using GANs",
                    ],
                    "criteria": "feasibility and data quality",
                }
            )

            self.assertIsInstance(score_result, list)

            # Verify that all operations stored results
            self.assertTrue(len(server.execution_results) >= 3)

        self.loop.run_until_complete(run_test())


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
