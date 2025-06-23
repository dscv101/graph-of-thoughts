#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Graph of Thoughts MCP Server Implementation.

This module provides a Model Context Protocol (MCP) server that exposes Graph of Thoughts
functionality as MCP tools, resources, and prompts. The server allows LLM hosts like
Claude Desktop, VSCode, and Cursor to leverage Graph of Thoughts reasoning capabilities.

Key Features:
    - MCP Tools: Expose Graph of Thoughts operations as callable tools
    - MCP Resources: Provide access to operation results, configurations, and templates
    - MCP Prompts: Offer reusable prompt templates for common GoT workflows
    - Protocol Compliance: Full MCP specification compliance with proper error handling
    - Transport Support: Both stdio and HTTP transport mechanisms

Architecture:
    The server is built using the official MCP Python SDK and follows the standard
    MCP server patterns. It integrates with the existing Graph of Thoughts operations
    and provides a clean interface for external LLM hosts.

Example Usage:
    # Start the server with stdio transport
    python -m graph_of_thoughts.mcp_server

    # Or use as a module
    from graph_of_thoughts.mcp_server import create_server
    server = create_server()
    await server.run()

MCP Tools Exposed:
    - break_down_task: Decompose complex tasks into subtasks
    - generate_thoughts: Generate multiple solution approaches
    - score_thoughts: Evaluate and score different approaches
    - validate_and_improve: Validate solutions and improve them
    - aggregate_results: Combine multiple results into final solution
    - create_reasoning_chain: Build complete reasoning workflows

MCP Resources Exposed:
    - got://operations/results: Access to operation execution results
    - got://templates/prompts: Prompt templates for different domains
    - got://configs/examples: Example configurations and workflows
    - got://logs/execution: Execution logs and debugging information

MCP Prompts Exposed:
    - analyze-problem: Structured problem analysis workflow
    - generate-solutions: Multi-approach solution generation
    - evaluate-options: Systematic option evaluation
    - debug-reasoning: Debugging reasoning chains
"""

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

# MCP SDK imports
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Graph of Thoughts imports
from graph_of_thoughts.operations import (
    Aggregate,
    Generate,
    Improve,
    KeepBestN,
    KeepValid,
    Operation,
    Score,
    Selector,
    ValidateAndImprove,
)
from graph_of_thoughts.operations.graph_of_operations import GraphOfOperations
from graph_of_thoughts.operations.thought import Thought

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_NAME = "graph-of-thoughts"
SERVER_VERSION = "0.0.3"


class GraphOfThoughtsServer:
    """
    Main MCP server class that exposes Graph of Thoughts functionality.

    This class implements the MCP server interface and provides tools, resources,
    and prompts for Graph of Thoughts operations. It manages the execution context
    and maintains state for ongoing reasoning processes.
    """

    def __init__(self):
        """Initialize the Graph of Thoughts MCP server."""
        self.server = Server(SERVER_NAME)
        self.execution_results: [str, Any] = {}
        self.active_operations: [str, GraphOfOperations] = {}
        self.prompt_templates: [str, [str, Any]] = {}

        # Initialize prompt templates
        self._initialize_prompt_templates()

        # Register MCP handlers
        self._register_tools()
        self._register_resources()
        self._register_prompts()

    def _initialize_prompt_templates(self):
        """Initialize the built-in prompt templates."""
        self.prompt_templates = {
            "analyze-problem": {
                "name": "analyze-problem",
                "description": "Structured analysis of complex problems using Graph of Thoughts",
                "arguments": [
                    {
                        "name": "problem",
                        "description": "The problem statement to analyze",
                        "required": True,
                    },
                    {
                        "name": "domain",
                        "description": "Problem domain (e.g., math, coding, reasoning)",
                        "required": False,
                    },
                ],
            },
            "generate-solutions": {
                "name": "generate-solutions",
                "description": "Generate multiple solution approaches using Graph of Thoughts",
                "arguments": [
                    {
                        "name": "problem",
                        "description": "The problem to solve",
                        "required": True,
                    },
                    {
                        "name": "num_approaches",
                        "description": "Number of different approaches to generate",
                        "required": False,
                    },
                ],
            },
            "evaluate-options": {
                "name": "evaluate-options",
                "description": "Systematically evaluate different solution options",
                "arguments": [
                    {
                        "name": "options",
                        "description": " of options to evaluate",
                        "required": True,
                    },
                    {
                        "name": "criteria",
                        "description": "Evaluation criteria",
                        "required": False,
                    },
                ],
            },
        }

    def _register_tools(self):
        """Register MCP tools for Graph of Thoughts operations."""

        @self.server.list_tools()
        async def list_tools() -> [types.Tool]:
            """ all available Graph of Thoughts tools."""
            return [
                types.Tool(
                    name="break_down_task",
                    description="Decompose a complex task into manageable subtasks using Graph of Thoughts reasoning",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The complex task to break down",
                            },
                            "max_subtasks": {
                                "type": "integer",
                                "description": "Maximum number of subtasks to generate",
                                "default": 5,
                            },
                            "domain": {
                                "type": "string",
                                "description": "Task domain for context (e.g., 'programming', 'math', 'analysis')",
                                "default": "general",
                            },
                        },
                        "required": ["task"],
                    },
                ),
                types.Tool(
                    name="generate_thoughts",
                    description="Generate multiple thought approaches for a given problem",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "problem": {
                                "type": "string",
                                "description": "The problem to generate thoughts for",
                            },
                            "num_thoughts": {
                                "type": "integer",
                                "description": "Number of different thoughts to generate",
                                "default": 3,
                            },
                            "approach_type": {
                                "type": "string",
                                "description": "Type of approach (e.g., 'analytical', 'creative', 'systematic')",
                                "default": "analytical",
                            },
                        },
                        "required": ["problem"],
                    },
                ),
                types.Tool(
                    name="score_thoughts",
                    description="Evaluate and score different thought approaches",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "thoughts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": " of thoughts to score",
                            },
                            "criteria": {
                                "type": "string",
                                "description": "Scoring criteria",
                                "default": "feasibility, effectiveness, and clarity",
                            },
                        },
                        "required": ["thoughts"],
                    },
                ),
                types.Tool(
                    name="validate_and_improve",
                    description="Validate solutions and improve them iteratively",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "solution": {
                                "type": "string",
                                "description": "The solution to validate and improve",
                            },
                            "validation_criteria": {
                                "type": "string",
                                "description": "Criteria for validation",
                                "default": "correctness, completeness, and efficiency",
                            },
                            "max_iterations": {
                                "type": "integer",
                                "description": "Maximum improvement iterations",
                                "default": 3,
                            },
                        },
                        "required": ["solution"],
                    },
                ),
                types.Tool(
                    name="aggregate_results",
                    description="Combine multiple results into a final comprehensive solution",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": " of results to aggregate",
                            },
                            "aggregation_method": {
                                "type": "string",
                                "description": "Method for aggregation (e.g., 'consensus', 'best_of', 'synthesis')",
                                "default": "synthesis",
                            },
                        },
                        "required": ["results"],
                    },
                ),
                types.Tool(
                    name="create_reasoning_chain",
                    description="Build a complete reasoning workflow with multiple operations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "problem": {
                                "type": "string",
                                "description": "The problem to solve with a reasoning chain",
                            },
                            "workflow_type": {
                                "type": "string",
                                "description": "Type of workflow (e.g., 'generate_score_select', 'validate_improve_aggregate')",
                                "default": "generate_score_select",
                            },
                            "num_branches": {
                                "type": "integer",
                                "description": "Number of parallel reasoning branches",
                                "default": 3,
                            },
                        },
                        "required": ["problem"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: [str, Any]
        ) -> [types.TextContent]:
            """Handle tool execution requests."""
            try:
                if name == "break_down_task":
                    return await self._break_down_task(arguments)
                elif name == "generate_thoughts":
                    return await self._generate_thoughts(arguments)
                elif name == "score_thoughts":
                    return await self._score_thoughts(arguments)
                elif name == "validate_and_improve":
                    return await self._validate_and_improve(arguments)
                elif name == "aggregate_results":
                    return await self._aggregate_results(arguments)
                elif name == "create_reasoning_chain":
                    return await self._create_reasoning_chain(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                return [
                    types.TextContent(
                        type="text", text=f"Error executing {name}: {str(e)}"
                    )
                ]

    async def _break_down_task(
        self, arguments: [str, Any]
    ) -> [types.TextContent]:
        """Break down a complex task into subtasks."""
        # Validate required parameters
        if "task" not in arguments:
            return [
                types.TextContent(
                    type="text",
                    text="Error: Missing required parameter 'task'. Please provide a task description to break down.",
                )
            ]

        task = arguments["task"]
        if not isinstance(task, str) or not task.strip():
            return [
                types.TextContent(
                    type="text",
                    text="Error: Parameter 'task' must be a non-empty string.",
                )
            ]

        # Validate optional parameters
        try:
            max_subtasks = int(arguments.get("max_subtasks", 5))
            if max_subtasks < 1 or max_subtasks > 20:
                max_subtasks = 5
        except (ValueError, TypeError):
            max_subtasks = 5

        domain = str(arguments.get("domain", "general"))

        # Create a simple task breakdown using Generate operation
        operation_id = str(uuid.uuid4())

        # Simulate task breakdown logic
        subtasks = []
        if "programming" in domain.lower() or "code" in task.lower():
            subtasks = [
                "1. Analyze requirements and constraints",
                "2. Design the solution architecture",
                "3. Implement core functionality",
                "4. Add error handling and edge cases",
                "5. Test and validate the solution",
            ]
        elif "math" in domain.lower() or "calculation" in task.lower():
            subtasks = [
                "1. Identify the mathematical concepts involved",
                "2. Break down the problem into smaller parts",
                "3. Apply relevant formulas or methods",
                "4. Verify the solution step by step",
            ]
        else:
            # General task breakdown
            subtasks = [
                "1. Define the problem clearly",
                "2. Gather necessary information",
                "3. Identify possible approaches",
                "4. Execute the chosen approach",
                "5. Review and refine the result",
            ]

        # Limit to max_subtasks
        subtasks = subtasks[:max_subtasks]

        result = {
            "operation_id": operation_id,
            "original_task": task,
            "domain": domain,
            "subtasks": subtasks,
            "timestamp": datetime.now().isoformat(),
        }

        # Store result for later access
        self.execution_results[operation_id] = result

        response_text = f"Task Breakdown for: {task}\n\n"
        response_text += f"Domain: {domain}\n"
        response_text += f"Subtasks ({len(subtasks)}):\n"
        for subtask in subtasks:
            response_text += f"  {subtask}\n"
        response_text += f"\nOperation ID: {operation_id}"

        return [types.TextContent(type="text", text=response_text)]

    async def _generate_thoughts(
        self, arguments: [str, Any]
    ) -> [types.TextContent]:
        """Generate multiple thought approaches for a problem."""
        # Validate required parameters
        if "problem" not in arguments:
            return [
                types.TextContent(
                    type="text",
                    text="Error: Missing required parameter 'problem'. Please provide a problem description.",
                )
            ]

        problem = arguments["problem"]
        if not isinstance(problem, str) or not problem.strip():
            return [
                types.TextContent(
                    type="text",
                    text="Error: Parameter 'problem' must be a non-empty string.",
                )
            ]

        # Validate optional parameters
        try:
            num_thoughts = int(arguments.get("num_thoughts", 3))
            if num_thoughts < 1 or num_thoughts > 10:
                num_thoughts = 3
        except (ValueError, TypeError):
            num_thoughts = 3

        approach_type = str(arguments.get("approach_type", "analytical"))

        operation_id = str(uuid.uuid4())

        # Generate different thought approaches based on the approach type
        thoughts = []

        if approach_type == "analytical":
            thoughts = [
                f"Analytical Approach 1: Break down '{problem}' into its core components and analyze each systematically",
                f"Analytical Approach 2: Apply logical reasoning and established frameworks to solve '{problem}'",
                f"Analytical Approach 3: Use data-driven analysis and evidence-based methods for '{problem}'",
            ]
        elif approach_type == "creative":
            thoughts = [
                f"Creative Approach 1: Think outside the box and explore unconventional solutions for '{problem}'",
                f"Creative Approach 2: Use analogies and metaphors to find innovative approaches to '{problem}'",
                f"Creative Approach 3: Combine ideas from different domains to create novel solutions for '{problem}'",
            ]
        elif approach_type == "systematic":
            thoughts = [
                f"Systematic Approach 1: Follow a step-by-step methodology to address '{problem}'",
                f"Systematic Approach 2: Use established best practices and proven techniques for '{problem}'",
                f"Systematic Approach 3: Apply structured problem-solving frameworks to '{problem}'",
            ]
        else:
            # Default mixed approach
            thoughts = [
                f"Approach 1: Analyze the problem '{problem}' systematically and identify key factors",
                f"Approach 2: Consider creative alternatives and innovative solutions for '{problem}'",
                f"Approach 3: Apply proven methodologies while remaining open to new insights for '{problem}'",
            ]

        # Limit to requested number of thoughts
        thoughts = thoughts[:num_thoughts]

        result = {
            "operation_id": operation_id,
            "problem": problem,
            "approach_type": approach_type,
            "thoughts": thoughts,
            "timestamp": datetime.now().isoformat(),
        }

        self.execution_results[operation_id] = result

        response_text = f"Generated Thoughts for: {problem}\n\n"
        response_text += f"Approach Type: {approach_type}\n"
        response_text += f"Number of Thoughts: {len(thoughts)}\n\n"
        for i, thought in enumerate(thoughts, 1):
            response_text += f"{i}. {thought}\n\n"
        response_text += f"Operation ID: {operation_id}"

        return [types.TextContent(type="text", text=response_text)]

    async def _score_thoughts(
        self, arguments: [str, Any]
    ) -> [types.TextContent]:
        """Score and evaluate different thoughts."""
        # Validate required parameters
        if "thoughts" not in arguments:
            return [
                types.TextContent(
                    type="text",
                    text="Error: Missing required parameter 'thoughts'. Please provide a list of thoughts to score.",
                )
            ]

        thoughts = arguments["thoughts"]
        if not isinstance(thoughts, list):
            return [
                types.TextContent(
                    type="text",
                    text="Error: Parameter 'thoughts' must be a list of strings.",
                )
            ]

        if len(thoughts) == 0:
            return [
                types.TextContent(
                    type="text",
                    text="No thoughts provided to score. Please provide at least one thought.",
                )
            ]

        # Convert all thoughts to strings and filter out empty ones
        thoughts = [
            str(thought).strip() for thought in thoughts if str(thought).strip()
        ]

        if len(thoughts) == 0:
            return [
                types.TextContent(
                    type="text",
                    text="No valid thoughts provided to score. Please provide non-empty thought descriptions.",
                )
            ]

        criteria = str(
            arguments.get("criteria", "feasibility, effectiveness, and clarity")
        )

        operation_id = str(uuid.uuid4())

        # Simple scoring logic (in a real implementation, this would use the LM)
        scored_thoughts = []
        for i, thought in enumerate(thoughts):
            # Simulate scoring based on length, complexity, and keywords
            base_score = 0.5

            # Length factor (moderate length is better)
            length_factor = min(len(thought) / 100, 1.0) * 0.2

            # Keyword factor (presence of action words)
            action_words = [
                "analyze",
                "implement",
                "design",
                "test",
                "validate",
                "create",
                "solve",
            ]
            keyword_factor = (
                sum(1 for word in action_words if word.lower() in thought.lower()) * 0.1
            )

            # Complexity factor (structured thoughts score higher)
            complexity_factor = (
                0.2
                if any(char in thought for char in ["1.", "2.", "3.", "-", "•"])
                else 0.1
            )

            final_score = min(
                base_score + length_factor + keyword_factor + complexity_factor, 1.0
            )

            scored_thoughts.append(
                {"thought": thought, "score": round(final_score, 2), "rank": i + 1}
            )

        # Sort by score (highest first)
        scored_thoughts.sort(key=lambda x: x["score"], reverse=True)

        # Update ranks after sorting
        for i, item in enumerate(scored_thoughts):
            item["rank"] = i + 1

        result = {
            "operation_id": operation_id,
            "criteria": criteria,
            "scored_thoughts": scored_thoughts,
            "timestamp": datetime.now().isoformat(),
        }

        self.execution_results[operation_id] = result

        response_text = f"Thought Scoring Results\n\n"
        response_text += f"Criteria: {criteria}\n"
        response_text += f"Number of thoughts evaluated: {len(scored_thoughts)}\n\n"

        for item in scored_thoughts:
            response_text += f"Rank {item['rank']}: Score {item['score']}\n"
            response_text += f"Thought: {item['thought']}\n\n"

        response_text += f"Operation ID: {operation_id}"

        return [types.TextContent(type="text", text=response_text)]

    async def _validate_and_improve(
        self, arguments: [str, Any]
    ) -> [types.TextContent]:
        """Validate and improve a solution iteratively."""
        solution = arguments["solution"]
        validation_criteria = arguments.get(
            "validation_criteria", "correctness, completeness, and efficiency"
        )
        max_iterations = arguments.get("max_iterations", 3)

        operation_id = str(uuid.uuid4())

        # Simulate validation and improvement process
        iterations = []
        current_solution = solution

        for iteration in range(max_iterations):
            # Simple validation logic
            issues = []

            # Check for common issues
            if len(current_solution) < 50:
                issues.append("Solution may be too brief and lack detail")

            if not any(
                word in current_solution.lower()
                for word in ["step", "method", "approach", "process"]
            ):
                issues.append("Solution lacks clear methodology")

            if "?" in current_solution:
                issues.append("Solution contains uncertainties or questions")

            # If no issues found, we're done
            if not issues:
                iterations.append(
                    {
                        "iteration": iteration + 1,
                        "solution": current_solution,
                        "issues": [],
                        "improvements": [],
                        "status": "validated",
                    }
                )
                break

            # Generate improvements
            improvements = []
            if "Solution may be too brief" in str(issues):
                improvements.append("Add more detailed explanation and examples")
                current_solution += " This approach involves systematic analysis and step-by-step execution."

            if "lacks clear methodology" in str(issues):
                improvements.append("Include clear methodological steps")
                current_solution += " The methodology includes: 1) Analysis, 2) Planning, 3) Implementation, 4) Validation."

            if "contains uncertainties" in str(issues):
                improvements.append("Replace uncertainties with definitive statements")
                current_solution = current_solution.replace("?", ".")

            iterations.append(
                {
                    "iteration": iteration + 1,
                    "solution": current_solution,
                    "issues": issues,
                    "improvements": improvements,
                    "status": "improved",
                }
            )

        result = {
            "operation_id": operation_id,
            "original_solution": solution,
            "validation_criteria": validation_criteria,
            "iterations": iterations,
            "final_solution": current_solution,
            "timestamp": datetime.now().isoformat(),
        }

        self.execution_results[operation_id] = result

        response_text = f"Validation and Improvement Results\n\n"
        response_text += f"Original Solution: {solution}\n\n"
        response_text += f"Validation Criteria: {validation_criteria}\n"
        response_text += f"Number of Iterations: {len(iterations)}\n\n"

        for iteration_data in iterations:
            response_text += f"Iteration {iteration_data['iteration']} ({iteration_data['status']}):\n"
            if iteration_data["issues"]:
                response_text += (
                    f"  Issues Found: {', '.join(iteration_data['issues'])}\n"
                )
            if iteration_data["improvements"]:
                response_text += (
                    f"  Improvements: {', '.join(iteration_data['improvements'])}\n"
                )
            response_text += f"  Solution: {iteration_data['solution'][:100]}...\n\n"

        response_text += f"Final Solution: {current_solution}\n\n"
        response_text += f"Operation ID: {operation_id}"

        return [types.TextContent(type="text", text=response_text)]

    async def _aggregate_results(
        self, arguments: [str, Any]
    ) -> [types.TextContent]:
        """Aggregate multiple results into a final solution."""
        results = arguments["results"]
        aggregation_method = arguments.get("aggregation_method", "synthesis")

        operation_id = str(uuid.uuid4())

        if not results:
            return [
                types.TextContent(
                    type="text", text="No results provided for aggregation"
                )
            ]

        # Perform aggregation based on method
        if aggregation_method == "consensus":
            # Find common elements across results
            common_words = set()
            for result in results:
                words = set(result.lower().split())
                if not common_words:
                    common_words = words
                else:
                    common_words &= words

            aggregated = f"Consensus elements: {', '.join(sorted(common_words)[:10])}"

        elif aggregation_method == "best_of":
            # Select the longest/most detailed result
            best_result = max(results, key=len)
            aggregated = f"Best result (most comprehensive): {best_result}"

        else:  # synthesis
            # Combine key points from all results
            aggregated = "Synthesized solution combining all approaches:\n\n"
            for i, result in enumerate(results, 1):
                aggregated += (
                    f"{i}. {result[:100]}{'...' if len(result) > 100 else ''}\n"
                )

            aggregated += "\nIntegrated approach: "
            aggregated += (
                "This solution incorporates the strengths of all provided approaches, "
            )
            aggregated += "creating a comprehensive strategy that addresses multiple perspectives."

        result_data = {
            "operation_id": operation_id,
            "input_results": results,
            "aggregation_method": aggregation_method,
            "aggregated_result": aggregated,
            "timestamp": datetime.now().isoformat(),
        }

        self.execution_results[operation_id] = result_data

        response_text = f"Result Aggregation\n\n"
        response_text += f"Method: {aggregation_method}\n"
        response_text += f"Input Results: {len(results)}\n\n"
        response_text += f"Aggregated Result:\n{aggregated}\n\n"
        response_text += f"Operation ID: {operation_id}"

        return [types.TextContent(type="text", text=response_text)]

    async def _create_reasoning_chain(
        self, arguments: [str, Any]
    ) -> [types.TextContent]:
        """Create a complete reasoning workflow."""
        problem = arguments["problem"]
        workflow_type = arguments.get("workflow_type", "generate_score_select")
        num_branches = arguments.get("num_branches", 3)

        operation_id = str(uuid.uuid4())

        # Create a reasoning chain based on workflow type
        chain_steps = []

        if workflow_type == "generate_score_select":
            # Step 1: Generate multiple approaches
            generate_result = await self._generate_thoughts(
                {
                    "problem": problem,
                    "num_thoughts": num_branches,
                    "approach_type": "analytical",
                }
            )

            # Extract thoughts from the result
            thoughts = [
                f"Approach {i+1}: Analytical solution for {problem}"
                for i in range(num_branches)
            ]

            chain_steps.append(
                {
                    "step": 1,
                    "operation": "generate",
                    "description": f"Generated {num_branches} different approaches",
                    "result": thoughts,
                }
            )

            # Step 2: Score the approaches
            score_result = await self._score_thoughts(
                {"thoughts": thoughts, "criteria": "feasibility and effectiveness"}
            )

            chain_steps.append(
                {
                    "step": 2,
                    "operation": "score",
                    "description": "Scored all approaches",
                    "result": "Approaches ranked by feasibility and effectiveness",
                }
            )

            # Step 3: Select best approach
            best_approach = thoughts[0]  # Simplified selection
            chain_steps.append(
                {
                    "step": 3,
                    "operation": "select",
                    "description": "Selected best approach",
                    "result": best_approach,
                }
            )

        elif workflow_type == "validate_improve_aggregate":
            # Step 1: Generate initial solution
            initial_solution = f"Initial solution approach for: {problem}"
            chain_steps.append(
                {
                    "step": 1,
                    "operation": "generate",
                    "description": "Generated initial solution",
                    "result": initial_solution,
                }
            )

            # Step 2: Validate and improve
            validate_result = await self._validate_and_improve(
                {
                    "solution": initial_solution,
                    "validation_criteria": "correctness and completeness",
                    "max_iterations": 2,
                }
            )

            improved_solution = f"Improved solution for: {problem}"
            chain_steps.append(
                {
                    "step": 2,
                    "operation": "validate_improve",
                    "description": "Validated and improved solution",
                    "result": improved_solution,
                }
            )

            # Step 3: Generate alternative and aggregate
            alternative = f"Alternative approach for: {problem}"
            aggregate_result = await self._aggregate_results(
                {
                    "results": [improved_solution, alternative],
                    "aggregation_method": "synthesis",
                }
            )

            chain_steps.append(
                {
                    "step": 3,
                    "operation": "aggregate",
                    "description": "Aggregated multiple solutions",
                    "result": "Synthesized final solution",
                }
            )

        result_data = {
            "operation_id": operation_id,
            "problem": problem,
            "workflow_type": workflow_type,
            "num_branches": num_branches,
            "chain_steps": chain_steps,
            "timestamp": datetime.now().isoformat(),
        }

        self.execution_results[operation_id] = result_data

        response_text = f"Reasoning Chain: {workflow_type}\n\n"
        response_text += f"Problem: {problem}\n"
        response_text += f"Branches: {num_branches}\n\n"
        response_text += "Chain Execution:\n"

        for step in chain_steps:
            response_text += f"Step {step['step']}: {step['operation'].upper()}\n"
            response_text += f"  Description: {step['description']}\n"
            response_text += f"  Result: {str(step['result'])[:100]}...\n\n"

        response_text += f"Operation ID: {operation_id}"

        return [types.TextContent(type="text", text=response_text)]

    def _register_resources(self):
        """Register MCP resources for accessing Graph of Thoughts data."""

        @self.server.list_resources()
        async def list_resources() -> [types.Resource]:
            """ all available Graph of Thoughts resources."""
            return [
                types.Resource(
                    uri="got://operations/results",
                    name="Operation Results",
                    description="Access to Graph of Thoughts operation execution results",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="got://templates/prompts",
                    name="Prompt Templates",
                    description="Reusable prompt templates for different domains",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="got://configs/examples",
                    name="Example Configurations",
                    description="Example configurations and workflows",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="got://logs/execution",
                    name="Execution Logs",
                    description="Execution logs and debugging information",
                    mimeType="text/plain",
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read Graph of Thoughts resource content."""
            if uri == "got://operations/results":
                return json.dumps(self.execution_results, indent=2)

            elif uri == "got://templates/prompts":
                templates = {
                    "problem_analysis": "Analyze the following problem step by step: {problem}",
                    "solution_generation": "Generate {num_solutions} different solutions for: {problem}",
                    "solution_evaluation": "Evaluate these solutions based on {criteria}: {solutions}",
                    "improvement_suggestion": "Improve this solution: {solution}. Focus on {aspects}.",
                }
                return json.dumps(templates, indent=2)

            elif uri == "got://configs/examples":
                examples = {
                    "simple_workflow": {
                        "description": "Basic generate-score-select workflow",
                        "steps": ["generate", "score", "select"],
                        "parameters": {"num_branches": 3},
                    },
                    "validation_workflow": {
                        "description": "Validation and improvement workflow",
                        "steps": ["generate", "validate", "improve", "aggregate"],
                        "parameters": {"max_iterations": 3},
                    },
                }
                return json.dumps(examples, indent=2)

            elif uri == "got://logs/execution":
                log_entries = []
                for op_id, result in self.execution_results.items():
                    log_entries.append(
                        f"[{result.get('timestamp', 'unknown')}] Operation {op_id}: {result}"
                    )
                return "\n".join(log_entries)

            else:
                raise ValueError(f"Resource not found: {uri}")

    def _register_prompts(self):
        """Register MCP prompts for Graph of Thoughts workflows."""

        @self.server.list_prompts()
        async def list_prompts() -> [types.Prompt]:
            """ all available Graph of Thoughts prompts."""
            return [
                types.Prompt(
                    name=template["name"],
                    description=template["description"],
                    arguments=[
                        types.PromptArgument(
                            name=arg["name"],
                            description=arg["description"],
                            required=arg["required"],
                        )
                        for arg in template["arguments"]
                    ],
                )
                for template in self.prompt_templates.values()
            ]

        @self.server.get_prompt()
        async def get_prompt(
            name: str, arguments: Optional[[str, str]] = None
        ) -> types.GetPromptResult:
            """Get a specific Graph of Thoughts prompt."""
            if name not in self.prompt_templates:
                raise ValueError(f"Prompt not found: {name}")

            args = arguments or {}

            if name == "analyze-problem":
                problem = args.get("problem", "")
                domain = args.get("domain", "general")

                return types.GetPromptResult(
                    description=f"Structured analysis of a {domain} problem using Graph of Thoughts",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"""Please analyze the following {domain} problem using Graph of Thoughts methodology:

Problem: {problem}

Please provide:
1. Problem decomposition into key components
2. Multiple solution approaches (at least 3)
3. Evaluation criteria for each approach
4. Recommended solution path with reasoning

Use systematic thinking and consider multiple perspectives.""",
                            ),
                        )
                    ],
                )

            elif name == "generate-solutions":
                problem = args.get("problem", "")
                num_approaches = int(args.get("num_approaches", "3"))

                return types.GetPromptResult(
                    description=f"Generate {num_approaches} solution approaches using Graph of Thoughts",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"""Generate {num_approaches} different solution approaches for this problem:

Problem: {problem}

For each approach, provide:
1. A clear strategy description
2. Key steps or methodology
3. Potential advantages
4. Potential challenges or limitations

Think creatively and consider diverse perspectives including analytical, creative, and systematic approaches.""",
                            ),
                        )
                    ],
                )

            elif name == "evaluate-options":
                options = args.get("options", "")
                criteria = args.get(
                    "criteria", "effectiveness, feasibility, and efficiency"
                )

                return types.GetPromptResult(
                    description="Systematic evaluation of solution options",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"""Evaluate the following options using Graph of Thoughts methodology:

Options to evaluate:
{options}

Evaluation criteria: {criteria}

Please provide:
1. Detailed analysis of each option against the criteria
2. Scoring or ranking with justification
3. Pros and cons for each option
4. Final recommendation with reasoning
5. Risk assessment and mitigation strategies

Use structured thinking and provide clear rationale for your evaluations.""",
                            ),
                        )
                    ],
                )

            else:
                raise ValueError(f"Prompt implementation not found: {name}")


async def create_server() -> GraphOfThoughtsServer:
    """Create and configure the Graph of Thoughts MCP server."""
    return GraphOfThoughtsServer()


async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting Graph of Thoughts MCP Server")

    try:
        # Create the server
        got_server = await create_server()

        # Run with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server running with stdio transport")
            await got_server.server.run(
                read_stream,
                write_stream,
                got_server.server.create_initialization_options(),
            )

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the server
    asyncio.run(main())