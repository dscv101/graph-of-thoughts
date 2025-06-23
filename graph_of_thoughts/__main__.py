#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Graph of Thoughts MCP Server Entry Point.

This module provides the command-line interface for running the Graph of Thoughts
MCP server. It supports both stdio and HTTP transport modes and includes
configuration options for different deployment scenarios.

Usage:
    # Run with stdio transport (default)
    python -m graph_of_thoughts

    # Run with HTTP transport
    python -m graph_of_thoughts --transport http --port 8080

    # Run with debug logging
    python -m graph_of_thoughts --debug

    # Show help
    python -m graph_of_thoughts --help
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_of_thoughts.mcp_server import main as server_main


def setup_logging(debug: bool = False):
    """Configure logging for the MCP server."""
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(
                sys.stderr
            )  # Use stderr to avoid interfering with stdio transport
        ],
    )

    # Reduce noise from some libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anyio").setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Graph of Thoughts MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with stdio transport (for Claude Desktop, VSCode, etc.)
    python -m graph_of_thoughts

    # Run with HTTP transport on port 8080
    python -m graph_of_thoughts --transport http --port 8080

    # Run with debug logging
    python -m graph_of_thoughts --debug

    # Show server information
    python -m graph_of_thoughts --info
        """,
    )

    _ = parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mechanism to use (default: stdio)",
    )

    _ = parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP transport (default: 8080)"
    )

    _ = parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transport (default: localhost)",
    )

    _ = parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    _ = parser.add_argument(
        "--info", action="store_true", help="Show server information and exit"
    )

    _ = parser.add_argument("--version", action="store_true", help="Show version and exit")

    return parser.parse_args()


def show_info():
    """Display server information."""
    print("Graph of Thoughts MCP Server")
    print("=" * 40)
    print(f"Version: 0.0.3")
    print(f"Protocol: Model Context Protocol (MCP)")
    print(f"Supported Transports: stdio, http")
    print()
    print("Tools Provided:")
    print("  - break_down_task: Decompose complex tasks")
    print("  - generate_thoughts: Generate multiple approaches")
    print("  - score_thoughts: Evaluate and rank approaches")
    print("  - validate_and_improve: Validate and improve solutions")
    print("  - aggregate_results: Combine multiple results")
    print("  - create_reasoning_chain: Build complete workflows")
    print()
    print("Resources Provided:")
    print("  - got://operations/results: Operation execution results")
    print("  - got://templates/prompts: Reusable prompt templates")
    print("  - got://configs/examples: Example configurations")
    print("  - got://logs/execution: Execution logs")
    print()
    print("Prompts Provided:")
    print("  - analyze-problem: Structured problem analysis")
    print("  - generate-solutions: Multi-approach solution generation")
    print("  - evaluate-options: Systematic option evaluation")
    print()
    print("Compatible Hosts:")
    print("  - Claude Desktop")
    print("  - VSCode with MCP extension")
    print("  - Cursor with MCP support")
    print("  - Any MCP-compatible client")


def show_version():
    """Display version information."""
    print("0.0.3")


async def run_http_server(host: str, port: int):
    """Run the server with HTTP transport."""
    print(f"HTTP transport not yet implemented. Use stdio transport instead.")
    print(f"Planned: Server would run on http://{host}:{port}")
    sys.exit(1)


def main():
    """Main entry point for the command-line interface."""
    args = parse_args()

    if args.version:
        show_version()
        return

    if args.info:
        show_info()
        return

    # Setup logging
    setup_logging(args.debug)

    logger = logging.getLogger(__name__)
    logger.info("Starting Graph of Thoughts MCP Server")

    try:
        if args.transport == "stdio":
            logger.info("Using stdio transport")
            asyncio.run(server_main())
        elif args.transport == "http":
            logger.info(f"Using HTTP transport on {args.host}:{args.port}")
            asyncio.run(run_http_server(args.host, args.port))
        else:
            logger.error(f"Unknown transport: {args.transport}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
