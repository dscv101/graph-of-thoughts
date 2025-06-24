#!/usr/bin/env python3
"""
MCP Setup Validator for Graph of Thoughts Server

This script helps diagnose and fix common MCP client connection issues.
Run this before configuring your MCP client to ensure everything works.
"""

import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

class MCPSetupValidator:
    """Validates MCP server setup and provides configuration recommendations."""

    def __init__(self):
        self.issues = []
        self.recommendations = []
        self.python_path = None
        self.system_info = {
            "platform": platform.system(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        }

    def log_issue(self, issue: str):
        """Log an issue found during validation."""
        self.issues.append(issue)
        logger.error(f"❌ {issue}")

    def log_recommendation(self, rec: str):
        """Log a recommendation for fixing issues."""
        self.recommendations.append(rec)
        logger.info(f"💡 {rec}")

    def log_success(self, message: str):
        """Log a successful validation step."""
        logger.info(f"✅ {message}")

    def check_python_version(self):
        """Check if Python version meets requirements."""
        logger.info("🔍 Checking Python version...")

        version_info = sys.version_info
        if version_info >= (3, 12):
            self.log_success(f"Python {version_info.major}.{version_info.minor}.{version_info.micro} meets requirements (3.12+)")
            return True
        else:
            self.log_issue(f"Python {version_info.major}.{version_info.minor}.{version_info.micro} is too old (3.12+ required)")
            self.log_recommendation("Upgrade to Python 3.12 or newer")
            return False

    def check_package_installation(self):
        """Check if graph_of_thoughts package is installed."""
        logger.info("🔍 Checking package installation...")

        try:
            import graph_of_thoughts
            self.log_success("graph_of_thoughts package is installed")

            # Check if MCP server module exists
            import graph_of_thoughts.mcp_server
            self.log_success("MCP server module is available")
            return True
        except ImportError as e:
            self.log_issue(f"graph_of_thoughts package not found: {e}")
            self.log_recommendation("Install with: pip install graph_of_thoughts")
            return False

    def check_mcp_dependency(self):
        """Check if MCP SDK is available."""
        logger.info("🔍 Checking MCP SDK...")

        try:
            import mcp
            import mcp.server
            import mcp.server.stdio
            self.log_success("MCP SDK is available")
            return True
        except ImportError as e:
            self.log_issue(f"MCP SDK not found: {e}")
            self.log_recommendation("Install with: pip install mcp")
            return False

    def test_server_creation(self):
        """Test if the MCP server can be created."""
        logger.info("🔍 Testing server creation...")

        try:
            # Import and test server creation
            from graph_of_thoughts.mcp_server import create_server

            async def test():
                server = await create_server()
                return server is not None

            result = asyncio.run(test())
            if result:
                self.log_success("MCP server can be created successfully")
                return True
            else:
                self.log_issue("MCP server creation returned None")
                return False
        except Exception as e:
            self.log_issue(f"MCP server creation failed: {e}")
            return False

    def test_command_execution(self):
        """Test if the server can be started via command line."""
        logger.info("🔍 Testing command line execution...")

        try:
            # Test --info command
            result = subprocess.run(
                [sys.executable, "-m", "graph_of_thoughts", "--info"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and "Graph of Thoughts MCP Server" in result.stdout:
                self.log_success("Command line execution works")
                return True
            else:
                self.log_issue(f"Command failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.log_issue("Command execution timed out")
            return False
        except Exception as e:
            self.log_issue(f"Command execution failed: {e}")
            return False

    def find_python_executable(self):
        """Find the best Python executable to use."""
        logger.info("🔍 Finding Python executable...")

        candidates = [
            sys.executable,
            shutil.which("python"),
            shutil.which("python3"),
            shutil.which("python.exe"),
        ]

        # Add platform-specific paths
        if platform.system() == "Windows":
            # Common Windows Python locations
            candidates.extend([
                "C:\\Python312\\python.exe",
                "C:\\Users\\{}\\AppData\\Local\\Programs\\Python\\Python312\\python.exe".format(os.getenv("USERNAME", "")),
            ])

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                self.python_path = candidate
                self.log_success(f"Found Python executable: {candidate}")
                return candidate

        self.log_issue("Could not find suitable Python executable")
        return None

    def generate_configurations(self):
        """Generate MCP client configurations."""
        if not self.python_path:
            self.python_path = sys.executable

        # Handle paths with spaces by using proper quoting
        safe_python_path = self.python_path
        if " " in safe_python_path and not safe_python_path.startswith('"'):
            safe_python_path = f'"{safe_python_path}"'

        configs = {
            "claude_desktop": {
                "mcpServers": {
                    "graph-of-thoughts": {
                        "command": self.python_path,
                        "args": ["-m", "graph_of_thoughts"]
                    }
                }
            },
            "vscode": {
                "mcp.servers": {
                    "graph-of-thoughts": {
                        "command": self.python_path,
                        "args": ["-m", "graph_of_thoughts"],
                        "cwd": "${workspaceFolder}"
                    }
                }
            },
            "cursor": {
                "mcp": {
                    "servers": {
                        "graph-of-thoughts": {
                            "command": f"{safe_python_path} -m graph_of_thoughts"
                        }
                    }
                }
            },
            "augment_code": {
                "augment.mcp.servers": {
                    "graph-of-thoughts": {
                        "command": self.python_path,
                        "args": ["-m", "graph_of_thoughts"],
                        "description": "Graph of Thoughts reasoning server",
                        "autoApprove": [
                            "break_down_task",
                            "generate_thoughts",
                            "score_thoughts",
                            "validate_and_improve",
                            "aggregate_results",
                            "create_reasoning_chain"
                        ]
                    }
                }
            }
        }

        return configs

    def run_validation(self):
        """Run all validation checks."""
        logger.info("🚀 Starting MCP Setup Validation")
        logger.info("=" * 50)

        checks = [
            self.check_python_version,
            self.check_package_installation,
            self.check_mcp_dependency,
            self.test_server_creation,
            self.test_command_execution,
        ]

        results = []
        for check in checks:
            try:
                result = check()
                results.append(result)
            except Exception as e:
                logger.error(f"Check failed with exception: {e}")
                results.append(False)

        # Find Python executable
        self.find_python_executable()

        # Generate report
        self.generate_report(results)

        return all(results)

    def generate_report(self, results):
        """Generate validation report and recommendations."""
        logger.info("\n" + "=" * 50)
        logger.info("📊 VALIDATION REPORT")
        logger.info("=" * 50)

        passed = sum(results)
        total = len(results)

        logger.info(f"Tests Passed: {passed}/{total}")

        if all(results):
            logger.info("🎉 ALL CHECKS PASSED! MCP server is ready for deployment.")

            # Generate configurations
            configs = self.generate_configurations()

            logger.info("\n📋 Recommended MCP Client Configurations:")
            logger.info("-" * 40)

            for client, config in configs.items():
                logger.info(f"\n{client.upper()}:")
                print(json.dumps(config, indent=2))

        else:
            logger.info("⚠️  Some checks failed. Please address the issues below:")

            if self.issues:
                logger.info("\n🔴 Issues Found:")
                for issue in self.issues:
                    logger.info(f"  • {issue}")

            if self.recommendations:
                logger.info("\n💡 Recommendations:")
                for rec in self.recommendations:
                    logger.info(f"  • {rec}")

        logger.info("\n" + "=" * 50)

def main():
    """Main entry point."""
    validator = MCPSetupValidator()
    success = validator.run_validation()

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
