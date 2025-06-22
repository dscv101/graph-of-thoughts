#!/usr/bin/env python3
"""
Test script for MCP integration with Graph of Thoughts.
This script tests the MCP implementation with different hosts and configurations.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports without dependencies that might not be installed
def test_imports():
    """Test that our MCP files can be imported."""
    try:
        # Test basic file existence
        files_to_check = [
            "graph_of_thoughts/language_models/mcp_config_template.json",
            "graph_of_thoughts/language_models/mcp_transport.py",
            "graph_of_thoughts/language_models/mcp_client.py",
            "graph_of_thoughts/language_models/mcp_sampling.py",
        ]

        for file_path in files_to_check:
            if not os.path.exists(file_path):
                print(f"❌ Missing file: {file_path}")
                return False
            else:
                print(f"✅ Found file: {file_path}")

        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


class MCPIntegrationTester:
    """
    Test suite for MCP integration.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_results = {}

    def setup_logging(self):
        """Set up logging for tests."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def test_config_loading(self) -> bool:
        """Test MCP configuration loading."""
        try:
            config_path = "graph_of_thoughts/language_models/mcp_config_template.json"
            
            if not os.path.exists(config_path):
                self.logger.error(f"Config template not found: {config_path}")
                return False

            with open(config_path, 'r') as f:
                config = json.load(f)

            required_configs = ["mcp_claude_desktop", "mcp_vscode", "mcp_cursor", "mcp_http_server"]
            for config_name in required_configs:
                if config_name not in config:
                    self.logger.error(f"Missing configuration: {config_name}")
                    return False

                cfg = config[config_name]
                required_fields = ["transport_type", "host_type", "model_preferences", "sampling_config"]
                for field in required_fields:
                    if field not in cfg:
                        self.logger.error(f"Missing field {field} in {config_name}")
                        return False

            self.logger.info("✓ Configuration loading test passed")
            return True

        except Exception as e:
            self.logger.error(f"Configuration loading test failed: {e}")
            return False

    def test_transport_creation(self) -> bool:
        """Test MCP transport creation."""
        try:
            config_path = "graph_of_thoughts/language_models/mcp_config_template.json"
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Test stdio transport
            stdio_config = config["mcp_claude_desktop"]
            stdio_transport = create_transport(stdio_config)
            if stdio_transport is None:
                self.logger.error("Failed to create stdio transport")
                return False

            # Test HTTP transport
            http_config = config["mcp_http_server"]
            http_transport = create_transport(http_config)
            if http_transport is None:
                self.logger.error("Failed to create HTTP transport")
                return False

            self.logger.info("✓ Transport creation test passed")
            return True

        except Exception as e:
            self.logger.error(f"Transport creation test failed: {e}")
            return False

    def test_mcp_language_model_creation(self) -> bool:
        """Test MCP language model instantiation."""
        try:
            config_path = "graph_of_thoughts/language_models/mcp_config_template.json"
            
            # Test each configuration
            configs_to_test = ["mcp_claude_desktop", "mcp_vscode", "mcp_cursor"]
            
            for config_name in configs_to_test:
                try:
                    lm = language_models.MCPLanguageModel(
                        config_path=config_path,
                        model_name=config_name,
                        cache=True
                    )
                    
                    # Check basic properties
                    if not hasattr(lm, 'transport'):
                        self.logger.error(f"Missing transport in {config_name}")
                        return False
                    
                    if not hasattr(lm, 'host_type'):
                        self.logger.error(f"Missing host_type in {config_name}")
                        return False

                    self.logger.info(f"✓ Created MCP language model: {config_name}")

                except Exception as e:
                    self.logger.error(f"Failed to create {config_name}: {e}")
                    return False

            self.logger.info("✓ MCP language model creation test passed")
            return True

        except Exception as e:
            self.logger.error(f"MCP language model creation test failed: {e}")
            return False

    def test_controller_integration(self) -> bool:
        """Test integration with Graph of Thoughts controller."""
        try:
            config_path = "graph_of_thoughts/language_models/mcp_config_template.json"
            
            # Create MCP language model
            lm = language_models.MCPLanguageModel(
                config_path=config_path,
                model_name="mcp_claude_desktop",
                cache=True
            )

            # Create simple operations graph
            gop = operations.GraphOfOperations()
            gop.append_operation(operations.Generate(1, 1))

            # Create controller
            ctrl = controller.Controller(
                lm,
                gop,
                SortingPrompter(),
                SortingParser(),
                {
                    "original": "[3, 1, 4, 1, 5]",
                    "current": "",
                    "method": "io"
                }
            )

            # Check that controller was created successfully
            if ctrl.lm != lm:
                self.logger.error("Controller language model mismatch")
                return False

            self.logger.info("✓ Controller integration test passed")
            return True

        except Exception as e:
            self.logger.error(f"Controller integration test failed: {e}")
            return False

    def test_error_handling(self) -> bool:
        """Test error handling in MCP implementation."""
        try:
            # Test invalid configuration
            try:
                lm = language_models.MCPLanguageModel(
                    config_path="nonexistent_config.json",
                    model_name="invalid_config"
                )
                self.logger.error("Should have failed with invalid config")
                return False
            except Exception:
                self.logger.info("✓ Correctly handled invalid configuration")

            # Test invalid model name
            try:
                config_path = "graph_of_thoughts/language_models/mcp_config_template.json"
                lm = language_models.MCPLanguageModel(
                    config_path=config_path,
                    model_name="nonexistent_model"
                )
                self.logger.error("Should have failed with invalid model name")
                return False
            except Exception:
                self.logger.info("✓ Correctly handled invalid model name")

            self.logger.info("✓ Error handling test passed")
            return True

        except Exception as e:
            self.logger.error(f"Error handling test failed: {e}")
            return False

    def test_response_parsing(self) -> bool:
        """Test response parsing functionality."""
        try:
            config_path = "graph_of_thoughts/language_models/mcp_config_template.json"
            lm = language_models.MCPLanguageModel(
                config_path=config_path,
                model_name="mcp_claude_desktop",
                cache=True
            )

            # Test single response parsing
            single_response = {
                "model": "claude-3-5-sonnet",
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": "This is a test response."
                },
                "stopReason": "endTurn"
            }

            texts = lm.get_response_texts(single_response)
            if len(texts) != 1 or texts[0] != "This is a test response.":
                self.logger.error("Single response parsing failed")
                return False

            # Test multiple response parsing
            multiple_responses = [single_response, single_response]
            texts = lm.get_response_texts(multiple_responses)
            if len(texts) != 2:
                self.logger.error("Multiple response parsing failed")
                return False

            self.logger.info("✓ Response parsing test passed")
            return True

        except Exception as e:
            self.logger.error(f"Response parsing test failed: {e}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        self.setup_logging()
        
        tests = [
            ("Config Loading", self.test_config_loading),
            ("Transport Creation", self.test_transport_creation),
            ("MCP Language Model Creation", self.test_mcp_language_model_creation),
            ("Controller Integration", self.test_controller_integration),
            ("Error Handling", self.test_error_handling),
            ("Response Parsing", self.test_response_parsing),
        ]

        results = {}
        passed = 0
        total = len(tests)

        self.logger.info("Starting MCP Integration Tests")
        self.logger.info("=" * 50)

        for test_name, test_func in tests:
            self.logger.info(f"Running: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                else:
                    self.logger.error(f"✗ {test_name} FAILED")
            except Exception as e:
                self.logger.error(f"✗ {test_name} FAILED with exception: {e}")
                results[test_name] = False

        self.logger.info("=" * 50)
        self.logger.info(f"Test Results: {passed}/{total} passed")
        
        if passed == total:
            self.logger.info("🎉 All tests passed!")
        else:
            self.logger.warning(f"⚠️  {total - passed} tests failed")

        return results

    def generate_test_report(self, results: Dict[str, bool]) -> str:
        """Generate a test report."""
        report = "# MCP Integration Test Report\n\n"
        report += f"**Test Date:** {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n\n"
        
        report += "## Test Results\n\n"
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            report += f"- **{test_name}:** {status}\n"
        
        passed = sum(results.values())
        total = len(results)
        report += f"\n**Summary:** {passed}/{total} tests passed\n\n"
        
        if passed == total:
            report += "🎉 **All tests passed!** The MCP integration is working correctly.\n"
        else:
            report += "⚠️ **Some tests failed.** Please check the logs for details.\n"
        
        report += "\n## Next Steps\n\n"
        if passed == total:
            report += "- You can now use MCP language models in your Graph of Thoughts applications\n"
            report += "- See the migration guide for instructions on updating your code\n"
            report += "- Try the example scripts to see MCP in action\n"
        else:
            report += "- Review the failed tests and check your MCP host configuration\n"
            report += "- Ensure your MCP host is running and accessible\n"
            report += "- Check the MCP configuration file for any errors\n"
        
        return report


def main():
    """Main test function."""
    print("MCP Integration Structure Test")
    print("=" * 40)

    # Test basic file structure
    if not test_imports():
        print("❌ Basic structure test failed")
        sys.exit(1)

    print("✅ Basic structure test passed")

    # Test configuration loading
    try:
        config_path = "graph_of_thoughts/language_models/mcp_config_template.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        required_configs = ["mcp_claude_desktop", "mcp_vscode", "mcp_cursor", "mcp_http_server"]
        for config_name in required_configs:
            if config_name in config:
                print(f"✅ Found configuration: {config_name}")
            else:
                print(f"❌ Missing configuration: {config_name}")
                sys.exit(1)

        print("✅ Configuration structure test passed")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        sys.exit(1)

    print("\n🎉 All structure tests passed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install mcp httpx anyio")
    print("2. Set up an MCP host (Claude Desktop, VSCode, or Cursor)")
    print("3. Run the full integration test")
    print("4. Try the MCP examples")

    sys.exit(0)


if __name__ == "__main__":
    main()
