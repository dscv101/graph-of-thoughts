#!/usr/bin/env python3
"""
Example script demonstrating MCP configuration validation system.

This script shows how to use the enhanced MCP configuration validation system
to validate configurations, generate templates, and handle validation errors.
"""

import json
import logging
import tempfile
from pathlib import Path

from graph_of_thoughts.language_models.mcp_protocol import (
    MCPConfigurationError,
    MCPConfigurationValidator,
    validate_mcp_config_file,
)


def main():
    """Demonstrate MCP configuration validation features."""

    # Set up logging to see validation details
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("🔧 MCP Configuration Validation System Demo")
    print("=" * 50)

    # Create validator instance
    validator = MCPConfigurationValidator(strict_mode=True, enable_security_checks=True)

    # Example 1: Generate configuration templates
    print("\n📝 1. Generating Configuration Templates")
    print("-" * 40)

    # Generate stdio template
    stdio_template = validator.generate_configuration_template(
        "stdio", "mcp_claude_desktop"
    )
    print("Generated stdio template:")
    print(stdio_template[:200] + "..." if len(stdio_template) > 200 else stdio_template)

    # Generate HTTP template
    http_template = validator.generate_configuration_template(
        "http", "mcp_remote_server"
    )
    print("\nGenerated HTTP template:")
    print(http_template[:200] + "..." if len(http_template) > 200 else http_template)

    # Example 2: Validate a valid configuration
    print("\n✅ 2. Validating Valid Configuration")
    print("-" * 40)

    valid_config = {
        "mcp_claude_desktop": {
            "transport": {
                "type": "stdio",
                "command": "claude-desktop",
                "args": ["--mcp-server"],
                "env": {},
            },
            "client_info": {"name": "graph-of-thoughts", "version": "0.0.3"},
            "capabilities": {"sampling": {}},
            "default_sampling_params": {
                "temperature": 1.0,
                "maxTokens": 4096,
                "includeContext": "thisServer",
            },
            "connection_config": {
                "timeout": 30.0,
                "retry_attempts": 3,
                "retry_delay": 1.0,
            },
            "cost_tracking": {"prompt_token_cost": 0.003, "response_token_cost": 0.015},
        }
    }

    # Create temporary file for validation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_config, f, indent=2)
        config_path = f.name

    try:
        result = validator.validate_startup_configuration(config_path)
        print(f"Configuration validation result: {result}")

        if validator.validation_warnings:
            print("Warnings:")
            for warning in validator.validation_warnings:
                print(f"  ⚠️  {warning}")
    except MCPConfigurationError as e:
        print(f"❌ Validation failed: {e}")
    finally:
        Path(config_path).unlink()

    # Example 3: Validate invalid configuration
    print("\n❌ 3. Validating Invalid Configuration")
    print("-" * 40)

    invalid_config = {
        "mcp_invalid": {
            "transport": {
                "type": "invalid_transport",  # Invalid transport type
                "command": "some-command",
            },
            "client_info": {
                "name": "",  # Empty name
                "version": "invalid-version-format",
            },
            "default_sampling_params": {
                "temperature": 5.0,  # Invalid temperature > 2
                "maxTokens": -100,  # Invalid negative tokens
                "includeContext": "invalidContext",  # Invalid context
            },
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(invalid_config, f, indent=2)
        invalid_path = f.name

    try:
        result = validator.validate_startup_configuration(invalid_path)
        print(f"Configuration validation result: {result}")
    except MCPConfigurationError as e:
        print(f"❌ Expected validation failure: {e}")
        print(f"Field path: {e.field_path}")
        if e.validation_errors:
            print("Validation errors:")
            for error in e.validation_errors:
                print(f"  - {error}")
    finally:
        Path(invalid_path).unlink()

    # Example 4: Security validation
    print("\n🔒 4. Security Validation Demo")
    print("-" * 40)

    security_config = {
        "mcp_security_test": {
            "transport": {
                "type": "stdio",
                "command": "rm -rf /",  # Dangerous command
                "env": {
                    "SECRET_KEY": "sensitive_value",  # Sensitive env var
                    "PASSWORD": "secret123",
                },
            },
            "client_info": {"name": "security-test", "version": "1.0.0"},
            "capabilities": {"sampling": {}},
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(security_config, f, indent=2)
        security_path = f.name

    try:
        # Clear previous warnings
        validator.validation_warnings.clear()

        result = validator.validate_startup_configuration(security_path)
        print(f"Configuration validation result: {result}")

        print("Security warnings:")
        for warning in validator.validation_warnings:
            print(f"  ⚠️  {warning}")
    except MCPConfigurationError as e:
        print(f"Validation error: {e}")
    finally:
        Path(security_path).unlink()

    # Example 5: Configuration summary
    print("\n📊 5. Configuration Summary")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_config, f, indent=2)
        summary_path = f.name

    try:
        summary = validator.get_configuration_summary(summary_path)
        print("Configuration summary:")
        print(f"  File size: {summary.get('file_size', 'unknown')} bytes")
        print(f"  Model count: {summary.get('model_count', 0)}")

        if "models" in summary:
            for model_name, model_info in summary["models"].items():
                print(f"  Model '{model_name}':")
                print(f"    Transport: {model_info.get('transport_type', 'unknown')}")
                print(
                    f"    Has capabilities: {model_info.get('has_capabilities', False)}"
                )
                print(
                    f"    Has sampling params: {model_info.get('has_sampling_params', False)}"
                )
                print(
                    f"    Has connection config: {model_info.get('has_connection_config', False)}"
                )
                print(
                    f"    Has cost tracking: {model_info.get('has_cost_tracking', False)}"
                )
    finally:
        Path(summary_path).unlink()

    # Example 6: Using the standalone validation function
    print("\n🔍 6. Standalone Validation Function")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_config, f, indent=2)
        standalone_path = f.name

    try:
        result = validate_mcp_config_file(
            standalone_path, strict_mode=True, enable_security_checks=True
        )
        print(f"Standalone validation result: {result}")
    finally:
        Path(standalone_path).unlink()

    print("\n🎉 Demo completed!")
    print("\nTo use the CLI tool, run:")
    print(
        "  python -m graph_of_thoughts.language_models.mcp_protocol validate <config_file>"
    )
    print(
        "  python -m graph_of_thoughts.language_models.mcp_protocol generate-template stdio"
    )
    print(
        "  python -m graph_of_thoughts.language_models.mcp_protocol summary <config_file>"
    )


if __name__ == "__main__":
    main()
