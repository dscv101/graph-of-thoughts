#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Plugin System Demonstration.

This script demonstrates the capabilities of the MCP host plugin system,
including plugin registration, configuration generation, host discovery,
and transport creation with host-specific optimizations.

The plugin system provides an extensible architecture for supporting
different MCP hosts without modifying core transport code.

Usage:
    python examples/mcp_plugin_system_demo.py

Features Demonstrated:
    - Plugin discovery and listing
    - Host capability querying
    - Configuration template generation
    - Custom plugin registration
    - Transport creation with plugin integration
    - Configuration validation
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.language_models.mcp_host_plugins import (
    HostCapabilities,
    MCPHostPlugin,
    register_host_plugin,
)
from graph_of_thoughts.language_models.mcp_plugin_manager import MCPPluginManager


class CustomMCPHostPlugin(MCPHostPlugin):
    """Example custom MCP host plugin for demonstration."""

    def get_host_name(self) -> str:
        return "custom_demo_host"

    def get_display_name(self) -> str:
        return "Custom Demo Host"

    def get_default_config(self) -> dict:
        return {
            "transport": {
                "type": "stdio",
                "command": "custom-mcp-host",
                "args": ["--demo-mode"],
                "env": {"DEMO_MODE": "true"},
            },
            "client_info": {"name": "graph-of-thoughts", "version": "0.0.3"},
            "capabilities": {"sampling": {}, "tools": {}, "resources": {}},
            "default_sampling_params": {
                "temperature": 0.8,
                "maxTokens": 2048,
                "includeContext": "thisServer",
            },
            "connection_config": {
                "timeout": 45.0,
                "retry_attempts": 2,
                "retry_delay": 1.5,
            },
            "cost_tracking": {"prompt_token_cost": 0.001, "response_token_cost": 0.005},
        }

    def get_capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            supports_resources=True,
            supports_prompts=False,
            supports_tools=True,
            supports_sampling=True,
            supports_roots=False,
            supports_discovery=True,
            transport_types=["stdio"],
            authentication_methods=[],
        )

    def _validate_host_specific_config(self, config: dict) -> bool:
        """Custom validation for demo host."""
        transport_config = config.get("transport", {})

        # Check for demo-specific requirements
        command = transport_config.get("command", "")
        if "custom" not in command.lower():
            print(f"Warning: Command '{command}' may not be the custom demo host")

        # Validate demo environment variable
        env = transport_config.get("env", {})
        if env.get("DEMO_MODE") != "true":
            print("Warning: DEMO_MODE environment variable not set to 'true'")

        return True


def demonstrate_plugin_discovery():
    """Demonstrate plugin discovery and listing capabilities."""
    print("🔍 Plugin Discovery and Listing")
    print("=" * 50)

    manager = MCPPluginManager()

    # List all available hosts
    hosts = manager.list_hosts()
    print(f"Available MCP hosts: {len(hosts)}")
    for host in hosts:
        print(f"  • {host}")
    print()

    # Get detailed information for each host
    print("Host Details:")
    for host_name in hosts:
        info = manager.get_host_info(host_name)
        if info:
            capabilities = info["capabilities"]
            print(f"\n📋 {info['display_name']} ({host_name})")
            print(f"   Transport Types: {capabilities['transport_types']}")
            print(f"   Supports Tools: {capabilities['supports_tools']}")
            print(f"   Supports Resources: {capabilities['supports_resources']}")
            print(f"   Supports Prompts: {capabilities['supports_prompts']}")
            print(f"   Supports Sampling: {capabilities['supports_sampling']}")
    print()


def demonstrate_capability_queries():
    """Demonstrate capability-based host queries."""
    print("🎯 Capability-Based Host Queries")
    print("=" * 50)

    manager = MCPPluginManager()

    # Find hosts that support specific capabilities
    capabilities_to_check = [
        "supports_tools",
        "supports_resources",
        "supports_prompts",
        "supports_sampling",
        "supports_discovery",
    ]

    for capability in capabilities_to_check:
        hosts = manager.get_hosts_by_capability(capability)
        print(f"{capability}: {hosts}")

    # Find hosts by transport type
    print(f"\nStdio transport hosts: {manager.get_hosts_by_transport('stdio')}")
    print(f"HTTP transport hosts: {manager.get_hosts_by_transport('http')}")
    print()


def demonstrate_config_generation():
    """Demonstrate configuration template generation."""
    print("⚙️  Configuration Template Generation")
    print("=" * 50)

    manager = MCPPluginManager()

    # Generate templates for each host
    hosts = manager.list_hosts()
    for host_name in hosts:
        print(f"\n📄 Configuration template for {host_name}:")
        template = manager.generate_config_template(host_name)
        if template:
            # Pretty print the JSON
            config_dict = json.loads(template)
            print(json.dumps(config_dict, indent=2))
        else:
            print("  Template not available")
    print()


def demonstrate_custom_plugin():
    """Demonstrate custom plugin registration."""
    print("🔧 Custom Plugin Registration")
    print("=" * 50)

    manager = MCPPluginManager()

    # Show hosts before registration
    print("Hosts before custom plugin registration:")
    print(f"  {manager.list_hosts()}")

    # Register custom plugin
    custom_plugin = CustomMCPHostPlugin()
    success = manager.register_plugin(custom_plugin)

    if success:
        print(
            f"\n✅ Successfully registered custom plugin: {custom_plugin.get_display_name()}"
        )

        # Show hosts after registration
        print("\nHosts after custom plugin registration:")
        print(f"  {manager.list_hosts()}")

        # Show custom plugin details
        info = manager.get_host_info("custom_demo_host")
        if info:
            print(f"\n📋 Custom Plugin Details:")
            print(f"   Display Name: {info['display_name']}")
            print(f"   Capabilities: {info['capabilities']}")

        # Generate config template for custom plugin
        template = manager.generate_config_template("custom_demo_host")
        if template:
            print(f"\n📄 Custom Plugin Configuration Template:")
            config_dict = json.loads(template)
            print(json.dumps(config_dict, indent=2))
    else:
        print("❌ Failed to register custom plugin")
    print()


def demonstrate_config_validation():
    """Demonstrate configuration validation."""
    print("✅ Configuration Validation")
    print("=" * 50)

    manager = MCPPluginManager()

    # Test valid configuration
    valid_config = {
        "transport": {
            "type": "stdio",
            "command": "claude-desktop",
            "args": ["--mcp-server"],
        },
        "client_info": {"name": "test-app", "version": "1.0.0"},
        "capabilities": {"sampling": {}},
    }

    is_valid = manager.validate_config("claude_desktop", valid_config)
    print(f"Valid Claude Desktop config: {is_valid}")

    # Test invalid configuration
    invalid_config = {"transport": {"type": "invalid_transport"}}

    is_valid = manager.validate_config("claude_desktop", invalid_config)
    print(f"Invalid config: {is_valid}")

    # Test HTTP configuration
    http_config = {
        "transport": {"type": "http", "url": "https://api.example.com/mcp"},
        "client_info": {"name": "test-app", "version": "1.0.0"},
        "capabilities": {"sampling": {}},
    }

    is_valid = manager.validate_config("http_server", http_config)
    print(f"Valid HTTP server config: {is_valid}")
    print()


def demonstrate_config_creation():
    """Demonstrate dynamic configuration creation."""
    print("🏗️  Dynamic Configuration Creation")
    print("=" * 50)

    manager = MCPPluginManager()

    # Create configuration with overrides
    config = manager.create_host_config(
        "claude_desktop",
        **{
            "transport.command": "my-custom-claude",
            "transport.args": ["--custom-mode"],
            "default_sampling_params.temperature": 0.5,
            "connection_config.timeout": 60.0,
        },
    )

    if config:
        print("📄 Dynamically created configuration with overrides:")
        print(json.dumps(config, indent=2))
    else:
        print("❌ Failed to create configuration")
    print()


def main():
    """Main demonstration function."""
    print("🚀 MCP Plugin System Demonstration")
    print("=" * 60)
    print()

    try:
        # Run all demonstrations
        demonstrate_plugin_discovery()
        demonstrate_capability_queries()
        demonstrate_config_generation()
        demonstrate_custom_plugin()
        demonstrate_config_validation()
        demonstrate_config_creation()

        print("✨ Plugin system demonstration completed successfully!")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
