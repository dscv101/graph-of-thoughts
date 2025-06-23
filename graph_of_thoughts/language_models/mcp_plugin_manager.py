# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Plugin Manager Implementation.

This module provides a high-level interface for managing MCP host plugins,
including plugin discovery, configuration management, and transport creation.
The plugin manager serves as the main entry point for working with the MCP plugin system.

Key Features:
    - Centralized plugin management
    - Configuration validation and generation
    - Host discovery and capability querying
    - Transport factory with plugin integration
    - Plugin lifecycle management
    - Configuration migration support

Example Usage:
    Basic plugin manager usage:

    ```python
    from graph_of_thoughts.language_models.mcp_plugin_manager import MCPPluginManager

    # Create plugin manager
    manager = MCPPluginManager()

    # List available hosts
    hosts = manager.list_hosts()
    print(f"Available hosts: {hosts}")

    # Get host capabilities
    capabilities = manager.get_host_capabilities("claude_desktop")
    print(f"Claude Desktop supports tools: {capabilities.supports_tools}")

    # Generate configuration template
    template = manager.generate_config_template("vscode")
    print(template)

    # Create transport from configuration
    config = manager.load_config("mcp_config.json")
    transport = manager.create_transport("claude_desktop", config)
    ```

    Advanced plugin management:

    ```python
    # Register custom plugin
    from graph_of_thoughts.language_models.mcp_host_plugins import MCPHostPlugin

    class MyCustomPlugin(MCPHostPlugin):
        # Implementation here
        pass

    manager.register_plugin(MyCustomPlugin())

    # Validate configuration
    config = {"transport": {"type": "stdio", "command": "my-app"}}
    is_valid = manager.validate_config("my_custom_host", config)

    # Export all templates
    manager.export_all_templates("config_templates.json")
    ```
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from .mcp_host_plugins import (
    HostCapabilities,
    MCPHostPlugin,
    create_transport_from_plugin,
    discover_available_hosts,
    export_all_config_templates,
    generate_config_template,
    get_host_capabilities,
    get_host_plugin,
    list_available_hosts,
    register_host_plugin,
    validate_host_config,
)

logger = logging.getLogger(__name__)


class MCPPluginManager:
    """
    High-level manager for MCP host plugins.

    This class provides a convenient interface for working with MCP host plugins,
    including configuration management, validation, and transport creation.
    """

    def __init__(self):
        """Initialize the plugin manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initialized MCP Plugin Manager")

    def list_hosts(self) -> [str]:
        """
        Get a list of all available MCP host names.

        Returns:
            [str]:  of registered host identifiers
        """
        return list_available_hosts()

    def get_host_info(self, host_name: str) -> Optional[[str, Any]]:
        """
        Get detailed information about a specific MCP host.

        Args:
            host_name: The host identifier

        Returns:
            [str, Any]: Host information including capabilities and config template
        """
        plugin = get_host_plugin(host_name)
        if not plugin:
            return None

        capabilities = plugin.get_capabilities()
        return {
            "host_name": host_name,
            "display_name": plugin.get_display_name(),
            "capabilities": {
                "supports_resources": capabilities.supports_resources,
                "supports_prompts": capabilities.supports_prompts,
                "supports_tools": capabilities.supports_tools,
                "supports_sampling": capabilities.supports_sampling,
                "supports_roots": capabilities.supports_roots,
                "supports_discovery": capabilities.supports_discovery,
                "transport_types": capabilities.transport_types,
                "authentication_methods": capabilities.authentication_methods,
            },
            "default_config": plugin.get_default_config(),
        }

    def get_host_capabilities(self, host_name: str) -> Optional[HostCapabilities]:
        """
        Get the capabilities for a specific MCP host.

        Args:
            host_name: The host identifier

        Returns:
            HostCapabilities: Host capabilities, or None if host not found
        """
        return get_host_capabilities(host_name)

    def register_plugin(self, plugin: MCPHostPlugin) -> bool:
        """
        Register a new MCP host plugin.

        Args:
            plugin: The plugin instance to register

        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            register_host_plugin(plugin)
            self.logger.info(
                f"Successfully registered plugin: {plugin.get_display_name()}"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to register plugin {plugin.get_display_name()}: {e}"
            )
            return False

    def generate_config_template(self, host_name: str) -> Optional[str]:
        """
        Generate a JSON configuration template for a specific host.

        Args:
            host_name: The host identifier

        Returns:
            str: JSON configuration template, or None if host not found
        """
        return generate_config_template(host_name)

    def validate_config(self, host_name: str, config: [str, Any]) -> bool:
        """
        Validate a configuration for a specific MCP host.

        Args:
            host_name: The host identifier
            config: Configuration dictionary to validate

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        return validate_host_config(host_name, config)

    def create_transport(self, config: [str, Any]) -> Any:
        """
        Create an MCP transport using the plugin system.

        Args:
            config: MCP configuration dictionary

        Returns:
            MCPTransport: Configured transport instance

        Raises:
            ValueError: If configuration is invalid or no suitable plugin found
        """
        return create_transport_from_plugin(config)

    def load_config(self, config_path: Union[str, Path]) -> [str, Any]:
        """
        Load MCP configuration from a JSON file.

        Args:
            config_path: Path to the configuration file

        Returns:
            [str, Any]: Loaded configuration

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file is invalid JSON
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            self.logger.info(f"Loaded configuration from: {config_path}")
            return config

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
            raise

    def save_config(
        self, config: [str, Any], config_path: Union[str, Path]
    ) -> None:
        """
        Save MCP configuration to a JSON file.

        Args:
            config: Configuration dictionary to save
            config_path: Path where to save the configuration
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=4)

            self.logger.info(f"Saved configuration to: {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise

    def discover_hosts(self) -> [str, [str, Any]]:
        """
        Discover all available MCP hosts and their information.

        Returns:
            [str, [str, Any]]: ionary mapping host names to their info
        """
        return discover_available_hosts()

    def export_all_templates(self, output_path: Union[str, Path]) -> None:
        """
        Export configuration templates for all registered hosts.

        Args:
            output_path: Path to write the configuration file
        """
        export_all_config_templates(str(output_path))

    def get_hosts_by_capability(self, capability: str) -> [str]:
        """
        Get hosts that support a specific capability.

        Args:
            capability: Capability name (e.g., 'supports_tools', 'supports_resources')

        Returns:
            [str]:  of host names that support the capability
        """
        matching_hosts = []

        for host_name in self.list_hosts():
            capabilities = self.get_host_capabilities(host_name)
            if capabilities and hasattr(capabilities, capability):
                if getattr(capabilities, capability):
                    matching_hosts.append(host_name)

        return matching_hosts

    def get_hosts_by_transport(self, transport_type: str) -> [str]:
        """
        Get hosts that support a specific transport type.

        Args:
            transport_type: Transport type (e.g., 'stdio', 'http')

        Returns:
            [str]:  of host names that support the transport type
        """
        matching_hosts = []

        for host_name in self.list_hosts():
            capabilities = self.get_host_capabilities(host_name)
            if capabilities and transport_type in capabilities.transport_types:
                matching_hosts.append(host_name)

        return matching_hosts

    def create_host_config(
        self, host_name: str, **overrides
    ) -> Optional[[str, Any]]:
        """
        Create a configuration for a specific host with optional overrides.

        Args:
            host_name: The host identifier
            **overrides: Configuration values to override

        Returns:
            [str, Any]: Generated configuration, or None if host not found
        """
        plugin = get_host_plugin(host_name)
        if not plugin:
            return None

        config = plugin.get_default_config().copy()

        # Apply overrides
        for key, value in overrides.items():
            if "." in key:
                # Handle nested keys like 'transport.command'
                keys = key.split(".")
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value

        return config