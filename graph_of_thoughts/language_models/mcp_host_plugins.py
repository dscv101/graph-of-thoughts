# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Host Plugin System Implementation.

This module provides an extensible plugin architecture for supporting different MCP host implementations.
The plugin system allows for easy addition of new MCP hosts without modifying core transport code,
enabling better maintainability and extensibility.

Key Features:
    - Extensible plugin architecture for MCP hosts
    - Automatic plugin discovery and registration
    - Host-specific configuration validation
    - Transport factory integration
    - Configuration template generation
    - Host capability detection and validation

Supported Plugin Types:
    - Stdio-based hosts (Claude Desktop, VSCode, Cursor, etc.)
    - HTTP-based hosts (Remote MCP servers)
    - Custom transport implementations

Example Usage:
    Register a new MCP host plugin:

    ```python
    from graph_of_thoughts.language_models.mcp_host_plugins import MCPHostPlugin, register_host_plugin

    class MyCustomHostPlugin(MCPHostPlugin):
        def get_host_name(self) -> str:
            return "my_custom_host"
        
        def get_default_config(self) -> Dict[str, Any]:
            return {
                "transport": {
                    "type": "stdio",
                    "command": "my-custom-host",
                    "args": ["--mcp-mode"]
                },
                "capabilities": {"sampling": {}, "tools": {}},
                "default_sampling_params": {
                    "temperature": 0.7,
                    "maxTokens": 4096
                }
            }
        
        def validate_config(self, config: Dict[str, Any]) -> bool:
            # Custom validation logic
            return True
    
    # Register the plugin
    register_host_plugin(MyCustomHostPlugin())
    ```

    Use the plugin system to create transports:

    ```python
    from graph_of_thoughts.language_models.mcp_host_plugins import create_transport_from_plugin

    # Create transport using plugin system
    config = get_host_config("my_custom_host")
    transport = create_transport_from_plugin(config)
    ```
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


@dataclass
class HostCapabilities:
    """
    Represents the capabilities of an MCP host.

    Attributes:
        supports_resources: Whether the host supports MCP resources
        supports_prompts: Whether the host supports MCP prompts
        supports_tools: Whether the host supports MCP tools
        supports_sampling: Whether the host supports MCP sampling
        supports_roots: Whether the host supports MCP roots
        supports_discovery: Whether the host supports dynamic discovery
        transport_types: List of supported transport types
        authentication_methods: List of supported authentication methods
    """

    supports_resources: bool = False
    supports_prompts: bool = False
    supports_tools: bool = False
    supports_sampling: bool = False
    supports_roots: bool = False
    supports_discovery: bool = False
    transport_types: List[str] = None
    authentication_methods: List[str] = None

    def __post_init__(self):
        if self.transport_types is None:
            self.transport_types = ["stdio"]
        if self.authentication_methods is None:
            self.authentication_methods = []


class MCPHostPlugin(ABC):
    """
    Abstract base class for MCP host plugins.

    Each MCP host (Claude Desktop, VSCode, Cursor, etc.) should implement this interface
    to provide host-specific configuration, validation, and capabilities.
    """

    @abstractmethod
    def get_host_name(self) -> str:
        """
        Get the unique identifier for this MCP host.

        Returns:
            str: Unique host identifier (e.g., "claude_desktop", "vscode", "cursor")
        """
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """
        Get the human-readable display name for this MCP host.

        Returns:
            str: Display name (e.g., "Claude Desktop", "VS Code", "Cursor")
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for this MCP host.

        Returns:
            Dict[str, Any]: Default configuration dictionary
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> HostCapabilities:
        """
        Get the capabilities supported by this MCP host.

        Returns:
            HostCapabilities: Host capability information
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration for this MCP host.

        Args:
            config: Configuration dictionary to validate

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Basic validation - check required fields
            if "transport" not in config:
                logger.error(f"Missing 'transport' in {self.get_host_name()} config")
                return False

            transport_config = config["transport"]
            if "type" not in transport_config:
                logger.error(
                    f"Missing 'type' in transport config for {self.get_host_name()}"
                )
                return False

            # Validate transport type is supported
            capabilities = self.get_capabilities()
            if transport_config["type"] not in capabilities.transport_types:
                logger.error(
                    f"Unsupported transport type '{transport_config['type']}' for {self.get_host_name()}"
                )
                return False

            return self._validate_host_specific_config(config)

        except Exception as e:
            logger.error(f"Config validation failed for {self.get_host_name()}: {e}")
            return False

    def _validate_host_specific_config(self, config: Dict[str, Any]) -> bool:
        """
        Perform host-specific configuration validation.

        Subclasses can override this method to add custom validation logic.

        Args:
            config: Configuration dictionary to validate

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        return True

    def get_config_template(self) -> str:
        """
        Get a JSON configuration template for this MCP host.

        Returns:
            str: JSON configuration template
        """
        config = self.get_default_config()
        return json.dumps({f"mcp_{self.get_host_name()}": config}, indent=4)

    def customize_transport_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Customize transport configuration for this host.

        Subclasses can override this method to modify transport configuration
        before transport creation.

        Args:
            config: Original configuration

        Returns:
            Dict[str, Any]: Modified configuration
        """
        return config


class ClaudeDesktopPlugin(MCPHostPlugin):
    """Plugin for Claude Desktop MCP host."""

    def get_host_name(self) -> str:
        return "claude_desktop"

    def get_display_name(self) -> str:
        return "Claude Desktop"

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "transport": {
                "type": "stdio",
                "command": "claude-desktop-mcp-server",
                "args": [],
                "env": {},
            },
            "client_info": {"name": "graph-of-thoughts", "version": "0.0.3"},
            "capabilities": {
                "sampling": {},
                "tools": {},
                "resources": {},
                "prompts": {},
            },
            "default_sampling_params": {
                "modelPreferences": {
                    "hints": [
                        {"name": "claude-3-5-sonnet"},
                        {"name": "claude-3-haiku"},
                    ],
                    "costPriority": 0.3,
                    "speedPriority": 0.4,
                    "intelligencePriority": 0.8,
                },
                "temperature": 1.0,
                "maxTokens": 4096,
                "stopSequences": [],
                "includeContext": "thisServer",
            },
            "connection_config": {
                "timeout": 30.0,
                "retry_attempts": 3,
                "retry_delay": 1.0,
            },
            "cost_tracking": {"prompt_token_cost": 0.003, "response_token_cost": 0.015},
        }

    def get_capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            supports_resources=True,
            supports_prompts=True,
            supports_tools=True,
            supports_sampling=True,
            supports_roots=False,
            supports_discovery=False,
            transport_types=["stdio"],
            authentication_methods=[],
        )

    def _validate_host_specific_config(self, config: Dict[str, Any]) -> bool:
        """Validate Claude Desktop specific configuration."""
        transport_config = config.get("transport", {})

        # Check for Claude Desktop specific command
        command = transport_config.get("command", "")
        if "claude" not in command.lower():
            logger.warning(f"Command '{command}' may not be Claude Desktop")

        return True


class VSCodePlugin(MCPHostPlugin):
    """Plugin for VS Code MCP host."""

    def get_host_name(self) -> str:
        return "vscode"

    def get_display_name(self) -> str:
        return "VS Code"

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "transport": {
                "type": "stdio",
                "command": "code",
                "args": ["--mcp-server"],
                "env": {},
            },
            "client_info": {"name": "graph-of-thoughts", "version": "0.0.3"},
            "capabilities": {
                "sampling": {},
                "tools": {},
                "resources": {},
                "prompts": {},
                "roots": {},
            },
            "default_sampling_params": {
                "modelPreferences": {
                    "hints": [{"name": "gpt-4"}, {"name": "gpt-3.5-turbo"}],
                    "costPriority": 0.5,
                    "speedPriority": 0.6,
                    "intelligencePriority": 0.7,
                },
                "temperature": 1.0,
                "maxTokens": 4096,
                "stopSequences": [],
                "includeContext": "thisServer",
            },
            "connection_config": {
                "timeout": 30.0,
                "retry_attempts": 3,
                "retry_delay": 1.0,
            },
            "cost_tracking": {"prompt_token_cost": 0.03, "response_token_cost": 0.06},
        }

    def get_capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            supports_resources=True,
            supports_prompts=True,
            supports_tools=True,
            supports_sampling=True,
            supports_roots=True,
            supports_discovery=True,
            transport_types=["stdio"],
            authentication_methods=[],
        )


class CursorPlugin(MCPHostPlugin):
    """Plugin for Cursor MCP host."""

    def get_host_name(self) -> str:
        return "cursor"

    def get_display_name(self) -> str:
        return "Cursor"

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "transport": {
                "type": "stdio",
                "command": "cursor",
                "args": ["--mcp-server"],
                "env": {},
            },
            "client_info": {"name": "graph-of-thoughts", "version": "0.0.3"},
            "capabilities": {"sampling": {}, "tools": {}},
            "default_sampling_params": {
                "modelPreferences": {
                    "hints": [{"name": "claude-3-5-sonnet"}, {"name": "gpt-4"}],
                    "costPriority": 0.4,
                    "speedPriority": 0.5,
                    "intelligencePriority": 0.8,
                },
                "temperature": 1.0,
                "maxTokens": 4096,
                "stopSequences": [],
                "includeContext": "thisServer",
            },
            "connection_config": {
                "timeout": 30.0,
                "retry_attempts": 3,
                "retry_delay": 1.0,
            },
            "cost_tracking": {"prompt_token_cost": 0.003, "response_token_cost": 0.015},
        }

    def get_capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            supports_resources=False,
            supports_prompts=False,
            supports_tools=True,
            supports_sampling=True,
            supports_roots=False,
            supports_discovery=False,
            transport_types=["stdio"],
            authentication_methods=[],
        )


class HTTPServerPlugin(MCPHostPlugin):
    """Plugin for HTTP-based MCP servers."""

    def get_host_name(self) -> str:
        return "http_server"

    def get_display_name(self) -> str:
        return "HTTP MCP Server"

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "transport": {
                "type": "http",
                "url": "http://localhost:8000/mcp",
                "headers": {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                "session_management": True,
            },
            "client_info": {"name": "graph-of-thoughts", "version": "0.0.3"},
            "capabilities": {
                "sampling": {},
                "tools": {},
                "resources": {},
                "prompts": {},
            },
            "default_sampling_params": {
                "modelPreferences": {
                    "hints": [{"name": "claude-3-5-sonnet"}],
                    "costPriority": 0.3,
                    "speedPriority": 0.5,
                    "intelligencePriority": 0.9,
                },
                "temperature": 1.0,
                "maxTokens": 4096,
                "stopSequences": [],
                "includeContext": "allServers",
            },
            "connection_config": {
                "timeout": 60.0,
                "retry_attempts": 5,
                "retry_delay": 2.0,
            },
            "cost_tracking": {"prompt_token_cost": 0.003, "response_token_cost": 0.015},
            "batch_processing": {
                "max_concurrent": 15,
                "batch_size": 100,
                "retry_attempts": 5,
                "retry_delay": 2.0,
                "timeout_per_request": 60.0,
                "enable_by_default": True,
            },
        }

    def get_capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            supports_resources=True,
            supports_prompts=True,
            supports_tools=True,
            supports_sampling=True,
            supports_roots=False,
            supports_discovery=True,
            transport_types=["http"],
            authentication_methods=["bearer", "basic", "oauth2"],
        )

    def _validate_host_specific_config(self, config: Dict[str, Any]) -> bool:
        """Validate HTTP server specific configuration."""
        transport_config = config.get("transport", {})

        # Check for required URL
        if "url" not in transport_config:
            logger.error("HTTP transport requires 'url' parameter")
            return False

        url = transport_config["url"]
        if not url.startswith(("http://", "https://")):
            logger.error(f"Invalid URL format: {url}")
            return False

        return True


# Global plugin registry
_plugin_registry: Dict[str, MCPHostPlugin] = {}


def register_host_plugin(plugin: MCPHostPlugin) -> None:
    """
    Register an MCP host plugin.

    Args:
        plugin: The plugin instance to register
    """
    host_name = plugin.get_host_name()
    if host_name in _plugin_registry:
        logger.warning(f"Overriding existing plugin for host: {host_name}")

    _plugin_registry[host_name] = plugin
    logger.info(
        f"Registered MCP host plugin: {plugin.get_display_name()} ({host_name})"
    )


def get_host_plugin(host_name: str) -> Optional[MCPHostPlugin]:
    """
    Get a registered MCP host plugin by name.

    Args:
        host_name: The host identifier

    Returns:
        MCPHostPlugin: The plugin instance, or None if not found
    """
    return _plugin_registry.get(host_name)


def list_available_hosts() -> List[str]:
    """
    Get a list of all registered MCP host names.

    Returns:
        List[str]: List of host identifiers
    """
    return list(_plugin_registry.keys())


def get_host_capabilities(host_name: str) -> Optional[HostCapabilities]:
    """
    Get the capabilities for a specific MCP host.

    Args:
        host_name: The host identifier

    Returns:
        HostCapabilities: Host capabilities, or None if host not found
    """
    plugin = get_host_plugin(host_name)
    return plugin.get_capabilities() if plugin else None


def create_transport_from_plugin(config: Dict[str, Any]) -> Any:
    """
    Create an MCP transport using the plugin system.

    This function determines the appropriate host plugin based on the configuration
    and creates the transport using host-specific customizations.

    Args:
        config: MCP configuration dictionary

    Returns:
        MCPTransport: Configured transport instance

    Raises:
        ValueError: If no suitable plugin is found or configuration is invalid
    """
    from .mcp_transport import create_transport

    # Try to determine host from config structure
    host_name = _detect_host_from_config(config)

    if host_name:
        plugin = get_host_plugin(host_name)
        if plugin:
            # Validate configuration using plugin
            if not plugin.validate_config(config):
                raise ValueError(f"Invalid configuration for host: {host_name}")

            # Allow plugin to customize transport config
            customized_config = plugin.customize_transport_config(config)
            logger.info(
                f"Creating transport for {plugin.get_display_name()} using plugin"
            )
            return create_transport(customized_config)

    # Fallback to standard transport creation
    logger.info("Creating transport using standard method (no plugin match)")
    return create_transport(config)


def _detect_host_from_config(config: Dict[str, Any]) -> Optional[str]:
    """
    Detect the MCP host type from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        str: Detected host name, or None if not detected
    """
    transport_config = config.get("transport", {})

    if transport_config.get("type") == "http":
        return "http_server"

    if transport_config.get("type") == "stdio":
        command = transport_config.get("command", "").lower()

        if "claude" in command:
            return "claude_desktop"
        elif "code" in command:
            return "vscode"
        elif "cursor" in command:
            return "cursor"

    return None


def generate_config_template(host_name: str) -> Optional[str]:
    """
    Generate a configuration template for a specific MCP host.

    Args:
        host_name: The host identifier

    Returns:
        str: JSON configuration template, or None if host not found
    """
    plugin = get_host_plugin(host_name)
    return plugin.get_config_template() if plugin else None


def validate_host_config(host_name: str, config: Dict[str, Any]) -> bool:
    """
    Validate a configuration for a specific MCP host.

    Args:
        host_name: The host identifier
        config: Configuration to validate

    Returns:
        bool: True if valid, False otherwise
    """
    plugin = get_host_plugin(host_name)
    return plugin.validate_config(config) if plugin else False


def discover_available_hosts() -> Dict[str, Dict[str, Any]]:
    """
    Discover all available MCP hosts and their information.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping host names to their info
    """
    hosts_info = {}

    for host_name, plugin in _plugin_registry.items():
        capabilities = plugin.get_capabilities()
        hosts_info[host_name] = {
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
            "config_template": plugin.get_config_template(),
        }

    return hosts_info


def export_all_config_templates(output_path: str) -> None:
    """
    Export configuration templates for all registered hosts to a file.

    Args:
        output_path: Path to write the configuration file
    """
    all_configs = {}

    for host_name, plugin in _plugin_registry.items():
        config_key = f"mcp_{host_name}"
        all_configs[config_key] = plugin.get_default_config()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_configs, f, indent=4)

    logger.info(f"Exported configuration templates to: {output_path}")


def _initialize_default_plugins() -> None:
    """Initialize and register default MCP host plugins."""
    default_plugins = [
        ClaudeDesktopPlugin(),
        VSCodePlugin(),
        CursorPlugin(),
        HTTPServerPlugin(),
    ]

    for plugin in default_plugins:
        register_host_plugin(plugin)


# Initialize default plugins when module is imported
_initialize_default_plugins()
