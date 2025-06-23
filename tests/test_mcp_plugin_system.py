#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Unit tests for the MCP Plugin System.

This module contains comprehensive tests for the MCP host plugin system,
including plugin registration, configuration validation, transport creation,
and plugin manager functionality.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.language_models.mcp_host_plugins import (
    ClaudeDesktopPlugin,
    CursorPlugin,
    HostCapabilities,
    HTTPServerPlugin,
    MCPHostPlugin,
    VSCodePlugin,
    _detect_host_from_config,
    discover_available_hosts,
    generate_config_template,
    get_host_capabilities,
    get_host_plugin,
    list_available_hosts,
    register_host_plugin,
    validate_host_config,
)
from graph_of_thoughts.language_models.mcp_plugin_manager import MCPPluginManager


class TestMCPHostPlugin(MCPHostPlugin):
    """Test plugin for unit testing."""

    def get_host_name(self) -> str:
        return "test_host"

    def get_display_name(self) -> str:
        return "Test Host"

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "transport": {"type": "stdio", "command": "test-host", "args": ["--test"]},
            "client_info": {"name": "test-client", "version": "1.0.0"},
            "capabilities": {"sampling": {}, "tools": {}},
        }

    def get_capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            supports_tools=True, supports_sampling=True, transport_types=["stdio"]
        )


class TestHostCapabilities(unittest.TestCase):
    """Test HostCapabilities dataclass."""

    def test_default_capabilities(self):
        """Test default capability values."""
        caps = HostCapabilities()

        self.assertFalse(caps.supports_resources)
        self.assertFalse(caps.supports_prompts)
        self.assertFalse(caps.supports_tools)
        self.assertFalse(caps.supports_sampling)
        self.assertFalse(caps.supports_roots)
        self.assertFalse(caps.supports_discovery)
        self.assertEqual(caps.transport_types, ["stdio"])
        self.assertEqual(caps.authentication_methods, [])

    def test_custom_capabilities(self):
        """Test custom capability values."""
        caps = HostCapabilities(
            supports_tools=True,
            supports_resources=True,
            transport_types=["stdio", "http"],
            authentication_methods=["bearer", "oauth2"],
        )

        self.assertTrue(caps.supports_tools)
        self.assertTrue(caps.supports_resources)
        self.assertFalse(caps.supports_prompts)
        self.assertEqual(caps.transport_types, ["stdio", "http"])
        self.assertEqual(caps.authentication_methods, ["bearer", "oauth2"])


class TestMCPHostPlugins(unittest.TestCase):
    """Test individual MCP host plugins."""

    def test_claude_desktop_plugin(self):
        """Test Claude Desktop plugin."""
        plugin = ClaudeDesktopPlugin()

        self.assertEqual(plugin.get_host_name(), "claude_desktop")
        self.assertEqual(plugin.get_display_name(), "Claude Desktop")

        config = plugin.get_default_config()
        self.assertIn("transport", config)
        self.assertEqual(config["transport"]["type"], "stdio")

        capabilities = plugin.get_capabilities()
        self.assertTrue(capabilities.supports_tools)
        self.assertTrue(capabilities.supports_resources)
        self.assertTrue(capabilities.supports_prompts)
        self.assertTrue(capabilities.supports_sampling)
        self.assertIn("stdio", capabilities.transport_types)

    def test_vscode_plugin(self):
        """Test VS Code plugin."""
        plugin = VSCodePlugin()

        self.assertEqual(plugin.get_host_name(), "vscode")
        self.assertEqual(plugin.get_display_name(), "VS Code")

        config = plugin.get_default_config()
        self.assertIn("transport", config)
        self.assertEqual(config["transport"]["type"], "stdio")
        self.assertEqual(config["transport"]["command"], "code")

        capabilities = plugin.get_capabilities()
        self.assertTrue(capabilities.supports_tools)
        self.assertTrue(capabilities.supports_resources)
        self.assertTrue(capabilities.supports_roots)
        self.assertTrue(capabilities.supports_discovery)

    def test_cursor_plugin(self):
        """Test Cursor plugin."""
        plugin = CursorPlugin()

        self.assertEqual(plugin.get_host_name(), "cursor")
        self.assertEqual(plugin.get_display_name(), "Cursor")

        config = plugin.get_default_config()
        self.assertEqual(config["transport"]["command"], "cursor")

        capabilities = plugin.get_capabilities()
        self.assertTrue(capabilities.supports_tools)
        self.assertFalse(capabilities.supports_resources)
        self.assertFalse(capabilities.supports_prompts)

    def test_http_server_plugin(self):
        """Test HTTP server plugin."""
        plugin = HTTPServerPlugin()

        self.assertEqual(plugin.get_host_name(), "http_server")
        self.assertEqual(plugin.get_display_name(), "HTTP MCP Server")

        config = plugin.get_default_config()
        self.assertEqual(config["transport"]["type"], "http")
        self.assertIn("url", config["transport"])

        capabilities = plugin.get_capabilities()
        self.assertTrue(capabilities.supports_discovery)
        self.assertIn("http", capabilities.transport_types)
        self.assertIn("bearer", capabilities.authentication_methods)


class TestPluginValidation(unittest.TestCase):
    """Test plugin configuration validation."""

    def setUp(self):
        """Set up test plugin."""
        self.plugin = TestMCPHostPlugin()

    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            "transport": {"type": "stdio", "command": "test-host"},
            "client_info": {"name": "test", "version": "1.0.0"},
        }

        self.assertTrue(self.plugin.validate_config(config))

    def test_missing_transport(self):
        """Test validation fails for missing transport."""
        config = {"client_info": {"name": "test", "version": "1.0.0"}}

        self.assertFalse(self.plugin.validate_config(config))

    def test_missing_transport_type(self):
        """Test validation fails for missing transport type."""
        config = {"transport": {"command": "test-host"}}

        self.assertFalse(self.plugin.validate_config(config))

    def test_unsupported_transport_type(self):
        """Test validation fails for unsupported transport type."""
        config = {"transport": {"type": "unsupported", "command": "test-host"}}

        self.assertFalse(self.plugin.validate_config(config))


class TestPluginRegistry(unittest.TestCase):
    """Test plugin registry functionality."""

    def setUp(self):
        """Set up test environment."""
        # Clear registry for clean tests
        from graph_of_thoughts.language_models.mcp_host_plugins import _plugin_registry

        self.original_registry = _plugin_registry.copy()
        _plugin_registry.clear()

    def tearDown(self):
        """Restore original registry."""
        from graph_of_thoughts.language_models.mcp_host_plugins import _plugin_registry

        _plugin_registry.clear()
        _plugin_registry.update(self.original_registry)

    def test_plugin_registration(self):
        """Test plugin registration."""
        plugin = TestMCPHostPlugin()
        register_host_plugin(plugin)

        self.assertIn("test_host", list_available_hosts())
        self.assertEqual(get_host_plugin("test_host"), plugin)

    def test_plugin_override(self):
        """Test plugin override warning."""
        plugin1 = TestMCPHostPlugin()
        plugin2 = TestMCPHostPlugin()

        register_host_plugin(plugin1)
        register_host_plugin(plugin2)  # Should override

        self.assertEqual(get_host_plugin("test_host"), plugin2)

    def test_get_nonexistent_plugin(self):
        """Test getting non-existent plugin."""
        self.assertIsNone(get_host_plugin("nonexistent"))

    def test_host_capabilities(self):
        """Test getting host capabilities."""
        plugin = TestMCPHostPlugin()
        register_host_plugin(plugin)

        capabilities = get_host_capabilities("test_host")
        self.assertIsNotNone(capabilities)
        self.assertTrue(capabilities.supports_tools)
        self.assertTrue(capabilities.supports_sampling)


class TestConfigTemplates(unittest.TestCase):
    """Test configuration template generation."""

    def setUp(self):
        """Set up test plugin."""
        from graph_of_thoughts.language_models.mcp_host_plugins import _plugin_registry

        self.original_registry = _plugin_registry.copy()
        _plugin_registry.clear()

        plugin = TestMCPHostPlugin()
        register_host_plugin(plugin)

    def tearDown(self):
        """Restore original registry."""
        from graph_of_thoughts.language_models.mcp_host_plugins import _plugin_registry

        _plugin_registry.clear()
        _plugin_registry.update(self.original_registry)

    def test_generate_template(self):
        """Test configuration template generation."""
        template = generate_config_template("test_host")
        self.assertIsNotNone(template)

        # Parse and validate JSON
        config = json.loads(template)
        self.assertIn("mcp_test_host", config)
        self.assertIn("transport", config["mcp_test_host"])

    def test_generate_nonexistent_template(self):
        """Test template generation for non-existent host."""
        template = generate_config_template("nonexistent")
        self.assertIsNone(template)


class TestHostDetection(unittest.TestCase):
    """Test host detection from configuration."""

    def test_detect_claude_desktop(self):
        """Test detection of Claude Desktop."""
        config = {"transport": {"type": "stdio", "command": "claude-desktop"}}

        host = _detect_host_from_config(config)
        self.assertEqual(host, "claude_desktop")

    def test_detect_vscode(self):
        """Test detection of VS Code."""
        config = {"transport": {"type": "stdio", "command": "code"}}

        host = _detect_host_from_config(config)
        self.assertEqual(host, "vscode")

    def test_detect_cursor(self):
        """Test detection of Cursor."""
        config = {"transport": {"type": "stdio", "command": "cursor"}}

        host = _detect_host_from_config(config)
        self.assertEqual(host, "cursor")

    def test_detect_http_server(self):
        """Test detection of HTTP server."""
        config = {"transport": {"type": "http", "url": "https://api.example.com"}}

        host = _detect_host_from_config(config)
        self.assertEqual(host, "http_server")

    def test_detect_unknown(self):
        """Test detection of unknown host."""
        config = {"transport": {"type": "stdio", "command": "unknown-host"}}

        host = _detect_host_from_config(config)
        self.assertIsNone(host)


class TestMCPPluginManager(unittest.TestCase):
    """Test MCP Plugin Manager functionality."""

    def setUp(self):
        """Set up test environment."""
        self.manager = MCPPluginManager()

        # Clear registry and add test plugin
        from graph_of_thoughts.language_models.mcp_host_plugins import _plugin_registry

        self.original_registry = _plugin_registry.copy()
        _plugin_registry.clear()

        plugin = TestMCPHostPlugin()
        register_host_plugin(plugin)

    def tearDown(self):
        """Restore original registry."""
        from graph_of_thoughts.language_models.mcp_host_plugins import _plugin_registry

        _plugin_registry.clear()
        _plugin_registry.update(self.original_registry)

    def test_list_hosts(self):
        """Test listing hosts."""
        hosts = self.manager.list_hosts()
        self.assertIn("test_host", hosts)

    def test_get_host_info(self):
        """Test getting host information."""
        info = self.manager.get_host_info("test_host")
        self.assertIsNotNone(info)
        self.assertEqual(info["host_name"], "test_host")
        self.assertEqual(info["display_name"], "Test Host")
        self.assertIn("capabilities", info)
        self.assertIn("default_config", info)

    def test_get_nonexistent_host_info(self):
        """Test getting info for non-existent host."""
        info = self.manager.get_host_info("nonexistent")
        self.assertIsNone(info)

    def test_register_plugin(self):
        """Test plugin registration through manager."""

        class AnotherTestPlugin(TestMCPHostPlugin):
            def get_host_name(self) -> str:
                return "another_test"

        plugin = AnotherTestPlugin()
        success = self.manager.register_plugin(plugin)

        self.assertTrue(success)
        self.assertIn("another_test", self.manager.list_hosts())

    def test_config_file_operations(self):
        """Test configuration file loading and saving."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_config = {"test": "value"}
            json.dump(test_config, f)
            temp_path = f.name

        try:
            # Test loading
            loaded_config = self.manager.load_config(temp_path)
            self.assertEqual(loaded_config, test_config)

            # Test saving
            new_config = {"new": "config"}
            self.manager.save_config(new_config, temp_path)

            # Verify saved config
            with open(temp_path, "r") as f:
                saved_config = json.load(f)
            self.assertEqual(saved_config, new_config)

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
