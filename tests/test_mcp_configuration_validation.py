#!/usr/bin/env python3
"""
Test suite for MCP configuration validation system.

This module tests the enhanced configuration validation system including:
- Startup configuration validation
- Runtime configuration validation
- Security validation
- Error reporting and messaging
- Configuration templates and summaries
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, mock_open

from graph_of_thoughts.language_models.mcp_protocol import (
    MCPConfigurationValidator,
    MCPConfigurationError,
    validate_mcp_config_file
)


class TestMCPConfigurationValidation(unittest.TestCase):
    """Test cases for MCP configuration validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = MCPConfigurationValidator(strict_mode=True, enable_security_checks=True)
        
        # Valid configuration for testing
        self.valid_config = {
            "transport": {
                "type": "stdio",
                "command": "claude-desktop",
                "args": ["--mcp-server"],
                "env": {}
            },
            "client_info": {
                "name": "graph-of-thoughts",
                "version": "0.0.3"
            },
            "capabilities": {
                "sampling": {}
            },
            "default_sampling_params": {
                "temperature": 1.0,
                "maxTokens": 4096,
                "includeContext": "thisServer"
            },
            "connection_config": {
                "timeout": 30.0,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "cost_tracking": {
                "prompt_token_cost": 0.003,
                "response_token_cost": 0.015
            }
        }
    
    def test_valid_stdio_configuration(self):
        """Test validation of valid stdio configuration."""
        self.validator._validate_model_configuration(self.valid_config, "test_model")
        # Should not raise any exceptions
    
    def test_valid_http_configuration(self):
        """Test validation of valid HTTP configuration."""
        http_config = self.valid_config.copy()
        http_config["transport"] = {
            "type": "http",
            "url": "https://api.example.com/mcp",
            "headers": {
                "Content-Type": "application/json"
            },
            "session_management": True
        }
        
        self.validator._validate_model_configuration(http_config, "test_model")
        # Should not raise any exceptions
    
    def test_missing_required_sections(self):
        """Test validation fails for missing required sections."""
        # Missing transport
        config = self.valid_config.copy()
        del config["transport"]
        
        with self.assertRaises(MCPConfigurationError) as cm:
            self.validator._validate_model_configuration(config, "test_model")
        
        self.assertIn("Missing required section: transport", str(cm.exception))
        self.assertEqual(cm.exception.field_path, "transport")
    
    def test_invalid_transport_type(self):
        """Test validation fails for invalid transport type."""
        config = self.valid_config.copy()
        config["transport"]["type"] = "invalid_transport"
        
        with self.assertRaises(MCPConfigurationError) as cm:
            self.validator._validate_model_configuration(config, "test_model")
        
        self.assertIn("Unsupported transport type", str(cm.exception))
        self.assertEqual(cm.exception.field_path, "transport.type")
    
    def test_missing_stdio_command(self):
        """Test validation fails for missing stdio command."""
        config = self.valid_config.copy()
        del config["transport"]["command"]
        
        with self.assertRaises(MCPConfigurationError) as cm:
            self.validator._validate_model_configuration(config, "test_model")
        
        self.assertIn("Missing required field for stdio transport: command", str(cm.exception))
        self.assertEqual(cm.exception.field_path, "transport.command")
    
    def test_invalid_temperature_range(self):
        """Test validation fails for invalid temperature range."""
        config = self.valid_config.copy()
        config["default_sampling_params"]["temperature"] = 3.0  # Invalid: > 2
        
        with self.assertRaises(MCPConfigurationError) as cm:
            self.validator._validate_model_configuration(config, "test_model")
        
        self.assertIn("Temperature must be a number between 0 and 2", str(cm.exception))
        self.assertEqual(cm.exception.field_path, "default_sampling_params.temperature")
    
    def test_invalid_max_tokens(self):
        """Test validation fails for invalid maxTokens."""
        config = self.valid_config.copy()
        config["default_sampling_params"]["maxTokens"] = -100  # Invalid: negative
        
        with self.assertRaises(MCPConfigurationError) as cm:
            self.validator._validate_model_configuration(config, "test_model")
        
        self.assertIn("maxTokens must be a positive integer", str(cm.exception))
        self.assertEqual(cm.exception.field_path, "default_sampling_params.maxTokens")
    
    def test_invalid_include_context(self):
        """Test validation fails for invalid includeContext."""
        config = self.valid_config.copy()
        config["default_sampling_params"]["includeContext"] = "invalidContext"
        
        with self.assertRaises(MCPConfigurationError) as cm:
            self.validator._validate_model_configuration(config, "test_model")
        
        self.assertIn("includeContext must be one of", str(cm.exception))
        self.assertEqual(cm.exception.field_path, "default_sampling_params.includeContext")
    
    def test_invalid_model_preferences(self):
        """Test validation fails for invalid model preferences."""
        config = self.valid_config.copy()
        config["default_sampling_params"]["modelPreferences"] = {
            "hints": [{"invalid": "hint"}],  # Missing 'name' field
            "costPriority": 1.5  # Invalid: > 1
        }
        
        with self.assertRaises(MCPConfigurationError) as cm:
            self.validator._validate_model_configuration(config, "test_model")
        
        # Should catch the first error (invalid hint structure)
        self.assertIn("Model hint at index 0 must be an object with 'name' field", str(cm.exception))
    
    def test_invalid_connection_config(self):
        """Test validation fails for invalid connection config."""
        config = self.valid_config.copy()
        config["connection_config"]["timeout"] = -5.0  # Invalid: negative
        
        with self.assertRaises(MCPConfigurationError) as cm:
            self.validator._validate_model_configuration(config, "test_model")
        
        self.assertIn("timeout must be a positive number", str(cm.exception))
        self.assertEqual(cm.exception.field_path, "connection_config.timeout")
    
    def test_invalid_cost_tracking(self):
        """Test validation fails for invalid cost tracking."""
        config = self.valid_config.copy()
        config["cost_tracking"]["prompt_token_cost"] = -0.001  # Invalid: negative
        
        with self.assertRaises(MCPConfigurationError) as cm:
            self.validator._validate_model_configuration(config, "test_model")
        
        self.assertIn("prompt_token_cost must be a non-negative number", str(cm.exception))
        self.assertEqual(cm.exception.field_path, "cost_tracking.prompt_token_cost")
    
    def test_security_validation_warnings(self):
        """Test security validation generates appropriate warnings."""
        config = self.valid_config.copy()
        config["transport"]["command"] = "rm -rf /"  # Dangerous command
        config["transport"]["env"] = {"SECRET_KEY": "value"}  # Sensitive env var
        
        # Should not raise exception but generate warnings
        self.validator._validate_model_configuration(config, "test_model")
        
        # Check that warnings were generated
        self.assertTrue(len(self.validator.validation_warnings) > 0)
        warning_text = " ".join(self.validator.validation_warnings)
        self.assertIn("dangerous command pattern", warning_text.lower())
        self.assertIn("sensitive data", warning_text.lower())
    
    def test_http_security_validation(self):
        """Test HTTP security validation."""
        config = self.valid_config.copy()
        config["transport"] = {
            "type": "http",
            "url": "http://remote-server.com/mcp",  # HTTP instead of HTTPS
            "headers": {
                "Authorization": "Bearer abc"  # Short token
            }
        }
        
        self.validator._validate_model_configuration(config, "test_model")
        
        # Check security warnings
        warning_text = " ".join(self.validator.validation_warnings)
        self.assertIn("HTTP instead of HTTPS", warning_text)
        self.assertIn("short token", warning_text)
    
    def test_startup_configuration_validation(self):
        """Test startup configuration validation with file operations."""
        config_data = {"test_model": self.valid_config}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f, indent=2)
            config_path = f.name
        
        try:
            # Should validate successfully
            result = self.validator.validate_startup_configuration(config_path)
            self.assertTrue(result)
        finally:
            os.unlink(config_path)
    
    def test_startup_validation_file_not_found(self):
        """Test startup validation fails for non-existent file."""
        with self.assertRaises(MCPConfigurationError) as cm:
            self.validator.validate_startup_configuration("/nonexistent/config.json")
        
        self.assertIn("Configuration file not found", str(cm.exception))
    
    def test_startup_validation_invalid_json(self):
        """Test startup validation fails for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            config_path = f.name
        
        try:
            with self.assertRaises(MCPConfigurationError) as cm:
                self.validator.validate_startup_configuration(config_path)
            
            self.assertIn("Invalid JSON", str(cm.exception))
        finally:
            os.unlink(config_path)
    
    def test_configuration_template_generation(self):
        """Test configuration template generation."""
        # Test stdio template
        stdio_template = self.validator.generate_configuration_template("stdio", "test_stdio")
        stdio_config = json.loads(stdio_template)
        
        self.assertIn("test_stdio", stdio_config)
        self.assertEqual(stdio_config["test_stdio"]["transport"]["type"], "stdio")
        
        # Test HTTP template
        http_template = self.validator.generate_configuration_template("http", "test_http")
        http_config = json.loads(http_template)
        
        self.assertIn("test_http", http_config)
        self.assertEqual(http_config["test_http"]["transport"]["type"], "http")
    
    def test_configuration_summary(self):
        """Test configuration summary generation."""
        config_data = {"test_model": self.valid_config}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f, indent=2)
            config_path = f.name
        
        try:
            summary = self.validator.get_configuration_summary(config_path)
            
            self.assertEqual(summary["model_count"], 1)
            self.assertIn("test_model", summary["models"])
            self.assertEqual(summary["models"]["test_model"]["transport_type"], "stdio")
            self.assertTrue(summary["models"]["test_model"]["has_capabilities"])
        finally:
            os.unlink(config_path)
    
    def test_validate_mcp_config_file_function(self):
        """Test the standalone validate_mcp_config_file function."""
        config_data = {"test_model": self.valid_config}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f, indent=2)
            config_path = f.name
        
        try:
            # Should validate successfully
            result = validate_mcp_config_file(config_path)
            self.assertTrue(result)
            
            # Test with invalid config
            invalid_config = {"test_model": {"invalid": "config"}}
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
                json.dump(invalid_config, f2, indent=2)
                invalid_path = f2.name
            
            try:
                result = validate_mcp_config_file(invalid_path)
                self.assertFalse(result)
            finally:
                os.unlink(invalid_path)
        finally:
            os.unlink(config_path)


if __name__ == '__main__':
    unittest.main()
