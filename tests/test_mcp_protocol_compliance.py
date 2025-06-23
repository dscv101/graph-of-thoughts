# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Comprehensive unit tests for MCP Protocol implementation.

This module tests the MCP protocol layer including message validation,
request/response formatting, and protocol compliance.
"""

import unittest
import json
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.language_models.mcp_protocol import (
    MCPProtocolValidator,
    create_sampling_request,
    create_text_message,
    create_image_message,
    MCPIncludeContext,
    MCPConfigurationError,
    MCPValidationError
)
from graph_of_thoughts.language_models.mcp_transport import (
    create_transport,
    StdioMCPTransport,
    HTTPMCPTransport,
    MCPTransportError,
    MCPConnectionError
)
from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel


class TestMCPProtocolValidator(unittest.TestCase):
    """Test the MCP protocol validator."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = MCPProtocolValidator()

    def test_validate_client_info_valid(self):
        """Test validation of valid client info."""
        client_info = {
            "name": "graph-of-thoughts",
            "version": "0.0.3"
        }
        self.assertTrue(self.validator.validate_client_info(client_info))

    def test_validate_client_info_missing_fields(self):
        """Test validation of client info with missing fields."""
        client_info = {"name": "graph-of-thoughts"}
        self.assertFalse(self.validator.validate_client_info(client_info))
    
    def test_validate_stdio_transport_valid(self):
        """Test validation of valid stdio transport."""
        transport = {
            "type": "stdio",
            "command": "claude-desktop-mcp-server",
            "args": [],
            "env": {}
        }
        self.assertTrue(self.validator.validate_transport_config(transport))

    def test_validate_http_transport_valid(self):
        """Test validation of valid HTTP transport."""
        transport = {
            "type": "http",
            "url": "http://localhost:8000/mcp",
            "headers": {"Content-Type": "application/json"}
        }
        self.assertTrue(self.validator.validate_transport_config(transport))

    def test_validate_transport_invalid_type(self):
        """Test validation of transport with invalid type."""
        transport = {
            "type": "invalid",
            "command": "test"
        }
        self.assertFalse(self.validator.validate_transport_config(transport))
    
    def test_validate_model_preferences_valid(self):
        """Test validation of valid model preferences."""
        preferences = {
            "hints": [{"name": "claude-3-5-sonnet"}],
            "costPriority": 0.3,
            "speedPriority": 0.4,
            "intelligencePriority": 0.8
        }
        self.assertTrue(self.validator.validate_model_preferences(preferences))

    def test_validate_model_preferences_invalid_priority(self):
        """Test validation of model preferences with invalid priority."""
        preferences = {
            "hints": [{"name": "claude-3-5-sonnet"}],
            "costPriority": 1.5  # Invalid: > 1
        }
        self.assertFalse(self.validator.validate_model_preferences(preferences))

    def test_validate_sampling_request_valid(self):
        """Test validation of valid sampling request."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": "Hello, world!"
                    }
                }
            ],
            "maxTokens": 1000,
            "includeContext": "none"
        }
        self.assertTrue(self.validator.validate_sampling_request(request))
    
    def test_validate_sampling_request_missing_messages(self):
        """Test validation of sampling request with missing messages."""
        request = {
            "maxTokens": 1000,
            "includeContext": "none"
        }
        assert not self.validator.validate_sampling_request(request)
    
    def test_validate_sampling_request_invalid_context(self):
        """Test validation of sampling request with invalid context."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": "Hello, world!"
                    }
                }
            ],
            "maxTokens": 1000,
            "includeContext": "invalid"
        }
        assert not self.validator.validate_sampling_request(request)
    
    def test_validate_complete_configuration(self):
        """Test validation of complete MCP configuration."""
        config = {
            "transport": {
                "type": "stdio",
                "command": "test-server",
                "args": []
            },
            "client_info": {
                "name": "test-client",
                "version": "1.0.0"
            },
            "capabilities": {
                "sampling": {}
            },
            "default_sampling_params": {
                "temperature": 0.7,
                "maxTokens": 1000,
                "includeContext": "none"
            }
        }
        assert self.validator.validate_configuration(config)


class TestMCPProtocolUtilities:
    """Test MCP protocol utility functions."""
    
    def test_create_sampling_request(self):
        """Test creation of sampling request."""
        messages = [create_text_message("user", "Hello")]
        request = create_sampling_request(
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            include_context="none"
        )
        
        assert request["messages"] == messages
        assert request["temperature"] == 0.7
        assert request["maxTokens"] == 1000
        assert request["includeContext"] == "none"
    
    def test_create_text_message(self):
        """Test creation of text message."""
        message = create_text_message("user", "Hello, world!")
        
        assert message["role"] == "user"
        assert message["content"]["type"] == "text"
        assert message["content"]["text"] == "Hello, world!"
    
    def test_create_image_message(self):
        """Test creation of image message."""
        message = create_image_message("user", "base64data", "image/png")
        
        assert message["role"] == "user"
        assert message["content"]["type"] == "image"
        assert message["content"]["data"] == "base64data"
        assert message["content"]["mimeType"] == "image/png"


class TestMCPTransport:
    """Test MCP transport implementations."""
    
    def test_create_stdio_transport(self):
        """Test creation of stdio transport."""
        config = {
            "transport": {
                "type": "stdio",
                "command": "test-server",
                "args": []
            },
            "client_info": {
                "name": "test",
                "version": "1.0"
            }
        }
        transport = create_transport(config)
        assert isinstance(transport, StdioMCPTransport)
    
    def test_create_http_transport(self):
        """Test creation of HTTP transport."""
        config = {
            "transport": {
                "type": "http",
                "url": "http://localhost:8000/mcp"
            },
            "client_info": {
                "name": "test",
                "version": "1.0"
            }
        }
        transport = create_transport(config)
        assert isinstance(transport, HTTPMCPTransport)
    
    def test_create_transport_invalid_type(self):
        """Test creation of transport with invalid type."""
        config = {
            "transport": {
                "type": "invalid"
            },
            "client_info": {
                "name": "test",
                "version": "1.0"
            }
        }
        with pytest.raises(ValueError):
            create_transport(config)


class TestMCPLanguageModel:
    """Test MCP language model implementation."""
    
    def test_initialization_with_valid_config(self):
        """Test initialization with valid configuration."""
        # Create a temporary config file
        config = {
            "test_config": {
                "transport": {
                    "type": "stdio",
                    "command": "test-server",
                    "args": []
                },
                "client_info": {
                    "name": "test",
                    "version": "1.0"
                },
                "capabilities": {
                    "sampling": {}
                },
                "default_sampling_params": {
                    "temperature": 0.7,
                    "maxTokens": 1000,
                    "includeContext": "none"
                },
                "cost_tracking": {
                    "prompt_token_cost": 0.001,
                    "response_token_cost": 0.002
                }
            }
        }
        
        with patch('builtins.open'), \
             patch('json.load', return_value=config):
            lm = MCPLanguageModel(
                config_path="test_config.json",
                model_name="test_config",
                cache=False
            )
            
            assert lm.transport_type == "stdio"
            assert lm.client_info["name"] == "test"
            assert lm.prompt_token_cost == 0.001
            assert lm.response_token_cost == 0.002
    
    def test_initialization_with_invalid_config(self):
        """Test initialization with invalid configuration."""
        config = {
            "invalid_config": {
                "transport": {
                    "type": "invalid"
                }
            }
        }
        
        with patch('builtins.open'), \
             patch('json.load', return_value=config):
            with pytest.raises(ValueError):
                MCPLanguageModel(
                    config_path="test_config.json",
                    model_name="invalid_config",
                    cache=False
                )


@pytest.mark.asyncio
class TestMCPIntegration:
    """Integration tests for MCP functionality."""
    
    async def test_sampling_request_flow(self):
        """Test the complete sampling request flow."""
        # Mock transport
        mock_transport = AsyncMock()
        mock_transport.connect.return_value = True
        mock_transport.send_sampling_request.return_value = {
            "model": "claude-3-5-sonnet",
            "role": "assistant",
            "content": {
                "type": "text",
                "text": "Hello! How can I help you?"
            },
            "stopReason": "endTurn"
        }
        
        # Create language model with mocked transport
        config = {
            "test_config": {
                "transport": {
                    "type": "stdio",
                    "command": "test-server",
                    "args": []
                },
                "client_info": {
                    "name": "test",
                    "version": "1.0"
                },
                "capabilities": {
                    "sampling": {}
                },
                "default_sampling_params": {
                    "temperature": 0.7,
                    "maxTokens": 1000,
                    "includeContext": "none"
                }
            }
        }
        
        with patch('builtins.open'), \
             patch('json.load', return_value=config), \
             patch('graph_of_thoughts.language_models.mcp_transport.create_transport', return_value=mock_transport):
            
            lm = MCPLanguageModel(
                config_path="test_config.json",
                model_name="test_config",
                cache=False
            )
            
            # Test query
            response = await lm.query_async("Hello, world!")
            
            # Verify the response
            assert len(response) == 1
            assert "Hello! How can I help you?" in response[0]
            
            # Verify transport was called correctly
            mock_transport.connect.assert_called_once()
            mock_transport.send_sampling_request.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
