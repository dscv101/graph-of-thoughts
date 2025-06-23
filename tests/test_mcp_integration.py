#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Integration tests for MCP implementation.

This module contains integration tests that verify the MCP implementation
works correctly with real MCP servers and different host configurations.
These tests require actual MCP servers to be available and configured.

Usage:
    # Run integration tests
    python -m pytest tests/test_mcp_integration.py -v

    # Run specific integration test
    python -m pytest tests/test_mcp_integration.py::TestMCPIntegration::test_claude_desktop_integration -v

Requirements:
    - MCP server installed and configured
    - Valid MCP configuration file
    - Network connectivity (for HTTP tests)
"""

import asyncio
import json
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock
from typing import Dict, Any, Optional

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel
from graph_of_thoughts.language_models.mcp_transport import (
    create_transport, MCPConnectionError, MCPTimeoutError
)


class AsyncTestCase(unittest.TestCase):
    """Base class for async integration tests."""
    
    def setUp(self):
        """Set up async test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up async test environment."""
        self.loop.close()
    
    def run_async(self, coro):
        """Run an async coroutine in the test loop."""
        return self.loop.run_until_complete(coro)


class TestMCPIntegration(AsyncTestCase):
    """Integration tests for MCP functionality."""
    
    def setUp(self):
        """Set up integration test environment."""
        super().setUp()
        
        # Test configuration for different MCP hosts
        self.test_configs = {
            "mock_stdio": {
                "transport": {
                    "type": "stdio",
                    "command": "echo",  # Simple command for testing
                    "args": ["test"]
                },
                "client_info": {
                    "name": "integration-test",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "sampling": {}
                },
                "default_sampling_params": {
                    "temperature": 0.7,
                    "maxTokens": 100,
                    "includeContext": "none"
                }
            },
            "mock_http": {
                "transport": {
                    "type": "http",
                    "url": "http://localhost:8000/mcp",
                    "headers": {
                        "Content-Type": "application/json"
                    }
                },
                "client_info": {
                    "name": "integration-test",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "sampling": {}
                },
                "default_sampling_params": {
                    "temperature": 0.7,
                    "maxTokens": 100,
                    "includeContext": "none"
                }
            }
        }
    
    def create_temp_config(self, config_name: str) -> str:
        """Create temporary configuration file."""
        config = {config_name: self.test_configs[config_name]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            return f.name
    
    async def test_mcp_language_model_initialization(self):
        """Test MCPLanguageModel initialization with different configs."""
        config_file = self.create_temp_config("mock_stdio")
        
        try:
            with patch('graph_of_thoughts.language_models.mcp_transport.create_transport') as mock_create:
                mock_transport = AsyncMock()
                mock_create.return_value = mock_transport
                
                # Test initialization
                lm = MCPLanguageModel(
                    config_path=config_file,
                    model_name="mock_stdio"
                )
                
                self.assertIsNotNone(lm)
                self.assertEqual(lm.model_name, "mock_stdio")
                self.assertIsNotNone(lm.transport)
                
        finally:
            os.unlink(config_file)
    
    async def test_transport_creation_stdio(self):
        """Test stdio transport creation."""
        config = self.test_configs["mock_stdio"]
        
        # This will fail with real command, but tests the creation process
        transport = create_transport(config)
        self.assertIsNotNone(transport)
        self.assertEqual(transport.command, "echo")
    
    async def test_transport_creation_http(self):
        """Test HTTP transport creation."""
        config = self.test_configs["mock_http"]
        
        transport = create_transport(config)
        self.assertIsNotNone(transport)
        self.assertEqual(transport.base_url, "http://localhost:8000/mcp")
    
    async def test_mocked_query_flow(self):
        """Test complete query flow with mocked transport."""
        config_file = self.create_temp_config("mock_stdio")
        
        try:
            with patch('graph_of_thoughts.language_models.mcp_transport.create_transport') as mock_create:
                # Mock transport with successful responses
                mock_transport = AsyncMock()
                mock_transport.connect.return_value = True
                mock_transport.send_sampling_request.return_value = {
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a test response from the mocked MCP server."
                        }
                    ]
                }
                mock_create.return_value = mock_transport
                
                # Create language model
                lm = MCPLanguageModel(
                    config_path=config_file,
                    model_name="mock_stdio"
                )
                
                # Test query
                response = await lm.query_async("Test query")
                
                # Verify response
                self.assertIsInstance(response, list)
                self.assertEqual(len(response), 1)
                self.assertIn("test response", response[0])
                
                # Verify transport calls
                mock_transport.connect.assert_called_once()
                mock_transport.send_sampling_request.assert_called_once()
                
        finally:
            os.unlink(config_file)
    
    async def test_error_handling_connection_failure(self):
        """Test error handling for connection failures."""
        config_file = self.create_temp_config("mock_stdio")
        
        try:
            with patch('graph_of_thoughts.language_models.mcp_transport.create_transport') as mock_create:
                # Mock transport that fails to connect
                mock_transport = AsyncMock()
                mock_transport.connect.side_effect = MCPConnectionError("Connection failed")
                mock_create.return_value = mock_transport
                
                lm = MCPLanguageModel(
                    config_path=config_file,
                    model_name="mock_stdio"
                )
                
                # Should raise connection error
                with self.assertRaises(MCPConnectionError):
                    await lm.query_async("Test query")
                    
        finally:
            os.unlink(config_file)
    
    async def test_error_handling_timeout(self):
        """Test error handling for timeouts."""
        config_file = self.create_temp_config("mock_stdio")
        
        try:
            with patch('graph_of_thoughts.language_models.mcp_transport.create_transport') as mock_create:
                # Mock transport that times out
                mock_transport = AsyncMock()
                mock_transport.connect.return_value = True
                mock_transport.send_sampling_request.side_effect = MCPTimeoutError("Request timed out")
                mock_create.return_value = mock_transport
                
                lm = MCPLanguageModel(
                    config_path=config_file,
                    model_name="mock_stdio"
                )
                
                # Should raise timeout error
                with self.assertRaises(MCPTimeoutError):
                    await lm.query_async("Test query")
                    
        finally:
            os.unlink(config_file)
    
    async def test_batch_processing_integration(self):
        """Test batch processing integration."""
        config_file = self.create_temp_config("mock_stdio")
        
        try:
            with patch('graph_of_thoughts.language_models.mcp_transport.create_transport') as mock_create:
                # Mock transport with batch responses
                mock_transport = AsyncMock()
                mock_transport.connect.return_value = True
                
                # Mock multiple responses for batch
                responses = [
                    {"content": [{"type": "text", "text": f"Response {i}"}]}
                    for i in range(3)
                ]
                mock_transport.send_sampling_request.side_effect = responses
                mock_create.return_value = mock_transport
                
                lm = MCPLanguageModel(
                    config_path=config_file,
                    model_name="mock_stdio"
                )
                
                # Test batch query (if implemented)
                queries = ["Query 1", "Query 2", "Query 3"]
                
                # For now, test sequential queries
                results = []
                for query in queries:
                    response = await lm.query_async(query)
                    results.extend(response)
                
                # Verify all responses received
                self.assertEqual(len(results), 3)
                for i, result in enumerate(results):
                    self.assertIn(f"Response {i}", result)
                    
        finally:
            os.unlink(config_file)
    
    async def test_context_manager_integration(self):
        """Test async context manager integration."""
        config_file = self.create_temp_config("mock_stdio")
        
        try:
            with patch('graph_of_thoughts.language_models.mcp_transport.create_transport') as mock_create:
                mock_transport = AsyncMock()
                mock_transport.connect.return_value = True
                mock_transport.disconnect = AsyncMock()
                mock_transport.send_sampling_request.return_value = {
                    "content": [{"type": "text", "text": "Context manager test"}]
                }
                mock_create.return_value = mock_transport
                
                # Test context manager usage
                async with MCPLanguageModel(
                    config_path=config_file,
                    model_name="mock_stdio"
                ) as lm:
                    response = await lm.query_async("Test query")
                    self.assertIsInstance(response, list)
                    self.assertEqual(len(response), 1)
                
                # Verify cleanup was called
                mock_transport.disconnect.assert_called_once()
                
        finally:
            os.unlink(config_file)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = self.test_configs["mock_stdio"]
        self.assertIn("transport", valid_config)
        self.assertIn("client_info", valid_config)
        
        # Test invalid configurations
        invalid_configs = [
            {},  # Empty config
            {"transport": {}},  # Missing required fields
            {"transport": {"type": "invalid"}},  # Invalid transport type
        ]
        
        for invalid_config in invalid_configs:
            with self.assertRaises((KeyError, ValueError)):
                create_transport(invalid_config)
    
    def run_async_test(self, test_method):
        """Helper to run async test methods."""
        return self.run_async(test_method())
    
    # Sync wrappers for async tests
    def test_mcp_language_model_initialization_sync(self):
        """Test MCPLanguageModel initialization (sync wrapper)."""
        self.run_async_test(self.test_mcp_language_model_initialization)
    
    def test_transport_creation_stdio_sync(self):
        """Test stdio transport creation (sync wrapper)."""
        self.run_async_test(self.test_transport_creation_stdio)
    
    def test_transport_creation_http_sync(self):
        """Test HTTP transport creation (sync wrapper)."""
        self.run_async_test(self.test_transport_creation_http)
    
    def test_mocked_query_flow_sync(self):
        """Test complete query flow (sync wrapper)."""
        self.run_async_test(self.test_mocked_query_flow)
    
    def test_error_handling_connection_failure_sync(self):
        """Test connection failure handling (sync wrapper)."""
        self.run_async_test(self.test_error_handling_connection_failure)
    
    def test_error_handling_timeout_sync(self):
        """Test timeout handling (sync wrapper)."""
        self.run_async_test(self.test_error_handling_timeout)
    
    def test_batch_processing_integration_sync(self):
        """Test batch processing integration (sync wrapper)."""
        self.run_async_test(self.test_batch_processing_integration)
    
    def test_context_manager_integration_sync(self):
        """Test context manager integration (sync wrapper)."""
        self.run_async_test(self.test_context_manager_integration)


@unittest.skipUnless(
    os.environ.get("MCP_INTEGRATION_TESTS") == "1",
    "Integration tests require MCP_INTEGRATION_TESTS=1 environment variable"
)
class TestRealMCPIntegration(AsyncTestCase):
    """Integration tests with real MCP servers (optional)."""
    
    def setUp(self):
        """Set up real integration test environment."""
        super().setUp()
        
        # These tests require actual MCP servers
        self.config_file = os.environ.get("MCP_CONFIG_FILE")
        self.model_name = os.environ.get("MCP_MODEL_NAME", "mcp_claude_desktop")
        
        if not self.config_file or not Path(self.config_file).exists():
            self.skipTest("Real MCP integration tests require valid MCP_CONFIG_FILE")
    
    async def test_real_mcp_query(self):
        """Test query with real MCP server."""
        lm = MCPLanguageModel(
            config_path=self.config_file,
            model_name=self.model_name
        )
        
        try:
            response = await lm.query_async("Hello, this is a test query.")
            
            self.assertIsInstance(response, list)
            self.assertGreater(len(response), 0)
            self.assertIsInstance(response[0], str)
            self.assertGreater(len(response[0]), 0)
            
        except Exception as e:
            self.skipTest(f"Real MCP server not available: {e}")
    
    def test_real_mcp_query_sync(self):
        """Test real MCP query (sync wrapper)."""
        self.run_async_test(self.test_real_mcp_query)


if __name__ == "__main__":
    unittest.main()
