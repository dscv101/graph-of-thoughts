#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Comprehensive unit tests for MCP Transport implementations.

This module tests the MCP transport layer including stdio and HTTP transports,
connection management, error handling, and protocol compliance.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import httpx

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.language_models.mcp_transport import (
    HTTPMCPTransport,
    MCPConnectionError,
    MCPProtocolError,
    MCPServerError,
    MCPTimeoutError,
    MCPTransport,
    MCPTransportError,
    MCPValidationError,
    StdioMCPTransport,
    create_transport,
)


class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases."""

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


class TestMCPTransportBase(unittest.TestCase):
    """Test MCPTransport base class."""

    def test_abstract_methods(self):
        """Test that MCPTransport is abstract."""
        with self.assertRaises(TypeError):
            MCPTransport({})


class TestStdioMCPTransport(AsyncTestCase):
    """Test StdioMCPTransport implementation."""

    def setUp(self):
        """Set up test transport."""
        super().setUp()
        self.config = {
            "transport": {
                "type": "stdio",
                "command": "test-server",
                "args": ["--test"],
                "env": {"TEST_MODE": "true"},
            },
            "client_info": {"name": "test-client", "version": "1.0.0"},
            "capabilities": {"sampling": {}},
            "default_sampling_params": {"temperature": 0.7, "maxTokens": 1000},
        }

    def test_initialization(self):
        """Test transport initialization."""
        transport = StdioMCPTransport(self.config)

        self.assertEqual(transport.command, "test-server")
        self.assertEqual(transport.args, ["--test"])
        self.assertEqual(transport.env["TEST_MODE"], "true")
        self.assertFalse(transport.connected)
        self.assertIsNone(transport.process)

    def test_invalid_config(self):
        """Test initialization with invalid config."""
        invalid_configs = [
            {},  # Missing transport
            {"transport": {}},  # Missing command
            {"transport": {"command": ""}},  # Empty command
        ]

        for config in invalid_configs:
            with self.assertRaises((KeyError, ValueError)):
                StdioMCPTransport(config)

    @patch("graph_of_thoughts.language_models.mcp_transport.stdio_client")
    @patch("graph_of_thoughts.language_models.mcp_transport.ClientSession")
    async def _test_connect_success(self, mock_session_class, mock_stdio_client):
        """Test successful connection."""
        # Mock the stdio client and session
        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_stdio_client.return_value.__aenter__.return_value = (mock_read_stream, mock_write_stream)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        transport = StdioMCPTransport(self.config)

        result = await transport.connect()

        self.assertTrue(result)
        self.assertTrue(transport.connected)
        self.assertIsNotNone(transport.session)
        mock_stdio_client.assert_called_once()
        mock_session.initialize.assert_called_once()

    @patch("graph_of_thoughts.language_models.mcp_transport.stdio_client")
    async def _test_connect_failure(self, mock_stdio_client):
        """Test connection failure."""
        mock_stdio_client.side_effect = OSError("Command not found")

        transport = StdioMCPTransport(self.config)

        with self.assertRaises(MCPConnectionError):
            await transport.connect()

        self.assertFalse(transport.connected)

    async def _test_disconnect(self):
        """Test disconnection."""
        transport = StdioMCPTransport(self.config)

        # Mock connected state with new MCP SDK attributes
        mock_exit_stack = AsyncMock()
        mock_session = AsyncMock()
        transport.exit_stack = mock_exit_stack
        transport.session = mock_session
        transport.connected = True

        await transport.disconnect()

        self.assertFalse(transport.connected)
        self.assertIsNone(transport.session)
        mock_exit_stack.aclose.assert_called_once()

    async def _test_send_request(self):
        """Test sending requests."""
        transport = StdioMCPTransport(self.config)

        # Mock connected state and session
        transport.connected = True
        mock_session = AsyncMock()
        transport.session = mock_session

        # Mock response
        mock_result = AsyncMock()
        mock_result.content = [{"type": "text", "text": "success"}]
        mock_session.send_request.return_value = mock_result

        result = await transport.send_request("sampling/createMessage", {"messages": []})

        # The result should be the converted response
        self.assertIsInstance(result, dict)
        mock_session.send_request.assert_called_once()

    async def _test_send_request_not_connected(self):
        """Test sending request when not connected."""
        transport = StdioMCPTransport(self.config)

        with self.assertRaises(RuntimeError):
            await transport.send_request("test_method", {})

    async def _test_send_notification(self):
        """Test sending notifications."""
        transport = StdioMCPTransport(self.config)
        transport.connected = True
        mock_session = AsyncMock()
        transport.session = mock_session

        await transport.send_notification("test_notification", {"param": "value"})

        # Should not raise any exceptions
        mock_session.send_notification.assert_called_once()

    def test_context_manager(self):
        """Test async context manager."""
        transport = StdioMCPTransport(self.config)

        async def test_context():
            with patch.object(transport, "connect", return_value=True):
                with patch.object(transport, "disconnect"):
                    async with transport as t:
                        self.assertEqual(t, transport)

        self.run_async(test_context())

    def run_async_test(self, test_method):
        """Helper to run async test methods."""
        return self.run_async(test_method())

    def test_connect_success_sync(self):
        """Test successful connection (sync wrapper)."""
        self.run_async_test(self._test_connect_success)

    def test_connect_failure_sync(self):
        """Test connection failure (sync wrapper)."""
        self.run_async_test(self._test_connect_failure)

    def test_disconnect_sync(self):
        """Test disconnection (sync wrapper)."""
        self.run_async_test(self._test_disconnect)

    def test_send_request_sync(self):
        """Test sending requests (sync wrapper)."""
        self.run_async_test(self._test_send_request)

    def test_send_request_not_connected_sync(self):
        """Test sending request when not connected (sync wrapper)."""
        self.run_async_test(self._test_send_request_not_connected)

    def test_send_notification_sync(self):
        """Test sending notifications (sync wrapper)."""
        self.run_async_test(self._test_send_notification)


class TestHTTPMCPTransport(AsyncTestCase):
    """Test HTTPMCPTransport implementation."""

    def setUp(self):
        """Set up test transport."""
        super().setUp()
        self.config = {
            "transport": {
                "type": "http",
                "url": "http://localhost:8000/mcp",
                "headers": {"Authorization": "Bearer test-token"},
                "session_management": True,
            },
            "client_info": {"name": "test-client", "version": "1.0.0"},
            "capabilities": {"sampling": {}},
        }

    def test_initialization(self):
        """Test transport initialization."""
        transport = HTTPMCPTransport(self.config)

        self.assertEqual(transport.base_url, "http://localhost:8000/mcp")
        self.assertIn("Authorization", transport.headers)
        self.assertTrue(transport.session_management)
        self.assertFalse(transport.connected)

    def test_invalid_config(self):
        """Test initialization with invalid config."""
        invalid_configs = [
            {},  # Missing transport
            {"transport": {}},  # Missing URL
            {"transport": {"url": ""}},  # Empty URL
            {"transport": {"url": "invalid-url"}},  # Invalid URL format
        ]

        for config in invalid_configs:
            with self.assertRaises((KeyError, ValueError)):
                HTTPMCPTransport(config)

    @patch("httpx.AsyncClient")
    async def _test_connect_success(self, mock_client_class):
        """Test successful HTTP connection."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock initialization response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"sampling": {}},
                "serverInfo": {"name": "test-server", "version": "1.0.0"},
            },
        })
        mock_client.post.return_value = mock_response

        transport = HTTPMCPTransport(self.config)
        result = await transport.connect()

        self.assertTrue(result)
        self.assertTrue(transport.connected)
        self.assertIsNotNone(transport.client)

    @patch("httpx.AsyncClient")
    async def _test_connect_failure(self, mock_client_class):
        """Test HTTP connection failure."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.post.side_effect = httpx.ConnectError("Connection failed")

        transport = HTTPMCPTransport(self.config)

        with self.assertRaises(MCPConnectionError):
            await transport.connect()

        self.assertFalse(transport.connected)

    async def _test_send_request(self):
        """Test sending HTTP requests."""
        transport = HTTPMCPTransport(self.config)
        transport.connected = True
        transport.client = AsyncMock()

        # Mock response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"message": "success"},
        }
        transport.client.post.return_value = mock_response

        result = await transport.send_request("test_method", {"param": "value"})

        self.assertEqual(result, {"message": "success"})
        transport.client.post.assert_called_once()

    async def _test_send_request_error_response(self):
        """Test handling error responses."""
        transport = HTTPMCPTransport(self.config)
        transport.connected = True
        transport.client = AsyncMock()

        # Mock error response
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "Internal Server Error"
        transport.client.post.return_value = mock_response

        with self.assertRaises(MCPServerError):
            await transport.send_request("test_method", {})

    def run_async_test(self, test_method):
        """Helper to run async test methods."""
        return self.run_async(test_method())

    def test_connect_success_sync(self):
        """Test successful HTTP connection (sync wrapper)."""
        self.run_async_test(self._test_connect_success)

    def test_connect_failure_sync(self):
        """Test HTTP connection failure (sync wrapper)."""
        self.run_async_test(self._test_connect_failure)

    def test_send_request_sync(self):
        """Test sending HTTP requests (sync wrapper)."""
        self.run_async_test(self._test_send_request)

    def test_send_request_error_response_sync(self):
        """Test handling error responses (sync wrapper)."""
        self.run_async_test(self._test_send_request_error_response)


class TestTransportFactory(unittest.TestCase):
    """Test transport factory function."""

    def test_create_stdio_transport(self):
        """Test creating stdio transport."""
        config = {"transport": {"type": "stdio", "command": "test-server"}}

        transport = create_transport(config)
        self.assertIsInstance(transport, StdioMCPTransport)

    def test_create_http_transport(self):
        """Test creating HTTP transport."""
        config = {"transport": {"type": "http", "url": "http://localhost:8000"}}

        transport = create_transport(config)
        self.assertIsInstance(transport, HTTPMCPTransport)

    def test_unsupported_transport_type(self):
        """Test creating transport with unsupported type."""
        config = {"transport": {"type": "unsupported"}}

        with self.assertRaises(ValueError):
            create_transport(config)


class TestMCPExceptions(unittest.TestCase):
    """Test MCP exception classes."""

    def test_exception_hierarchy(self):
        """Test exception inheritance hierarchy."""
        self.assertTrue(issubclass(MCPConnectionError, MCPTransportError))
        self.assertTrue(issubclass(MCPTimeoutError, MCPTransportError))
        self.assertTrue(issubclass(MCPProtocolError, MCPTransportError))
        self.assertTrue(issubclass(MCPServerError, MCPTransportError))
        self.assertTrue(issubclass(MCPValidationError, MCPTransportError))

    def test_exception_messages(self):
        """Test exception message handling."""
        error = MCPConnectionError("Connection failed")
        self.assertEqual(str(error), "Connection failed")

        error = MCPTimeoutError("Request timed out")
        self.assertEqual(str(error), "Request timed out")


if __name__ == "__main__":
    unittest.main()
