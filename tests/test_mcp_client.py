#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Comprehensive unit tests for MCP Client implementation.

This module tests the MCPLanguageModel class including configuration loading,
query processing, response parsing, error handling, and integration features.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel
from graph_of_thoughts.language_models.mcp_transport import (
    MCPConnectionError,
    MCPTransportError,
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


class TestMCPLanguageModel(AsyncTestCase):
    """Test MCPLanguageModel functionality."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.test_config = {
            "mcp_test": {
                "transport": {
                    "type": "stdio",
                    "command": "test-server",
                    "args": ["--test"],
                },
                "client_info": {"name": "test-client", "version": "1.0.0"},
                "capabilities": {"sampling": {}},
                "default_sampling_params": {
                    "temperature": 0.7,
                    "maxTokens": 1000,
                    "includeContext": "none",
                },
                "cost_tracking": {
                    "prompt_token_cost": 0.001,
                    "response_token_cost": 0.002,
                },
            }
        }

    def test_initialization_with_config_dict(self):
        """Test initialization with configuration dictionary."""
        with patch(
            "graph_of_thoughts.language_models.mcp_transport.create_transport"
        ) as mock_create:
            mock_transport = AsyncMock()
            mock_create.return_value = mock_transport

            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            self.assertEqual(lm.model_name, "mcp_test")
            self.assertEqual(lm.config, self.test_config["mcp_test"])
            self.assertIsNotNone(lm.transport)

    def test_initialization_with_config_file(self):
        """Test initialization with configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.test_config, f)
            config_path = f.name

        try:
            with patch(
                "graph_of_thoughts.language_models.mcp_transport.create_transport"
            ) as mock_create:
                mock_transport = AsyncMock()
                mock_create.return_value = mock_transport

                lm = MCPLanguageModel(config_path=config_path, model_name="mcp_test")

                self.assertEqual(lm.model_name, "mcp_test")
                self.assertEqual(lm.config["transport"]["command"], "test-server")
        finally:
            os.unlink(config_path)

    def test_initialization_missing_config(self):
        """Test initialization with missing configuration."""
        with self.assertRaises(FileNotFoundError):
            MCPLanguageModel(model_name="nonexistent")

    def test_initialization_invalid_model_name(self):
        """Test initialization with invalid model name."""
        with self.assertRaises(ValueError):
            MCPLanguageModel(config=self.test_config, model_name="invalid_model")

    async def _test_query_async_success(self):
        """Test successful async query."""
        with patch(
            "graph_of_thoughts.language_models.mcp_transport.create_transport"
        ) as mock_create:
            mock_transport = AsyncMock()
            mock_transport.connect.return_value = True
            mock_transport.send_sampling_request.return_value = {
                "content": [
                    {"type": "text", "text": "Hello! How can I help you today?"}
                ]
            }
            mock_create.return_value = mock_transport

            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            response = await lm.query_async("Hello, world!")

            # Extract text using the proper API
            texts = lm.get_response_texts(response)
            self.assertEqual(len(texts), 1)
            self.assertIn("Hello! How can I help you today?", texts[0])
            mock_transport.connect.assert_called_once()
            mock_transport.send_sampling_request.assert_called_once()

    async def _test_query_async_with_params(self):
        """Test async query with custom parameters."""
        with patch(
            "graph_of_thoughts.language_models.mcp_transport.create_transport"
        ) as mock_create:
            mock_transport = AsyncMock()
            mock_transport.connect.return_value = True
            mock_transport.send_sampling_request.return_value = {
                "content": [{"type": "text", "text": "Response"}]
            }
            mock_create.return_value = mock_transport

            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            await lm.query_async(
                "Test query", temperature=0.5, max_tokens=500, num_responses=2
            )

            # Verify the sampling request was called with correct parameters
            call_args = mock_transport.send_sampling_request.call_args[0][0]
            self.assertEqual(call_args["temperature"], 0.5)
            self.assertEqual(call_args["maxTokens"], 500)

    async def _test_query_async_connection_error(self):
        """Test async query with connection error."""
        with patch(
            "graph_of_thoughts.language_models.mcp_transport.create_transport"
        ) as mock_create:
            mock_transport = AsyncMock()
            mock_transport.connect.side_effect = MCPConnectionError("Connection failed")
            mock_create.return_value = mock_transport

            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            with self.assertRaises(MCPConnectionError):
                await lm.query_async("Test query")

    def test_query_sync(self):
        """Test synchronous query wrapper."""
        with patch(
            "graph_of_thoughts.language_models.mcp_transport.create_transport"
        ) as mock_create:
            mock_transport = AsyncMock()
            mock_transport.connect.return_value = True
            mock_transport.send_sampling_request.return_value = {
                "content": [{"type": "text", "text": "Sync response"}]
            }
            mock_create.return_value = mock_transport

            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            response = lm.query("Test sync query")

            # Extract text using the proper API
            texts = lm.get_response_texts(response)
            self.assertEqual(len(texts), 1)
            self.assertIn("Sync response", texts[0])

    def test_get_response_texts_single(self):
        """Test extracting text from single response."""
        with patch("graph_of_thoughts.language_models.mcp_transport.create_transport"):
            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            response = {
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                ]
            }

            texts = lm.get_response_texts(response)
            self.assertEqual(len(texts), 2)
            self.assertEqual(texts[0], "First part")
            self.assertEqual(texts[1], "Second part")

    def test_get_response_texts_multiple(self):
        """Test extracting text from multiple responses."""
        with patch("graph_of_thoughts.language_models.mcp_transport.create_transport"):
            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            responses = [
                {"content": [{"type": "text", "text": "Response 1"}]},
                {"content": [{"type": "text", "text": "Response 2"}]},
            ]

            texts = lm.get_response_texts(responses)
            self.assertEqual(len(texts), 2)
            self.assertEqual(texts[0], "Response 1")
            self.assertEqual(texts[1], "Response 2")

    def test_cost_tracking(self):
        """Test cost tracking functionality."""
        with patch("graph_of_thoughts.language_models.mcp_transport.create_transport"):
            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            # Test cost calculation
            prompt_tokens = 100
            response_tokens = 50

            cost = lm._calculate_cost(prompt_tokens, response_tokens)
            # Cost calculation is per 1000 tokens
            expected_cost = (100 / 1000.0 * 0.001) + (50 / 1000.0 * 0.002)
            self.assertEqual(cost, expected_cost)

    def test_token_estimation(self):
        """Test token estimation."""
        with patch("graph_of_thoughts.language_models.mcp_transport.create_transport"):
            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            # Test word-based estimation
            text = "This is a test message with several words"
            tokens = lm._estimate_tokens(text)

            # Should be approximately the number of words
            word_count = len(text.split())
            self.assertGreater(tokens, word_count * 0.5)
            self.assertLess(tokens, word_count * 2)

    async def _test_context_manager(self):
        """Test async context manager."""
        with patch(
            "graph_of_thoughts.language_models.mcp_transport.create_transport"
        ) as mock_create:
            mock_transport = AsyncMock()
            mock_transport.connect.return_value = True
            mock_transport.disconnect = AsyncMock()
            mock_create.return_value = mock_transport

            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            async with lm as client:
                self.assertEqual(client, lm)
                mock_transport.connect.assert_called_once()

            mock_transport.disconnect.assert_called_once()

    def test_circuit_breaker_status(self):
        """Test circuit breaker status methods."""
        with patch(
            "graph_of_thoughts.language_models.mcp_transport.create_transport"
        ) as mock_create:
            mock_transport = AsyncMock()

            # Mock circuit breaker methods - use MagicMock instead of AsyncMock for these
            mock_transport.get_circuit_breaker_metrics = MagicMock(return_value=MagicMock(
                total_requests=10,
                successful_requests=8,
                failed_requests=2,
                circuit_open_count=1,
                last_failure_time=1234567890,
                state_change_time=1234567890,
            ))
            # Import the actual enum for proper mocking
            from graph_of_thoughts.language_models.mcp_circuit_breaker import CircuitBreakerState
            mock_transport.get_circuit_breaker_state = MagicMock(return_value=CircuitBreakerState.CLOSED)
            mock_transport.is_circuit_healthy = MagicMock(return_value=True)

            mock_create.return_value = mock_transport

            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            # Test circuit breaker status
            status = lm.get_circuit_breaker_status()
            self.assertIsNotNone(status)
            self.assertEqual(status["state"], "closed")
            self.assertTrue(status["is_healthy"])
            self.assertEqual(status["total_requests"], 10)

            # Test service health check
            self.assertTrue(lm.is_service_healthy())

    def test_circuit_breaker_status_not_available(self):
        """Test circuit breaker status when not available."""
        with patch(
            "graph_of_thoughts.language_models.mcp_transport.create_transport"
        ) as mock_create:
            mock_transport = AsyncMock()
            # Don't add circuit breaker methods
            mock_create.return_value = mock_transport

            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            # Should return None when circuit breaker not available
            status = lm.get_circuit_breaker_status()
            self.assertIsNone(status)

            # Should assume healthy when no circuit breaker
            self.assertTrue(lm.is_service_healthy())

    def run_async_test(self, test_method):
        """Helper to run async test methods."""
        return self.run_async(test_method())

    def test_query_async_success_sync(self):
        """Test successful async query (sync wrapper)."""
        self.run_async_test(self._test_query_async_success)

    def test_query_async_with_params_sync(self):
        """Test async query with custom parameters (sync wrapper)."""
        self.run_async_test(self._test_query_async_with_params)

    def test_query_async_connection_error_sync(self):
        """Test async query with connection error (sync wrapper)."""
        self.run_async_test(self._test_query_async_connection_error)

    def test_context_manager_sync(self):
        """Test async context manager (sync wrapper)."""
        self.run_async_test(self._test_context_manager)


class TestMCPLanguageModelEdgeCases(AsyncTestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.test_config = {
            "mcp_test": {
                "transport": {"type": "stdio", "command": "test"},
                "client_info": {"name": "test", "version": "1.0"},
                "capabilities": {"sampling": {}},
                "default_sampling_params": {"temperature": 0.7},
            }
        }

    def test_empty_response(self):
        """Test handling of empty response."""
        with patch("graph_of_thoughts.language_models.mcp_transport.create_transport"):
            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            # Test empty content
            response = {"content": []}
            texts = lm.get_response_texts(response)
            self.assertEqual(texts, [])

    def test_malformed_response(self):
        """Test handling of malformed response."""
        with patch("graph_of_thoughts.language_models.mcp_transport.create_transport"):
            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            # Test missing content field - new implementation provides descriptive message
            response = {"invalid": "structure"}
            texts = lm.get_response_texts(response)
            self.assertEqual(len(texts), 1)
            self.assertIn("Unknown content type", texts[0])

    def test_mixed_content_types(self):
        """Test handling of mixed content types."""
        with patch("graph_of_thoughts.language_models.mcp_transport.create_transport"):
            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            response = {
                "content": [
                    {"type": "text", "text": "Text content"},
                    {"type": "image", "data": "base64data"},
                    {"type": "text", "text": "More text"},
                ]
            }

            texts = lm.get_response_texts(response)
            # New implementation includes image content as descriptive placeholder
            self.assertEqual(len(texts), 3)
            self.assertEqual(texts[0], "Text content")
            self.assertIn("Image content", texts[1])  # Image placeholder
            self.assertEqual(texts[2], "More text")

    def test_large_token_estimation(self):
        """Test token estimation with large text."""
        with patch("graph_of_thoughts.language_models.mcp_transport.create_transport"):
            lm = MCPLanguageModel(config=self.test_config, model_name="mcp_test")

            # Test with large text
            large_text = "word " * 10000
            tokens = lm._estimate_tokens(large_text)

            # Should handle large text without issues
            self.assertGreater(tokens, 5000)
            self.assertLess(tokens, 20000)


if __name__ == "__main__":
    unittest.main()
