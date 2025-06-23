#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Comprehensive unit tests for MCP Sampling implementation.

This module tests the MCP sampling functionality including request creation,
batch processing, parameter validation, and response handling.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.language_models.mcp_sampling import (
    MCPSamplingManager
)
from graph_of_thoughts.language_models.mcp_protocol import (
    create_sampling_request, create_text_message, create_image_message, MCPIncludeContext
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


class TestSamplingRequestCreation(unittest.TestCase):
    """Test sampling request creation functions."""
    
    def test_create_text_message(self):
        """Test creating text messages."""
        message = create_text_message("user", "Hello, world!")
        
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"]["type"], "text")
        self.assertEqual(message["content"]["text"], "Hello, world!")
    
    def test_create_text_message_with_role(self):
        """Test creating text messages with custom role."""
        message = create_text_message("system", "System message")
        
        self.assertEqual(message["role"], "system")
        self.assertEqual(message["content"]["text"], "System message")
    
    def test_create_image_message(self):
        """Test creating image messages."""
        image_data = "base64encodeddata"
        message = create_image_message("user", image_data, "image/png")
        
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"]["type"], "image")
        self.assertEqual(message["content"]["data"], image_data)
        self.assertEqual(message["content"]["mimeType"], "image/png")
    
    def test_create_sampling_request_basic(self):
        """Test creating basic sampling request."""
        messages = [create_text_message("user", "Test prompt")]
        request = create_sampling_request(messages)
        
        self.assertEqual(request["messages"], messages)
        self.assertEqual(request["maxTokens"], 1000)
        self.assertEqual(request["includeContext"], "none")
    
    def test_create_sampling_request_with_params(self):
        """Test creating sampling request with custom parameters."""
        messages = [create_text_message("user", "Test prompt")]
        request = create_sampling_request(
            messages,
            temperature=0.5,
            max_tokens=500,
            stop_sequences=["END"],
            include_context=MCPIncludeContext.THIS_SERVER.value
        )
        
        self.assertEqual(request["temperature"], 0.5)
        self.assertEqual(request["maxTokens"], 500)
        self.assertEqual(request["stopSequences"], ["END"])
        self.assertEqual(request["includeContext"], "thisServer")
    
    def test_create_sampling_request_with_model_preferences(self):
        """Test creating sampling request with model preferences."""
        messages = [create_text_message("user", "Test prompt")]
        model_preferences = {
            "hints": [{"name": "claude-3-5-sonnet"}],
            "costPriority": 0.3,
            "speedPriority": 0.4,
            "intelligencePriority": 0.8
        }
        
        request = create_sampling_request(
            messages,
            model_preferences=model_preferences
        )
        
        self.assertEqual(request["modelPreferences"], model_preferences)
    
    def test_include_context_enum(self):
        """Test MCPIncludeContext enum values."""
        self.assertEqual(MCPIncludeContext.NONE.value, "none")
        self.assertEqual(MCPIncludeContext.THIS_SERVER.value, "thisServer")
        self.assertEqual(MCPIncludeContext.ALL_SERVERS.value, "allServers")


class TestMCPSamplingManager(AsyncTestCase):
    """Test MCPSamplingManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.mock_transport = AsyncMock()
        self.config = {
            "default_sampling_params": {
                "temperature": 0.7,
                "maxTokens": 1000,
                "includeContext": "none"
            },
            "batch_processing": {
                "max_concurrent": 5,
                "batch_size": 10,
                "retry_attempts": 3,
                "retry_delay": 1.0,
                "timeout_per_request": 30.0,
                "enable_by_default": True
            }
        }
        self.manager = MCPSamplingManager(self.mock_transport, self.config)
    
    async def test_single_request(self):
        """Test sending single sampling request."""
        self.mock_transport.send_sampling_request.return_value = {
            "content": [{"type": "text", "text": "Response"}]
        }
        
        messages = [create_text_message("user", "Test prompt")]
        response = await self.manager.create_message(
            messages,
            temperature=0.5
        )
        
        self.assertEqual(response["content"][0]["text"], "Response")
        self.mock_transport.send_sampling_request.assert_called_once()
        
        # Check that custom temperature was used
        call_args = self.mock_transport.send_sampling_request.call_args[0][0]
        self.assertEqual(call_args["params"]["temperature"], 0.5)
    
    async def test_batch_request(self):
        """Test batch processing of multiple requests."""
        # Mock responses for batch
        self.mock_transport.send_sampling_request.return_value = {
            "content": [{"type": "text", "text": "Response"}]
        }

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = await self.manager.create_batch_completions(prompts)

        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0], "Response")
        self.assertEqual(responses[1], "Response")
        self.assertEqual(responses[2], "Response")

        # Should have made 3 calls
        self.assertEqual(self.mock_transport.send_sampling_request.call_count, 3)
    
    async def test_batch_with_custom_params(self):
        """Test batch processing with custom parameters."""
        self.mock_transport.send_sampling_request.return_value = {
            "content": [{"type": "text", "text": "Response"}]
        }

        prompts = ["Prompt 1", "Prompt 2"]
        await self.manager.create_batch_completions(
            prompts,
            temperature=0.3,
            max_tokens=500
        )

        # Check that all requests used custom parameters
        for call in self.mock_transport.send_sampling_request.call_args_list:
            params = call[0][0]["params"]
            self.assertEqual(params["temperature"], 0.3)
            self.assertEqual(params["maxTokens"], 500)
    
    async def test_batch_error_handling(self):
        """Test error handling in batch processing."""
        # Mock one success and one failure
        self.mock_transport.send_sampling_request.side_effect = [
            {"content": [{"type": "text", "text": "Success"}]},
            Exception("Request failed")
        ]

        prompts = ["Good prompt", "Bad prompt"]
        responses = await self.manager.create_batch_completions(prompts)

        # Should return partial results
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0], "Success")
        self.assertIsNone(responses[1])  # Failed request should be None
    
    async def test_batch_size_limiting(self):
        """Test batch size limiting."""
        # Create more prompts than batch size
        prompts = [f"Prompt {i}" for i in range(15)]

        self.mock_transport.send_sampling_request.return_value = {
            "content": [{"type": "text", "text": "Response"}]
        }

        responses = await self.manager.create_batch_completions(prompts)

        # Should process all prompts despite batch size limit
        self.assertEqual(len(responses), 15)
        self.assertEqual(self.mock_transport.send_sampling_request.call_count, 15)
    
    async def test_concurrent_limiting(self):
        """Test concurrent request limiting."""
        # Create many prompts to test concurrency limiting
        prompts = [f"Prompt {i}" for i in range(20)]
        
        # Track concurrent calls
        active_calls = 0
        max_concurrent = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal active_calls, max_concurrent
            active_calls += 1
            max_concurrent = max(max_concurrent, active_calls)
            await asyncio.sleep(0.01)  # Simulate processing time
            active_calls -= 1
            return {"content": [{"type": "text", "text": "Response"}]}
        
        self.mock_transport.send_sampling_request.side_effect = mock_request
        
        await self.manager.create_batch_completions(prompts)
        
        # Should not exceed configured max_concurrent
        self.assertLessEqual(max_concurrent, self.config["batch_processing"]["max_concurrent"])
    
    # Parameter merging and validation tests removed - methods don't exist in current implementation
    
    def run_async_test(self, test_method):
        """Helper to run async test methods."""
        return self.run_async(test_method())
    
    def test_single_request_sync(self):
        """Test sending single sampling request (sync wrapper)."""
        self.run_async_test(self.test_single_request)
    
    def test_batch_request_sync(self):
        """Test batch processing (sync wrapper)."""
        self.run_async_test(self.test_batch_request)
    
    def test_batch_with_custom_params_sync(self):
        """Test batch with custom parameters (sync wrapper)."""
        self.run_async_test(self.test_batch_with_custom_params)
    
    def test_batch_error_handling_sync(self):
        """Test batch error handling (sync wrapper)."""
        self.run_async_test(self.test_batch_error_handling)
    
    def test_batch_size_limiting_sync(self):
        """Test batch size limiting (sync wrapper)."""
        self.run_async_test(self.test_batch_size_limiting)
    
    def test_concurrent_limiting_sync(self):
        """Test concurrent limiting (sync wrapper)."""
        self.run_async_test(self.test_concurrent_limiting)


# BatchProcessor tests removed - class doesn't exist in current implementation


if __name__ == "__main__":
    unittest.main()
