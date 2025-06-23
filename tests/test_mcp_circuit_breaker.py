#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Unit tests for the MCP Circuit Breaker implementation.

This module contains comprehensive tests for the circuit breaker pattern
implementation, including state transitions, failure detection, recovery
mechanisms, and metrics collection.
"""

import asyncio
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.language_models.mcp_circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    CircuitBreakerTransportWrapper,
    MCPCircuitBreaker,
    create_circuit_breaker_from_config,
)


class TestCircuitBreakerConfig(unittest.TestCase):
    """Test CircuitBreakerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        self.assertEqual(config.failure_threshold, 5)
        self.assertEqual(config.recovery_timeout, 30.0)
        self.assertEqual(config.half_open_max_calls, 3)
        self.assertEqual(config.expected_exceptions, (Exception,))
        self.assertEqual(config.success_threshold, 2)
        self.assertEqual(config.monitoring_window, 60.0)
        self.assertEqual(config.minimum_throughput, 10)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=15.0,
            expected_exceptions=(ValueError, ConnectionError),
        )

        self.assertEqual(config.failure_threshold, 3)
        self.assertEqual(config.recovery_timeout, 15.0)
        self.assertEqual(config.expected_exceptions, (ValueError, ConnectionError))


class TestCircuitBreakerMetrics(unittest.TestCase):
    """Test CircuitBreakerMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = CircuitBreakerMetrics()

        self.assertEqual(metrics.total_requests, 0)
        self.assertEqual(metrics.successful_requests, 0)
        self.assertEqual(metrics.failed_requests, 0)
        self.assertEqual(metrics.circuit_open_count, 0)
        self.assertIsNone(metrics.last_failure_time)
        self.assertEqual(metrics.current_state, CircuitBreakerState.CLOSED)


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


class TestMCPCircuitBreaker(AsyncTestCase):
    """Test MCPCircuitBreaker functionality."""

    def setUp(self):
        """Set up test circuit breaker."""
        super().setUp()
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            half_open_max_calls=2,
            success_threshold=2,
            expected_exceptions=(ValueError, ConnectionError),
        )
        self.circuit_breaker = MCPCircuitBreaker(self.config)

    def test_initial_state(self):
        """Test initial circuit breaker state."""
        self.assertEqual(self.circuit_breaker.get_state(), CircuitBreakerState.CLOSED)
        self.assertTrue(self.circuit_breaker.is_closed())
        self.assertFalse(self.circuit_breaker.is_open())
        self.assertFalse(self.circuit_breaker.is_half_open())

    async def test_successful_operation(self):
        """Test successful operation through circuit breaker."""

        async def successful_operation():
            return "success"

        async with self.circuit_breaker:
            result = await successful_operation()

        self.assertEqual(result, "success")
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics.successful_requests, 1)
        self.assertEqual(metrics.failed_requests, 0)
        self.assertEqual(metrics.total_requests, 1)

    async def test_failed_operation(self):
        """Test failed operation through circuit breaker."""

        async def failing_operation():
            raise ValueError("Test failure")

        with self.assertRaises(ValueError):
            async with self.circuit_breaker:
                await failing_operation()

        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics.successful_requests, 0)
        self.assertEqual(metrics.failed_requests, 1)
        self.assertEqual(metrics.total_requests, 1)

    async def test_circuit_opening(self):
        """Test circuit breaker opening after failures."""

        async def failing_operation():
            raise ValueError("Test failure")

        # Generate enough failures to open circuit
        for i in range(self.config.failure_threshold):
            with self.assertRaises(ValueError):
                async with self.circuit_breaker:
                    await failing_operation()

        # Circuit should now be open
        self.assertTrue(self.circuit_breaker.is_open())

        # Next call should fail fast
        with self.assertRaises(CircuitBreakerOpenError):
            async with self.circuit_breaker:
                await failing_operation()

    async def test_recovery_timeout(self):
        """Test circuit breaker recovery after timeout."""

        async def failing_operation():
            raise ValueError("Test failure")

        # Open the circuit
        for i in range(self.config.failure_threshold):
            with self.assertRaises(ValueError):
                async with self.circuit_breaker:
                    await failing_operation()

        self.assertTrue(self.circuit_breaker.is_open())

        # Wait for recovery timeout
        await asyncio.sleep(self.config.recovery_timeout + 0.1)

        # Circuit should transition to half-open on next call
        with self.assertRaises(ValueError):
            async with self.circuit_breaker:
                await failing_operation()

        self.assertTrue(self.circuit_breaker.is_open())  # Should re-open after failure

    async def test_half_open_recovery(self):
        """Test successful recovery through half-open state."""
        call_count = 0

        async def sometimes_failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= self.config.failure_threshold:
                raise ValueError("Initial failures")
            return "success"

        # Open the circuit with initial failures
        for i in range(self.config.failure_threshold):
            with self.assertRaises(ValueError):
                async with self.circuit_breaker:
                    await sometimes_failing_operation()

        self.assertTrue(self.circuit_breaker.is_open())

        # Wait for recovery timeout
        await asyncio.sleep(self.config.recovery_timeout + 0.1)

        # Successful calls in half-open state should close circuit
        for i in range(self.config.success_threshold):
            async with self.circuit_breaker:
                result = await sometimes_failing_operation()
                self.assertEqual(result, "success")

        # Circuit should be closed now
        self.assertTrue(self.circuit_breaker.is_closed())

    async def test_call_method(self):
        """Test the call method."""

        async def test_operation(value):
            return f"result: {value}"

        result = await self.circuit_breaker.call(test_operation, "test")
        self.assertEqual(result, "result: test")

    async def test_protect_context_manager(self):
        """Test the protect context manager."""

        async def test_operation():
            return "protected result"

        async with self.circuit_breaker.protect(test_operation) as result:
            self.assertEqual(result, "protected result")

    def test_reset(self):
        """Test circuit breaker reset."""
        # Manually set some state
        self.circuit_breaker.state = CircuitBreakerState.OPEN
        self.circuit_breaker.metrics.failed_requests = 5
        self.circuit_breaker.failure_times = [time.time()]

        # Reset
        self.circuit_breaker.reset()

        # Check reset state
        self.assertTrue(self.circuit_breaker.is_closed())
        self.assertEqual(self.circuit_breaker.metrics.failed_requests, 0)
        self.assertEqual(len(self.circuit_breaker.failure_times), 0)

    def run_async_test(self, test_method):
        """Helper to run async test methods."""
        return self.run_async(test_method())

    def test_successful_operation_sync(self):
        """Test successful operation (sync wrapper)."""
        self.run_async_test(self.test_successful_operation)

    def test_failed_operation_sync(self):
        """Test failed operation (sync wrapper)."""
        self.run_async_test(self.test_failed_operation)

    def test_circuit_opening_sync(self):
        """Test circuit opening (sync wrapper)."""
        self.run_async_test(self.test_circuit_opening)

    def test_recovery_timeout_sync(self):
        """Test recovery timeout (sync wrapper)."""
        self.run_async_test(self.test_recovery_timeout)

    def test_half_open_recovery_sync(self):
        """Test half-open recovery (sync wrapper)."""
        self.run_async_test(self.test_half_open_recovery)

    def test_call_method_sync(self):
        """Test call method (sync wrapper)."""
        self.run_async_test(self.test_call_method)

    def test_protect_context_manager_sync(self):
        """Test protect context manager (sync wrapper)."""
        self.run_async_test(self.test_protect_context_manager)


class TestCircuitBreakerConfig(unittest.TestCase):
    """Test circuit breaker configuration creation."""

    def test_create_from_config_enabled(self):
        """Test creating circuit breaker from enabled config."""
        config = {
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 3,
                "recovery_timeout": 15.0,
            }
        }

        cb = create_circuit_breaker_from_config(config)
        self.assertIsNotNone(cb)
        self.assertEqual(cb.config.failure_threshold, 3)
        self.assertEqual(cb.config.recovery_timeout, 15.0)

    def test_create_from_config_disabled(self):
        """Test creating circuit breaker from disabled config."""
        config = {"circuit_breaker": {"enabled": False}}

        cb = create_circuit_breaker_from_config(config)
        self.assertIsNone(cb)

    def test_create_from_config_missing(self):
        """Test creating circuit breaker from config without circuit_breaker section."""
        config = {}

        cb = create_circuit_breaker_from_config(config)
        self.assertIsNone(cb)


class TestCircuitBreakerTransportWrapper(AsyncTestCase):
    """Test CircuitBreakerTransportWrapper."""

    def setUp(self):
        """Set up test wrapper."""
        super().setUp()
        self.mock_transport = AsyncMock()
        self.circuit_breaker = MCPCircuitBreaker(
            CircuitBreakerConfig(failure_threshold=2)
        )
        self.wrapper = CircuitBreakerTransportWrapper(
            self.mock_transport, self.circuit_breaker
        )

    async def test_successful_request(self):
        """Test successful request through wrapper."""
        self.mock_transport.send_request.return_value = {"result": "success"}

        result = await self.wrapper.send_request("test_method", {"param": "value"})

        self.assertEqual(result, {"result": "success"})
        self.mock_transport.send_request.assert_called_once_with(
            "test_method", {"param": "value"}
        )

    async def test_failed_request(self):
        """Test failed request through wrapper."""
        self.mock_transport.send_request.side_effect = ValueError("Connection failed")

        with self.assertRaises(ValueError):
            await self.wrapper.send_request("test_method", {"param": "value"})

        # Circuit should record the failure
        metrics = self.wrapper.get_circuit_breaker_metrics()
        self.assertEqual(metrics.failed_requests, 1)

    async def test_circuit_protection(self):
        """Test circuit breaker protection in wrapper."""
        # Cause failures to open circuit
        self.mock_transport.send_request.side_effect = ValueError("Service down")

        # Generate failures to open circuit
        for i in range(2):
            with self.assertRaises(ValueError):
                await self.wrapper.send_request("test", {})

        # Next call should be blocked by circuit breaker
        with self.assertRaises(CircuitBreakerOpenError):
            await self.wrapper.send_request("test", {})

        # Transport should not be called for the blocked request
        self.assertEqual(self.mock_transport.send_request.call_count, 2)

    def test_attribute_delegation(self):
        """Test attribute delegation to underlying transport."""
        self.mock_transport.some_attribute = "test_value"

        self.assertEqual(self.wrapper.some_attribute, "test_value")

    def test_circuit_status_methods(self):
        """Test circuit breaker status methods."""
        self.assertTrue(self.wrapper.is_circuit_healthy())

        state = self.wrapper.get_circuit_breaker_state()
        self.assertEqual(state, CircuitBreakerState.CLOSED)

        metrics = self.wrapper.get_circuit_breaker_metrics()
        self.assertIsInstance(metrics, CircuitBreakerMetrics)

    def run_async_test(self, test_method):
        """Helper to run async test methods."""
        return self.run_async(test_method())

    def test_successful_request_sync(self):
        """Test successful request (sync wrapper)."""
        self.run_async_test(self.test_successful_request)

    def test_failed_request_sync(self):
        """Test failed request (sync wrapper)."""
        self.run_async_test(self.test_failed_request)

    def test_circuit_protection_sync(self):
        """Test circuit protection (sync wrapper)."""
        self.run_async_test(self.test_circuit_protection)


if __name__ == "__main__":
    unittest.main()
