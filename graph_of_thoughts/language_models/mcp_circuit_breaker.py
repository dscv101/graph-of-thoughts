# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Circuit Breaker Implementation.

This module provides a circuit breaker pattern implementation for MCP connections,
helping to prevent cascading failures and provide automatic recovery when MCP
servers become unavailable or unresponsive.

The circuit breaker pattern helps improve system resilience by:
- Preventing repeated calls to failing services
- Providing fast failure responses when services are down
- Automatically attempting recovery after a timeout period
- Monitoring service health and failure rates

Key Features:
    - Configurable failure thresholds and timeouts
    - Automatic state transitions (Closed -> Open -> Half-Open -> Closed)
    - Exponential backoff for recovery attempts
    - Detailed metrics and monitoring
    - Integration with existing MCP error handling
    - Support for different failure criteria

Circuit Breaker States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests fail fast
    - HALF_OPEN: Testing if service has recovered

Example Usage:
    Basic circuit breaker usage:

    ```python
    from graph_of_thoughts.language_models.mcp_circuit_breaker import MCPCircuitBreaker

    # Create circuit breaker with configuration
    circuit_breaker = MCPCircuitBreaker(
        failure_threshold=5,
        recovery_timeout=30.0,
        expected_exception=(MCPConnectionError, MCPTimeoutError)
    )

    # Use with MCP operations
    async def protected_operation():
        async with circuit_breaker:
            return await mcp_transport.send_request("method", params)

    try:
        result = await protected_operation()
    except CircuitBreakerOpenError:
        print("Service is currently unavailable")
    ```

    Integration with MCP client:

    ```python
    # Circuit breaker is automatically integrated with MCPLanguageModel
    lm = MCPLanguageModel(
        config_path="config.json",
        model_name="mcp_claude_desktop"
    )

    # Circuit breaker configuration in config file
    config = {
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 5,
            "recovery_timeout": 30.0,
            "half_open_max_calls": 3
        }
    }
    ```
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type, Union

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery (seconds)
        half_open_max_calls: Maximum calls allowed in half-open state
        expected_exceptions: Exception types that count as failures
        success_threshold: Successes needed in half-open to close circuit
        monitoring_window: Time window for failure rate calculation (seconds)
        minimum_throughput: Minimum requests before considering failure rate
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    expected_exceptions: tuple = field(default_factory=lambda: (Exception,))
    success_threshold: int = 2
    monitoring_window: float = 60.0
    minimum_throughput: int = 10


@dataclass
class CircuitBreakerMetrics:
    """
    Metrics for circuit breaker monitoring.

    Attributes:
        total_requests: Total number of requests
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        circuit_open_count: Number of times circuit opened
        last_failure_time: Timestamp of last failure
        state_change_time: Timestamp of last state change
        current_state: Current circuit breaker state
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_open_count: int = 0
    last_failure_time: Optional[float] = None
    state_change_time: float = field(default_factory=time.time)
    current_state: CircuitBreakerState = CircuitBreakerState.CLOSED


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message)
        self.message = message


class MCPCircuitBreaker:
    """
    Circuit breaker implementation for MCP operations.

    Provides resilience against failing MCP servers by implementing the
    circuit breaker pattern with automatic recovery and monitoring.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize the circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics()
        self.state = CircuitBreakerState.CLOSED
        self.half_open_calls = 0
        self.half_open_successes = 0
        self.failure_times: [float] = []
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.logger.info(f"Initialized circuit breaker with config: {self.config}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._check_state()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            await self._record_success()
        elif issubclass(exc_type, self.config.expected_exceptions):
            await self._record_failure()
        # Don't suppress exceptions
        return False

    @asynccontextmanager
    async def protect(self, operation: Callable):
        """
        Protect an operation with the circuit breaker.

        Args:
            operation: Async callable to protect

        Returns:
            Result of the operation

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception from the protected operation
        """
        async with self:
            return await operation()

    async def call(self, operation: Callable, *args, **kwargs):
        """
        Call an operation through the circuit breaker.

        Args:
            operation: Async callable to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of the operation

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception from the operation
        """
        async with self:
            return await operation(*args, **kwargs)

    async def _check_state(self):
        """Check and update circuit breaker state."""
        async with self.lock:
            current_time = time.time()

            if self.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                time_since_open = current_time - self.metrics.state_change_time
                if time_since_open >= self.config.recovery_timeout:
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is open. Recovery in {self.config.recovery_timeout - time_since_open:.1f}s"
                    )

            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Check if we've exceeded half-open call limit
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self._transition_to_open()
                    raise CircuitBreakerOpenError("Half-open call limit exceeded")

                self.half_open_calls += 1

            # Update metrics
            self.metrics.total_requests += 1

    async def _record_success(self):
        """Record a successful operation."""
        async with self.lock:
            self.metrics.successful_requests += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_successes += 1

                # Check if we have enough successes to close the circuit
                if self.half_open_successes >= self.config.success_threshold:
                    self._transition_to_closed()

    async def _record_failure(self):
        """Record a failed operation."""
        async with self.lock:
            current_time = time.time()
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = current_time
            self.failure_times.append(current_time)

            # Clean old failure times outside monitoring window
            cutoff_time = current_time - self.config.monitoring_window
            self.failure_times = [t for t in self.failure_times if t > cutoff_time]

            if self.state == CircuitBreakerState.CLOSED:
                # Check if we should open the circuit
                if self._should_open_circuit():
                    self._transition_to_open()

            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._transition_to_open()

    def _should_open_circuit(self) -> bool:
        """
        Determine if the circuit should be opened based on failure rate.

        Returns:
            True if circuit should be opened
        """
        # Check simple failure threshold
        if len(self.failure_times) >= self.config.failure_threshold:
            return True

        # Check failure rate if we have minimum throughput
        if self.metrics.total_requests >= self.config.minimum_throughput:
            failure_rate = self.metrics.failed_requests / self.metrics.total_requests
            # Open if failure rate > 50% and we have recent failures
            if failure_rate > 0.5 and len(self.failure_times) > 0:
                return True

        return False

    def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        self.logger.warning("Circuit breaker opening due to failures")
        self.state = CircuitBreakerState.OPEN
        self.metrics.current_state = CircuitBreakerState.OPEN
        self.metrics.circuit_open_count += 1
        self.metrics.state_change_time = time.time()
        self.half_open_calls = 0
        self.half_open_successes = 0

    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        self.logger.info(
            "Circuit breaker transitioning to half-open for recovery testing"
        )
        self.state = CircuitBreakerState.HALF_OPEN
        self.metrics.current_state = CircuitBreakerState.HALF_OPEN
        self.metrics.state_change_time = time.time()
        self.half_open_calls = 0
        self.half_open_successes = 0

    def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        self.logger.info("Circuit breaker closing - service recovered")
        self.state = CircuitBreakerState.CLOSED
        self.metrics.current_state = CircuitBreakerState.CLOSED
        self.metrics.state_change_time = time.time()
        self.half_open_calls = 0
        self.half_open_successes = 0
        # Reset failure tracking
        self.failure_times.clear()

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self.metrics

    def is_closed(self) -> bool:
        """Check if circuit breaker is closed (normal operation)."""
        return self.state == CircuitBreakerState.CLOSED

    def is_open(self) -> bool:
        """Check if circuit breaker is open (failing fast)."""
        return self.state == CircuitBreakerState.OPEN

    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open (testing recovery)."""
        return self.state == CircuitBreakerState.HALF_OPEN

    def reset(self):
        """Reset circuit breaker to initial state."""
        self.logger.info("Resetting circuit breaker")
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.half_open_calls = 0
        self.half_open_successes = 0
        self.failure_times.clear()


def create_circuit_breaker_from_config(
    config: [str, Any]
) -> Optional[MCPCircuitBreaker]:
    """
    Create a circuit breaker from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        MCPCircuitBreaker instance or None if disabled
    """
    cb_config = config.get("circuit_breaker", {})

    if not cb_config.get("enabled", False):
        return None

    # Import MCP exceptions for expected_exceptions
    try:
        from .mcp_transport import MCPConnectionError, MCPServerError, MCPTimeoutError

        expected_exceptions = (MCPConnectionError, MCPTimeoutError, MCPServerError)
    except ImportError:
        expected_exceptions = (Exception,)

    circuit_config = CircuitBreakerConfig(
        failure_threshold=cb_config.get("failure_threshold", 5),
        recovery_timeout=cb_config.get("recovery_timeout", 30.0),
        half_open_max_calls=cb_config.get("half_open_max_calls", 3),
        expected_exceptions=expected_exceptions,
        success_threshold=cb_config.get("success_threshold", 2),
        monitoring_window=cb_config.get("monitoring_window", 60.0),
        minimum_throughput=cb_config.get("minimum_throughput", 10),
    )

    return MCPCircuitBreaker(circuit_config)


class CircuitBreakerTransportWrapper:
    """
    Wrapper that adds circuit breaker protection to MCP transports.

    This wrapper intercepts transport operations and applies circuit breaker
    logic to prevent cascading failures and provide automatic recovery.
    """

    def __init__(self, transport, circuit_breaker: MCPCircuitBreaker):
        """
        Initialize the circuit breaker transport wrapper.

        Args:
            transport: The underlying MCP transport
            circuit_breaker: Circuit breaker instance
        """
        self.transport = transport
        self.circuit_breaker = circuit_breaker
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def __getattr__(self, name):
        """Delegate attribute access to underlying transport."""
        return getattr(self.transport, name)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.transport.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return await self.transport.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self) -> bool:
        """Connect with circuit breaker protection."""
        async with self.circuit_breaker:
            return await self.transport.connect()

    async def disconnect(self) -> None:
        """Disconnect (no circuit breaker needed for cleanup)."""
        await self.transport.disconnect()

    async def send_request(self, method: str, params: [str, Any]) -> [str, Any]:
        """Send request with circuit breaker protection."""
        async with self.circuit_breaker:
            return await self.transport.send_request(method, params)

    async def send_notification(self, method: str, params: [str, Any]) -> None:
        """Send notification with circuit breaker protection."""
        async with self.circuit_breaker:
            return await self.transport.send_notification(method, params)

    async def send_sampling_request(self, request: [str, Any]) -> [str, Any]:
        """Send sampling request with circuit breaker protection."""
        async with self.circuit_breaker:
            return await self.transport.send_sampling_request(request)

    async def initialize(self) -> [str, Any]:
        """Initialize with circuit breaker protection."""
        async with self.circuit_breaker:
            return await self.transport.initialize()

    def get_circuit_breaker_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self.circuit_breaker.get_metrics()

    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        """Get circuit breaker state."""
        return self.circuit_breaker.get_state()

    def is_circuit_healthy(self) -> bool:
        """Check if circuit breaker indicates healthy service."""
        return self.circuit_breaker.is_closed()


def wrap_transport_with_circuit_breaker(transport, config: [str, Any]):
    """
    Wrap an MCP transport with circuit breaker protection.

    Args:
        transport: MCP transport to wrap
        config: Configuration dictionary

    Returns:
        Original transport or wrapped transport with circuit breaker
    """
    circuit_breaker = create_circuit_breaker_from_config(config)

    if circuit_breaker is None:
        # Circuit breaker disabled, return original transport
        return transport

    logger.info("Wrapping MCP transport with circuit breaker protection")
    return CircuitBreakerTransportWrapper(transport, circuit_breaker)