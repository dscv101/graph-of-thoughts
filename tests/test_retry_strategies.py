"""
Tests for advanced retry and backoff strategies.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graph_of_thoughts.language_models.mcp_client import (
    BackoffJitterType,
    MCPConnectionError,
    MCPServerError,
    MCPTimeoutError,
    MCPValidationError,
    RetryConfig,
    RetryManager,
    RetryStrategy,
)


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.jitter_type == BackoffJitterType.EQUAL
        assert config.timeout_multiplier == 1.0
        assert config.circuit_breaker_integration is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            strategy=RetryStrategy.ADAPTIVE,
            jitter_type=BackoffJitterType.DECORRELATED,
            connection_error_max_attempts=7,
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.strategy == RetryStrategy.ADAPTIVE
        assert config.jitter_type == BackoffJitterType.DECORRELATED
        assert config.connection_error_max_attempts == 7


class TestRetryManager:
    """Test RetryManager functionality."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_multiplier=2.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter_type=BackoffJitterType.NONE,
        )
        manager = RetryManager(config)

        # Test exponential progression
        assert manager.calculate_delay(0) == 1.0
        assert manager.calculate_delay(1) == 2.0
        assert manager.calculate_delay(2) == 4.0
        assert manager.calculate_delay(3) == 8.0

    def test_linear_backoff(self):
        """Test linear backoff calculation."""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.LINEAR,
            jitter_type=BackoffJitterType.NONE,
        )
        manager = RetryManager(config)

        # Test linear progression
        assert manager.calculate_delay(0) == 1.0
        assert manager.calculate_delay(1) == 2.0
        assert manager.calculate_delay(2) == 3.0
        assert manager.calculate_delay(3) == 4.0

    def test_fixed_backoff(self):
        """Test fixed backoff calculation."""
        config = RetryConfig(
            base_delay=2.5,
            strategy=RetryStrategy.FIXED,
            jitter_type=BackoffJitterType.NONE,
        )
        manager = RetryManager(config)

        # Test fixed delay
        assert manager.calculate_delay(0) == 2.5
        assert manager.calculate_delay(1) == 2.5
        assert manager.calculate_delay(2) == 2.5
        assert manager.calculate_delay(3) == 2.5

    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            backoff_multiplier=2.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter_type=BackoffJitterType.NONE,
        )
        manager = RetryManager(config)

        # Should be capped at max_delay
        assert manager.calculate_delay(10) == 5.0

    def test_minimum_delay(self):
        """Test minimum delay enforcement."""
        config = RetryConfig(
            base_delay=0.01,
            strategy=RetryStrategy.FIXED,
            jitter_type=BackoffJitterType.NONE,
        )
        manager = RetryManager(config)

        # Should enforce minimum 100ms delay
        assert manager.calculate_delay(0) == 0.1

    def test_equal_jitter(self):
        """Test equal jitter application."""
        config = RetryConfig(
            base_delay=10.0,
            strategy=RetryStrategy.FIXED,
            jitter_type=BackoffJitterType.EQUAL,
        )
        manager = RetryManager(config)

        # Test multiple calculations to ensure jitter varies
        delays = [manager.calculate_delay(0) for _ in range(10)]

        # All delays should be close to base_delay but not identical
        assert all(9.0 <= delay <= 11.0 for delay in delays)
        assert len(set(delays)) > 1  # Should have variation

    def test_full_jitter(self):
        """Test full jitter application."""
        config = RetryConfig(
            base_delay=10.0,
            strategy=RetryStrategy.FIXED,
            jitter_type=BackoffJitterType.FULL,
        )
        manager = RetryManager(config)

        # Test multiple calculations
        delays = [manager.calculate_delay(0) for _ in range(10)]

        # All delays should be between 0 and base_delay
        assert all(0.1 <= delay <= 10.0 for delay in delays)  # 0.1 is minimum
        assert len(set(delays)) > 1  # Should have variation

    def test_should_retry_validation_error(self):
        """Test that validation errors are not retried."""
        config = RetryConfig()
        manager = RetryManager(config)

        error = MCPValidationError("Invalid request")
        assert not manager.should_retry(error, 0)

    def test_should_retry_connection_error(self):
        """Test that connection errors are retried."""
        config = RetryConfig(max_attempts=3)
        manager = RetryManager(config)

        error = MCPConnectionError("Connection failed")
        assert manager.should_retry(error, 0)
        assert manager.should_retry(error, 1)
        assert not manager.should_retry(error, 3)

    def test_error_specific_max_attempts(self):
        """Test error-specific max attempts."""
        config = RetryConfig(
            max_attempts=3,
            connection_error_max_attempts=5,
            timeout_error_max_attempts=1,
        )
        manager = RetryManager(config)

        # Connection errors get more attempts
        conn_error = MCPConnectionError("Connection failed")
        assert manager.get_max_attempts_for_error(type(conn_error)) == 5

        # Timeout errors get fewer attempts
        timeout_error = MCPTimeoutError("Timeout")
        assert manager.get_max_attempts_for_error(type(timeout_error)) == 1

        # Other errors use default
        server_error = MCPServerError("Server error")
        assert manager.get_max_attempts_for_error(type(server_error)) == 3

    def test_adaptive_strategy_success_reduction(self):
        """Test adaptive strategy reduces delays after successes."""
        config = RetryConfig(
            strategy=RetryStrategy.ADAPTIVE,
            success_threshold_for_reduction=3,
            jitter_type=BackoffJitterType.NONE,
        )
        manager = RetryManager(config)

        # Record successes to trigger reduction
        for _ in range(3):
            manager.record_success()

        # Delay should be reduced (adaptive multiplier should be < 1.0)
        delay = manager.calculate_delay(1)
        expected_base = config.base_delay * (config.backoff_multiplier**1)
        # The adaptive multiplier should reduce the delay
        assert delay < expected_base
        assert manager.adaptive_delay_multiplier < 1.0

    def test_adaptive_strategy_failure_increase(self):
        """Test adaptive strategy increases delays after failures."""
        config = RetryConfig(
            strategy=RetryStrategy.ADAPTIVE,
            failure_threshold_for_increase=2,
            jitter_type=BackoffJitterType.NONE,
        )
        manager = RetryManager(config)

        # Record failures to trigger increase
        for _ in range(2):
            manager.record_failure()

        # Delay should be increased (adaptive multiplier should be > 1.0)
        delay = manager.calculate_delay(1)
        expected_base = config.base_delay * (config.backoff_multiplier**1)
        # The adaptive multiplier should increase the delay
        assert delay > expected_base
        assert manager.adaptive_delay_multiplier > 1.0

    def test_server_error_4xx_no_retry(self):
        """Test that 4xx server errors are not retried."""
        config = RetryConfig()
        manager = RetryManager(config)

        # 4xx errors should not be retried
        error_400 = MCPServerError("Bad request", "400")
        assert not manager.should_retry(error_400, 0)

        error_404 = MCPServerError("Not found", "404")
        assert not manager.should_retry(error_404, 0)

        # 5xx errors should be retried
        error_500 = MCPServerError("Internal error", "500")
        assert manager.should_retry(error_500, 0)

        # Non-numeric error codes should be retried
        error_unknown = MCPServerError("Unknown error", "UNKNOWN")
        assert manager.should_retry(error_unknown, 0)


@pytest.mark.asyncio
class TestRetryIntegration:
    """Test retry integration with MCP client."""

    async def test_retry_on_connection_error(self):
        """Test retry behavior on connection errors."""
        import json
        import tempfile

        from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel

        # Mock transport that fails then succeeds
        mock_transport = AsyncMock()
        mock_transport.send_sampling_request.side_effect = [
            MCPConnectionError("Connection failed"),
            MCPConnectionError("Connection failed"),
            {"content": {"type": "text", "text": "Success"}},
        ]

        # Create temporary config file
        config_data = {
            "test": {
                "transport": {"type": "stdio", "command": "test-command"},
                "client_info": {"name": "test", "version": "1.0.0"},
                "capabilities": {"sampling": {}},
                "retry_config": {
                    "max_attempts": 3,
                    "base_delay": 0.1,  # Fast for testing
                    "strategy": "exponential",
                    "jitter_type": "none",
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            # Mock other dependencies
            with patch(
                "graph_of_thoughts.language_models.mcp_client.create_transport",
                return_value=mock_transport,
            ):
                with patch(
                    "graph_of_thoughts.language_models.mcp_client.MCPProtocolValidator"
                ):
                    lm = MCPLanguageModel(config_path, "test")
                    lm.transport = mock_transport
                    lm._connected = True

                    # Should succeed after retries
                    response = await lm._send_sampling_request({"test": "request"})
                    assert response["content"]["text"] == "Success"
                    assert mock_transport.send_sampling_request.call_count == 3
        finally:
            import os

            os.unlink(config_path)

    async def test_no_retry_on_validation_error(self):
        """Test that validation errors are not retried."""
        import json
        import tempfile

        from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel

        # Mock transport that always fails with validation error
        mock_transport = AsyncMock()
        mock_transport.send_sampling_request.side_effect = MCPValidationError(
            "Invalid request"
        )

        # Create temporary config file
        config_data = {
            "test": {
                "transport": {"type": "stdio", "command": "test-command"},
                "client_info": {"name": "test", "version": "1.0.0"},
                "capabilities": {"sampling": {}},
                "retry_config": {"max_attempts": 3, "base_delay": 0.1},
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch(
                "graph_of_thoughts.language_models.mcp_client.create_transport",
                return_value=mock_transport,
            ):
                with patch(
                    "graph_of_thoughts.language_models.mcp_client.MCPProtocolValidator"
                ):
                    lm = MCPLanguageModel(config_path, "test")
                    lm.transport = mock_transport
                    lm._connected = True

                    # Should fail immediately without retries
                    with pytest.raises(Exception):
                        await lm._send_sampling_request({"test": "request"})

                    # Should only be called once (no retries)
                    assert mock_transport.send_sampling_request.call_count == 1
        finally:
            import os

            os.unlink(config_path)
