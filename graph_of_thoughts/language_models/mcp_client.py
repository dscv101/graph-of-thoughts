# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Language Model Client Implementation.

This module provides a comprehensive implementation of a language model client that communicates
with MCP (Model Context Protocol) hosts such as Claude Desktop, VSCode, Cursor, and remote MCP servers.
The implementation follows the official MCP specification for protocol compliance and proper message formatting.

Key Features:
    - Full MCP protocol compliance with JSON-RPC 2.0 messaging
    - Support for both stdio and HTTP transports
    - Automatic configuration migration from legacy formats
    - Robust error handling with exponential backoff retry logic
    - Async context manager support for proper resource management
    - Token usage tracking and cost estimation
    - Response caching capabilities

Supported MCP Hosts:
    - Claude Desktop (stdio transport)
    - VSCode with MCP extension (stdio transport)
    - Cursor with MCP integration (stdio transport)
    - Remote MCP servers (HTTP transport)

Example Usage:
    Basic usage with Claude Desktop:

    ```python
    from graph_of_thoughts.language_models import MCPLanguageModel

    # Initialize with Claude Desktop configuration
    lm = MCPLanguageModel(
        config_path="path/to/mcp_config.json",
        model_name="mcp_claude_desktop",
        cache=True
    )

    # Query the model
    response = lm.query("What is the capital of France?")
    texts = lm.get_response_texts(response)
    print(texts[0])  # "The capital of France is Paris."
    ```

    Async usage with proper resource management:

    ```python
    async def example_async_usage():
        async with MCPLanguageModel("config.json", "mcp_claude_desktop") as lm:
            response = await lm._query_async("Explain quantum computing")
            return lm.get_response_texts(response)

    import asyncio
    result = asyncio.run(example_async_usage())
    ```

    Multiple responses:

    ```python
    lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
    responses = lm.query("Generate 3 creative story ideas", num_responses=3)
    for i, text in enumerate(lm.get_response_texts(responses)):
        print(f"Idea {i+1}: {text}")
    ```

Configuration Format:
    The configuration file should contain MCP host configurations:

    ```json
    {
        "mcp_claude_desktop": {
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
                "retry_attempts": 3
            },
            "cost_tracking": {
                "prompt_token_cost": 0.003,
                "response_token_cost": 0.015
            }
        }
    }
    ```

Error Handling:
    The client provides comprehensive error handling for various failure scenarios:

    - MCPConnectionError: Connection establishment failures
    - MCPTimeoutError: Request timeout scenarios
    - MCPProtocolError: Protocol-level errors
    - MCPValidationError: Invalid request/response formats
    - MCPServerError: Server-side errors with error codes

    All errors include detailed error messages and original exception context for debugging.

Thread Safety:
    The MCPLanguageModel is designed to be thread-safe for the synchronous query() method,
    which handles event loop management automatically. For async usage, use the async methods
    directly within an async context.

Performance Considerations:
    - Connection pooling: Connections are reused across requests
    - Response caching: Optional caching to avoid redundant requests
    - Retry logic: Exponential backoff for transient failures
    - Token estimation: Efficient token counting for cost tracking
"""

# Standard library imports
import asyncio
import concurrent.futures
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

# Third-party imports
import backoff

# Local imports
from .abstract_language_model import AbstractLanguageModel
from .mcp_metrics import (
    MCPMetricsCollector,
    create_metrics_collector_from_config,
    integrate_metrics_with_circuit_breaker,
    set_global_metrics_collector,
    setup_default_export_callbacks,
)
from .mcp_protocol import MCPProtocolValidator, create_sampling_request
from .mcp_transport import (
    MCPConnectionError,
    MCPProtocolError,
    MCPServerError,
    MCPTimeoutError,
    MCPTransport,
    MCPTransportError,
    MCPValidationError,
    create_transport,
)
from .token_estimation import TokenEstimationConfig, TokenEstimator


class RetryStrategy(Enum):
    """Enumeration of available retry strategies."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class BackoffJitterType(Enum):
    """Types of jitter to apply to backoff delays."""

    NONE = "none"
    FULL = "full"
    EQUAL = "equal"
    DECORRELATED = "decorrelated"


@dataclass
class RetryConfig:
    """
    Configuration for retry and backoff strategies.

    Attributes:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds for backoff calculation
        max_delay: Maximum delay in seconds between retries
        backoff_multiplier: Multiplier for exponential backoff
        strategy: Retry strategy to use
        jitter_type: Type of jitter to apply
        timeout_multiplier: Multiplier for timeout on retries
        circuit_breaker_integration: Whether to integrate with circuit breaker
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter_type: BackoffJitterType = BackoffJitterType.EQUAL
    timeout_multiplier: float = 1.0
    circuit_breaker_integration: bool = True

    # Error-specific retry configurations
    connection_error_max_attempts: int | None = None
    timeout_error_max_attempts: int | None = None
    server_error_max_attempts: int | None = None

    # Adaptive strategy parameters
    success_threshold_for_reduction: int = 5
    failure_threshold_for_increase: int = 3


class RetryManager:
    """
    Advanced retry manager with configurable strategies and jitter support.
    """

    def __init__(self, config: RetryConfig):
        self.config = config
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.adaptive_delay_multiplier = 1.0

    def calculate_delay(self, attempt: int, error_type: type | None = None) -> float:
        """
        Calculate the delay for a retry attempt with jitter and strategy.

        :param attempt: Current attempt number (0-based)
        :param error_type: Type of error that occurred
        :return: Delay in seconds
        """
        # Get base delay based on strategy
        if self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier**attempt)
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        elif self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.ADAPTIVE:
            delay = self._calculate_adaptive_delay(attempt)
        else:
            delay = self.config.base_delay * (self.config.backoff_multiplier**attempt)

        # Apply adaptive multiplier for adaptive strategy
        if self.config.strategy == RetryStrategy.ADAPTIVE:
            delay *= self.adaptive_delay_multiplier

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Apply jitter
        delay = self._apply_jitter(delay, attempt)

        return max(0.1, delay)  # Minimum 100ms delay

    def _calculate_adaptive_delay(self, attempt: int) -> float:
        """Calculate delay for adaptive strategy."""
        base = self.config.base_delay * (self.config.backoff_multiplier**attempt)

        # Adjust based on recent success/failure patterns
        if self.consecutive_successes >= self.config.success_threshold_for_reduction:
            # Recent successes, reduce delay
            return base * 0.5
        elif self.consecutive_failures >= self.config.failure_threshold_for_increase:
            # Recent failures, increase delay
            return base * 2.0

        return base

    def _apply_jitter(self, delay: float, attempt: int) -> float:
        """Apply jitter to the delay."""
        if self.config.jitter_type == BackoffJitterType.NONE:
            return delay
        elif self.config.jitter_type == BackoffJitterType.FULL:
            return random.uniform(0, delay)
        elif self.config.jitter_type == BackoffJitterType.EQUAL:
            jitter = delay * 0.1 * random.uniform(-1, 1)
            return delay + jitter
        elif self.config.jitter_type == BackoffJitterType.DECORRELATED:
            # Decorrelated jitter as described in AWS blog
            if attempt == 0:
                return delay
            prev_delay = self.calculate_delay(attempt - 1)
            return random.uniform(self.config.base_delay, prev_delay * 3)

        return delay

    def get_max_attempts_for_error(self, error_type: type) -> int:
        """Get maximum attempts for specific error type."""
        if (
            issubclass(error_type, MCPConnectionError)
            and self.config.connection_error_max_attempts
        ):
            return self.config.connection_error_max_attempts
        elif (
            issubclass(error_type, MCPTimeoutError)
            and self.config.timeout_error_max_attempts
        ):
            return self.config.timeout_error_max_attempts
        elif (
            issubclass(error_type, MCPServerError)
            and self.config.server_error_max_attempts
        ):
            return self.config.server_error_max_attempts

        return self.config.max_attempts

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if an error should be retried.

        :param error: The exception that occurred
        :param attempt: Current attempt number (0-based)
        :return: True if should retry, False otherwise
        """
        # Don't retry validation errors or certain server errors
        if isinstance(error, (MCPValidationError,)):
            return False

        # Don't retry certain server errors (4xx client errors)
        if isinstance(error, MCPServerError):
            if hasattr(error, "error_code") and error.error_code:
                try:
                    code = int(error.error_code)
                    if 400 <= code < 500:  # Client errors shouldn't be retried
                        return False
                except (ValueError, TypeError):
                    pass

        # Check attempt limit for error type
        max_attempts = self.get_max_attempts_for_error(type(error))
        return attempt < max_attempts

    def record_success(self):
        """Record a successful operation for adaptive strategy."""
        self.consecutive_successes += 1
        self.consecutive_failures = 0

        # Adjust adaptive multiplier
        if (
            self.config.strategy == RetryStrategy.ADAPTIVE
            and self.consecutive_successes
            >= self.config.success_threshold_for_reduction
        ):
            self.adaptive_delay_multiplier = max(
                0.5, self.adaptive_delay_multiplier * 0.9
            )

    def record_failure(self):
        """Record a failed operation for adaptive strategy."""
        self.consecutive_failures += 1
        self.consecutive_successes = 0

        # Adjust adaptive multiplier
        if (
            self.config.strategy == RetryStrategy.ADAPTIVE
            and self.consecutive_failures >= self.config.failure_threshold_for_increase
        ):
            self.adaptive_delay_multiplier = min(
                3.0, self.adaptive_delay_multiplier * 1.1
            )


class MCPLanguageModel(AbstractLanguageModel):
    """
    The MCPLanguageModel class handles interactions with language models through the Model Context Protocol (MCP).
    This implementation follows the official MCP specification for protocol compliance and proper message formatting.

    This class provides a high-level interface for communicating with MCP hosts while handling all the
    low-level protocol details, connection management, and error recovery automatically.

    Attributes:
        config (): The MCP configuration for the selected model
        transport (MCPTransport): The transport layer (stdio or HTTP)
        validator (MCPProtocolValidator): Protocol message validator
        transport_type (str): Type of transport being used ("stdio" or "http")
        host_type (str): The MCP host command or URL
        prompt_tokens (int): Total prompt tokens used (for cost tracking)
        completion_tokens (int): Total completion tokens used (for cost tracking)
        cost (float): Estimated total cost based on token usage

    Example:
        Basic synchronous usage:

        ```python
        # Initialize the model
        lm = MCPLanguageModel(
            config_path="mcp_config.json",
            model_name="mcp_claude_desktop",
            cache=True
        )

        # Single query
        response = lm.query("What are the benefits of renewable energy?")
        text = lm.get_response_texts(response)[0]
        print(f"Response: {text}")

        # Multiple responses for creative tasks
        responses = lm.query("Write a haiku about technology", num_responses=3)
        for i, haiku in enumerate(lm.get_response_texts(responses)):
            print(f"Haiku {i+1}:\\n{haiku}\\n")

        # Check cost and usage
        print(f"Total cost: ${lm.cost:.4f}")
        print(f"Tokens used: {lm.prompt_tokens + lm.completion_tokens}")
        ```

        Async usage with context manager:

        ```python
        async def process_documents(documents):
            async with MCPLanguageModel("config.json", "mcp_claude_desktop") as lm:
                summaries = []
                for doc in documents:
                    response = await lm._query_async(f"Summarize: {doc}")
                    summary = lm.get_response_texts(response)[0]
                    summaries.append(summary)
                return summaries

        # Run the async function
        import asyncio
        docs = ["Document 1 content...", "Document 2 content..."]
        summaries = asyncio.run(process_documents(docs))
        ```

        Error handling:

        ```python
        from graph_of_thoughts.language_models.mcp_transport import (
            MCPConnectionError, MCPTimeoutError, MCPServerError
        )

        try:
            lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
            response = lm.query("Complex query that might fail")
        except MCPConnectionError as e:
            print(f"Connection failed: {e}")
        except MCPTimeoutError as e:
            print(f"Request timed out: {e}")
        except MCPServerError as e:
            print(f"Server error {e.error_code}: {e}")
        ```

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self,
        config_path: str = "",
        model_name: str = "mcp_claude_desktop",
        cache: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the MCPLanguageModel instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "". Ignored if config is provided.
        :type config_path: str
        :param model_name: Name of the model configuration, default is 'mcp_claude_desktop'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        :param config: Configuration dictionary. If provided, config_path is ignored.
        :type config: dict[str, Any] | None
        """
        # Initialize cache configuration from model config if available
        cache_config = None
        if cache:
            from .caching import CacheConfig

            cache_config = (
                CacheConfig()
            )  # Use defaults, will be overridden if config specifies

        # Handle config parameter - if provided, use it directly instead of loading from file
        if config is not None:
            # Initialize parent class manually to avoid file loading
            self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
            self.config: dict[str, Any] = config
            self.model_name: str = model_name
            self.cache: bool = cache
            self.config_path: str = config_path

            # Initialize intelligent caching system
            if self.cache:
                from .caching import get_cache_manager
                self.cache_manager = get_cache_manager(cache_config)
                # Keep legacy cache for backward compatibility
                self.response_cache: "dict[str, list[Any]]" = {}
            else:
                self.cache_manager = None

            # Initialize cost tracking attributes
            self.prompt_tokens: int = 0
            self.completion_tokens: int = 0
            self.cost: float = 0.0
        else:
            # Use traditional file-based config loading
            super().__init__(config_path, model_name, cache, cache_config)

        # Extract model-specific configuration
        if model_name not in self.config:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        self.config: dict[str, Any] = self.config[model_name]

        # Migrate old configuration format to new format if needed
        self.config = self._migrate_config_if_needed(self.config)

        # Validate configuration using enhanced validator
        from .mcp_protocol import MCPConfigurationError, MCPConfigurationValidator

        self.config_validator = MCPConfigurationValidator(
            strict_mode=True, enable_security_checks=True
        )
        try:
            # Validate runtime configuration
            self.config_validator.validate_runtime_configuration(
                self.config, model_name
            )
        except MCPConfigurationError as e:
            raise ValueError(f"Invalid MCP configuration for {model_name}: {e}")

        # Keep legacy validator for backward compatibility
        self.validator = MCPProtocolValidator()
        if not self.validator.validate_configuration(self.config):
            raise ValueError(f"Invalid MCP configuration for {model_name}")

        # Extract configuration sections
        self.transport_config: dict[str, Any] = self.config["transport"]
        self.client_info: dict[str, Any] = self.config["client_info"]
        self.capabilities: dict[str, Any] = self.config["capabilities"]
        self.default_sampling_params: dict[str, Any] = self.config.get(
            "default_sampling_params", {}
        )
        self.connection_config: dict[str, Any] = self.config.get("connection_config", {})

        # Cost tracking (application-specific, not part of MCP protocol)
        cost_tracking = self.config.get("cost_tracking", {})
        self.prompt_token_cost: float = cost_tracking.get("prompt_token_cost", 0.0)
        self.response_token_cost: float = cost_tracking.get("response_token_cost", 0.0)

        # Batch processing configuration
        batch_config = self.config.get("batch_processing", {})
        self.default_max_concurrent: int = batch_config.get("max_concurrent", 10)
        self.default_batch_size: int = batch_config.get("batch_size", 50)
        self.default_retry_attempts: int = batch_config.get("retry_attempts", 3)
        self.default_retry_delay: float = batch_config.get("retry_delay", 1.0)

        # Advanced retry configuration
        retry_config_dict = self.config.get("retry_config", {})
        self.retry_config = RetryConfig(
            max_attempts=retry_config_dict.get(
                "max_attempts", self.default_retry_attempts
            ),
            base_delay=retry_config_dict.get("base_delay", self.default_retry_delay),
            max_delay=retry_config_dict.get("max_delay", 60.0),
            backoff_multiplier=retry_config_dict.get("backoff_multiplier", 2.0),
            strategy=RetryStrategy(retry_config_dict.get("strategy", "exponential")),
            jitter_type=BackoffJitterType(
                retry_config_dict.get("jitter_type", "equal")
            ),
            timeout_multiplier=retry_config_dict.get("timeout_multiplier", 1.0),
            circuit_breaker_integration=retry_config_dict.get(
                "circuit_breaker_integration", True
            ),
            connection_error_max_attempts=retry_config_dict.get(
                "connection_error_max_attempts"
            ),
            timeout_error_max_attempts=retry_config_dict.get(
                "timeout_error_max_attempts"
            ),
            server_error_max_attempts=retry_config_dict.get(
                "server_error_max_attempts"
            ),
            success_threshold_for_reduction=retry_config_dict.get(
                "success_threshold_for_reduction", 5
            ),
            failure_threshold_for_increase=retry_config_dict.get(
                "failure_threshold_for_increase", 3
            ),
        )

        # Initialize retry manager
        self.retry_manager = RetryManager(self.retry_config)

        # Initialize transport using plugin system for enhanced functionality
        try:
            from .mcp_host_plugins import create_transport_from_plugin

            base_transport = create_transport_from_plugin(self.config)
            self.logger.info("Created transport using MCP plugin system")
        except ImportError:
            # Fallback to standard transport creation if plugin system not available
            base_transport = create_transport(self.config)
            self.logger.info("Created transport using standard method")
        except Exception as e:
            # If plugin system fails, fallback to standard method
            self.logger.warning(
                f"Plugin system failed, falling back to standard transport: {e}"
            )
            base_transport = create_transport(self.config)

        # Wrap transport with circuit breaker if enabled
        try:
            from .mcp_circuit_breaker import wrap_transport_with_circuit_breaker

            self.transport: MCPTransport = wrap_transport_with_circuit_breaker(
                base_transport, self.config
            )
            if hasattr(self.transport, "circuit_breaker"):
                self.logger.info("Enabled circuit breaker protection for MCP transport")
        except ImportError:
            self.transport: MCPTransport = base_transport
        except Exception as e:
            self.logger.warning(
                f"Failed to enable circuit breaker, using transport without protection: {e}"
            )
            self.transport: MCPTransport = base_transport

        self._connection_established = False

        # Initialize improved token estimator
        token_config = TokenEstimationConfig()
        # Allow configuration override for token estimation
        if "token_estimation" in self.config:
            token_est_config = self.config["token_estimation"]
            token_config.avg_chars_per_token = token_est_config.get(
                "avg_chars_per_token", 3.5
            )
            token_config.enable_subword_estimation = token_est_config.get(
                "enable_subword_estimation", True
            )
            token_config.code_token_multiplier = token_est_config.get(
                "code_token_multiplier", 1.3
            )

        self.token_estimator = TokenEstimator(token_config)

        # Initialize metrics collection system
        self.metrics_collector = create_metrics_collector_from_config(self.config)
        if self.metrics_collector:
            # Set up default export callbacks
            setup_default_export_callbacks(self.metrics_collector, self.config)

            # Set as global metrics collector for easy access
            set_global_metrics_collector(self.metrics_collector)

            # Integrate with circuit breaker if available
            if hasattr(self.transport, "circuit_breaker"):
                integrate_metrics_with_circuit_breaker(
                    self.transport.circuit_breaker, self.metrics_collector
                )

            self.logger.info("Initialized MCP metrics collection system")
        else:
            self.logger.debug("Metrics collection disabled")

        # Configure intelligent caching if enabled
        if self.cache and self.cache_manager:
            cache_settings = self.config.get("caching", {})
            if cache_settings:
                from .caching import CacheConfig

                # Update cache configuration from model config
                cache_config = CacheConfig(
                    max_size=cache_settings.get("max_size", 1000),
                    default_ttl=cache_settings.get("default_ttl", 3600.0),
                    response_cache_size=cache_settings.get("response_cache_size", 500),
                    config_cache_size=cache_settings.get("config_cache_size", 50),
                    metadata_cache_size=cache_settings.get("metadata_cache_size", 200),
                    response_ttl=cache_settings.get("response_ttl", 1800.0),
                    config_ttl=cache_settings.get("config_ttl", 7200.0),
                    metadata_ttl=cache_settings.get("metadata_ttl", 3600.0),
                )
                # Reinitialize cache manager with updated config
                from .caching import MultiLevelCacheManager

                self.cache_manager = MultiLevelCacheManager(cache_config)

        # Legacy compatibility properties
        self.transport_type: str = self.transport_config.get("type", "stdio")
        self.host_type: str = self.transport_config.get(
            "command", "unknown"
        )  # For backward compatibility

    def _migrate_config_if_needed(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Migrate old configuration format to new MCP protocol-compliant format.
        Provides backward compatibility for existing configuration files.

        :param config: Configuration dictionary (potentially in old format)
        :type config: dict[str, Any]
        :return: Configuration in new format
        :rtype: dict[str, Any]
        """
        # Check if this is already in new format
        if "transport" in config and "client_info" in config:
            return config  # Already in new format

        # Check if this is in old format
        if "transport_type" in config and "host_type" in config:
            self.logger.info(
                "Migrating configuration from old format to new MCP protocol format"
            )

            # Create new format configuration
            new_config = {
                "transport": {
                    "type": config.get("transport_type", "stdio"),
                    "command": config.get("host_type", "unknown"),
                    "args": config.get("args", []),
                    "env": config.get("env", {}),
                },
                "client_info": {"name": "graph-of-thoughts", "version": "0.0.3"},
                "capabilities": {"sampling": {}},
                "default_sampling_params": {},
                "connection_config": config.get("connection_config", {}),
                "cost_tracking": {},
            }

            # Migrate model_preferences to default_sampling_params
            if "model_preferences" in config:
                new_config["default_sampling_params"]["modelPreferences"] = config[
                    "model_preferences"
                ]

            # Migrate sampling_config to default_sampling_params
            if "sampling_config" in config:
                sampling_config = config["sampling_config"]
                new_config["default_sampling_params"].update(
                    {
                        "temperature": sampling_config.get("temperature"),
                        "maxTokens": sampling_config.get("max_tokens", 1000),
                        "stopSequences": sampling_config.get("stop_sequences"),
                        "includeContext": sampling_config.get(
                            "include_context", "none"
                        ),
                    }
                )

            # Migrate cost tracking
            if "prompt_token_cost" in config or "response_token_cost" in config:
                new_config["cost_tracking"] = {
                    "prompt_token_cost": config.get("prompt_token_cost", 0.0),
                    "response_token_cost": config.get("response_token_cost", 0.0),
                }

            return new_config

        # If neither old nor new format, return as-is and let validation catch issues
        return config

    async def _ensure_connection(self) -> None:
        """
        Ensure that the MCP connection is established and initialized.
        """
        if not self._connection_established:
            try:
                success = await self.transport.connect()
                if not success:
                    raise MCPConnectionError(f"Failed to connect to MCP server")
                self._connection_established = True
                self.logger.info(f"Connected to MCP server via {self.transport_type}")
            except Exception as e:
                raise MCPConnectionError(f"Connection failed: {e}")

    async def _disconnect(self) -> None:
        """
        Disconnect from the MCP server.
        """
        if self._connection_established:
            try:
                await self.transport.disconnect()
                self._connection_established = False
                self.logger.info("Disconnected from MCP server")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")

    def _create_sampling_request(
        self, query: str, num_responses: int = 1, **kwargs
    ) -> dict[str, Any]:
        """
        Create a properly formatted MCP sampling request following the specification.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :param kwargs: Additional parameters (temperature, max_tokens, etc.)
        :return: The sampling request
        :rtype: dict[str, Any]
        """
        # Create messages in MCP format
        messages = [{"role": "user", "content": {"type": "text", "text": query}}]

        # Merge default params with kwargs, giving priority to kwargs
        params = {
            "model_preferences": self.default_sampling_params.get("modelPreferences"),
            "system_prompt": self.default_sampling_params.get("systemPrompt"),
            "include_context": self.default_sampling_params.get("includeContext", "none"),
            "temperature": self.default_sampling_params.get("temperature"),
            "max_tokens": self.default_sampling_params.get("maxTokens", 1000),
            "stop_sequences": self.default_sampling_params.get("stopSequences"),
        }

        # Override with kwargs, handling parameter name mapping
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        if "maxTokens" in kwargs:
            params["max_tokens"] = kwargs["maxTokens"]
        if "stop_sequences" in kwargs:
            params["stop_sequences"] = kwargs["stop_sequences"]
        if "stopSequences" in kwargs:
            params["stop_sequences"] = kwargs["stopSequences"]
        if "system_prompt" in kwargs:
            params["system_prompt"] = kwargs["system_prompt"]
        if "systemPrompt" in kwargs:
            params["system_prompt"] = kwargs["systemPrompt"]
        if "model_preferences" in kwargs:
            params["model_preferences"] = kwargs["model_preferences"]
        if "modelPreferences" in kwargs:
            params["model_preferences"] = kwargs["modelPreferences"]
        if "include_context" in kwargs:
            params["include_context"] = kwargs["include_context"]
        if "includeContext" in kwargs:
            params["include_context"] = kwargs["includeContext"]

        # Use the protocol utility to create the request
        return create_sampling_request(
            messages=messages,
            model_preferences=params["model_preferences"],
            system_prompt=params["system_prompt"],
            include_context=params["include_context"],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            stop_sequences=params["stop_sequences"],
            metadata={
                "num_responses": num_responses,
                "source": "graph_of_thoughts",
                "client": self.client_info["name"],
            },
        )

    async def _send_sampling_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Send a sampling request to the MCP server with advanced retry logic.

        :param request: The sampling request
        :type request: dict[str, Any]
        :return: The response from the server
        :rtype: dict[str, Any]
        """
        await self._ensure_connection()

        # Validate the request before sending
        if not self.validator.validate_sampling_request(request):
            raise ValueError("Invalid sampling request format")

        # Use advanced retry logic
        last_exception = None

        for attempt in range(self.retry_config.max_attempts):
            try:
                # Track the request with metrics if available
                if self.metrics_collector:
                    with self.metrics_collector.track_request(
                        "sampling/createMessage"
                    ) as tracker:
                        response = await self.transport.send_sampling_request(request)
                        if tracker:
                            tracker.record_success(response)
                        self.logger.debug("Received MCP sampling response")

                        # Record success for adaptive strategy
                        self.retry_manager.record_success()
                        return response
                else:
                    # No metrics tracking
                    response = await self.transport.send_sampling_request(request)
                    self.logger.debug("Received MCP sampling response")

                    # Record success for adaptive strategy
                    self.retry_manager.record_success()
                    return response

            except Exception as e:
                last_exception = e

                # Track error with metrics if available
                if self.metrics_collector:
                    with self.metrics_collector.track_request(
                        "sampling/createMessage"
                    ) as tracker:
                        if tracker:
                            tracker.record_error(type(e).__name__)

                # Check if we should retry this error
                if not self.retry_manager.should_retry(e, attempt):
                    self.retry_manager.record_failure()
                    self.logger.error(f"Non-retryable error: {e}")
                    raise MCPTransportError(f"Sampling request failed: {e}")

                # Don't retry on the last attempt
                if attempt >= self.retry_config.max_attempts - 1:
                    self.retry_manager.record_failure()
                    break

                # Calculate delay with jitter and strategy
                delay = self.retry_manager.calculate_delay(attempt, type(e))

                self.logger.warning(
                    f"Sampling request attempt {attempt + 1} failed: {e}, "
                    f"retrying in {delay:.2f}s (strategy: {self.retry_config.strategy.value})"
                )

                # Record failure for adaptive strategy
                self.retry_manager.record_failure()

                # Wait before retry
                await asyncio.sleep(delay)

        # All retries exhausted
        self.logger.error(
            f"All {self.retry_config.max_attempts} attempts failed for sampling request"
        )
        raise MCPTransportError(
            f"Sampling request failed after {self.retry_config.max_attempts} attempts: {last_exception}"
        )

    def query(self, query: str, num_responses: int = 1) -> dict | list[dict]:
        """
        Query the MCP host for responses.

        This is the main method for interacting with the language model. It handles
        connection management, request formatting, and response processing automatically.
        The method is thread-safe and can be called from any context.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the MCP host. Single dict if num_responses=1, list of dicts otherwise.
        :rtype: dict | list[dict]

        Example:
            Single response:

            ```python
            lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
            response = lm.query("What is machine learning?")
            text = lm.get_response_texts(response)[0]
            print(text)
            ```

            Multiple responses for creative tasks:

            ```python
            responses = lm.query("Generate a creative story opening", num_responses=3)
            for i, story in enumerate(lm.get_response_texts(responses)):
                print(f"Story {i+1}: {story}")
            ```

            With caching enabled:

            ```python
            lm = MCPLanguageModel("config.json", "mcp_claude_desktop", cache=True)

            # First call - makes actual request
            response1 = lm.query("What is Python?")

            # Second call - returns cached result
            response2 = lm.query("What is Python?")  # Same query, cached response
            ```

        Raises:
            MCPConnectionError: If connection to MCP host fails
            MCPTimeoutError: If request times out
            MCPProtocolError: If there's a protocol-level error
            MCPValidationError: If request validation fails
            MCPServerError: If the MCP server returns an error
        """
        # Check intelligent cache first
        if self.cache and self.cache_manager:
            # Create cache key including all relevant parameters
            cache_params = {
                "num_responses": num_responses,
                "model_name": self.model_name,
                "sampling_params": self.default_sampling_params,
            }
            cached_response = self.cache_manager.get_response(query, **cache_params)
            if cached_response is not None:
                self.logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_response

        # Fallback to legacy cache for backward compatibility
        if (
            self.cache
            and hasattr(self, "response_cache")
            and query in self.response_cache
        ):
            return self.response_cache[query]

        # Use proper event loop management
        response = self._run_async_query(query, num_responses)

        # Cache the response using intelligent cache
        if self.cache and self.cache_manager:
            cache_params = {
                "num_responses": num_responses,
                "model_name": self.model_name,
                "sampling_params": self.default_sampling_params,
            }
            self.cache_manager.put_response(query, response, **cache_params)
            self.logger.debug(f"Cached response for query: {query[:50]}...")

        # Also cache in legacy cache for backward compatibility
        if self.cache and hasattr(self, "response_cache"):
            self.response_cache[query] = response

        return response

    def _run_async_query(
        self, query: str, num_responses: int = 1
    ) -> dict | list[dict]:
        """
        Helper method to run async query with optimized event loop management.

        Uses asyncio.run() as the primary method for running async code, with
        proper handling for nested event loop scenarios.

        :param query: The query to be posed to the language model
        :type query: str
        :param num_responses: Number of desired responses
        :type num_responses: int
        :return: Response(s) from the MCP host
        :rtype: dict | list[dict]
        :raises RuntimeError: If called from within an async context (use _query_async instead)
        """

        async def _run_query():
            async with self:  # Use async context manager
                return await self._query_async(query, num_responses)

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # If we reach here, we're in an async context
            raise RuntimeError(
                "Cannot call _run_async_query from within an async context. "
                "Use 'await _query_async()' instead or call from a synchronous context."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No event loop running, safe to use asyncio.run
                return asyncio.run(_run_query())
            else:
                # Re-raise the error about being in async context
                raise

    async def query_async(
        self, query: str, num_responses: int = 1, **kwargs
    ) -> dict | list[dict]:
        """
        Public async interface for querying the MCP host.

        This method provides the async functionality for querying the MCP host.
        It handles connection management, request formatting, and response processing.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :param kwargs: Additional parameters (temperature, max_tokens, etc.)
        :return: Response(s) from the MCP host. Single dict if num_responses=1, list of dicts otherwise.
        :rtype: dict | list[dict]
        """
        return await self._query_async(query, num_responses, **kwargs)

    async def _query_async(
        self, query: str, num_responses: int = 1, **kwargs
    ) -> dict | list[dict]:
        """
        Async implementation of query.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :param kwargs: Additional parameters (temperature, max_tokens, etc.)
        :return: Response(s) from the MCP host.
        :rtype: dict | list[dict]
        """
        request = self._create_sampling_request(query, num_responses, **kwargs)

        if num_responses == 1:
            response = await self._send_sampling_request(request)
            self._update_token_usage(response, prompt_text=query)
            return response
        else:
            # For multiple responses, use concurrent batch processing
            # Pass kwargs to each query in the batch
            return await self.query_batch([query] * num_responses, **kwargs)

    async def query_batch(
        self,
        queries: list[str],
        max_concurrent: int | None = None,
        batch_size: int | None = None,
        retry_attempts: int | None = None,
        retry_delay: float | None = None,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Process multiple queries concurrently with batch processing optimizations.

        This method efficiently handles large numbers of queries by:
        - Processing queries in configurable batch sizes
        - Limiting concurrent requests to prevent overwhelming the server
        - Implementing retry logic with exponential backoff
        - Providing detailed error handling and logging

        :param queries:  of query strings to process
        :type queries: [str]
        :param max_concurrent: Maximum number of concurrent requests (default: 10)
        :type max_concurrent: int
        :param batch_size: Maximum batch size to process at once (default: 50)
        :type batch_size: int
        :param retry_attempts: Number of retry attempts for failed requests (default: 3)
        :type retry_attempts: int
        :param retry_delay: Initial delay between retries in seconds (default: 1.0)
        :type retry_delay: float
        :return:  of response dictionaries in the same order as input queries
        :rtype: [[str, Any]]
        :raises ConnectionError: If not connected to MCP server
        :raises ValueError: If queries list is empty

        Example:
            Process multiple queries efficiently:

            ```python
            lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
            queries = [
                "What is machine learning?",
                "Explain quantum computing",
                "Describe blockchain technology"
            ]

            async with lm:
                responses = await lm.query_batch(queries, max_concurrent=5)
                for i, response in enumerate(responses):
                    print(f"Response {i+1}: {response}")
            ```
        """
        # Connection will be established automatically by _send_sampling_request

        if not queries:
            raise ValueError("queries list cannot be empty")

        # Use default values if not provided
        max_concurrent = max_concurrent or self.default_max_concurrent
        batch_size = batch_size or self.default_batch_size
        retry_attempts = retry_attempts or self.default_retry_attempts
        retry_delay = retry_delay or self.default_retry_delay

        self.logger.info(
            f"Processing batch of {len(queries)} queries with max_concurrent={max_concurrent}"
        )

        # Process queries in batches to manage memory and server load
        all_responses = []
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            self.logger.debug(
                f"Processing batch {i//batch_size + 1}: {len(batch_queries)} queries"
            )

            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)

            # Create tasks for concurrent processing
            tasks = [
                self._process_single_query_with_retry(
                    query, semaphore, retry_attempts, retry_delay, **kwargs
                )
                for query in batch_queries
            ]

            # Execute batch concurrently
            batch_responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions and convert to proper responses
            processed_responses = []
            for j, response in enumerate(batch_responses):
                if isinstance(response, Exception):
                    self.logger.error(f"Query {i+j} failed permanently: {response}")
                    # Create error response in expected format
                    error_response = {
                        "content": {"type": "text", "text": f"Error: {str(response)}"},
                        "metadata": {"error": True, "error_message": str(response)},
                    }
                    processed_responses.append(error_response)
                else:
                    processed_responses.append(response)

            all_responses.extend(processed_responses)

        self.logger.info(f"Completed batch processing: {len(all_responses)} responses")
        return all_responses

    async def _process_single_query_with_retry(
        self,
        query: str,
        semaphore: asyncio.Semaphore,
        retry_attempts: int,
        retry_delay: float,
        **kwargs
    ) -> [str, Any]:
        """
        Process a single query with advanced retry logic and concurrency control.

        :param query: The query string to process
        :type query: str
        :param semaphore: Semaphore for controlling concurrency
        :type semaphore: asyncio.Semaphore
        :param retry_attempts: Number of retry attempts (legacy parameter, uses retry_config instead)
        :type retry_attempts: int
        :param retry_delay: Initial delay between retries (legacy parameter, uses retry_config instead)
        :type retry_delay: float
        :return: Response dictionary
        :rtype: [str, Any]
        :raises Exception: If all retry attempts fail
        """
        async with semaphore:
            # Create a temporary retry manager for this specific query if needed
            # This allows batch processing to use different retry settings
            temp_retry_config = RetryConfig(
                max_attempts=retry_attempts,
                base_delay=retry_delay,
                max_delay=self.retry_config.max_delay,
                backoff_multiplier=self.retry_config.backoff_multiplier,
                strategy=self.retry_config.strategy,
                jitter_type=self.retry_config.jitter_type,
                timeout_multiplier=self.retry_config.timeout_multiplier,
                circuit_breaker_integration=self.retry_config.circuit_breaker_integration,
            )
            temp_retry_manager = RetryManager(temp_retry_config)

            last_exception = None

            for attempt in range(retry_attempts):
                try:
                    request = self._create_sampling_request(query, 1, **kwargs)
                    response = await self._send_sampling_request(request)
                    self._update_token_usage(response, prompt_text=query)

                    # Record success for adaptive strategy
                    temp_retry_manager.record_success()
                    return response

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this error
                    if not temp_retry_manager.should_retry(e, attempt):
                        temp_retry_manager.record_failure()
                        break

                    # Don't retry on the last attempt
                    if attempt >= retry_attempts - 1:
                        temp_retry_manager.record_failure()
                        break

                    # Calculate delay with jitter and strategy
                    delay = temp_retry_manager.calculate_delay(attempt, type(e))

                    self.logger.warning(
                        f"Batch query attempt {attempt + 1} failed: {e}, "
                        f"retrying in {delay:.2f}s (strategy: {temp_retry_config.strategy.value})"
                    )

                    # Record failure for adaptive strategy
                    temp_retry_manager.record_failure()

                    # Wait before retry
                    await asyncio.sleep(delay)

            raise last_exception

    def _update_token_usage(
        self, response: dict[str, Any], prompt_text: str | None = None
    ) -> None:
        """
        Update token usage and cost tracking based on response using improved estimation.
        Note: This is application-specific functionality, not part of the MCP protocol.

        :param response: The response from the MCP server
        :type response: dict[str, Any]
        :param prompt_text: Optional prompt text for better estimation
        :type prompt_text: str | None
        """
        try:
            # Try to extract actual token usage from response metadata if available
            metadata = response.get("metadata", {})
            if "usage" in metadata:
                usage = metadata["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.logger.debug(
                    f"Using actual token usage: {prompt_tokens} prompt, {completion_tokens} completion"
                )
            else:
                # Use improved token estimation as fallback
                content = response.get("content", {})
                if isinstance(content, dict) and content.get("type") == "text":
                    response_text = content.get("text", "")

                    # Estimate completion tokens using improved algorithm
                    estimated_completion_tokens = self.token_estimator.estimate_tokens(
                        response_text, context="response"
                    )
                    self.completion_tokens += estimated_completion_tokens

                    # Estimate prompt tokens if prompt text is available
                    if prompt_text:
                        estimated_prompt_tokens = self.token_estimator.estimate_tokens(
                            prompt_text, context="prompt"
                        )
                        self.prompt_tokens += estimated_prompt_tokens
                    else:
                        # Fallback: estimate based on response length and typical prompt/response ratio
                        estimated_prompt_tokens = max(
                            20, estimated_completion_tokens // 3
                        )
                        self.prompt_tokens += estimated_prompt_tokens

                    self.logger.debug(
                        f"Using improved token estimation: {estimated_prompt_tokens} prompt, "
                        f"{estimated_completion_tokens} completion"
                    )

            # Update cost calculation
            prompt_tokens_k = float(self.prompt_tokens) / 1000.0
            completion_tokens_k = float(self.completion_tokens) / 1000.0
            self.cost = (
                self.prompt_token_cost * prompt_tokens_k
                + self.response_token_cost * completion_tokens_k
            )

            self.logger.debug(
                f"Token usage updated. Total: {self.prompt_tokens + self.completion_tokens} tokens, Cost: ${self.cost:.4f}"
            )

        except Exception as e:
            self.logger.warning(f"Failed to update token usage: {e}")

    def get_response_texts(self, query_response: dict | list[dict]) -> list[str]:
        """
        Extract the response texts from the query response following MCP response format.

        This method handles the MCP response format and extracts readable text content.
        It supports various content types including text and image content, providing
        appropriate representations for each type.

        :param query_response: The response (or list of responses) from the MCP server.
        :type query_response: dict | list[dict]
        :return: List of response strings. Always returns a list, even for single responses.
        :rtype: list[str]

        Example:
            Extract text from single response:

            ```python
            lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
            response = lm.query("Explain photosynthesis")
            texts = lm.get_response_texts(response)
            print(f"Explanation: {texts[0]}")
            ```

            Extract text from multiple responses:

            ```python
            responses = lm.query("Give me 3 recipe ideas", num_responses=3)
            recipes = lm.get_response_texts(responses)
            for i, recipe in enumerate(recipes):
                print(f"Recipe {i+1}: {recipe}")
            ```

            Handle different content types:

            ```python
            response = lm.query("Describe this image and provide analysis")
            texts = lm.get_response_texts(response)
            for text in texts:
                if "[Image content:" in text:
                    print(f"Image detected: {text}")
                else:
                    print(f"Text response: {text}")
            ```

        Note:
            - Text content is extracted directly from the "text" field
            - Image content returns a descriptive placeholder with MIME type
            - Unknown content types return a descriptive placeholder
            - Malformed responses return error messages for debugging
        """
        if not isinstance(query_response, list):
            query_response = [query_response]

        texts = []
        for response in query_response:
            try:
                # Handle MCP response format
                content = response.get("content", {})

                # Handle both single content dict and list of content items
                if isinstance(content, list):
                    # Content is a list of items
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text = item.get("text", "")
                                texts.append(text)
                            elif item.get("type") == "image":
                                # For image content, return a description
                                mime_type = item.get("mimeType", "unknown")
                                texts.append(f"[Image content: {mime_type}]")
                            else:
                                # Unknown content type
                                texts.append(
                                    f"[Unknown content type: {item.get('type', 'none')}]"
                                )
                elif isinstance(content, dict):
                    # Content is a single item
                    if content.get("type") == "text":
                        text = content.get("text", "")
                        texts.append(text)
                    elif content.get("type") == "image":
                        # For image content, return a description
                        mime_type = content.get("mimeType", "unknown")
                        texts.append(f"[Image content: {mime_type}]")
                    else:
                        # Unknown content type
                        texts.append(
                            f"[Unknown content type: {content.get('type', 'none')}]"
                        )
                else:
                    # Fallback for unexpected response format
                    self.logger.warning(f"Unexpected response format: {response}")
                    texts.append(str(response))
            except Exception as e:
                self.logger.error(f"Error extracting text from response: {e}")
                texts.append(f"[Error extracting response: {e}]")

        return texts

    def get_circuit_breaker_status(self) -> dict[str, Any] | None:
        """
        Get circuit breaker status and metrics if available.

        Returns:
            Dict with circuit breaker status or None if not enabled
        """
        if hasattr(self.transport, "get_circuit_breaker_metrics"):
            try:
                metrics = self.transport.get_circuit_breaker_metrics()
                state = self.transport.get_circuit_breaker_state()

                return {
                    "state": state.value,
                    "is_healthy": self.transport.is_circuit_healthy(),
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "circuit_open_count": metrics.circuit_open_count,
                    "last_failure_time": metrics.last_failure_time,
                    "state_change_time": metrics.state_change_time,
                }
            except Exception as e:
                self.logger.warning(f"Failed to get circuit breaker status: {e}")
                return None
        return None

    def is_service_healthy(self) -> bool:
        """
        Check if the MCP service is healthy based on circuit breaker state.

        Returns:
            True if service is healthy or circuit breaker not enabled
        """
        if hasattr(self.transport, "is_circuit_healthy"):
            return self.transport.is_circuit_healthy()
        return True  # Assume healthy if no circuit breaker

    def _calculate_cost(self, prompt_tokens: int, response_tokens: int) -> float:
        """
        Calculate the cost of a request based on token usage.

        :param prompt_tokens: Number of prompt tokens
        :type prompt_tokens: int
        :param response_tokens: Number of response tokens
        :type response_tokens: int
        :return: Estimated cost in dollars
        :rtype: float
        """
        prompt_cost = (prompt_tokens / 1000.0) * self.prompt_token_cost
        response_cost = (response_tokens / 1000.0) * self.response_token_cost
        return prompt_cost + response_cost

    def _estimate_tokens(self, text: str, context: str = "general") -> int:
        """
        Estimate the number of tokens in the given text.

        :param text: The text to analyze
        :type text: str
        :param context: Context hint for better estimation ("general", "code", "prompt", "response")
        :type context: str
        :return: Estimated number of tokens
        :rtype: int
        """
        return self.token_estimator.estimate_tokens(text, context)

    def get_metrics(self) -> dict[str, Any] | None:
        """
        Get comprehensive metrics for the MCP client.

        Returns:
            Dictionary containing current metrics or None if metrics disabled
        """
        if self.metrics_collector:
            return self.metrics_collector.get_current_metrics()
        return None

    def get_method_metrics(self, method: str) -> dict[str, Any] | None:
        """
        Get metrics for a specific MCP method.

        Args:
            method: The MCP method name

        Returns:
            ionary containing method-specific metrics or None if metrics disabled
        """
        if self.metrics_collector:
            return self.metrics_collector.get_method_metrics(method)
        return None

    def get_error_summary(self) -> dict[str, Any] | None:
        """
        Get summary of errors encountered.

        Returns:
            Dictionary containing error statistics or None if metrics disabled
        """
        if self.metrics_collector:
            return self.metrics_collector.get_error_summary()
        return None

    def get_health_status(self) -> [str, Any]:
        """
        Get overall health status based on metrics and circuit breaker state.

        Returns:
            ionary containing comprehensive health assessment
        """
        health_status = {
            "timestamp": time.time(),
            "overall_status": "unknown",
            "components": {},
        }

        # Check metrics-based health
        if self.metrics_collector:
            metrics_health = self.metrics_collector.get_health_status()
            health_status["components"]["metrics"] = metrics_health

        # Check circuit breaker health
        cb_status = self.get_circuit_breaker_status()
        if cb_status:
            cb_health = {
                "status": "healthy" if cb_status["is_healthy"] else "unhealthy",
                "state": cb_status["state"],
                "error_rate": (
                    cb_status["failed_requests"] / max(cb_status["total_requests"], 1)
                )
                * 100,
            }
            health_status["components"]["circuit_breaker"] = cb_health

        # Check connection health
        connection_health = {
            "status": "healthy" if self._connection_established else "disconnected",
            "connected": self._connection_established,
        }
        health_status["components"]["connection"] = connection_health

        # Determine overall status
        component_statuses = [
            comp.get("status", "unknown")
            for comp in health_status["components"].values()
        ]
        if "unhealthy" in component_statuses:
            health_status["overall_status"] = "unhealthy"
        elif "degraded" in component_statuses:
            health_status["overall_status"] = "degraded"
        elif "disconnected" in component_statuses:
            health_status["overall_status"] = "disconnected"
        elif all(status == "healthy" for status in component_statuses):
            health_status["overall_status"] = "healthy"

        return health_status

    def export_metrics(self, format_type: str = "json") -> str:
        """
        Export current metrics in specified format.

        Args:
            format_type: Export format ("json", "prometheus", "csv")

        Returns:
            Formatted metrics string or empty string if metrics disabled
        """
        if self.metrics_collector:
            return self.metrics_collector.export_metrics(format_type)
        return ""

    def trigger_metrics_export(self):
        """Trigger metrics export if enabled and due."""
        if self.metrics_collector:
            self.metrics_collector.trigger_export()

    def reset_metrics(self):
        """Reset all collected metrics."""
        if self.metrics_collector:
            self.metrics_collector.reset_metrics()

    async def __aenter__(self) -> "MCPLanguageModel":
        """
        Async context manager entry.
        Establishes connection to the MCP server.

        :return: Self for use in async with statement
        :rtype: MCPLanguageModel
        """
        await self._ensure_connection()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any | None,
    ) -> None:
        """
        Async context manager exit.
        Properly disconnects from the MCP server.

        :param exc_type: Exception type if an exception occurred
        :param exc_val: Exception value if an exception occurred
        :param exc_tb: Exception traceback if an exception occurred
        :return: None (don't suppress exceptions)
        :rtype: None
        """
        await self._disconnect()
        return None

    async def close(self) -> None:
        """
        Explicitly close the connection to the MCP server.
        This method can be called manually for cleanup.
        """
        await self._disconnect()
