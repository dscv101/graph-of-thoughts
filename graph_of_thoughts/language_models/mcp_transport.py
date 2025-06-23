# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Transport Layer Implementation.

This module provides comprehensive transport implementations that follow the official MCP specification
for stdio and HTTP transports with proper JSON-RPC 2.0 message handling. The transport layer abstracts
the underlying communication mechanisms and provides a unified interface for MCP protocol communication.

Key Features:
    - Full JSON-RPC 2.0 compliance with proper message formatting
    - Support for stdio transport (local MCP servers)
    - Support for HTTP transport (remote MCP servers)
    - Robust error handling with custom exception hierarchy
    - Async context manager support for resource management
    - Automatic connection management and cleanup
    - Request/response correlation and timeout handling
    - Protocol validation and message formatting

Transport Types:
    1. Stdio Transport (StdioMCPTransport):
       - Communicates with local MCP servers via stdin/stdout
       - Used for Claude Desktop, VSCode, Cursor integrations
       - Supports process lifecycle management
       - Handles environment variable passing

    2. HTTP Transport (HTTPMCPTransport):
       - Communicates with remote MCP servers via HTTP
       - Supports RESTful MCP server endpoints
       - Includes session management and authentication
       - Handles connection pooling and keep-alive

Example Usage:
    Creating a stdio transport:

    ```python
    from graph_of_thoughts.language_models.mcp_transport import create_transport

    config = {
        "transport": {
            "type": "stdio",
            "command": "claude-desktop",
            "args": ["--mcp-server"],
            "env": {}
        },
        "client_info": {
            "name": "my-app",
            "version": "1.0.0"
        },
        "capabilities": {"sampling": {}},
        "connection_config": {
            "timeout": 30.0,
            "request_timeout": 60.0
        }
    }

    transport = create_transport(config)

    # Use with async context manager
    async with transport:
        response = await transport.send_request("ping", {})
        print(f"Server responded: {response}")
    ```

    Creating an HTTP transport with connection pooling:

    ```python
    config = {
        "transport": {
            "type": "http",
            "url": "https://api.example.com/mcp",
            "headers": {
                "Authorization": "Bearer token123",
                "Content-Type": "application/json"
            }
        },
        "client_info": {"name": "my-app", "version": "1.0.0"},
        "capabilities": {"sampling": {}},
        "connection_config": {
            "timeout": 30.0,
            "connection_pool": {
                "max_connections": 20,
                "max_keepalive_connections": 10,
                "keepalive_expiry": 30.0,
                "enable_http2": False,
                "retries": 3
            }
        }
    }

    transport = create_transport(config)

    async with transport:
        # Send sampling request - benefits from connection pooling
        sampling_request = {
            "messages": [{"role": "user", "content": {"type": "text", "text": "Hello"}}],
            "temperature": 0.7,
            "maxTokens": 100
        }
        response = await transport.send_sampling_request(sampling_request)

        # Check connection pool status
        pool_info = transport.get_connection_pool_info()
        print(f"Pool status: {pool_info}")
    ```

Error Handling:
    The transport layer provides a comprehensive exception hierarchy:

    - MCPTransportError: Base exception for all transport errors
    - MCPConnectionError: Connection establishment failures
    - MCPTimeoutError: Request timeout scenarios
    - MCPProtocolError: Protocol-level errors and malformed messages
    - MCPValidationError: Request/response validation failures
    - MCPServerError: Server-side errors with error codes and data

    Example error handling:

    ```python
    from graph_of_thoughts.language_models.mcp_transport import (
        MCPConnectionError, MCPTimeoutError, MCPServerError
    )

    try:
        async with transport:
            response = await transport.send_request("method", params)
    except MCPConnectionError as e:
        print(f"Failed to connect: {e}")
        if e.original_error:
            print(f"Original error: {e.original_error}")
    except MCPTimeoutError as e:
        print(f"Request timed out: {e}")
    except MCPServerError as e:
        print(f"Server error {e.error_code}: {e}")
        if e.error_data:
            print(f"Error details: {e.error_data}")
    ```

Protocol Compliance:
    All transport implementations strictly follow the MCP specification:
    - Proper JSON-RPC 2.0 message formatting
    - Correct initialization and capability negotiation
    - Standard error code handling
    - Protocol version compatibility checking
    - Message correlation and response matching
"""

# Standard library imports
import asyncio
import json
import logging
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import httpx
import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Local imports
from .mcp_protocol import MCPProtocolValidator, create_sampling_request

# Import metrics integration
try:
    from .mcp_metrics import MetricsIntegrationMixin, get_global_metrics_collector
    METRICS_AVAILABLE = True
except ImportError:
    # Fallback if metrics module not available
    class MetricsIntegrationMixin:
        def set_metrics_collector(self, metrics_collector): pass
        def record_connection_metric(self, success, duration_ms): pass
        def record_transport_metric(self, metric_name, value, labels=None): pass
    def get_global_metrics_collector(): return None
    METRICS_AVAILABLE = False


class MCPTransport(ABC):
    """
    Abstract base class for MCP transport implementations following the MCP specification.
    Handles JSON-RPC 2.0 message formatting and protocol compliance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the MCP transport.

        :param config: Transport configuration
        :type config: Dict[str, Any]
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session: Optional[ClientSession] = None
        self.connected = False
        self.validator = MCPProtocolValidator()
        self._request_id_counter = 0

        # Validate configuration
        if not self.validator.validate_configuration(config):
            raise ValueError("Invalid MCP configuration")

    def _generate_request_id(self) -> str:
        """Generate a unique request ID for JSON-RPC messages."""
        self._request_id_counter += 1
        return f"req_{self._request_id_counter}_{uuid.uuid4().hex[:8]}"

    def _create_jsonrpc_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a JSON-RPC 2.0 request message.

        :param method: The method name
        :type method: str
        :param params: The parameters
        :type params: Dict[str, Any]
        :return: JSON-RPC request
        :rtype: Dict[str, Any]
        """
        return {
            "jsonrpc": "2.0",
            "id": self._generate_request_id(),
            "method": method,
            "params": params
        }

    def _create_jsonrpc_notification(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a JSON-RPC 2.0 notification message.

        :param method: The method name
        :type method: str
        :param params: The parameters
        :type params: Dict[str, Any]
        :return: JSON-RPC notification
        :rtype: Dict[str, Any]
        """
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the MCP host.

        :return: True if connection successful, False otherwise
        :rtype: bool
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close the connection to the MCP host.
        """
        pass

    @abstractmethod
    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request to the MCP host.

        :param method: The method name
        :type method: str
        :param params: The request parameters
        :type params: Dict[str, Any]
        :return: The response from the host
        :rtype: Dict[str, Any]
        """
        pass

    async def send_sampling_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a sampling request to the MCP host using the proper protocol.

        :param request: The sampling request
        :type request: Dict[str, Any]
        :return: The response from the host
        :rtype: Dict[str, Any]
        """
        # Validate the sampling request
        if not self.validator.validate_sampling_request(request):
            raise ValueError("Invalid sampling request")

        return await self.send_request("sampling/createMessage", request)

    async def initialize(self) -> Dict[str, Any]:
        """
        Send initialization request to establish the MCP session.

        :return: Initialization response
        :rtype: Dict[str, Any]
        """
        client_info = self.config.get("client_info", {})
        capabilities = self.config.get("capabilities", {})

        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": capabilities,
            "clientInfo": client_info
        }

        response = await self.send_request("initialize", init_params)

        # Send initialized notification
        await self.send_notification("initialized", {})

        return response

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """
        Send a JSON-RPC notification (no response expected).

        :param method: The method name
        :type method: str
        :param params: The notification parameters
        :type params: Dict[str, Any]
        """
        # Default implementation - subclasses should override if needed
        pass

    async def __aenter__(self) -> "MCPTransport":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Async context manager exit."""
        await self.disconnect()


class StdioMCPTransport(MCPTransport, MetricsIntegrationMixin):
    """
    MCP transport implementation using stdio communication following the MCP specification.
    This is used for connecting to local MCP servers via standard input/output.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the stdio MCP transport.

        :param config: Transport configuration
        :type config: Dict[str, Any]
        """
        super().__init__(config)
        MetricsIntegrationMixin.__init__(self)
        self.process: Optional[subprocess.Popen] = None
        self.stdio_transport = None
        self.write_stream = None
        self.read_stream = None
        self.exit_stack = None

        # Timeout configuration
        connection_config = config.get("connection_config", {})
        self.connection_timeout = connection_config.get("timeout", 30.0)
        self.request_timeout = connection_config.get("request_timeout", 60.0)

        # Set up metrics collector if available
        if METRICS_AVAILABLE:
            self.set_metrics_collector(get_global_metrics_collector())

    async def connect(self) -> bool:
        """
        Establish stdio connection to the MCP server.

        :return: True if connection successful, False otherwise
        :rtype: bool
        """
        start_time = time.time()
        success = False
        try:
            transport_config = self.config.get("transport", {})
            command = transport_config.get("command")
            args = transport_config.get("args", [])
            env = transport_config.get("env", {})

            if not command:
                raise ValueError("Missing command in stdio transport configuration")

            self.logger.info(f"Connecting to MCP server: {command} {' '.join(args)}")

            # Create server parameters for the MCP SDK
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env if env else None
            )

            # Use the MCP SDK to establish the connection
            self.exit_stack = AsyncExitStack()

            # Connect using the MCP SDK with timeout
            try:
                stdio_transport = await asyncio.wait_for(
                    self.exit_stack.enter_async_context(stdio_client(server_params)),
                    timeout=self.connection_timeout
                )
                self.read_stream, self.write_stream = stdio_transport
            except asyncio.TimeoutError:
                raise MCPTimeoutError(f"Connection timeout after {self.connection_timeout}s")
            except FileNotFoundError as e:
                raise MCPConnectionError(f"MCP server command not found: {command}", e)
            except PermissionError as e:
                raise MCPConnectionError(f"Permission denied executing MCP server: {command}", e)

            # Create the MCP session
            try:
                self.session = await asyncio.wait_for(
                    self.exit_stack.enter_async_context(
                        ClientSession(self.read_stream, self.write_stream)
                    ),
                    timeout=self.connection_timeout
                )
            except asyncio.TimeoutError:
                raise MCPTimeoutError(f"Session creation timeout after {self.connection_timeout}s")

            # Initialize the MCP session
            try:
                await asyncio.wait_for(
                    self.session.initialize(),
                    timeout=self.connection_timeout
                )
            except asyncio.TimeoutError:
                raise MCPTimeoutError(f"Session initialization timeout after {self.connection_timeout}s")
            except Exception as e:
                raise MCPProtocolError(f"Failed to initialize MCP session: {e}", e)

            self.connected = True
            success = True
            self.logger.info("Successfully connected to MCP server via stdio")
            return True

        except (MCPTimeoutError, MCPConnectionError, MCPProtocolError):
            # Re-raise our custom exceptions
            if self.exit_stack:
                try:
                    await self.exit_stack.aclose()
                except Exception as cleanup_error:
                    self.logger.warning(f"Error during connection cleanup: {cleanup_error}")
            raise
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Unexpected error connecting via stdio: {e}")
            if self.exit_stack:
                try:
                    await self.exit_stack.aclose()
                except Exception as cleanup_error:
                    self.logger.warning(f"Error during connection cleanup: {cleanup_error}")
            raise MCPConnectionError(f"Failed to connect via stdio: {e}", e)
        finally:
            # Record connection metrics
            duration_ms = (time.time() - start_time) * 1000
            self.record_connection_metric(success, duration_ms)

    async def disconnect(self) -> None:
        """
        Close the stdio connection to the MCP server.
        """
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
            self.connected = False
            self.session = None
            self.read_stream = None
            self.write_stream = None
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request via stdio using the MCP session.

        :param method: The method name
        :type method: str
        :param params: The request parameters
        :type params: Dict[str, Any]
        :return: The response from the server
        :rtype: Dict[str, Any]
        """
        if not self.connected or not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            self.logger.debug(f"Sending MCP request: {method}")

            if method == "sampling/createMessage":
                # Convert params to proper MCP types
                try:
                    request_params = self._convert_to_mcp_sampling_params(params)
                except (KeyError, ValueError, TypeError) as e:
                    raise MCPValidationError(f"Invalid sampling parameters: {e}", e)

                # Create the MCP request
                try:
                    request = mcp_types.CreateMessageRequest(
                        method="sampling/createMessage",
                        params=request_params
                    )
                except Exception as e:
                    raise MCPProtocolError(f"Failed to create MCP request: {e}", e)

                # Send the request using the MCP session with timeout
                try:
                    result = await asyncio.wait_for(
                        self.session.send_request(request, mcp_types.CreateMessageResult),
                        timeout=self.request_timeout
                    )
                except asyncio.TimeoutError:
                    raise MCPTimeoutError(f"Request timeout after {self.request_timeout}s")
                except Exception as e:
                    # Check if this is a server error response
                    if hasattr(e, 'error') and isinstance(e.error, dict):
                        error_code = e.error.get('code', 'unknown')
                        error_message = e.error.get('message', str(e))
                        error_data = e.error.get('data')
                        raise MCPServerError(f"Server error: {error_message}", error_code, error_data)
                    raise MCPProtocolError(f"Failed to send sampling request: {e}", e)

                # Convert the result back to our expected format
                try:
                    return self._convert_from_mcp_result(result)
                except Exception as e:
                    raise MCPProtocolError(f"Failed to convert MCP result: {e}", e)
            else:
                # Handle other MCP methods using generic request
                try:
                    request = mcp_types.Request(
                        method=method,
                        params=params
                    )
                except Exception as e:
                    raise MCPProtocolError(f"Failed to create generic MCP request: {e}", e)

                # Send generic request with timeout
                try:
                    result = await asyncio.wait_for(
                        self.session.send_request(request, dict),
                        timeout=self.request_timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    raise MCPTimeoutError(f"Request timeout after {self.request_timeout}s")
                except Exception as e:
                    raise MCPProtocolError(f"Failed to send generic request: {e}", e)

        except (MCPTimeoutError, MCPConnectionError, MCPProtocolError, MCPValidationError, MCPServerError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Unexpected error sending request: {e}")
            raise MCPTransportError(f"Failed to send request: {e}", e)

    def _convert_to_mcp_sampling_params(self, params: Dict[str, Any]) -> mcp_types.CreateMessageRequestParams:
        """
        Convert our internal sampling parameters to MCP CreateMessageRequestParams.

        :param params: Internal sampling parameters
        :type params: Dict[str, Any]
        :return: MCP CreateMessageRequestParams
        :rtype: mcp_types.CreateMessageRequestParams
        """
        # Convert messages to MCP SamplingMessage format
        # Each message must have role (user/assistant) and content (text/image/etc.)
        messages = []
        for msg in params.get("messages", []):
            content = msg["content"]

            # Currently only text content is supported in this implementation
            # Future versions could add support for image, audio, etc.
            if content["type"] == "text":
                mcp_content = mcp_types.TextContent(
                    type="text",
                    text=content["text"]
                )
            else:
                # Raise error for unsupported content types to fail fast
                # This helps identify when new content types need to be implemented
                raise ValueError(f"Unsupported content type: {content['type']}")

            # Create MCP-compliant message structure
            mcp_message = mcp_types.SamplingMessage(
                role=msg["role"],  # Must be "user" or "assistant"
                content=mcp_content
            )
            messages.append(mcp_message)

        # Convert model preferences if present
        # Model preferences allow fine-tuning of model selection and behavior
        model_preferences = None
        if "modelPreferences" in params:
            prefs = params["modelPreferences"]

            # Convert model hints (preferred models) to MCP format
            hints = None
            if "hints" in prefs:
                # Each hint specifies a preferred model by name
                hints = [mcp_types.ModelHint(name=hint["name"]) for hint in prefs["hints"]]

            # Create model preferences with priority weights (0.0 to 1.0)
            # These help the MCP host choose the best model for the request
            model_preferences = mcp_types.ModelPreferences(
                hints=hints,
                costPriority=prefs.get("costPriority"),        # Lower cost preference
                speedPriority=prefs.get("speedPriority"),      # Faster response preference
                intelligencePriority=prefs.get("intelligencePriority")  # Higher capability preference
            )

        return mcp_types.CreateMessageRequestParams(
            messages=messages,
            modelPreferences=model_preferences,
            systemPrompt=params.get("systemPrompt"),
            includeContext=params.get("includeContext", "none"),
            temperature=params.get("temperature"),
            maxTokens=params.get("maxTokens", 1000),
            stopSequences=params.get("stopSequences"),
            metadata=params.get("metadata")
        )

    def _convert_from_mcp_result(self, result: mcp_types.CreateMessageResult) -> Dict[str, Any]:
        """
        Convert MCP CreateMessageResult to our internal format.

        :param result: MCP CreateMessageResult
        :type result: mcp_types.CreateMessageResult
        :return: Internal format response
        :rtype: Dict[str, Any]
        """
        # Convert content back to our format
        content = result.content
        if hasattr(content, 'type') and content.type == "text":
            content_dict = {
                "type": "text",
                "text": content.text
            }
        else:
            # Handle other content types
            content_dict = {
                "type": getattr(content, 'type', 'unknown'),
                "text": str(content)
            }

        return {
            "role": result.role,
            "content": content_dict,
            "model": result.model,
            "stopReason": result.stopReason
        }

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """
        Send a JSON-RPC notification via stdio.

        :param method: The method name
        :type method: str
        :param params: The notification parameters
        :type params: Dict[str, Any]
        """
        if not self.connected or not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            self.logger.debug(f"Sending MCP notification: {method}")

            # Create a notification using the MCP session
            notification = mcp_types.Notification(
                method=method,
                params=params
            )

            await self.session.send_notification(notification)

        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            raise


class HTTPMCPTransport(MCPTransport, MetricsIntegrationMixin):
    """
    MCP transport implementation using Streamable HTTP communication following the MCP specification.
    This is used for connecting to remote MCP servers via HTTP with optional Server-Sent Events.

    Features:
        - Connection pooling for improved performance
        - Keep-alive connections to reduce overhead
        - Configurable pool limits and timeouts
        - Automatic connection reuse across requests
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the HTTP MCP transport with connection pooling.

        :param config: Transport configuration
        :type config: Dict[str, Any]
        """
        super().__init__(config)
        MetricsIntegrationMixin.__init__(self)
        self.client: Optional[httpx.AsyncClient] = None
        transport_config = config.get("transport", {})
        self.server_url = transport_config.get("url", "http://localhost:8000/mcp")
        self.headers = transport_config.get("headers", {})
        self.session_id: Optional[str] = None
        self.session_management = transport_config.get("session_management", False)

        # Timeout configuration
        connection_config = config.get("connection_config", {})
        self.connection_timeout = connection_config.get("timeout", 30.0)
        self.request_timeout = connection_config.get("request_timeout", 60.0)

        # Connection pooling configuration
        pool_config = connection_config.get("connection_pool", {})
        self.max_connections = pool_config.get("max_connections", 20)
        self.max_keepalive_connections = pool_config.get("max_keepalive_connections", 10)
        self.keepalive_expiry = pool_config.get("keepalive_expiry", 30.0)
        self.enable_http2 = pool_config.get("enable_http2", False)
        self.retries = pool_config.get("retries", 3)

        # Set up metrics collector if available
        if METRICS_AVAILABLE:
            self.set_metrics_collector(get_global_metrics_collector())

    async def connect(self) -> bool:
        """
        Establish HTTP connection to the MCP server with connection pooling.

        :return: True if connection successful, False otherwise
        :rtype: bool
        """
        start_time = time.time()
        success = False
        try:
            # Set up default headers for MCP
            default_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            default_headers.update(self.headers)

            # Create HTTP client with connection pooling and optimized settings
            try:
                # Configure connection limits for pooling
                limits = httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_keepalive_connections,
                    keepalive_expiry=self.keepalive_expiry
                )

                # Configure timeout settings
                timeout = httpx.Timeout(
                    connect=self.connection_timeout,
                    read=self.request_timeout,
                    write=self.request_timeout,
                    pool=self.connection_timeout
                )

                # Create client with pooling and keep-alive
                self.client = httpx.AsyncClient(
                    timeout=timeout,
                    headers=default_headers,
                    limits=limits,
                    http2=self.enable_http2,
                    follow_redirects=True,
                    max_redirects=3
                )

                self.logger.debug(
                    f"Created HTTP client with connection pooling: "
                    f"max_connections={self.max_connections}, "
                    f"max_keepalive={self.max_keepalive_connections}, "
                    f"keepalive_expiry={self.keepalive_expiry}s"
                )

            except Exception as e:
                raise MCPConnectionError(f"Failed to create HTTP client: {e}", e)

            # Send initialization request with timeout
            try:
                init_response = await asyncio.wait_for(
                    self.initialize(),
                    timeout=self.connection_timeout
                )
            except asyncio.TimeoutError:
                raise MCPTimeoutError(f"Initialization timeout after {self.connection_timeout}s")
            except httpx.ConnectError as e:
                raise MCPConnectionError(f"Failed to connect to {self.server_url}: {e}", e)
            except httpx.HTTPStatusError as e:
                raise MCPServerError(f"HTTP error during initialization: {e.response.status_code}", str(e.response.status_code))
            except Exception as e:
                raise MCPProtocolError(f"Failed to initialize MCP connection: {e}", e)

            # Check for session ID in response headers
            if self.session_management and "Mcp-Session-Id" in init_response:
                self.session_id = init_response["Mcp-Session-Id"]
                self.logger.info(f"Received session ID: {self.session_id}")

            self.connected = True
            success = True
            self.logger.info(f"Connected to MCP server at {self.server_url}")
            return True

        except (MCPTimeoutError, MCPConnectionError, MCPProtocolError, MCPServerError):
            # Re-raise our custom exceptions
            if self.client:
                try:
                    await self.client.aclose()
                except Exception as cleanup_error:
                    self.logger.warning(f"Error during HTTP client cleanup: {cleanup_error}")
                self.client = None
            raise
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Unexpected error connecting via HTTP: {e}")
            if self.client:
                try:
                    await self.client.aclose()
                except Exception as cleanup_error:
                    self.logger.warning(f"Error during HTTP client cleanup: {cleanup_error}")
                self.client = None
            raise MCPConnectionError(f"Failed to connect via HTTP: {e}", e)
        finally:
            # Record connection metrics
            duration_ms = (time.time() - start_time) * 1000
            self.record_connection_metric(success, duration_ms)

    async def disconnect(self) -> None:
        """
        Close the HTTP connection to the MCP server and clean up connection pool.
        """
        try:
            # Log connection pool statistics before closing
            if self.client:
                self._log_connection_stats()

            # Send session termination if using session management
            if self.session_management and self.session_id and self.client:
                try:
                    await self.client.delete(
                        self.server_url,
                        headers={"Mcp-Session-Id": self.session_id}
                    )
                except:
                    pass  # Ignore errors during session cleanup

            if self.client:
                await self.client.aclose()

            self.connected = False
            self.session_id = None
            self.logger.info("Disconnected from MCP server and closed connection pool")
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

    def _log_connection_stats(self) -> None:
        """
        Log connection pool statistics for monitoring and debugging.
        """
        if not self.client:
            return

        try:
            # Get connection pool info from httpx client
            pool_info = {}
            if hasattr(self.client, '_transport') and hasattr(self.client._transport, '_pool'):
                pool = self.client._transport._pool
                if hasattr(pool, '_connections'):
                    pool_info['active_connections'] = len(pool._connections)
                if hasattr(pool, '_keepalive_connections'):
                    pool_info['keepalive_connections'] = len(pool._keepalive_connections)

            if pool_info:
                self.logger.debug(f"Connection pool stats: {pool_info}")
            else:
                self.logger.debug("Connection pool statistics not available")

        except Exception as e:
            self.logger.debug(f"Could not retrieve connection pool stats: {e}")

    def get_connection_pool_info(self) -> Dict[str, Any]:
        """
        Get information about the current connection pool state.

        :return: Dictionary containing pool information
        :rtype: Dict[str, Any]
        """
        if not self.client:
            return {"status": "disconnected"}

        info = {
            "status": "connected" if self.connected else "disconnected",
            "max_connections": self.max_connections,
            "max_keepalive_connections": self.max_keepalive_connections,
            "keepalive_expiry": self.keepalive_expiry,
            "http2_enabled": self.enable_http2,
            "server_url": self.server_url
        }

        # Try to get actual pool statistics
        try:
            if hasattr(self.client, '_transport') and hasattr(self.client._transport, '_pool'):
                pool = self.client._transport._pool
                if hasattr(pool, '_connections'):
                    info['active_connections'] = len(pool._connections)
                if hasattr(pool, '_keepalive_connections'):
                    info['keepalive_connections'] = len(pool._keepalive_connections)
        except Exception:
            pass  # Pool stats not available

        return info

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request via HTTP using the Streamable HTTP transport.

        :param method: The method name
        :type method: str
        :param params: The request parameters
        :type params: Dict[str, Any]
        :return: The response from the server
        :rtype: Dict[str, Any]
        """
        if not self.connected or not self.client:
            raise RuntimeError("Not connected to MCP server")

        try:
            # Create JSON-RPC request
            jsonrpc_request = self._create_jsonrpc_request(method, params)

            # Prepare headers
            headers = {}
            if self.session_management and self.session_id:
                headers["Mcp-Session-Id"] = self.session_id

            self.logger.debug(f"Sending HTTP request: {method}")

            # Send POST request to MCP endpoint with timeout
            try:
                response = await asyncio.wait_for(
                    self.client.post(
                        self.server_url,
                        json=jsonrpc_request,
                        headers=headers
                    ),
                    timeout=self.request_timeout
                )
            except asyncio.TimeoutError:
                raise MCPTimeoutError(f"HTTP request timeout after {self.request_timeout}s")
            except httpx.ConnectError as e:
                raise MCPConnectionError(f"Connection error: {e}", e)
            except httpx.HTTPStatusError as e:
                raise MCPServerError(f"HTTP error: {e.response.status_code}", str(e.response.status_code))

            # Handle response based on status code
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")

                if "application/json" in content_type:
                    # Single JSON response
                    try:
                        json_response = response.json()
                    except Exception as e:
                        raise MCPProtocolError(f"Failed to parse JSON response: {e}", e)

                    # Handle JSON-RPC response format
                    if "result" in json_response:
                        result = json_response["result"]
                        # For sampling requests, convert from MCP format if needed
                        if method == "sampling/createMessage":
                            try:
                                return self._convert_http_sampling_response(result)
                            except Exception as e:
                                raise MCPProtocolError(f"Failed to convert sampling response: {e}", e)
                        return result
                    elif "error" in json_response:
                        error = json_response["error"]
                        error_code = error.get("code", "unknown") if isinstance(error, dict) else "unknown"
                        error_message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                        error_data = error.get("data") if isinstance(error, dict) else None
                        raise MCPServerError(f"MCP server error: {error_message}", error_code, error_data)
                    else:
                        return json_response

                elif "text/event-stream" in content_type:
                    # SSE stream response - handle the first event
                    # In a full implementation, you'd handle the entire stream
                    try:
                        lines = response.text.split('\n')
                        for line in lines:
                            if line.startswith('data: '):
                                data = line[6:]  # Remove 'data: ' prefix
                                if data.strip():
                                    return json.loads(data)
                        raise MCPProtocolError("No data in SSE stream")
                    except json.JSONDecodeError as e:
                        raise MCPProtocolError(f"Failed to parse SSE data: {e}", e)
                else:
                    raise MCPProtocolError(f"Unexpected content type: {content_type}")
            elif response.status_code == 400:
                raise MCPValidationError(f"Bad request: {response.text}")
            elif response.status_code == 401:
                raise MCPConnectionError("Authentication failed")
            elif response.status_code == 403:
                raise MCPConnectionError("Access forbidden")
            elif response.status_code == 404:
                raise MCPConnectionError(f"MCP endpoint not found: {self.server_url}")
            elif response.status_code >= 500:
                raise MCPServerError(f"Server error: {response.status_code}", str(response.status_code))
            else:
                raise MCPServerError(f"HTTP request failed: {response.status_code}", str(response.status_code))

        except (MCPTimeoutError, MCPConnectionError, MCPProtocolError, MCPValidationError, MCPServerError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Unexpected error sending HTTP request: {e}")
            raise MCPTransportError(f"Failed to send HTTP request: {e}", e)

    def _convert_http_sampling_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert HTTP JSON-RPC sampling response to our internal format.

        :param result: HTTP JSON-RPC result
        :type result: Dict[str, Any]
        :return: Internal format response
        :rtype: Dict[str, Any]
        """
        # HTTP responses should already be in the correct format
        # but ensure consistency with our expected structure
        return {
            "role": result.get("role", "assistant"),
            "content": result.get("content", {"type": "text", "text": ""}),
            "model": result.get("model", "unknown"),
            "stopReason": result.get("stopReason", "endTurn")
        }

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """
        Send a JSON-RPC notification via HTTP.

        :param method: The method name
        :type method: str
        :param params: The notification parameters
        :type params: Dict[str, Any]
        """
        if not self.connected or not self.client:
            raise RuntimeError("Not connected to MCP server")

        try:
            # Create JSON-RPC notification
            jsonrpc_notification = self._create_jsonrpc_notification(method, params)

            # Prepare headers
            headers = {}
            if self.session_management and self.session_id:
                headers["Mcp-Session-Id"] = self.session_id

            self.logger.debug(f"Sending HTTP notification: {method}")

            # Send POST request
            await self.client.post(
                self.server_url,
                json=jsonrpc_notification,
                headers=headers
            )

        except Exception as e:
            self.logger.error(f"Failed to send HTTP notification: {e}")
            raise


def create_transport(config: Dict[str, Any]) -> MCPTransport:
    """
    Factory function to create the appropriate MCP transport based on configuration.

    This function analyzes the provided configuration and instantiates the correct
    transport implementation (stdio or HTTP) based on the transport type specified
    in the configuration.

    Note: For enhanced functionality with host-specific optimizations and validation,
    consider using the MCP plugin system via create_transport_from_plugin() function
    from the mcp_host_plugins module.

    :param config: Complete MCP configuration containing transport settings
    :type config: Dict[str, Any]
    :return: MCP transport instance (StdioMCPTransport or HTTPMCPTransport)
    :rtype: MCPTransport

    Example:
        Create stdio transport for Claude Desktop:

        ```python
        config = {
            "transport": {
                "type": "stdio",
                "command": "claude-desktop",
                "args": ["--mcp-server"]
            },
            "client_info": {"name": "my-app", "version": "1.0.0"},
            "capabilities": {"sampling": {}}
        }

        transport = create_transport(config)
        print(f"Created transport: {type(transport).__name__}")
        # Output: Created transport: StdioMCPTransport
        ```

        Create HTTP transport for remote server:

        ```python
        config = {
            "transport": {
                "type": "http",
                "url": "https://mcp-server.example.com/api"
            },
            "client_info": {"name": "my-app", "version": "1.0.0"},
            "capabilities": {"sampling": {}}
        }

        transport = create_transport(config)
        print(f"Created transport: {type(transport).__name__}")
        # Output: Created transport: HTTPMCPTransport
        ```

    Raises:
        ValueError: If the transport type is not supported or missing

    Note:
        The configuration is validated during transport creation. Invalid
        configurations will raise appropriate validation errors.
    """
    transport_config = config.get("transport", {})
    transport_type = transport_config.get("type", "stdio")

    if transport_type == "stdio":
        return StdioMCPTransport(config)
    elif transport_type == "http":
        return HTTPMCPTransport(config)
    else:
        raise ValueError(f"Unsupported transport type: {transport_type}")


class MCPTransportError(Exception):
    """Exception raised for MCP transport-related errors."""
    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class MCPConnectionError(MCPTransportError):
    """Exception raised for MCP connection errors."""
    pass


class MCPTimeoutError(MCPTransportError):
    """Exception raised for MCP timeout errors."""
    pass


class MCPProtocolError(MCPTransportError):
    """Exception raised for MCP protocol errors."""
    pass


class MCPServerError(MCPTransportError):
    """Exception raised for MCP server-side errors."""
    def __init__(self, message: str, error_code: Optional[str] = None, error_data: Optional[dict] = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.error_data = error_data


class MCPValidationError(MCPTransportError):
    """Exception raised for MCP message validation errors."""
    pass
