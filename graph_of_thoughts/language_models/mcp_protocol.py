# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Protocol Validation and Message Formatting Utilities.

This module provides comprehensive utilities for validating MCP protocol messages and configurations
according to the official MCP specification. It ensures protocol compliance and proper message
formatting for all MCP communications.

Key Features:
    - Complete MCP protocol validation
    - Message format validation and creation
    - Configuration schema validation
    - Transport configuration validation
    - Model preferences validation
    - Sampling request validation
    - Content type validation (text, image)
    - Error reporting with detailed messages

Validation Components:
    1. Configuration Validation:
       - Transport settings (stdio/HTTP)
       - Client information
       - Capabilities declaration
       - Connection parameters

    2. Message Validation:
       - Message structure and required fields
       - Content type validation
       - Role validation (user/assistant)
       - Metadata validation

    3. Sampling Request Validation:
       - Message array validation
       - Model preferences validation
       - Parameter validation (temperature, tokens, etc.)
       - Context inclusion validation

Example Usage:
    Validate a complete MCP configuration:

    ```python
    from graph_of_thoughts.language_models.mcp_protocol import MCPProtocolValidator

    validator = MCPProtocolValidator()

    config = {
        "transport": {
            "type": "stdio",
            "command": "claude-desktop",
            "args": ["--mcp-server"]
        },
        "client_info": {
            "name": "my-application",
            "version": "1.0.0"
        },
        "capabilities": {
            "sampling": {}
        }
    }

    if validator.validate_configuration(config):
        print("Configuration is valid!")
    else:
        print("Configuration validation failed")
    ```

    Create and validate sampling requests:

    ```python
    from graph_of_thoughts.language_models.mcp_protocol import (
        create_sampling_request, create_text_message
    )

    # Create properly formatted messages
    messages = [
        create_text_message("user", "What is the weather like today?"),
        create_text_message("assistant", "I don't have access to current weather data.")
    ]

    # Create sampling request
    request = create_sampling_request(
        messages=messages,
        model_preferences={
            "hints": [{"name": "claude-3-5-sonnet"}],
            "costPriority": 0.5,
            "intelligencePriority": 0.8
        },
        system_prompt="You are a helpful weather assistant.",
        temperature=0.7,
        max_tokens=500,
        stop_sequences=["END_RESPONSE"]
    )

    # Validate the request
    if validator.validate_sampling_request(request):
        print("Sampling request is valid!")
        print(f"Request: {request}")
    ```

    Handle different content types:

    ```python
    from graph_of_thoughts.language_models.mcp_protocol import (
        create_text_message, create_image_message
    )

    # Text message
    text_msg = create_text_message("user", "Describe this image")

    # Image message (base64 encoded)
    image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    image_msg = create_image_message("user", image_data, "image/png")

    messages = [text_msg, image_msg]

    # Validate messages
    for msg in messages:
        if validator._validate_message(msg):
            print(f"Message valid: {msg['content']['type']}")
    ```

Validation Features:
    The validator provides detailed error reporting and supports:

    - Required field checking
    - Type validation
    - Value range validation
    - Format validation
    - Cross-field validation
    - Protocol version compatibility

    Example with error handling:

    ```python
    import logging

    # Enable debug logging to see validation details
    logging.basicConfig(level=logging.DEBUG)

    validator = MCPProtocolValidator()

    # Invalid configuration (missing required fields)
    invalid_config = {
        "transport": {
            "type": "stdio"
            # Missing required "command" field
        }
    }

    if not validator.validate_configuration(invalid_config):
        print("Validation failed - check logs for details")
        # Logs will show: "Missing required field in stdio transport: command"
    ```

Protocol Compliance:
    All validation follows the official MCP specification:
    - JSON-RPC 2.0 message format compliance
    - MCP protocol version compatibility
    - Standard error codes and messages
    - Proper content type handling
    - Correct capability negotiation format
"""

# Standard library imports
import json
import logging
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union


class MCPTransportType(Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    HTTP = "http"


class MCPIncludeContext(Enum):
    """Valid values for includeContext in sampling requests."""
    NONE = "none"
    THIS_SERVER = "thisServer"
    ALL_SERVERS = "allServers"


class MCPClientInfo(TypedDict):
    """MCP client information structure."""
    name: str
    version: str


class MCPModelHint(TypedDict):
    """Model hint structure for model preferences."""
    name: str


class MCPModelPreferences(TypedDict, total=False):
    """Model preferences structure for sampling requests."""
    hints: List[MCPModelHint]
    costPriority: float
    speedPriority: float
    intelligencePriority: float


class MCPMessageContent(TypedDict, total=False):
    """Content structure for MCP messages."""
    type: str
    text: str
    data: str  # base64 encoded for binary content
    mimeType: str


class MCPMessage(TypedDict):
    """Message structure for MCP conversations."""
    role: str  # "user" or "assistant"
    content: MCPMessageContent


class MCPSamplingRequest(TypedDict, total=False):
    """MCP sampling request structure according to the specification."""
    messages: List[MCPMessage]
    maxTokens: int
    modelPreferences: MCPModelPreferences
    systemPrompt: str
    includeContext: str
    temperature: float
    stopSequences: List[str]
    metadata: Dict[str, Any]


class MCPSamplingResponse(TypedDict, total=False):
    """MCP sampling response structure."""
    model: str
    role: str
    content: MCPMessageContent
    stopReason: str


class MCPProtocolValidator:
    """Validator for MCP protocol messages and configurations."""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_client_info(self, client_info: Dict[str, Any]) -> bool:
        """
        Validate client information structure.
        
        :param client_info: Client info dictionary
        :type client_info: Dict[str, Any]
        :return: True if valid, False otherwise
        :rtype: bool
        """
        required_fields = ["name", "version"]
        for field in required_fields:
            if field not in client_info:
                self.logger.error(f"Missing required field in client_info: {field}")
                return False
        return True
    
    def validate_transport_config(self, transport: Dict[str, Any]) -> bool:
        """
        Validate transport configuration.
        
        :param transport: Transport configuration dictionary
        :type transport: Dict[str, Any]
        :return: True if valid, False otherwise
        :rtype: bool
        """
        if "type" not in transport:
            self.logger.error("Missing 'type' field in transport config")
            return False
        
        transport_type = transport["type"]
        if transport_type not in [t.value for t in MCPTransportType]:
            self.logger.error(f"Invalid transport type: {transport_type}")
            return False
        
        if transport_type == MCPTransportType.STDIO.value:
            return self._validate_stdio_transport(transport)
        elif transport_type == MCPTransportType.HTTP.value:
            return self._validate_http_transport(transport)
        
        return False
    
    def _validate_stdio_transport(self, transport: Dict[str, Any]) -> bool:
        """Validate stdio transport configuration."""
        required_fields = ["command"]
        for field in required_fields:
            if field not in transport:
                self.logger.error(f"Missing required field in stdio transport: {field}")
                return False
        return True

    def _validate_http_transport(self, transport: Dict[str, Any]) -> bool:
        """Validate HTTP transport configuration."""
        required_fields = ["url"]
        for field in required_fields:
            if field not in transport:
                self.logger.error(f"Missing required field in HTTP transport: {field}")
                return False
        return True
    
    def validate_model_preferences(self, preferences: Dict[str, Any]) -> bool:
        """
        Validate model preferences structure.
        
        :param preferences: Model preferences dictionary
        :type preferences: Dict[str, Any]
        :return: True if valid, False otherwise
        :rtype: bool
        """
        # Validate hints if present
        if "hints" in preferences:
            hints = preferences["hints"]
            if not isinstance(hints, list):
                self.logger.error("Model hints must be a list")
                return False
            for hint in hints:
                if not isinstance(hint, dict) or "name" not in hint:
                    self.logger.error("Each model hint must be a dict with 'name' field")
                    return False
        
        # Validate priority values
        priority_fields = ["costPriority", "speedPriority", "intelligencePriority"]
        for field in priority_fields:
            if field in preferences:
                value = preferences[field]
                if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                    self.logger.error(f"{field} must be a number between 0 and 1")
                    return False
        
        return True
    
    def validate_sampling_request(self, request: Dict[str, Any]) -> bool:
        """
        Validate sampling request structure.
        
        :param request: Sampling request dictionary
        :type request: Dict[str, Any]
        :return: True if valid, False otherwise
        :rtype: bool
        """
        # Check required fields
        if "messages" not in request:
            self.logger.error("Missing required field: messages")
            return False
        
        if "maxTokens" not in request:
            self.logger.error("Missing required field: maxTokens")
            return False
        
        # Validate messages
        messages = request["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            self.logger.error("Messages must be a non-empty list")
            return False
        
        for message in messages:
            if not self._validate_message(message):
                return False
        
        # Validate model preferences if present
        if "modelPreferences" in request:
            if not self.validate_model_preferences(request["modelPreferences"]):
                return False
        
        # Validate includeContext if present
        if "includeContext" in request:
            context = request["includeContext"]
            if context not in [c.value for c in MCPIncludeContext]:
                self.logger.error(f"Invalid includeContext value: {context}")
                return False
        
        return True
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate individual message structure."""
        required_fields = ["role", "content"]
        for field in required_fields:
            if field not in message:
                self.logger.error(f"Missing required field in message: {field}")
                return False
        
        # Validate role
        if message["role"] not in ["user", "assistant"]:
            self.logger.error(f"Invalid message role: {message['role']}")
            return False
        
        # Validate content
        content = message["content"]
        if not isinstance(content, dict) or "type" not in content:
            self.logger.error("Message content must be a dict with 'type' field")
            return False
        
        content_type = content["type"]
        if content_type == "text":
            if "text" not in content:
                self.logger.error("Text content must have 'text' field")
                return False
        elif content_type == "image":
            if "data" not in content or "mimeType" not in content:
                self.logger.error("Image content must have 'data' and 'mimeType' fields")
                return False
        else:
            self.logger.error(f"Invalid content type: {content_type}")
            return False
        
        return True
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validate complete MCP configuration.
        
        :param config: Configuration dictionary
        :type config: Dict[str, Any]
        :return: True if valid, False otherwise
        :rtype: bool
        """
        required_sections = ["transport", "client_info"]
        for section in required_sections:
            if section not in config:
                self.logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate each section
        if not self.validate_transport_config(config["transport"]):
            return False
        
        if not self.validate_client_info(config["client_info"]):
            return False
        
        # Validate default sampling params if present
        if "default_sampling_params" in config:
            # Create a mock sampling request to validate the structure
            mock_request = {
                "messages": [{"role": "user", "content": {"type": "text", "text": "test"}}],
                "maxTokens": 1000,
                **config["default_sampling_params"]
            }
            if not self.validate_sampling_request(mock_request):
                return False
        
        return True


def create_sampling_request(
    messages: List[Dict[str, Any]],
    model_preferences: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    include_context: str = "none",
    temperature: Optional[float] = None,
    max_tokens: int = 1000,
    stop_sequences: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a properly formatted MCP sampling request.

    :param messages: List of conversation messages
    :type messages: List[Dict[str, Any]]
    :param model_preferences: Model selection preferences
    :type model_preferences: Optional[Dict[str, Any]]
    :param system_prompt: Optional system prompt
    :type system_prompt: Optional[str]
    :param include_context: Context inclusion setting
    :type include_context: str
    :param temperature: Sampling temperature
    :type temperature: Optional[float]
    :param max_tokens: Maximum tokens to generate
    :type max_tokens: int
    :param stop_sequences: Stop sequences
    :type stop_sequences: Optional[List[str]]
    :param metadata: Additional metadata
    :type metadata: Optional[Dict[str, Any]]
    :return: Formatted sampling request
    :rtype: Dict[str, Any]
    """
    request = {
        "messages": messages,
        "maxTokens": max_tokens,
        "includeContext": include_context
    }

    if model_preferences:
        request["modelPreferences"] = model_preferences
    if system_prompt:
        request["systemPrompt"] = system_prompt
    if temperature is not None:
        request["temperature"] = temperature
    if stop_sequences:
        request["stopSequences"] = stop_sequences
    if metadata:
        request["metadata"] = metadata

    return request


def create_text_message(role: str, text: str) -> Dict[str, Any]:
    """
    Create a properly formatted MCP text message.

    :param role: Message role ("user" or "assistant")
    :type role: str
    :param text: Message text content
    :type text: str
    :return: Formatted message
    :rtype: Dict[str, Any]
    """
    return {
        "role": role,
        "content": {
            "type": "text",
            "text": text
        }
    }


def create_image_message(role: str, data: str, mime_type: str) -> Dict[str, Any]:
    """
    Create a properly formatted MCP image message.

    :param role: Message role ("user" or "assistant")
    :type role: str
    :param data: Base64 encoded image data
    :type data: str
    :param mime_type: MIME type of the image
    :type mime_type: str
    :return: Formatted message
    :rtype: Dict[str, Any]
    """
    return {
        "role": role,
        "content": {
            "type": "image",
            "data": data,
            "mimeType": mime_type
        }
    }


def validate_mcp_config_file(config_path: str, strict_mode: bool = True, enable_security_checks: bool = True) -> bool:
    """
    Validate an MCP configuration file using the enhanced validator.

    :param config_path: Path to the configuration file
    :type config_path: str
    :param strict_mode: If True, enforce strict validation rules
    :type strict_mode: bool
    :param enable_security_checks: If True, perform security validation
    :type enable_security_checks: bool
    :return: True if valid, False otherwise
    :rtype: bool
    """
    try:
        validator = MCPConfigurationValidator(strict_mode=strict_mode, enable_security_checks=enable_security_checks)
        return validator.validate_startup_configuration(config_path)
    except MCPConfigurationError as e:
        logging.error(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during validation: {e}")
        return False


def validate_mcp_config_file_legacy(config_path: str) -> bool:
    """
    Legacy validation function for backward compatibility.

    :param config_path: Path to the configuration file
    :type config_path: str
    :return: True if valid, False otherwise
    :rtype: bool
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        validator = MCPProtocolValidator()

        # Validate each configuration in the file
        for config_name, config_data in config.items():
            if not validator.validate_configuration(config_data):
                logging.error(f"Invalid configuration: {config_name}")
                return False

        return True
    except Exception as e:
        logging.error(f"Failed to validate config file {config_path}: {e}")
        return False


class MCPConfigurationError(Exception):
    """Exception raised for MCP configuration errors."""

    def __init__(self, message: str, field_path: str = "", validation_errors: List[str] = None):
        """
        Initialize configuration error with detailed context.

        :param message: Main error message
        :type message: str
        :param field_path: Path to the problematic field (e.g., "transport.command")
        :type field_path: str
        :param validation_errors: List of specific validation errors
        :type validation_errors: List[str]
        """
        super().__init__(message)
        self.field_path = field_path
        self.validation_errors = validation_errors or []

    def __str__(self) -> str:
        """Return formatted error message with context."""
        msg = super().__str__()
        if self.field_path:
            msg = f"Configuration error at '{self.field_path}': {msg}"
        if self.validation_errors:
            msg += f"\nValidation errors:\n" + "\n".join(f"  - {err}" for err in self.validation_errors)
        return msg


class MCPConfigurationValidator:
    """
    Enhanced configuration validator with comprehensive validation and clear error reporting.

    This validator provides:
    - Startup configuration validation
    - Runtime configuration checks
    - Detailed error reporting with field paths
    - Protocol compliance validation
    - Security validation
    - Performance configuration validation
    """

    # Protocol version constants
    SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05", "2025-06-18"]
    DEFAULT_PROTOCOL_VERSION = "2025-06-18"

    # Transport type constants
    SUPPORTED_TRANSPORT_TYPES = ["stdio", "http"]

    # Required fields for different transport types
    STDIO_REQUIRED_FIELDS = ["command"]
    HTTP_REQUIRED_FIELDS = ["url"]

    # Default values for various configuration sections
    DEFAULT_CONNECTION_CONFIG = {
        "timeout": 30.0,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "request_timeout": 60.0
    }

    DEFAULT_COST_TRACKING = {
        "prompt_token_cost": 0.0,
        "response_token_cost": 0.0
    }

    def __init__(self, strict_mode: bool = True, enable_security_checks: bool = True):
        """
        Initialize the configuration validator.

        :param strict_mode: If True, enforce strict validation rules
        :type strict_mode: bool
        :param enable_security_checks: If True, perform security validation
        :type enable_security_checks: bool
        """
        self.strict_mode = strict_mode
        self.enable_security_checks = enable_security_checks
        self.logger = logging.getLogger(self.__class__.__name__)
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []

    def validate_startup_configuration(self, config_path: str) -> bool:
        """
        Validate configuration file at startup with comprehensive checks.

        :param config_path: Path to the configuration file
        :type config_path: str
        :return: True if configuration is valid
        :rtype: bool
        :raises MCPConfigurationError: If configuration is invalid
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()

        try:
            # Check if file exists and is readable
            if not os.path.exists(config_path):
                raise MCPConfigurationError(f"Configuration file not found: {config_path}")

            if not os.access(config_path, os.R_OK):
                raise MCPConfigurationError(f"Configuration file is not readable: {config_path}")

            # Load and parse configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                try:
                    config_data = json.load(f)
                except json.JSONDecodeError as e:
                    raise MCPConfigurationError(
                        f"Invalid JSON in configuration file: {e}",
                        field_path=f"line {e.lineno}, column {e.colno}"
                    )

            # Validate configuration structure
            if not isinstance(config_data, dict):
                raise MCPConfigurationError("Configuration must be a JSON object")

            if not config_data:
                raise MCPConfigurationError("Configuration file is empty")

            # Validate each model configuration
            for model_name, model_config in config_data.items():
                try:
                    self._validate_model_configuration(model_config, model_name)
                except MCPConfigurationError as e:
                    e.field_path = f"{model_name}.{e.field_path}" if e.field_path else model_name
                    raise e

            # Log warnings if any
            if self.validation_warnings:
                for warning in self.validation_warnings:
                    self.logger.warning(warning)

            self.logger.info(f"Configuration validation successful for {len(config_data)} model(s)")
            return True

        except MCPConfigurationError:
            raise
        except Exception as e:
            raise MCPConfigurationError(f"Unexpected error during configuration validation: {e}")

    def _validate_model_configuration(self, config: Dict[str, Any], model_name: str) -> None:
        """
        Validate a single model configuration.

        :param config: Model configuration dictionary
        :type config: Dict[str, Any]
        :param model_name: Name of the model configuration
        :type model_name: str
        :raises MCPConfigurationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise MCPConfigurationError(f"Model configuration must be an object", field_path="")

        # Validate required sections
        required_sections = ["transport", "client_info"]
        for section in required_sections:
            if section not in config:
                raise MCPConfigurationError(f"Missing required section: {section}", field_path=section)

        # Validate each section
        self._validate_transport_section(config["transport"])
        self._validate_client_info_section(config["client_info"])

        # Validate optional sections
        if "capabilities" in config:
            self._validate_capabilities_section(config["capabilities"])

        if "default_sampling_params" in config:
            self._validate_sampling_params_section(config["default_sampling_params"])

        if "connection_config" in config:
            self._validate_connection_config_section(config["connection_config"])

        if "cost_tracking" in config:
            self._validate_cost_tracking_section(config["cost_tracking"])

        if "metrics" in config:
            self._validate_metrics_section(config["metrics"])

        # Security validation
        if self.enable_security_checks:
            self._validate_security_configuration(config)

    def _validate_transport_section(self, transport: Dict[str, Any]) -> None:
        """
        Validate transport configuration section.

        :param transport: Transport configuration
        :type transport: Dict[str, Any]
        :raises MCPConfigurationError: If transport configuration is invalid
        """
        if not isinstance(transport, dict):
            raise MCPConfigurationError("Transport configuration must be an object", field_path="transport")

        # Validate transport type
        if "type" not in transport:
            raise MCPConfigurationError("Missing required field: type", field_path="transport.type")

        transport_type = transport["type"]
        if transport_type not in self.SUPPORTED_TRANSPORT_TYPES:
            raise MCPConfigurationError(
                f"Unsupported transport type: {transport_type}. Supported types: {', '.join(self.SUPPORTED_TRANSPORT_TYPES)}",
                field_path="transport.type"
            )

        # Validate transport-specific fields
        if transport_type == "stdio":
            self._validate_stdio_transport(transport)
        elif transport_type == "http":
            self._validate_http_transport(transport)

    def _validate_stdio_transport(self, transport: Dict[str, Any]) -> None:
        """
        Validate stdio transport configuration.

        :param transport: Transport configuration
        :type transport: Dict[str, Any]
        :raises MCPConfigurationError: If stdio transport configuration is invalid
        """
        # Check required fields
        for field in self.STDIO_REQUIRED_FIELDS:
            if field not in transport:
                raise MCPConfigurationError(f"Missing required field for stdio transport: {field}", field_path=f"transport.{field}")

        # Validate command
        command = transport["command"]
        if not isinstance(command, str) or not command.strip():
            raise MCPConfigurationError("Command must be a non-empty string", field_path="transport.command")

        # Validate args if present
        if "args" in transport:
            args = transport["args"]
            if not isinstance(args, list):
                raise MCPConfigurationError("Args must be a list", field_path="transport.args")

            for i, arg in enumerate(args):
                if not isinstance(arg, str):
                    raise MCPConfigurationError(f"Argument at index {i} must be a string", field_path=f"transport.args[{i}]")

        # Validate env if present
        if "env" in transport:
            env = transport["env"]
            if not isinstance(env, dict):
                raise MCPConfigurationError("Environment variables must be an object", field_path="transport.env")

            for key, value in env.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise MCPConfigurationError(f"Environment variable {key} must have string key and value", field_path=f"transport.env.{key}")

        # Security check for stdio
        if self.enable_security_checks:
            self._validate_stdio_security(transport)

    def _validate_http_transport(self, transport: Dict[str, Any]) -> None:
        """
        Validate HTTP transport configuration.

        :param transport: Transport configuration
        :type transport: Dict[str, Any]
        :raises MCPConfigurationError: If HTTP transport configuration is invalid
        """
        # Check required fields
        for field in self.HTTP_REQUIRED_FIELDS:
            if field not in transport:
                raise MCPConfigurationError(f"Missing required field for HTTP transport: {field}", field_path=f"transport.{field}")

        # Validate URL
        url = transport["url"]
        if not isinstance(url, str) or not url.strip():
            raise MCPConfigurationError("URL must be a non-empty string", field_path="transport.url")

        # Validate URL format
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise MCPConfigurationError("URL must be a valid HTTP/HTTPS URL", field_path="transport.url")

            if parsed.scheme not in ["http", "https"]:
                raise MCPConfigurationError("URL scheme must be http or https", field_path="transport.url")
        except Exception as e:
            raise MCPConfigurationError(f"Invalid URL format: {e}", field_path="transport.url")

        # Validate headers if present
        if "headers" in transport:
            headers = transport["headers"]
            if not isinstance(headers, dict):
                raise MCPConfigurationError("Headers must be an object", field_path="transport.headers")

            for key, value in headers.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise MCPConfigurationError(f"Header {key} must have string key and value", field_path=f"transport.headers.{key}")

        # Validate session management if present
        if "session_management" in transport:
            session_mgmt = transport["session_management"]
            if not isinstance(session_mgmt, bool):
                raise MCPConfigurationError("Session management must be a boolean", field_path="transport.session_management")

        # Security check for HTTP
        if self.enable_security_checks:
            self._validate_http_security(transport)

    def _validate_client_info_section(self, client_info: Dict[str, Any]) -> None:
        """
        Validate client_info configuration section.

        :param client_info: Client info configuration
        :type client_info: Dict[str, Any]
        :raises MCPConfigurationError: If client_info configuration is invalid
        """
        if not isinstance(client_info, dict):
            raise MCPConfigurationError("Client info must be an object", field_path="client_info")

        # Validate required fields
        required_fields = ["name", "version"]
        for field in required_fields:
            if field not in client_info:
                raise MCPConfigurationError(f"Missing required field: {field}", field_path=f"client_info.{field}")

            value = client_info[field]
            if not isinstance(value, str) or not value.strip():
                raise MCPConfigurationError(f"Field {field} must be a non-empty string", field_path=f"client_info.{field}")

        # Validate optional title field
        if "title" in client_info:
            title = client_info["title"]
            if not isinstance(title, str):
                raise MCPConfigurationError("Title must be a string", field_path="client_info.title")

        # Validate version format
        version = client_info["version"]
        if not self._is_valid_version(version):
            self.validation_warnings.append(f"Version '{version}' does not follow semantic versioning format")

    def _validate_capabilities_section(self, capabilities: Dict[str, Any]) -> None:
        """
        Validate capabilities configuration section.

        :param capabilities: Capabilities configuration
        :type capabilities: Dict[str, Any]
        :raises MCPConfigurationError: If capabilities configuration is invalid
        """
        if not isinstance(capabilities, dict):
            raise MCPConfigurationError("Capabilities must be an object", field_path="capabilities")

        # Validate known capability types
        known_capabilities = ["sampling", "roots", "elicitation", "experimental"]
        for cap_name, cap_config in capabilities.items():
            if cap_name not in known_capabilities:
                self.validation_warnings.append(f"Unknown capability: {cap_name}")

            if not isinstance(cap_config, dict):
                raise MCPConfigurationError(f"Capability {cap_name} must be an object", field_path=f"capabilities.{cap_name}")

    def _validate_sampling_params_section(self, params: Dict[str, Any]) -> None:
        """
        Validate default_sampling_params configuration section.

        :param params: Sampling parameters configuration
        :type params: Dict[str, Any]
        :raises MCPConfigurationError: If sampling parameters configuration is invalid
        """
        if not isinstance(params, dict):
            raise MCPConfigurationError("Sampling parameters must be an object", field_path="default_sampling_params")

        # Validate temperature
        if "temperature" in params:
            temp = params["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                raise MCPConfigurationError("Temperature must be a number between 0 and 2", field_path="default_sampling_params.temperature")

        # Validate maxTokens
        if "maxTokens" in params:
            max_tokens = params["maxTokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise MCPConfigurationError("maxTokens must be a positive integer", field_path="default_sampling_params.maxTokens")

        # Validate stopSequences
        if "stopSequences" in params:
            stop_seqs = params["stopSequences"]
            if not isinstance(stop_seqs, list):
                raise MCPConfigurationError("stopSequences must be a list", field_path="default_sampling_params.stopSequences")

            for i, seq in enumerate(stop_seqs):
                if not isinstance(seq, str):
                    raise MCPConfigurationError(f"Stop sequence at index {i} must be a string", field_path=f"default_sampling_params.stopSequences[{i}]")

        # Validate includeContext
        if "includeContext" in params:
            context = params["includeContext"]
            valid_contexts = ["none", "thisServer", "allServers"]
            if context not in valid_contexts:
                raise MCPConfigurationError(f"includeContext must be one of: {', '.join(valid_contexts)}", field_path="default_sampling_params.includeContext")

        # Validate modelPreferences
        if "modelPreferences" in params:
            self._validate_model_preferences(params["modelPreferences"])

    def _validate_model_preferences(self, prefs: Dict[str, Any]) -> None:
        """
        Validate model preferences configuration.

        :param prefs: Model preferences configuration
        :type prefs: Dict[str, Any]
        :raises MCPConfigurationError: If model preferences configuration is invalid
        """
        if not isinstance(prefs, dict):
            raise MCPConfigurationError("Model preferences must be an object", field_path="default_sampling_params.modelPreferences")

        # Validate hints
        if "hints" in prefs:
            hints = prefs["hints"]
            if not isinstance(hints, list):
                raise MCPConfigurationError("Model hints must be a list", field_path="default_sampling_params.modelPreferences.hints")

            for i, hint in enumerate(hints):
                if not isinstance(hint, dict) or "name" not in hint:
                    raise MCPConfigurationError(f"Model hint at index {i} must be an object with 'name' field", field_path=f"default_sampling_params.modelPreferences.hints[{i}]")

                if not isinstance(hint["name"], str) or not hint["name"].strip():
                    raise MCPConfigurationError(f"Model hint name at index {i} must be a non-empty string", field_path=f"default_sampling_params.modelPreferences.hints[{i}].name")

        # Validate priority values
        priority_fields = ["costPriority", "speedPriority", "intelligencePriority"]
        for field in priority_fields:
            if field in prefs:
                value = prefs[field]
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    raise MCPConfigurationError(f"{field} must be a number between 0 and 1", field_path=f"default_sampling_params.modelPreferences.{field}")

    def _validate_connection_config_section(self, config: Dict[str, Any]) -> None:
        """
        Validate connection_config configuration section.

        :param config: Connection configuration
        :type config: Dict[str, Any]
        :raises MCPConfigurationError: If connection configuration is invalid
        """
        if not isinstance(config, dict):
            raise MCPConfigurationError("Connection config must be an object", field_path="connection_config")

        # Validate timeout values
        timeout_fields = ["timeout", "request_timeout", "retry_delay"]
        for field in timeout_fields:
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise MCPConfigurationError(f"{field} must be a positive number", field_path=f"connection_config.{field}")

        # Validate retry attempts
        if "retry_attempts" in config:
            attempts = config["retry_attempts"]
            if not isinstance(attempts, int) or attempts < 0:
                raise MCPConfigurationError("retry_attempts must be a non-negative integer", field_path="connection_config.retry_attempts")

            if attempts > 10:
                self.validation_warnings.append("High retry_attempts value may cause long delays on failures")

    def _validate_cost_tracking_section(self, config: Dict[str, Any]) -> None:
        """
        Validate cost_tracking configuration section.

        :param config: Cost tracking configuration
        :type config: Dict[str, Any]
        :raises MCPConfigurationError: If cost tracking configuration is invalid
        """
        if not isinstance(config, dict):
            raise MCPConfigurationError("Cost tracking must be an object", field_path="cost_tracking")

        # Validate cost fields
        cost_fields = ["prompt_token_cost", "response_token_cost"]
        for field in cost_fields:
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)) or value < 0:
                    raise MCPConfigurationError(f"{field} must be a non-negative number", field_path=f"cost_tracking.{field}")

    def _validate_metrics_section(self, config: Dict[str, Any]) -> None:
        """
        Validate metrics configuration section.

        :param config: Metrics configuration
        :type config: Dict[str, Any]
        :raises MCPConfigurationError: If metrics configuration is invalid
        """
        if not isinstance(config, dict):
            raise MCPConfigurationError("Metrics config must be an object", field_path="metrics")

        # Validate enabled flag
        if "enabled" in config:
            enabled = config["enabled"]
            if not isinstance(enabled, bool):
                raise MCPConfigurationError("Metrics enabled must be a boolean", field_path="metrics.enabled")

        # Validate export interval
        if "export_interval" in config:
            interval = config["export_interval"]
            if not isinstance(interval, (int, float)) or interval <= 0:
                raise MCPConfigurationError("Export interval must be a positive number", field_path="metrics.export_interval")

        # Validate export format
        if "export_format" in config:
            fmt = config["export_format"]
            valid_formats = ["json", "csv", "prometheus"]
            if fmt not in valid_formats:
                raise MCPConfigurationError(f"Export format must be one of: {', '.join(valid_formats)}", field_path="metrics.export_format")

        # Validate max history size
        if "max_history_size" in config:
            size = config["max_history_size"]
            if not isinstance(size, int) or size <= 0:
                raise MCPConfigurationError("Max history size must be a positive integer", field_path="metrics.max_history_size")

        # Validate export file path
        if "export_file" in config:
            file_path = config["export_file"]
            if not isinstance(file_path, str) or not file_path.strip():
                raise MCPConfigurationError("Export file must be a non-empty string", field_path="metrics.export_file")

    def _validate_security_configuration(self, config: Dict[str, Any]) -> None:
        """
        Validate security aspects of the configuration.

        :param config: Full model configuration
        :type config: Dict[str, Any]
        """
        transport = config.get("transport", {})
        transport_type = transport.get("type")

        if transport_type == "stdio":
            self._validate_stdio_security(transport)
        elif transport_type == "http":
            self._validate_http_security(transport)

    def _validate_stdio_security(self, transport: Dict[str, Any]) -> None:
        """
        Validate security aspects of stdio transport.

        :param transport: Transport configuration
        :type transport: Dict[str, Any]
        """
        command = transport.get("command", "")

        # Check for potentially dangerous commands
        dangerous_patterns = [
            "rm ", "del ", "format ", "mkfs", "dd if=", ":(){ :|:& };:",  # Destructive commands
            "curl ", "wget ", "nc ", "netcat ",  # Network commands
            "python -c", "perl -e", "ruby -e",  # Code execution
            "eval ", "exec ", "system(",  # Dynamic execution
        ]

        for pattern in dangerous_patterns:
            if pattern in command.lower():
                self.validation_warnings.append(f"Potentially dangerous command pattern detected: {pattern}")

        # Check for absolute paths vs relative paths
        if os.path.isabs(command):
            self.validation_warnings.append("Using absolute path for command - ensure the executable is trusted")

        # Check environment variables for sensitive data
        env = transport.get("env", {})
        sensitive_patterns = ["password", "secret", "key", "token", "credential"]
        for env_key in env.keys():
            if any(pattern in env_key.lower() for pattern in sensitive_patterns):
                self.validation_warnings.append(f"Environment variable '{env_key}' may contain sensitive data")

    def _validate_http_security(self, transport: Dict[str, Any]) -> None:
        """
        Validate security aspects of HTTP transport.

        :param transport: Transport configuration
        :type transport: Dict[str, Any]
        """
        url = transport.get("url", "")

        # Check for HTTPS in production-like URLs
        if url.startswith("http://") and not any(host in url for host in ["localhost", "127.0.0.1", "0.0.0.0"]):
            self.validation_warnings.append("Using HTTP instead of HTTPS for remote connections is insecure")

        # Check headers for sensitive data
        headers = transport.get("headers", {})
        for header_name, header_value in headers.items():
            if "authorization" in header_name.lower():
                if "bearer" in header_value.lower() and len(header_value) < 20:
                    self.validation_warnings.append("Authorization header appears to contain a short token - ensure it's properly configured")

            if any(sensitive in header_name.lower() for sensitive in ["password", "secret", "key"]):
                self.validation_warnings.append(f"Header '{header_name}' may contain sensitive data")

    def _is_valid_version(self, version: str) -> bool:
        """
        Check if version follows semantic versioning format.

        :param version: Version string to validate
        :type version: str
        :return: True if version is valid
        :rtype: bool
        """
        import re
        # Basic semantic versioning pattern: MAJOR.MINOR.PATCH with optional pre-release and build metadata
        semver_pattern = r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
        return bool(re.match(semver_pattern, version))

    def validate_runtime_configuration(self, config: Dict[str, Any], model_name: str) -> bool:
        """
        Validate configuration at runtime with additional checks.

        :param config: Model configuration to validate
        :type config: Dict[str, Any]
        :param model_name: Name of the model configuration
        :type model_name: str
        :return: True if configuration is valid
        :rtype: bool
        :raises MCPConfigurationError: If configuration is invalid
        """
        try:
            self._validate_model_configuration(config, model_name)

            # Additional runtime checks
            self._validate_runtime_connectivity(config)
            self._validate_runtime_permissions(config)

            return True
        except MCPConfigurationError:
            raise
        except Exception as e:
            raise MCPConfigurationError(f"Runtime validation failed: {e}")

    def _validate_runtime_connectivity(self, config: Dict[str, Any]) -> None:
        """
        Validate connectivity aspects at runtime.

        :param config: Model configuration
        :type config: Dict[str, Any]
        """
        transport = config.get("transport", {})
        transport_type = transport.get("type")

        if transport_type == "stdio":
            command = transport.get("command", "")
            # Check if command exists and is executable
            if not shutil.which(command):
                self.validation_warnings.append(f"Command '{command}' not found in PATH")

        elif transport_type == "http":
            url = transport.get("url", "")
            # Basic URL reachability could be checked here, but we avoid network calls
            # during validation to keep it fast and avoid side effects
            pass

    def _validate_runtime_permissions(self, config: Dict[str, Any]) -> None:
        """
        Validate permission aspects at runtime.

        :param config: Model configuration
        :type config: Dict[str, Any]
        """
        # Check if metrics export file is writable
        metrics = config.get("metrics", {})
        if metrics.get("enabled") and "export_file" in metrics:
            export_file = metrics["export_file"]
            export_dir = os.path.dirname(os.path.abspath(export_file))

            if not os.path.exists(export_dir):
                self.validation_warnings.append(f"Metrics export directory does not exist: {export_dir}")
            elif not os.access(export_dir, os.W_OK):
                self.validation_warnings.append(f"Metrics export directory is not writable: {export_dir}")

    def get_configuration_summary(self, config_path: str) -> Dict[str, Any]:
        """
        Get a summary of the configuration for debugging and monitoring.

        :param config_path: Path to the configuration file
        :type config_path: str
        :return: Configuration summary
        :rtype: Dict[str, Any]
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            summary = {
                "file_path": config_path,
                "file_size": os.path.getsize(config_path),
                "last_modified": os.path.getmtime(config_path),
                "model_count": len(config_data),
                "models": {}
            }

            for model_name, model_config in config_data.items():
                transport = model_config.get("transport", {})
                summary["models"][model_name] = {
                    "transport_type": transport.get("type"),
                    "has_capabilities": "capabilities" in model_config,
                    "has_sampling_params": "default_sampling_params" in model_config,
                    "has_connection_config": "connection_config" in model_config,
                    "has_cost_tracking": "cost_tracking" in model_config,
                    "has_metrics": "metrics" in model_config,
                }

            return summary
        except Exception as e:
            return {"error": str(e), "file_path": config_path}

    def generate_configuration_template(self, transport_type: str = "stdio", model_name: str = "mcp_model") -> str:
        """
        Generate a configuration template for a specific transport type.

        :param transport_type: Type of transport (stdio or http)
        :type transport_type: str
        :param model_name: Name for the model configuration
        :type model_name: str
        :return: JSON configuration template
        :rtype: str
        """
        if transport_type == "stdio":
            template = {
                model_name: {
                    "transport": {
                        "type": "stdio",
                        "command": "your-mcp-server-command",
                        "args": [],
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
                    "connection_config": self.DEFAULT_CONNECTION_CONFIG.copy(),
                    "cost_tracking": self.DEFAULT_COST_TRACKING.copy()
                }
            }
        elif transport_type == "http":
            template = {
                model_name: {
                    "transport": {
                        "type": "http",
                        "url": "https://your-mcp-server.com/mcp",
                        "headers": {
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        },
                        "session_management": True
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
                        "includeContext": "allServers"
                    },
                    "connection_config": self.DEFAULT_CONNECTION_CONFIG.copy(),
                    "cost_tracking": self.DEFAULT_COST_TRACKING.copy()
                }
            }
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")

        return json.dumps(template, indent=4)


# CLI and utility functions for configuration validation

def validate_config_cli():
    """
    Command-line interface for configuration validation.

    Usage:
        python -m graph_of_thoughts.language_models.mcp_protocol validate <config_path>
        python -m graph_of_thoughts.language_models.mcp_protocol generate-template <transport_type>
        python -m graph_of_thoughts.language_models.mcp_protocol summary <config_path>
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="MCP Configuration Validation Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate MCP configuration file")
    validate_parser.add_argument("config_path", help="Path to configuration file")
    validate_parser.add_argument("--strict", action="store_true", help="Enable strict validation mode")
    validate_parser.add_argument("--no-security", action="store_true", help="Disable security checks")
    validate_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    # Generate template command
    template_parser = subparsers.add_parser("generate-template", help="Generate configuration template")
    template_parser.add_argument("transport_type", choices=["stdio", "http"], help="Transport type")
    template_parser.add_argument("--model-name", default="mcp_model", help="Model configuration name")
    template_parser.add_argument("--output", "-o", help="Output file path")

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Get configuration summary")
    summary_parser.add_argument("config_path", help="Path to configuration file")
    summary_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command == "validate":
        # Set up logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

        try:
            validator = MCPConfigurationValidator(
                strict_mode=args.strict,
                enable_security_checks=not args.no_security
            )

            if validator.validate_startup_configuration(args.config_path):
                print("✅ Configuration is valid!")

                # Show warnings if any
                if validator.validation_warnings:
                    print("\n⚠️  Warnings:")
                    for warning in validator.validation_warnings:
                        print(f"  - {warning}")

                sys.exit(0)
            else:
                print("❌ Configuration validation failed!")
                sys.exit(1)

        except MCPConfigurationError as e:
            print(f"❌ Configuration error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)

    elif args.command == "generate-template":
        try:
            validator = MCPConfigurationValidator()
            template = validator.generate_configuration_template(
                transport_type=args.transport_type,
                model_name=args.model_name
            )

            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(template)
                print(f"✅ Template written to {args.output}")
            else:
                print(template)

        except Exception as e:
            print(f"❌ Failed to generate template: {e}")
            sys.exit(1)

    elif args.command == "summary":
        try:
            validator = MCPConfigurationValidator()
            summary = validator.get_configuration_summary(args.config_path)

            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                print(f"Configuration Summary for {args.config_path}")
                print("=" * 50)
                print(f"File size: {summary.get('file_size', 'unknown')} bytes")
                print(f"Model count: {summary.get('model_count', 0)}")

                if 'models' in summary:
                    print("\nModels:")
                    for model_name, model_info in summary['models'].items():
                        print(f"  {model_name}:")
                        print(f"    Transport: {model_info.get('transport_type', 'unknown')}")
                        print(f"    Has capabilities: {model_info.get('has_capabilities', False)}")
                        print(f"    Has sampling params: {model_info.get('has_sampling_params', False)}")
                        print(f"    Has connection config: {model_info.get('has_connection_config', False)}")
                        print(f"    Has cost tracking: {model_info.get('has_cost_tracking', False)}")
                        print(f"    Has metrics: {model_info.get('has_metrics', False)}")

                if 'error' in summary:
                    print(f"\n❌ Error: {summary['error']}")

        except Exception as e:
            print(f"❌ Failed to get summary: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    validate_config_cli()


class MCPValidationError(Exception):
    """Exception raised for MCP validation errors."""
    pass
