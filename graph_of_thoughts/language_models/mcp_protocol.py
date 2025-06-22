# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Protocol validation and message formatting utilities.
This module provides utilities for validating MCP protocol messages and configurations
according to the official MCP specification.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class MCPTransportType(Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    HTTP = "http"


class MCPIncludeContext(Enum):
    """Valid values for includeContext in sampling requests."""
    NONE = "none"
    THIS_SERVER = "thisServer"
    ALL_SERVERS = "allServers"


@dataclass
class MCPClientInfo:
    """MCP client information structure."""
    name: str
    version: str


@dataclass
class MCPModelHint:
    """Model hint structure for model preferences."""
    name: str


@dataclass
class MCPModelPreferences:
    """Model preferences structure for sampling requests."""
    hints: Optional[List[MCPModelHint]] = None
    costPriority: Optional[float] = None
    speedPriority: Optional[float] = None
    intelligencePriority: Optional[float] = None


@dataclass
class MCPMessageContent:
    """Content structure for MCP messages."""
    type: str
    text: Optional[str] = None
    data: Optional[str] = None  # base64 encoded for binary content
    mimeType: Optional[str] = None


@dataclass
class MCPMessage:
    """Message structure for MCP conversations."""
    role: str  # "user" or "assistant"
    content: MCPMessageContent


@dataclass
class MCPSamplingRequest:
    """MCP sampling request structure according to the specification."""
    messages: List[MCPMessage]
    modelPreferences: Optional[MCPModelPreferences] = None
    systemPrompt: Optional[str] = None
    includeContext: Optional[str] = None
    temperature: Optional[float] = None
    maxTokens: int = 1000
    stopSequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPSamplingResponse:
    """MCP sampling response structure."""
    model: str
    stopReason: Optional[str] = None
    role: str = "assistant"
    content: MCPMessageContent = None


class MCPProtocolValidator:
    """Validator for MCP protocol messages and configurations."""
    
    def __init__(self):
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


def validate_mcp_config_file(config_path: str) -> bool:
    """
    Validate an MCP configuration file.

    :param config_path: Path to the configuration file
    :type config_path: str
    :return: True if valid, False otherwise
    :rtype: bool
    """
    try:
        import json
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
    pass


class MCPValidationError(Exception):
    """Exception raised for MCP validation errors."""
    pass
