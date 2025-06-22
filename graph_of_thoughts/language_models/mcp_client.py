# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

import asyncio
import backoff
import time
import random
from typing import List, Dict, Union, Any
from .abstract_language_model import AbstractLanguageModel
from .mcp_transport import create_transport, MCPTransport, MCPTransportError, MCPConnectionError
from .mcp_protocol import MCPProtocolValidator, create_sampling_request


class MCPLanguageModel(AbstractLanguageModel):
    """
    The MCPLanguageModel class handles interactions with language models through the Model Context Protocol (MCP).
    This implementation follows the official MCP specification for protocol compliance and proper message formatting.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "mcp_claude_desktop", cache: bool = False
    ) -> None:
        """
        Initialize the MCPLanguageModel instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model configuration, default is 'mcp_claude_desktop'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]

        # Validate configuration
        self.validator = MCPProtocolValidator()
        if not self.validator.validate_configuration(self.config):
            raise ValueError(f"Invalid MCP configuration for {model_name}")

        # Extract configuration sections
        self.transport_config: Dict = self.config["transport"]
        self.client_info: Dict = self.config["client_info"]
        self.capabilities: Dict = self.config["capabilities"]
        self.default_sampling_params: Dict = self.config.get("default_sampling_params", {})
        self.connection_config: Dict = self.config.get("connection_config", {})

        # Cost tracking (application-specific, not part of MCP protocol)
        cost_tracking = self.config.get("cost_tracking", {})
        self.prompt_token_cost: float = cost_tracking.get("prompt_token_cost", 0.0)
        self.response_token_cost: float = cost_tracking.get("response_token_cost", 0.0)

        # Initialize transport
        self.transport: MCPTransport = create_transport(self.config)
        self._connection_established = False

        # Legacy compatibility properties
        self.transport_type: str = self.transport_config.get("type", "stdio")
        self.host_type: str = self.transport_config.get("command", "unknown")  # For backward compatibility

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

    def _create_sampling_request(self, query: str, num_responses: int = 1) -> Dict[str, Any]:
        """
        Create a properly formatted MCP sampling request following the specification.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The sampling request
        :rtype: Dict[str, Any]
        """
        # Create messages in MCP format
        messages = [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": query
                }
            }
        ]

        # Use the protocol utility to create the request
        return create_sampling_request(
            messages=messages,
            model_preferences=self.default_sampling_params.get("modelPreferences"),
            system_prompt=self.default_sampling_params.get("systemPrompt"),
            include_context=self.default_sampling_params.get("includeContext", "none"),
            temperature=self.default_sampling_params.get("temperature"),
            max_tokens=self.default_sampling_params.get("maxTokens", 1000),
            stop_sequences=self.default_sampling_params.get("stopSequences"),
            metadata={
                "num_responses": num_responses,
                "source": "graph_of_thoughts",
                "client": self.client_info["name"]
            }
        )

    @backoff.on_exception(
        backoff.expo,
        (MCPTransportError, MCPConnectionError),
        max_time=10,
        max_tries=3
    )
    async def _send_sampling_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a sampling request to the MCP server with retry logic.

        :param request: The sampling request
        :type request: Dict[str, Any]
        :return: The response from the server
        :rtype: Dict[str, Any]
        """
        await self._ensure_connection()

        # Validate the request before sending
        if not self.validator.validate_sampling_request(request):
            raise ValueError("Invalid sampling request format")

        try:
            response = await self.transport.send_sampling_request(request)
            self.logger.debug("Received MCP sampling response")
            return response
        except Exception as e:
            self.logger.error(f"Failed to send sampling request: {e}")
            raise MCPTransportError(f"Sampling request failed: {e}")

    def query(self, query: str, num_responses: int = 1) -> Union[List[Dict], Dict]:
        """
        Query the MCP host for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the MCP host.
        :rtype: Union[List[Dict], Dict]
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        # Run the async query in a new event loop or existing one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if num_responses == 1:
            response = loop.run_until_complete(self._query_async(query, num_responses))
        else:
            # Handle multiple responses
            responses = []
            remaining_responses = num_responses
            total_attempts = num_responses
            
            while remaining_responses > 0 and total_attempts > 0:
                try:
                    batch_size = min(remaining_responses, 5)  # Limit batch size
                    batch_response = loop.run_until_complete(self._query_async(query, batch_size))
                    if isinstance(batch_response, list):
                        responses.extend(batch_response)
                    else:
                        responses.append(batch_response)
                    remaining_responses -= batch_size
                except Exception as e:
                    self.logger.warning(f"Error in MCP query: {e}, retrying with smaller batch")
                    batch_size = max(1, batch_size // 2)
                    time.sleep(random.randint(1, 3))
                    total_attempts -= 1
            
            response = responses

        if self.cache:
            self.response_cache[query] = response
        
        return response

    async def _query_async(self, query: str, num_responses: int = 1) -> Union[List[Dict], Dict]:
        """
        Async implementation of query.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the MCP host.
        :rtype: Union[List[Dict], Dict]
        """
        request = self._create_sampling_request(query, num_responses)
        
        if num_responses == 1:
            response = await self._send_sampling_request(request)
            self._update_token_usage(response)
            return response
        else:
            # For multiple responses, send multiple requests
            responses = []
            for _ in range(num_responses):
                response = await self._send_sampling_request(request)
                self._update_token_usage(response)
                responses.append(response)
            return responses

    def _update_token_usage(self, response: Dict[str, Any]) -> None:
        """
        Update token usage and cost tracking based on response.
        Note: This is application-specific functionality, not part of the MCP protocol.

        :param response: The response from the MCP server
        :type response: Dict[str, Any]
        """
        try:
            # Try to extract actual token usage from response metadata if available
            metadata = response.get("metadata", {})
            if "usage" in metadata:
                usage = metadata["usage"]
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
            else:
                # Fallback to estimation if no usage data available
                content = response.get("content", {})
                if isinstance(content, dict) and content.get("type") == "text":
                    text = content.get("text", "")
                    # Rough estimation: 1 token ≈ 4 characters
                    estimated_tokens = len(text) // 4
                    self.completion_tokens += estimated_tokens
                    # Estimate prompt tokens similarly
                    self.prompt_tokens += 50  # Rough estimate for prompt overhead

            # Update cost calculation
            prompt_tokens_k = float(self.prompt_tokens) / 1000.0
            completion_tokens_k = float(self.completion_tokens) / 1000.0
            self.cost = (
                self.prompt_token_cost * prompt_tokens_k
                + self.response_token_cost * completion_tokens_k
            )

            self.logger.debug(f"Token usage updated. Estimated cost: ${self.cost:.4f}")

        except Exception as e:
            self.logger.warning(f"Failed to update token usage: {e}")

    def get_response_texts(self, query_response: Union[List[Dict], Dict]) -> List[str]:
        """
        Extract the response texts from the query response following MCP response format.

        :param query_response: The response (or list of responses) from the MCP server.
        :type query_response: Union[List[Dict], Dict]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, list):
            query_response = [query_response]

        texts = []
        for response in query_response:
            try:
                # Handle MCP response format
                content = response.get("content", {})
                if isinstance(content, dict):
                    if content.get("type") == "text":
                        text = content.get("text", "")
                        texts.append(text)
                    elif content.get("type") == "image":
                        # For image content, return a description
                        mime_type = content.get("mimeType", "unknown")
                        texts.append(f"[Image content: {mime_type}]")
                    else:
                        # Unknown content type
                        texts.append(f"[Unknown content type: {content.get('type', 'none')}]")
                else:
                    # Fallback for unexpected response format
                    self.logger.warning(f"Unexpected response format: {response}")
                    texts.append(str(response))
            except Exception as e:
                self.logger.error(f"Error extracting text from response: {e}")
                texts.append(f"[Error extracting response: {e}]")

        return texts

    def __del__(self):
        """
        Cleanup when the object is destroyed.
        """
        if self._connection_established:
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._disconnect())
            except:
                pass  # Ignore errors during cleanup
