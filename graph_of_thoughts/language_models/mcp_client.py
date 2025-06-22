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
from .mcp_transport import create_transport, MCPTransport


class MCPLanguageModel(AbstractLanguageModel):
    """
    The MCPLanguageModel class handles interactions with language models through the Model Context Protocol (MCP).
    This allows connecting to various MCP hosts like Claude Desktop, VSCode, Cursor, or remote MCP servers.

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
        
        # Extract configuration
        self.transport_type: str = self.config["transport_type"]
        self.host_type: str = self.config["host_type"]
        self.model_preferences: Dict = self.config["model_preferences"]
        self.sampling_config: Dict = self.config["sampling_config"]
        self.connection_config: Dict = self.config["connection_config"]
        
        # Cost tracking
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        
        # Initialize transport
        self.transport: MCPTransport = create_transport(self.config)
        self._connection_established = False

    async def _ensure_connection(self) -> None:
        """
        Ensure that the MCP connection is established.
        """
        if not self._connection_established:
            success = await self.transport.connect()
            if not success:
                raise RuntimeError(f"Failed to connect to MCP host: {self.host_type}")
            self._connection_established = True
            self.logger.info(f"Connected to MCP host: {self.host_type}")

    async def _disconnect(self) -> None:
        """
        Disconnect from the MCP host.
        """
        if self._connection_established:
            await self.transport.disconnect()
            self._connection_established = False

    def _create_sampling_request(self, query: str, num_responses: int = 1) -> Dict[str, Any]:
        """
        Create a sampling request for the MCP host.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The sampling request
        :rtype: Dict[str, Any]
        """
        return {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": query
                    }
                }
            ],
            "modelPreferences": self.model_preferences,
            "temperature": self.sampling_config["temperature"],
            "maxTokens": self.sampling_config["max_tokens"],
            "stopSequences": self.sampling_config["stop_sequences"],
            "includeContext": self.sampling_config["include_context"],
            "metadata": {
                "num_responses": num_responses,
                "source": "graph_of_thoughts"
            }
        }

    @backoff.on_exception(backoff.expo, Exception, max_time=10, max_tries=3)
    async def _send_sampling_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a sampling request to the MCP host with retry logic.

        :param request: The sampling request
        :type request: Dict[str, Any]
        :return: The response from the host
        :rtype: Dict[str, Any]
        """
        await self._ensure_connection()
        return await self.transport.send_sampling_request(request)

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

        :param response: The response from the MCP host
        :type response: Dict[str, Any]
        """
        # Extract token usage from response metadata if available
        # This is a simplified implementation - actual token counting would depend on the MCP host
        content = response.get("content", {})
        if isinstance(content, dict) and content.get("type") == "text":
            text = content.get("text", "")
            # Rough estimation: 1 token ≈ 4 characters
            estimated_tokens = len(text) // 4
            self.completion_tokens += estimated_tokens
            # Estimate prompt tokens similarly
            self.prompt_tokens += 50  # Rough estimate for prompt overhead
            
            # Update cost
            prompt_tokens_k = float(self.prompt_tokens) / 1000.0
            completion_tokens_k = float(self.completion_tokens) / 1000.0
            self.cost = (
                self.prompt_token_cost * prompt_tokens_k
                + self.response_token_cost * completion_tokens_k
            )
            
            self.logger.info(f"MCP response received. Estimated cost: ${self.cost:.4f}")

    def get_response_texts(self, query_response: Union[List[Dict], Dict]) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response (or list of responses) from the MCP host.
        :type query_response: Union[List[Dict], Dict]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, list):
            query_response = [query_response]
        
        texts = []
        for response in query_response:
            content = response.get("content", {})
            if isinstance(content, dict) and content.get("type") == "text":
                text = content.get("text", "")
                texts.append(text)
            else:
                # Fallback for unexpected response format
                texts.append(str(response))
        
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
