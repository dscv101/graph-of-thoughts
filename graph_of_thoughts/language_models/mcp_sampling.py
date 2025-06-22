# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Sampling Implementation.
This module provides a comprehensive implementation of MCP sampling following the official specification.
It includes conversation management, context handling, and advanced sampling features.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from .mcp_transport import MCPTransport, MCPTransportError
from .mcp_protocol import (
    MCPProtocolValidator,
    create_sampling_request,
    MCPMessage,
    MCPMessageContent,
    MCPModelPreferences,
    MCPIncludeContext
)


class MCPSamplingManager:
    """
    Manager class for handling MCP sampling requests following the official MCP specification.
    Provides advanced features like conversation context, system prompts, model preferences,
    and proper message formatting according to the MCP protocol.
    """

    def __init__(self, transport: MCPTransport, config: Dict[str, Any]):
        """
        Initialize the MCP sampling manager.

        :param transport: The MCP transport to use for communication
        :type transport: MCPTransport
        :param config: Configuration for sampling
        :type config: Dict[str, Any]
        """
        self.transport = transport
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.validator = MCPProtocolValidator()
        self.conversation_history: List[Dict[str, Any]] = []
        self.default_sampling_params = config.get("default_sampling_params", {})

    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        model_preferences: Optional[Dict[str, Any]] = None,
        include_context: str = "none",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a message using MCP sampling following the official specification.

        :param messages: List of messages in the conversation (MCP format)
        :type messages: List[Dict[str, Any]]
        :param system_prompt: Optional system prompt
        :type system_prompt: Optional[str]
        :param model_preferences: Model selection preferences
        :type model_preferences: Optional[Dict[str, Any]]
        :param include_context: Context inclusion setting ("none", "thisServer", "allServers")
        :type include_context: str
        :param temperature: Sampling temperature
        :type temperature: Optional[float]
        :param max_tokens: Maximum tokens to generate
        :type max_tokens: Optional[int]
        :param stop_sequences: Stop sequences
        :type stop_sequences: Optional[List[str]]
        :param metadata: Additional metadata
        :type metadata: Optional[Dict[str, Any]]
        :return: The response from the MCP server
        :rtype: Dict[str, Any]
        """
        # Use defaults from config if not provided
        if model_preferences is None:
            model_preferences = self.default_sampling_params.get("modelPreferences")
        if temperature is None:
            temperature = self.default_sampling_params.get("temperature")
        if max_tokens is None:
            max_tokens = self.default_sampling_params.get("maxTokens", 1000)
        if stop_sequences is None:
            stop_sequences = self.default_sampling_params.get("stopSequences")
        if metadata is None:
            metadata = {}

        # Validate include_context
        valid_contexts = [c.value for c in MCPIncludeContext]
        if include_context not in valid_contexts:
            raise ValueError(f"Invalid include_context: {include_context}. Must be one of {valid_contexts}")

        # Create the sampling request using the protocol utility
        request = create_sampling_request(
            messages=messages,
            model_preferences=model_preferences,
            system_prompt=system_prompt,
            include_context=include_context,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            metadata={
                **metadata,
                "source": "graph_of_thoughts_sampling",
                "manager_version": "1.0.0"
            }
        )

        # Validate the request
        if not self.validator.validate_sampling_request(request):
            raise ValueError("Invalid sampling request format")

        self.logger.debug(f"Sending MCP sampling request: {request}")

        try:
            response = await self.transport.send_sampling_request(request)

            # Update conversation history with properly formatted messages
            self.conversation_history.extend(messages)
            if response.get("content"):
                self.conversation_history.append({
                    "role": response.get("role", "assistant"),
                    "content": response["content"]
                })

            return response

        except MCPTransportError as e:
            self.logger.error(f"MCP transport error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"MCP sampling request failed: {e}")
            raise MCPTransportError(f"Sampling failed: {e}")

    async def create_simple_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create a simple text completion using MCP sampling.

        :param prompt: The user prompt
        :type prompt: str
        :param system_prompt: Optional system prompt
        :type system_prompt: Optional[str]
        :param kwargs: Additional arguments for create_message
        :return: The text response
        :rtype: str
        """
        messages = [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": prompt
                }
            }
        ]

        response = await self.create_message(
            messages=messages,
            system_prompt=system_prompt,
            **kwargs
        )

        # Extract text from response
        content = response.get("content", {})
        if isinstance(content, dict) and content.get("type") == "text":
            return content.get("text", "")
        else:
            return str(response)

    async def create_conversation_completion(
        self,
        new_message: str,
        use_history: bool = True,
        **kwargs
    ) -> str:
        """
        Create a completion as part of an ongoing conversation.

        :param new_message: The new user message
        :type new_message: str
        :param use_history: Whether to include conversation history
        :type use_history: bool
        :param kwargs: Additional arguments for create_message
        :return: The text response
        :rtype: str
        """
        messages = []
        
        if use_history:
            messages.extend(self.conversation_history)
        
        messages.append({
            "role": "user",
            "content": {
                "type": "text",
                "text": new_message
            }
        })

        response = await self.create_message(messages=messages, **kwargs)

        # Extract text from response
        content = response.get("content", {})
        if isinstance(content, dict) and content.get("type") == "text":
            return content.get("text", "")
        else:
            return str(response)

    async def create_multi_turn_completion(
        self,
        conversation: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Create a completion for a multi-turn conversation.

        :param conversation: List of conversation turns with 'role' and 'content' keys
        :type conversation: List[Dict[str, str]]
        :param kwargs: Additional arguments for create_message
        :return: The text response
        :rtype: str
        """
        messages = []
        for turn in conversation:
            messages.append({
                "role": turn["role"],
                "content": {
                    "type": "text",
                    "text": turn["content"]
                }
            })

        response = await self.create_message(messages=messages, **kwargs)

        # Extract text from response
        content = response.get("content", {})
        if isinstance(content, dict) and content.get("type") == "text":
            return content.get("text", "")
        else:
            return str(response)

    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.conversation_history.clear()
        self.logger.debug("Conversation history cleared")

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.

        :return: The conversation history
        :rtype: List[Dict[str, Any]]
        """
        return self.conversation_history.copy()

    async def create_batch_completions(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Create multiple completions in batch.

        :param prompts: List of prompts to process
        :type prompts: List[str]
        :param system_prompt: Optional system prompt
        :type system_prompt: Optional[str]
        :param kwargs: Additional arguments for create_message
        :return: List of text responses
        :rtype: List[str]
        """
        tasks = []
        for prompt in prompts:
            task = self.create_simple_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and convert to strings
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Batch completion {i} failed: {response}")
                results.append(f"Error: {str(response)}")
            else:
                results.append(response)
        
        return results

    async def create_completion_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> str:
        """
        Create a completion with retry logic.

        :param prompt: The user prompt
        :type prompt: str
        :param max_retries: Maximum number of retries
        :type max_retries: int
        :param retry_delay: Delay between retries in seconds
        :type retry_delay: float
        :param kwargs: Additional arguments for create_simple_completion
        :return: The text response
        :rtype: str
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await self.create_simple_completion(prompt, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    self.logger.warning(f"Completion attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error(f"All {max_retries + 1} completion attempts failed")
        
        raise last_exception
