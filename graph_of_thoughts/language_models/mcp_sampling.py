# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from .mcp_transport import MCPTransport


class MCPSamplingManager:
    """
    Manager class for handling MCP sampling requests with advanced features like
    conversation context, system prompts, and model preferences.
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
        self.conversation_history: List[Dict[str, Any]] = []

    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        model_preferences: Optional[Dict[str, Any]] = None,
        sampling_config: Optional[Dict[str, Any]] = None,
        include_context: str = "thisServer",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a message using MCP sampling.

        :param messages: List of messages in the conversation
        :type messages: List[Dict[str, Any]]
        :param system_prompt: Optional system prompt
        :type system_prompt: Optional[str]
        :param model_preferences: Model selection preferences
        :type model_preferences: Optional[Dict[str, Any]]
        :param sampling_config: Sampling configuration
        :type sampling_config: Optional[Dict[str, Any]]
        :param include_context: Context inclusion setting
        :type include_context: str
        :param metadata: Additional metadata
        :type metadata: Optional[Dict[str, Any]]
        :return: The response from the MCP host
        :rtype: Dict[str, Any]
        """
        # Use defaults from config if not provided
        if model_preferences is None:
            model_preferences = self.config.get("model_preferences", {})
        if sampling_config is None:
            sampling_config = self.config.get("sampling_config", {})
        if metadata is None:
            metadata = {}

        # Build the sampling request
        request = {
            "messages": messages,
            "modelPreferences": model_preferences,
            "includeContext": include_context,
            "temperature": sampling_config.get("temperature", 1.0),
            "maxTokens": sampling_config.get("max_tokens", 4096),
            "stopSequences": sampling_config.get("stop_sequences", []),
            "metadata": {
                **metadata,
                "source": "graph_of_thoughts_sampling",
                "timestamp": asyncio.get_event_loop().time()
            }
        }

        # Add system prompt if provided
        if system_prompt:
            request["systemPrompt"] = system_prompt

        self.logger.debug(f"Sending MCP sampling request: {request}")

        try:
            response = await self.transport.send_sampling_request(request)
            
            # Update conversation history
            self.conversation_history.extend(messages)
            if response.get("content"):
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response["content"]
                })

            return response

        except Exception as e:
            self.logger.error(f"MCP sampling request failed: {e}")
            raise

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
