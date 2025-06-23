# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Optional
import json
import os
import logging

from .caching import get_cache_manager, CacheConfig


class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "", cache: bool = False, cache_config: Optional[CacheConfig] = None
    ) -> None:
        """
        Initialize the AbstractLanguageModel instance with configuration, model details, and caching options.

        :param config_path: Path to the config file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the language model. Defaults to "".
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        :param cache_config: Optional cache configuration for advanced caching features.
        :type cache_config: Optional[CacheConfig]
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict = None
        self.model_name: str = model_name
        self.cache = cache

        # Initialize intelligent caching system
        if self.cache:
            self.cache_manager = get_cache_manager(cache_config)
            # Keep legacy cache for backward compatibility
            self.response_cache: Dict[str, List[Any]] = {}
        else:
            self.cache_manager = None

        self.config_path = config_path
        self.load_config(config_path)
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cost: float = 0.0

    def load_config(self, path: str) -> None:
        """
        Load configuration from a specified path with intelligent caching.

        :param path: Path to the config file. If an empty path provided,
                     default is `config.json` in the current directory.
        :type path: str
        """
        if path == "":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "config.json")

        # Try to get from cache first
        if self.cache and self.cache_manager:
            cached_config = self.cache_manager.get_config(path, self.model_name)
            if cached_config is not None:
                self.config = cached_config
                self.logger.debug(f"Loaded config from cache for {path} and {self.model_name}")
                return

        # Load from file
        with open(path, "r") as f:
            self.config = json.load(f)

        # Cache the loaded configuration
        if self.cache and self.cache_manager:
            self.cache_manager.put_config(path, self.model_name, self.config)

        self.logger.debug(f"Loaded config from {path} for {self.model_name}")

    def clear_cache(self) -> None:
        """
        Clear the response cache and intelligent cache.
        """
        if hasattr(self, 'response_cache'):
            self.response_cache.clear()
        if self.cache_manager:
            self.cache_manager.clear_all()

    def get_cache_stats(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Get cache statistics for monitoring and debugging.

        :return: Cache statistics or None if caching is disabled
        :rtype: Optional[Dict[str, Dict[str, Any]]]
        """
        if not self.cache or not self.cache_manager:
            return None
        return self.cache_manager.get_all_stats()

    @abstractmethod
    def query(self, query: str, num_responses: int = 1) -> Any:
        """
        Abstract method to query the language model.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :return: The language model's response(s).
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_response_texts(self, query_responses: Union[List[Any], Any]) -> List[str]:
        """
        Abstract method to extract response texts from the language model's response(s).

        :param query_responses: The responses returned from the language model.
        :type query_responses: Union[List[Any], Any]
        :return: List of textual responses.
        :rtype: List[str]
        """
        pass
