# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Robert Gerstenberger, Nils Blach

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple


class Parser(ABC):
    """
    Abstract base class that defines the interface for all parsers.
    Parsers are used to parse the responses from the language models.
    """

    @abstractmethod
    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for a aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the response from the language model.
        :rtype: Union[Dict, List[Dict]]
        """
        pass

    @abstractmethod
    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the response from the language model.
        :rtype: Dict
        """
        pass

    @abstractmethod
    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the response from the language model.
        :rtype: List[Dict]
        """
        pass

    @abstractmethod
    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass

    @abstractmethod
    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        """
        pass


class BatchParser(Parser):
    """
    Enhanced parser that provides batch processing capabilities for handling multiple responses efficiently.

    This class extends the base Parser with batch-aware methods that can:
    - Process multiple responses concurrently
    - Handle partial failures gracefully
    - Provide detailed error reporting
    - Support configurable error handling strategies
    """

    def __init__(self, error_handling_strategy: str = "skip_errors") -> None:
        """
        Initialize the BatchParser with configurable error handling.

        :param error_handling_strategy: Strategy for handling parsing errors.
                                      Options: "skip_errors", "raise_on_error", "return_partial"
        :type error_handling_strategy: str
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_handling_strategy = error_handling_strategy

    def parse_batch_responses(
        self,
        parse_method: str,
        states: List[Dict],
        batch_texts: List[List[str]],
        **kwargs
    ) -> Tuple[List[Union[Dict, List[Dict], float, bool]], List[Optional[Exception]]]:
        """
        Parse multiple responses using the specified parsing method with batch processing.

        :param parse_method: Name of the parsing method to use
        :type parse_method: str
        :param states: List of thought states corresponding to each batch
        :type states: List[Dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: List[List[str]]
        :param kwargs: Additional arguments for the parsing method
        :return: Tuple of (parsed_results, errors)
        :rtype: Tuple[List[Union[Dict, List[Dict], float, bool]], List[Optional[Exception]]]
        """
        results = []
        errors = []

        for i, (state, texts) in enumerate(zip(states, batch_texts)):
            try:
                if parse_method == "parse_generate_answer":
                    result = self.parse_generate_answer(state, texts)
                elif parse_method == "parse_score_answer":
                    result = self.parse_score_answer([state], texts)
                    result = result[0] if result else 0.0  # Extract single score
                elif parse_method == "parse_aggregation_answer":
                    result = self.parse_aggregation_answer([state], texts)
                elif parse_method == "parse_improve_answer":
                    result = self.parse_improve_answer(state, texts)
                elif parse_method == "parse_validation_answer":
                    result = self.parse_validation_answer(state, texts)
                else:
                    raise ValueError(f"Unknown parsing method: {parse_method}")

                results.append(result)
                errors.append(None)

            except Exception as e:
                self.logger.error(f"Error parsing batch item {i} with method {parse_method}: {e}")

                if self.error_handling_strategy == "raise_on_error":
                    raise e
                elif self.error_handling_strategy == "skip_errors":
                    # Skip this item and continue
                    continue
                elif self.error_handling_strategy == "return_partial":
                    # Return a default/error value
                    if parse_method == "parse_generate_answer":
                        results.append([{"error": str(e), "parsing_failed": True}])
                    elif parse_method == "parse_score_answer":
                        results.append(0.0)  # Default score
                    elif parse_method == "parse_aggregation_answer":
                        results.append({"error": str(e), "parsing_failed": True})
                    elif parse_method == "parse_improve_answer":
                        results.append({"error": str(e), "parsing_failed": True})
                    elif parse_method == "parse_validation_answer":
                        results.append(False)  # Default to invalid

                    errors.append(e)
                else:
                    raise ValueError(f"Unknown error handling strategy: {self.error_handling_strategy}")

        return results, errors

    def parse_batch_generate_answers(
        self,
        states: List[Dict],
        batch_texts: List[List[str]]
    ) -> Tuple[List[List[Dict]], List[Optional[Exception]]]:
        """
        Parse multiple generate responses in batch.

        :param states: List of thought states
        :type states: List[Dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: List[List[str]]
        :return: Tuple of (list of generated states for each input, errors)
        :rtype: Tuple[List[List[Dict]], List[Optional[Exception]]]
        """
        return self.parse_batch_responses("parse_generate_answer", states, batch_texts)

    def parse_batch_score_answers(
        self,
        states: List[Dict],
        batch_texts: List[List[str]]
    ) -> Tuple[List[float], List[Optional[Exception]]]:
        """
        Parse multiple score responses in batch.

        :param states: List of thought states
        :type states: List[Dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: List[List[str]]
        :return: Tuple of (list of scores, errors)
        :rtype: Tuple[List[float], List[Optional[Exception]]]
        """
        return self.parse_batch_responses("parse_score_answer", states, batch_texts)

    def parse_batch_aggregation_answers(
        self,
        states: List[Dict],
        batch_texts: List[List[str]]
    ) -> Tuple[List[Union[Dict, List[Dict]]], List[Optional[Exception]]]:
        """
        Parse multiple aggregation responses in batch.

        :param states: List of thought states
        :type states: List[Dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: List[List[str]]
        :return: Tuple of (list of aggregated states, errors)
        :rtype: Tuple[List[Union[Dict, List[Dict]]], List[Optional[Exception]]]
        """
        return self.parse_batch_responses("parse_aggregation_answer", states, batch_texts)

    def parse_batch_improve_answers(
        self,
        states: List[Dict],
        batch_texts: List[List[str]]
    ) -> Tuple[List[Dict], List[Optional[Exception]]]:
        """
        Parse multiple improve responses in batch.

        :param states: List of thought states
        :type states: List[Dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: List[List[str]]
        :return: Tuple of (list of improved states, errors)
        :rtype: Tuple[List[Dict], List[Optional[Exception]]]
        """
        return self.parse_batch_responses("parse_improve_answer", states, batch_texts)

    def parse_batch_validation_answers(
        self,
        states: List[Dict],
        batch_texts: List[List[str]]
    ) -> Tuple[List[bool], List[Optional[Exception]]]:
        """
        Parse multiple validation responses in batch.

        :param states: List of thought states
        :type states: List[Dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: List[List[str]]
        :return: Tuple of (list of validation results, errors)
        :rtype: Tuple[List[bool], List[Optional[Exception]]]
        """
        return self.parse_batch_responses("parse_validation_answer", states, batch_texts)

    def get_batch_statistics(self, errors: List[Optional[Exception]]) -> Dict[str, Union[int, float]]:
        """
        Get statistics about batch processing results.

        :param errors: List of errors from batch processing
        :type errors: List[Optional[Exception]]
        :return: Dictionary with statistics
        :rtype: Dict[str, Union[int, float]]
        """
        total_items = len(errors)
        error_count = sum(1 for error in errors if error is not None)
        success_count = total_items - error_count

        return {
            "total_items": total_items,
            "successful_items": success_count,
            "failed_items": error_count,
            "success_rate": success_count / total_items if total_items > 0 else 0.0,
            "error_rate": error_count / total_items if total_items > 0 else 0.0
        }
