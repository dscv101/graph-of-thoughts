# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Robert Gerstenberger, Nils Blach

import logging
from abc import ABC, abstractmethod


class Parser(ABC):
    """
    Abstract base class that defines the interface for all parsers.
    Parsers are used to parse the responses from the language models.
    """

    @abstractmethod
    def parse_aggregation_answer(
        self, states: list[dict], texts: list[str]
    ) -> dict | list[dict]:
        """
        Parse the response from the language model for a aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: list[dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: list[str]
        :return: The new thought states after parsing the response from the language model.
        :rtype: dict | list[dict]
        """
        pass

    @abstractmethod
    def parse_improve_answer(self, state: dict, texts: list[str]) -> dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: dict
        :param texts: The responses to the prompt from the language model.
        :type texts: list[str]
        :return: The new thought state after parsing the response from the language model.
        :rtype: dict
        """
        pass

    @abstractmethod
    def parse_generate_answer(self, state: dict, texts: list[str]) -> list[dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: dict
        :param texts: The responses to the prompt from the language model.
        :type texts: list[str]
        :return: The new thought states after parsing the response from the language model.
        :rtype: list[dict]
        """
        pass

    @abstractmethod
    def parse_validation_answer(self, state: dict, texts: list[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: dict
        :param texts: The responses to the prompt from the language model.
        :type texts: list[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass

    @abstractmethod
    def parse_score_answer(self, states: list[dict], texts: list[str]) -> list[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: list[dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: list[str]
        :return: The scores for the thought states.
        :rtype: list[float]
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
        states: list[dict],
        batch_texts: list[list[str]],
        **kwargs,
    ) -> tuple[list[dict | list[dict] | float | bool], list[Exception | None]]:
        """
        Parse multiple responses using the specified parsing method with batch processing.

        :param parse_method: Name of the parsing method to use
        :type parse_method: str
        :param states: List of thought states corresponding to each batch
        :type states: list[dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: list[list[str]]
        :param kwargs: Additional arguments for the parsing method
        :return: Tuple of (parsed_results, errors)
        :rtype: tuple[list[dict | list[dict] | float | bool], list[Exception | None]]
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
                self.logger.error(
                    f"Error parsing batch item {i} with method {parse_method}: {e}"
                )

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
                    raise ValueError(
                        f"Unknown error handling strategy: {self.error_handling_strategy}"
                    )

        return results, errors

    def parse_batch_generate_answers(
        self, states: list[dict], batch_texts: list[list[str]]
    ) -> tuple[list[list[dict]], list[Exception | None]]:
        """
        Parse multiple generate responses in batch.

        :param states: List of thought states
        :type states: list[dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: list[list[str]]
        :return: Tuple of (list of generated states for each input, errors)
        :rtype: tuple[list[list[dict]], list[Exception | None]]
        """
        return self.parse_batch_responses("parse_generate_answer", states, batch_texts)

    def parse_batch_score_answers(
        self, states: list[dict], batch_texts: list[list[str]]
    ) -> tuple[list[float], list[Exception | None]]:
        """
        Parse multiple score responses in batch.

        :param states: List of thought states
        :type states: list[dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: list[list[str]]
        :return: Tuple of (list of scores, errors)
        :rtype: tuple[list[float], list[Exception | None]]
        """
        return self.parse_batch_responses("parse_score_answer", states, batch_texts)

    def parse_batch_aggregation_answers(
        self, states: list[dict], batch_texts: list[list[str]]
    ) -> tuple[list[dict | list[dict]], list[Exception | None]]:
        """
        Parse multiple aggregation responses in batch.

        :param states: List of thought states
        :type states: list[dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: list[list[str]]
        :return: Tuple of (list of aggregated states, errors)
        :rtype: tuple[list[dict | list[dict]], list[Exception | None]]
        """
        return self.parse_batch_responses(
            "parse_aggregation_answer", states, batch_texts
        )

    def parse_batch_improve_answers(
        self, states: list[dict], batch_texts: list[list[str]]
    ) -> tuple[list[dict], list[Exception | None]]:
        """
        Parse multiple improve responses in batch.

        :param states: List of thought states
        :type states: list[dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: list[list[str]]
        :return: Tuple of (list of improved states, errors)
        :rtype: tuple[list[dict], list[Exception | None]]
        """
        return self.parse_batch_responses("parse_improve_answer", states, batch_texts)

    def parse_batch_validation_answers(
        self, states: list[dict], batch_texts: list[list[str]]
    ) -> tuple[list[bool], list[Exception | None]]:
        """
        Parse multiple validation responses in batch.

        :param states: List of thought states
        :type states: list[dict]
        :param batch_texts: List of text responses for each state
        :type batch_texts: list[list[str]]
        :return: Tuple of (list of validation results, errors)
        :rtype: tuple[list[bool], list[Exception | None]]
        """
        return self.parse_batch_responses(
            "parse_validation_answer", states, batch_texts
        )

    def get_batch_statistics(
        self, errors: list[Exception | None]
    ) -> dict[str, int | float]:
        """
        Get statistics about batch processing results.

        :param errors: List of errors from batch processing
        :type errors: list[Exception | None]
        :return: Dictionary with statistics
        :rtype: dict[str, int | float]
        """
        total_items = len(errors)
        error_count = sum(1 for error in errors if error is not None)
        success_count = total_items - error_count

        return {
            "total_items": total_items,
            "successful_items": success_count,
            "failed_items": error_count,
            "success_rate": success_count / total_items if total_items > 0 else 0.0,
            "error_rate": error_count / total_items if total_items > 0 else 0.0,
        }
