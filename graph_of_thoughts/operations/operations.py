# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Iterator, Optional

from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.operations.thought import Thought
from graph_of_thoughts.parser import Parser
from graph_of_thoughts.prompter import Prompter


class OperationType(Enum):
    """
    Enum to represent different operation types that can be used as unique identifiers.
    """

    score: int = 0
    validate_and_improve: int = 1
    generate: int = 2
    improve: int = 3
    aggregate: int = 4
    keep_best_n: int = 5
    keep_valid: int = 6
    ground_truth_evaluator: int = 7
    selector: int = 8
    batch_generate: int = 9
    batch_score: int = 10
    batch_aggregate: int = 11


class Operation(ABC):
    """
    Abstract base class that defines the interface for all operations.
    """

    _ids: Iterator[int] = itertools.count(0)

    operation_type: OperationType = None

    def __init__(self) -> None:
        """
        Initializes a new Operation instance with a unique id, and empty predecessors and successors.
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Operation._ids)
        self.predecessors: list[Operation] = []
        self.successors: list[Operation] = []
        self.executed: bool = False

    def can_be_executed(self) -> bool:
        """
        Checks if the operation can be executed based on its predecessors.

        :return: True if all predecessors have been executed, False otherwise.
        :rtype: bool
        """
        return all(predecessor.executed for predecessor in self.predecessors)

    def get_previous_thoughts(self) -> list[Thought]:
        """
        Iterates over all predecessors and aggregates their thoughts.

        :return: A list of all thoughts from the predecessors.
        :rtype: list[Thought]
        """
        previous_thoughts: list[Thought] = [
            thought
            for predecessor in self.predecessors
            for thought in predecessor.get_thoughts()
        ]

        return previous_thoughts

    def add_predecessor(self, operation: "Operation") -> None:
        """
        Add a preceding operation and update the relationships.

        :param operation: The operation to be set as a predecessor.
        :type operation: "Operation"
        """
        self.predecessors.append(operation)
        operation.successors.append(self)

    def add_successor(self, operation: "Operation") -> None:
        """
        Add a succeeding operation and update the relationships.

        :param operation: The operation to be set as a successor.
        :type operation: "Operation"
        """
        self.successors.append(operation)
        operation.predecessors.append(self)

    def execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Execute the operation, assuring that all predecessors have been executed.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If not all predecessors have been executed.
        """
        assert self.can_be_executed(), "Not all predecessors have been executed"
        self.logger.info(
            "Executing operation %d of type %s", self.id, self.operation_type
        )
        self._execute(lm, prompter, parser, **kwargs)
        self.logger.debug("Operation %d executed", self.id)
        self.executed = True

    @abstractmethod
    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Abstract method for the actual execution of the operation.
        This should be implemented in derived classes.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        """
        pass

    @abstractmethod
    def get_thoughts(self) -> list[Thought]:
        """
        Abstract method to retrieve the thoughts associated with the operation.
        This should be implemented in derived classes.

        :return: List of associated thoughts.
        :rtype: list[Thought]
        """
        pass


class Score(Operation):
    """
    Operation to score thoughts.
    """

    operation_type: OperationType = OperationType.score

    def __init__(
        self,
        num_samples: int = 1,
        combined_scoring: bool = False,
        scoring_function: Callable[
            [list[dict] | dict], list[float] | float
        ] | None = None,
    ) -> None:
        """
        Initializes a new Score operation.

        :param num_samples: Number of samples to use for scoring. Defaults to 1.
        :type num_samples: int
        :param combined_scoring: Whether to score all thoughts together or individually. Defaults to False.
        :type combined_scoring: bool
        :param scoring_function: A function to score thoughts (if not using LM). Defaults to None.
        :type scoring_function: Takes a list of thought states or a single thought state and
                                returns a list of scores or a single score.
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.combined_scoring: bool = combined_scoring
        self.thoughts: list[Thought] = []
        self.scoring_function: Callable[
            [list[dict] | dict], list[float] | float
        ] | None = scoring_function

    def get_thoughts(self) -> list[Thought]:
        """
        Returns the thoughts associated with the operation.

        :return: List of scored thoughts.
        :rtype: list[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the scoring operation by scoring the thoughts from the predecessors.
        If combined scoring is used, the thoughts are scored together, otherwise individually.
        If a scoring function is provided, it is used, otherwise the LM is prompted.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        previous_thoughts: list[Thought] = self.get_previous_thoughts()

        assert (
            len(self.predecessors) > 0
        ), "Score operation needs at least one predecessor"

        if self.combined_scoring:
            previous_thoughts_states = [thought.state for thought in previous_thoughts]
            if self.scoring_function is not None:
                self.logger.debug(
                    "Using scoring function %s to score states", self.scoring_function
                )
                scores = self.scoring_function(previous_thoughts_states)
            else:
                prompt = prompter.score_prompt(previous_thoughts_states)
                self.logger.debug("Prompt for LM: %s", prompt)

                responses = lm.get_response_texts(
                    lm.query(prompt, num_responses=self.num_samples)
                )
                self.logger.debug("Responses from LM: %s", responses)
                scores = parser.parse_score_answer(previous_thoughts_states, responses)
            for thought, score in zip(previous_thoughts, scores):
                new_thought = Thought.from_thought(thought)
                new_thought.score = score
                self.thoughts.append(new_thought)
        else:
            for thought in previous_thoughts:
                new_thought = Thought.from_thought(thought)
                if self.scoring_function is not None:
                    self.logger.debug(
                        "Using scoring function %s to score state",
                        self.scoring_function,
                    )
                    score = self.scoring_function(thought.state)
                else:
                    prompt = prompter.score_prompt([thought.state])
                    self.logger.debug("Prompt for LM: %s", prompt)

                    responses = lm.get_response_texts(
                        lm.query(prompt, num_responses=self.num_samples)
                    )
                    self.logger.debug("Responses from LM: %s", responses)
                    score = parser.parse_score_answer([thought.state], responses)[0]

                new_thought.score = score
                self.thoughts.append(new_thought)

        self.logger.info(
            "Score operation %d scored %d thoughts",
            self.id,
            len(self.thoughts),
        )


class ValidateAndImprove(Operation):
    """
    Operation to validate and improve thoughts.
    """

    operation_type: OperationType = OperationType.validate_and_improve

    def __init__(
        self,
        num_samples: int = 1,
        improve: bool = True,
        num_tries: int = 3,
        validate_function: Callable[[dict], bool] | None = None,
    ) -> None:
        """
        Initializes a new ValidateAndImprove operation.

        :param num_samples: Number of samples to use for validation. Defaults to 1.
        :type num_samples: int
        :param improve: Whether to improve the thought if it is not valid. Defaults to True.
        :type improve: bool
        :param num_tries: Number of tries to improve the thought before giving up. Defaults to 3.
        :type num_tries: int
        :param validate_function: A function to validate thoughts (if not using LM). Defaults to None.
        :type validate_function: Takes a thought state and returns a boolean.
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.improve: bool = improve
        self.num_tries: int = num_tries
        self.validate_function: Callable[[dict], bool] | None = validate_function
        self.thoughts: list[list[Thought]] = []

    def get_thoughts(self) -> list[Thought]:
        """
        Returns the list of final thoughts, after validation and improvement.

        :return: List of final validated and improved thoughts.
        :rtype: list[Thought]
        """
        return [thought_list[-1] for thought_list in self.thoughts]

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the ValidateAndImprove operation by validating and improving the predecessors' thoughts.
        If a validation function is provided, it is used, otherwise the LM is prompted.
        If improvement is enabled, the LM is prompted to improve the thought, if it is not valid.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        previous_thoughts: list[Thought] = self.get_previous_thoughts()

        assert (
            len(self.predecessors) > 0
        ), "ValidateAndImprove operation needs at least one predecessor"

        for thought in previous_thoughts:
            thought_list = []
            current_thought = Thought.from_thought(thought)
            current_try = 0
            while True:
                if self.validate_function is not None:
                    self.logger.debug(
                        "Using validate function %s to score states",
                        self.validate_function,
                    )
                    valid = self.validate_function(current_thought.state)
                else:
                    prompt = prompter.validation_prompt(**current_thought.state)
                    self.logger.debug("Prompt for LM: %s", prompt)
                    responses = lm.get_response_texts(
                        lm.query(prompt, num_responses=self.num_samples)
                    )
                    self.logger.debug("Responses from LM: %s", responses)

                    valid = parser.parse_validation_answer(
                        current_thought.state, responses
                    )
                current_thought.valid = valid
                thought_list.append(current_thought)
                if (
                    not self.improve
                    or current_thought.valid
                    or current_try >= self.num_tries
                ):
                    break
                improve_prompt = prompter.improve_prompt(**current_thought.state)
                self.logger.debug("Prompt for LM: %s", improve_prompt)
                responses = lm.get_response_texts(
                    lm.query(improve_prompt, num_responses=1)
                )
                self.logger.debug("Responses from LM: %s", responses)
                state_update = parser.parse_improve_answer(
                    current_thought.state, responses
                )
                current_thought = Thought({**current_thought.state, **state_update})
                current_try += 1
            self.thoughts.append(thought_list)

        self.logger.info(
            "Validate and improve operation %d created %d valid thoughts from %d previous thoughts",
            self.id,
            len(
                [
                    thought_list[-1]
                    for thought_list in self.thoughts
                    if thought_list[-1].valid
                ]
            ),
            len(previous_thoughts),
        )


class Generate(Operation):
    """
    Operation to generate thoughts.
    """

    operation_type: OperationType = OperationType.generate

    def __init__(
        self, num_branches_prompt: int = 1, num_branches_response: int = 1
    ) -> None:
        """
        Initializes a new Generate operation.

        :param num_branches_prompt: Number of responses that each prompt should generate (passed to prompter). Defaults to 1.
        :type num_branches_prompt: int
        :param num_branches_response: Number of responses the LM should generate for each prompt. Defaults to 1.
        :type num_branches_response: int
        """
        super().__init__()
        self.num_branches_prompt: int = num_branches_prompt
        self.num_branches_response: int = num_branches_response
        self.thoughts: list[Thought] = []

    def get_thoughts(self) -> list[Thought]:
        """
        Returns the thoughts associated with the operation.

        :return: List of generated thoughts.
        :rtype: list[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Generate operation by generating thoughts from the predecessors.
        The thoughts are generated by prompting the LM with the predecessors' thought states.
        If there are no predecessors, the kwargs are used as a base state.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        """
        previous_thoughts: list[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
            return

        if len(previous_thoughts) == 0:
            # no predecessors, use kwargs as base state
            previous_thoughts = [Thought(state=kwargs)]

        for thought in previous_thoughts:
            base_state = thought.state
            prompt = prompter.generate_prompt(self.num_branches_prompt, **base_state)
            self.logger.debug("Prompt for LM: %s", prompt)
            responses = lm.get_response_texts(
                lm.query(prompt, num_responses=self.num_branches_response)
            )
            self.logger.debug("Responses from LM: %s", responses)
            for new_state in parser.parse_generate_answer(base_state, responses):
                new_state = {**base_state, **new_state}
                self.thoughts.append(Thought(new_state))
                self.logger.debug(
                    "New thought %d created with state %s",
                    self.thoughts[-1].id,
                    self.thoughts[-1].state,
                )
        if (
            len(self.thoughts)
            > self.num_branches_prompt
            * self.num_branches_response
            * len(previous_thoughts)
            and self.num_branches_prompt > 0
        ):
            self.logger.warning(
                "Generate operation %d created more thoughts than expected",
                self.id,
            )
        self.logger.info(
            "Generate operation %d created %d new thoughts", self.id, len(self.thoughts)
        )


class Improve(Operation):
    """
    Operation to improve thoughts.
    """

    operation_type: OperationType = OperationType.improve

    def __init__(self) -> None:
        """
        Initializes a new Improve operation.
        """
        super().__init__()
        self.thoughts: list[Thought] = []

    def get_thoughts(self) -> list[Thought]:
        """
        Returns the thoughts associated with the operation after improvement.

        :return: List of improved thoughts.
        :rtype: list[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Improve operation by improving the predecessors' thoughts.
        The thoughts are improved by prompting the LM with the predecessors' thought states.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        previous_thoughts: list[Thought] = self.get_previous_thoughts()

        assert len(self.predecessors) > 0, "Needs at least one predecessor"

        for thought in previous_thoughts:
            improve_prompt = prompter.improve_prompt(**thought.state)
            self.logger.debug("Prompt for LM: %s", improve_prompt)
            responses = lm.get_response_texts(lm.query(improve_prompt, num_responses=1))
            self.logger.debug("Responses from LM: %s", responses)
            state_update = parser.parse_improve_answer(thought.state, responses)
            self.thoughts.append(Thought({**thought.state, **state_update}))

        self.logger.info(
            "Improve operation %d improved %d thoughts", self.id, len(self.thoughts)
        )


class Aggregate(Operation):
    """
    Operation to aggregate thoughts.
    """

    operation_type: OperationType = OperationType.aggregate

    def __init__(self, num_responses: int = 1) -> None:
        """
        Initializes a new Aggregate operation.

        :param num_responses: Number of responses to use for aggregation. Defaults to 1.
        :type num_responses: int
        """
        super().__init__()
        self.thoughts: list[Thought] = []
        self.num_responses: int = num_responses

    def get_thoughts(self) -> list[Thought]:
        """
        Returns the thoughts associated with the operation after aggregation.

        :return: List of aggregated thoughts.
        :rtype: list[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Aggregate operation by aggregating the predecessors' thoughts.
        The thoughts are aggregated by prompting the LM with the predecessors' thought states.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        assert (
            len(self.predecessors) >= 1
        ), "Aggregate operation must have at least one predecessor"

        previous_thoughts: list[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            return

        # applied in order of score
        base_state: dict = {}
        for thought in sorted(previous_thoughts, key=lambda thought: thought.score):
            base_state = {**base_state, **thought.state}

        previous_thought_states = [thought.state for thought in previous_thoughts]
        prompt = prompter.aggregation_prompt(previous_thought_states)

        self.logger.debug("Prompt for LM: %s", prompt)

        responses = lm.get_response_texts(
            lm.query(prompt, num_responses=self.num_responses)
        )

        self.logger.debug("Responses from LM: %s", responses)

        parsed = parser.parse_aggregation_answer(previous_thought_states, responses)

        if isinstance(parsed, dict):
            parsed = [parsed]
        for new_state in parsed:
            self.thoughts.append(Thought({**base_state, **new_state}))


class KeepBestN(Operation):
    """
    Operation to keep the best N thoughts from predecessors based on their score.
    """

    operation_type: OperationType = OperationType.keep_best_n

    def __init__(self, n: int, higher_is_better: bool = True) -> None:
        """
        Initializes a new KeepBestN operation.

        :param n: Maximum number of thoughts to keep.
        :type n: int
        :param higher_is_better: Whether higher scores are better. Defaults to True.
        :type higher_is_better: bool
        :raises AssertionError: If `n` is not greater than zero.
        """
        super().__init__()
        self.n: int = n
        assert self.n > 0, "KeepBestN operation must keep at least one thought"
        self.higher_is_better: bool = higher_is_better
        self.thoughts: list[Thought] = []

    def get_best_n(self) -> list[Thought]:
        """
        Returns the best N thoughts from the predecessors based on their score.

        :return: List of best N thoughts.
        :rtype: list[Thought]
        :raises AssertionError: If not all predecessors have been executed.
        :raises AssertionError: If not all thoughts have been scored.
        """
        previous_thoughts: list[Thought] = self.get_previous_thoughts()
        assert all(
            previous_thought.scored for previous_thought in previous_thoughts
        ), "Not all thoughts have been scored"

        try:
            return sorted(
                previous_thoughts,
                key=lambda thought: thought.score,
                reverse=self.higher_is_better,
            )[: self.n]
        except:
            self.logger.error("Error in KeepBestN operation")
            self.logger.error(
                "Previous operation: %s", [op.id for op in self.predecessors]
            )
            self.logger.error("Previous thoughts: %s", previous_thoughts)
            self.logger.error(
                "Scores: %s", [thought.score for thought in previous_thoughts]
            )
            return sorted(
                [i for i in previous_thoughts if isinstance(i.score, float)],
                key=lambda thought: thought.score,
                reverse=self.higher_is_better,
            )[: self.n]

    def get_thoughts(self) -> list[Thought]:
        """
        Returns the thoughts kept by the operation.

        :return: List of kept thoughts.
        :rtype: list[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the KeepBestN operation by keeping the best N thoughts from the predecessors according to their score.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        :raises AssertionError: If not all predecessors have been executed.
        :raises AssertionError: If not all thoughts have been scored.
        """
        assert (
            len(self.predecessors) >= 1
        ), "KeepBestN operation must have at least one predecessor"

        self.thoughts = [Thought.from_thought(thought) for thought in self.get_best_n()]

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s kept", thought.id, thought.state
            )

        self.logger.info(
            "KeepBestN operation %d kept %d thoughts", self.id, len(self.thoughts)
        )


class KeepValid(Operation):
    """
    Operation to keep valid thoughts from predecessors.
    """

    operation_type: OperationType = OperationType.keep_valid

    def __init__(self) -> None:
        """
        Initializes a new KeepValid operation.
        """
        super().__init__()
        self.thoughts: list[Thought] = []

    def get_thoughts(self) -> list[Thought]:
        """
        Returns the thoughts kept by the operation.

        :return: List of kept thoughts.
        :rtype: list[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the KeepValid operation by keeping the valid thoughts from the predecessors.
        Keeps unvalidated thoughts as well.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        assert (
            len(self.predecessors) >= 1
        ), "KeepValid operation must have at least one predecessor"

        self.thoughts: list[Thought] = [
            Thought.from_thought(thought)
            for thought in self.get_previous_thoughts()
            if not thought.validated or thought.valid
        ]

        if any(not thought.validated for thought in self.thoughts):
            self.logger.warning(
                "KeepValid operation %d has unvalidated thoughts", self.id
            )

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s kept", thought.id, thought.state
            )

        self.logger.info(
            "KeepValid operation %d kept %d thoughts", self.id, len(self.thoughts)
        )


class GroundTruth(Operation):
    """
    Operation to evaluate if thoughts correctly solve the problem, using a ground truth evaluator
    """

    operation_type: OperationType = OperationType.ground_truth_evaluator

    def __init__(self, ground_truth_evaluator: Callable[[dict], bool]) -> None:
        """
        Initializes a new GroundTruth operation.

        :param ground_truth_evaluator: A function to evaluate if a thought solves the problem.
        :type ground_truth_evaluator: A function that takes a thought state and returns a boolean.
        """
        super().__init__()
        self.ground_truth_evaluator: Callable[[dict], bool] = ground_truth_evaluator
        self.thoughts: list[Thought] = []

    def get_thoughts(self) -> "list[list[Thought]]":
        """
        Returns the thoughts associated with the operation.

        :return: List of evaluated thoughts.
        :rtype: "list[list[Thought]]"
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the GroundTruth operation by evaluating the predecessors' thoughts using the ground truth evaluator function.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessor.
        """
        assert (
            len(self.predecessors) >= 1
        ), "GroundTruth operation must have at least one predecessor"

        previous_thoughts: "list[list[Thought]]" = self.get_previous_thoughts()

        for thought in previous_thoughts:
            new_thought = Thought.from_thought(thought)
            try:
                new_thought.solved = self.ground_truth_evaluator(new_thought.state)
            except:
                new_thought.solved = False
            self.thoughts.append(new_thought)

        self.logger.info(
            "GroundTruth operation %d evaluated %d thoughts and %d solved the problem",
            self.id,
            len(self.thoughts),
            len([thought for thought in self.thoughts if thought.solved]),
        )


class Selector(Operation):
    """
    Operation to select thoughts from predecessors.
    Useful for separating thoughts to perform different, subsequent operations on them.
    """

    operation_type: OperationType = OperationType.selector

    def __init__(self, selector: Callable[["list[list[Thought]]"], "list[list[Thought]]"]) -> None:
        """
        Initializes a new Selector operation.

        :param selector: A function to select thoughts from the predecessors' thoughts.
        :type selector: A function that takes a list of thoughts and returns a list of thoughts.
        """
        super().__init__()
        self.selector: Callable[["list[list[Thought]]"], "list[list[Thought]]"] = selector
        self.thoughts: "list[list[Thought]]" = []

    def get_thoughts(self) -> "list[list[Thought]]":
        """
        Returns the thoughts selected by the operation.

        :return: List of selected thoughts.
        :rtype: "list[list[Thought]]"
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Selector operation by selecting thoughts from the predecessors using the selector function.
        If the Selector has no predecessors, the selector function is called with a thought containing the kwargs as state.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        """
        previous_thoughts: "list[list[Thought]]" = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            previous_thoughts = [Thought(kwargs)]

        self.thoughts = [
            Thought.from_thought(thought)
            for thought in self.selector(previous_thoughts)
        ]

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s selected", thought.id, thought.state
            )

        self.logger.info(
            "Selector operation %d selected %d thoughts", self.id, len(self.thoughts)
        )


class BatchGenerate(Operation):
    """
    Batch-aware operation to generate thoughts efficiently using concurrent processing.

    This operation optimizes the generation process by:
    - Processing multiple thoughts concurrently
    - Using batch processing for language model queries
    - Implementing configurable concurrency limits
    - Providing enhanced error handling and retry logic
    """

    operation_type: OperationType = OperationType.batch_generate

    def __init__(
        self,
        num_branches_prompt: int = 1,
        num_branches_response: int = 1,
        max_concurrent: Optional[int] = None,
        batch_size: Optional[int] = None,
        enable_batch_processing: bool = True,
    ) -> None:
        """
        Initializes a new BatchGenerate operation.

        :param num_branches_prompt: Number of responses that each prompt should generate. Defaults to 1.
        :type num_branches_prompt: int
        :param num_branches_response: Number of responses the LM should generate for each prompt. Defaults to 1.
        :type num_branches_response: int
        :param max_concurrent: Maximum number of concurrent requests. Uses LM default if None.
        :type max_concurrent: Optional[int]
        :param batch_size: Maximum batch size for processing. Uses LM default if None.
        :type batch_size: Optional[int]
        :param enable_batch_processing: Whether to use batch processing optimizations. Defaults to True.
        :type enable_batch_processing: bool
        """
        super().__init__()
        self.num_branches_prompt: int = num_branches_prompt
        self.num_branches_response: int = num_branches_response
        self.max_concurrent: Optional[int] = max_concurrent
        self.batch_size: Optional[int] = batch_size
        self.enable_batch_processing: bool = enable_batch_processing
        self.thoughts: "list[list[Thought]]" = []

    def get_thoughts(self) -> "list[list[Thought]]":
        """
        Returns the thoughts associated with the operation.

        :return: List of generated thoughts.
        :rtype: "list[list[Thought]]"
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the BatchGenerate operation using efficient batch processing.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        """
        previous_thoughts: "list[list[Thought]]" = self.get_previous_thoughts()

        if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
            return

        if len(previous_thoughts) == 0:
            # no predecessors, use kwargs as base state
            previous_thoughts = [Thought(state=kwargs)]

        # Check if the language model supports batch processing
        if (
            self.enable_batch_processing
            and hasattr(lm, "query_batch")
            and len(previous_thoughts) > 1
        ):
            self._execute_batch(lm, prompter, parser, previous_thoughts)
        else:
            self._execute_sequential(lm, prompter, parser, previous_thoughts)

        if (
            len(self.thoughts)
            > self.num_branches_prompt
            * self.num_branches_response
            * len(previous_thoughts)
            and self.num_branches_prompt > 0
        ):
            self.logger.warning(
                "BatchGenerate operation %d created more thoughts than expected",
                self.id,
            )
        self.logger.info(
            "BatchGenerate operation %d created %d new thoughts",
            self.id,
            len(self.thoughts),
        )

    def _execute_batch(
        self,
        lm: AbstractLanguageModel,
        prompter: Prompter,
        parser: Parser,
        previous_thoughts: "list[list[Thought]]",
    ) -> None:
        """
        Execute generation using batch processing for improved performance.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param previous_thoughts: List of thoughts to process.
        :type previous_thoughts: "list[list[Thought]]"
        """
        self.logger.info(
            f"Using batch processing for {len(previous_thoughts)} thoughts"
        )

        # Prepare all prompts for batch processing
        prompts = []
        thought_prompt_mapping = []  # Track which prompt belongs to which thought

        for thought_idx, thought in enumerate(previous_thoughts):
            base_state = thought.state
            for _ in range(self.num_branches_response):
                prompt = prompter.generate_prompt(
                    self.num_branches_prompt, **base_state
                )
                prompts.append(prompt)
                thought_prompt_mapping.append((thought_idx, thought, base_state))

        self.logger.debug(f"Prepared {len(prompts)} prompts for batch processing")

        # Use optimized async execution for batch processing
        responses = self._run_async_batch_safely(lm, prompts)

        # Process responses and create thoughts
        for i, (response, (thought_idx, original_thought, base_state)) in enumerate(
            zip(responses, thought_prompt_mapping)
        ):
            try:
                # Extract text from response
                response_texts = lm.get_response_texts([response])
                self.logger.debug(f"Response {i}: {response_texts}")

                # Parse the response
                for new_state in parser.parse_generate_answer(
                    base_state, response_texts
                ):
                    new_state = {**base_state, **new_state}
                    self.thoughts.append(Thought(new_state))
                    self.logger.debug(
                        "New thought %d created with state %s",
                        self.thoughts[-1].id,
                        self.thoughts[-1].state,
                    )
            except Exception as e:
                self.logger.error(f"Error processing response {i}: {e}")
                # Create a fallback thought with error information
                error_state = {
                    **base_state,
                    "error": str(e),
                    "batch_processing_failed": True,
                }
                self.thoughts.append(Thought(error_state))

    def _run_async_batch_safely(
        self, lm: AbstractLanguageModel, prompts: "list[list[str]]"
    ) -> list[list[dict]]:
        """
        Safely run batch processing with optimized event loop management.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompts: List of prompts to process.
        :type prompts: "list[list[str]]"
        :return: List of responses.
        :rtype: list[list[dict]]
        :raises RuntimeError: If called from within an async context
        """

        async def _run_batch():
            return await self._run_batch_async(lm, prompts)

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # If we reach here, we're in an async context
            raise RuntimeError(
                "Cannot call _run_async_batch_safely from within an async context. "
                "Use 'await _run_batch_async()' instead or call from a synchronous context."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No event loop running, safe to use asyncio.run
                return asyncio.run(_run_batch())
            else:
                # Re-raise the error about being in async context
                raise

    async def _run_batch_async(
        self, lm: AbstractLanguageModel, prompts: "list[list[str]]"
    ) -> list[list[dict]]:
        """
        Run batch processing asynchronously.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompts: List of prompts to process.
        :type prompts: "list[list[str]]"
        :return: List of responses.
        :rtype: list[list[dict]]
        """
        async with lm:  # Use async context manager
            return await lm.query_batch(
                prompts, max_concurrent=self.max_concurrent, batch_size=self.batch_size
            )

    def _execute_sequential(
        self,
        lm: AbstractLanguageModel,
        prompter: Prompter,
        parser: Parser,
        previous_thoughts: "list[list[Thought]]",
    ) -> None:
        """
        Execute generation using sequential processing (fallback method).

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param previous_thoughts: List of thoughts to process.
        :type previous_thoughts: "list[list[Thought]]"
        """
        self.logger.info(
            f"Using sequential processing for {len(previous_thoughts)} thoughts"
        )

        for thought in previous_thoughts:
            base_state = thought.state
            prompt = prompter.generate_prompt(self.num_branches_prompt, **base_state)
            self.logger.debug("Prompt for LM: %s", prompt)
            responses = lm.get_response_texts(
                lm.query(prompt, num_responses=self.num_branches_response)
            )
            self.logger.debug("Responses from LM: %s", responses)
            for new_state in parser.parse_generate_answer(base_state, responses):
                new_state = {**base_state, **new_state}
                self.thoughts.append(Thought(new_state))
                self.logger.debug(
                    "New thought %d created with state %s",
                    self.thoughts[-1].id,
                    self.thoughts[-1].state,
                )


class BatchScore(Operation):
    """
    Batch-aware operation to score thoughts efficiently using concurrent processing.

    This operation optimizes the scoring process by:
    - Processing multiple thoughts concurrently
    - Using batch processing for language model queries
    - Supporting both individual and combined scoring modes
    - Implementing configurable concurrency limits
    """

    operation_type: OperationType = OperationType.batch_score

    def __init__(
        self,
        num_samples: int = 1,
        combined_scoring: bool = False,
        scoring_function: Callable[
            [list[dict] | dict], list[float] | float
        ] | None = None,
        max_concurrent: Optional[int] = None,
        batch_size: Optional[int] = None,
        enable_batch_processing: bool = True,
    ) -> None:
        """
        Initializes a new BatchScore operation.

        :param num_samples: Number of samples to use for scoring. Defaults to 1.
        :type num_samples: int
        :param combined_scoring: Whether to score all thoughts together or individually. Defaults to False.
        :type combined_scoring: bool
        :param scoring_function: A function to score thoughts (if not using LM). Defaults to None.
        :type scoring_function: Takes a list of thought states or a single thought state and
                                returns a list of scores or a single score.
        :param max_concurrent: Maximum number of concurrent requests. Uses LM default if None.
        :type max_concurrent: Optional[int]
        :param batch_size: Maximum batch size for processing. Uses LM default if None.
        :type batch_size: Optional[int]
        :param enable_batch_processing: Whether to use batch processing optimizations. Defaults to True.
        :type enable_batch_processing: bool
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.combined_scoring: bool = combined_scoring
        self.scoring_function: Callable[
            [list[dict] | dict], list[float] | float
        ] | None = scoring_function
        self.max_concurrent: Optional[int] = max_concurrent
        self.batch_size: Optional[int] = batch_size
        self.enable_batch_processing: bool = enable_batch_processing
        self.thoughts: "list[list[Thought]]" = []

    def get_thoughts(self) -> "list[list[Thought]]":
        """
        Returns the thoughts associated with the operation.

        :return: List of scored thoughts.
        :rtype: "list[list[Thought]]"
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the BatchScore operation using efficient batch processing.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        previous_thoughts: "list[list[Thought]]" = self.get_previous_thoughts()

        assert (
            len(self.predecessors) > 0
        ), "BatchScore operation needs at least one predecessor"

        if self.combined_scoring:
            self._execute_combined_scoring(lm, prompter, parser, previous_thoughts)
        else:
            # Check if the language model supports batch processing
            if (
                self.enable_batch_processing
                and hasattr(lm, "query_batch")
                and len(previous_thoughts) > 1
                and self.scoring_function is None
            ):  # Only use batch for LM scoring
                self._execute_batch_scoring(lm, prompter, parser, previous_thoughts)
            else:
                self._execute_sequential_scoring(
                    lm, prompter, parser, previous_thoughts
                )

        self.logger.info(
            "BatchScore operation %d scored %d thoughts",
            self.id,
            len(self.thoughts),
        )

    def _execute_combined_scoring(
        self,
        lm: AbstractLanguageModel,
        prompter: Prompter,
        parser: Parser,
        previous_thoughts: "list[list[Thought]]",
    ) -> None:
        """
        Execute combined scoring for all thoughts together.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param previous_thoughts: List of thoughts to score.
        :type previous_thoughts: "list[list[Thought]]"
        """
        previous_thoughts_states = [thought.state for thought in previous_thoughts]
        if self.scoring_function is not None:
            self.logger.debug(
                "Using scoring function %s to score states", self.scoring_function
            )
            scores = self.scoring_function(previous_thoughts_states)
        else:
            prompt = prompter.score_prompt(previous_thoughts_states)
            self.logger.debug("Prompt for LM: %s", prompt)

            responses = lm.get_response_texts(
                lm.query(prompt, num_responses=self.num_samples)
            )
            self.logger.debug("Responses from LM: %s", responses)
            scores = parser.parse_score_answer(previous_thoughts_states, responses)

        for thought, score in zip(previous_thoughts, scores):
            new_thought = Thought.from_thought(thought)
            new_thought.score = score
            self.thoughts.append(new_thought)

    def _execute_batch_scoring(
        self,
        lm: AbstractLanguageModel,
        prompter: Prompter,
        parser: Parser,
        previous_thoughts: "list[list[Thought]]",
    ) -> None:
        """
        Execute scoring using batch processing for improved performance.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param previous_thoughts: List of thoughts to score.
        :type previous_thoughts: "list[list[Thought]]"
        """
        self.logger.info(f"Using batch scoring for {len(previous_thoughts)} thoughts")

        # Prepare all prompts for batch processing
        prompts = []
        thought_mapping = []  # Track which prompt belongs to which thought

        for thought in previous_thoughts:
            for _ in range(self.num_samples):
                prompt = prompter.score_prompt([thought.state])
                prompts.append(prompt)
                thought_mapping.append(thought)

        self.logger.debug(
            f"Prepared {len(prompts)} scoring prompts for batch processing"
        )

        # Use optimized async execution for batch processing
        responses = self._run_async_batch_safely(lm, prompts)

        # Process responses and assign scores
        for i in range(0, len(responses), self.num_samples):
            thought = thought_mapping[i]
            thought_responses = responses[i : i + self.num_samples]

            try:
                # Extract text from responses
                response_texts = []
                for response in thought_responses:
                    response_texts.extend(lm.get_response_texts([response]))

                self.logger.debug(
                    f"Scoring responses for thought {thought.id}: {response_texts}"
                )

                # Parse the score
                score = parser.parse_score_answer([thought.state], response_texts)[0]

                new_thought = Thought.from_thought(thought)
                new_thought.score = score
                self.thoughts.append(new_thought)

            except Exception as e:
                self.logger.error(
                    f"Error processing score for thought {thought.id}: {e}"
                )
                # Create a fallback thought with default score
                new_thought = Thought.from_thought(thought)
                new_thought.score = 0.0  # Default score for failed scoring
                self.thoughts.append(new_thought)

    def _run_async_batch_safely(
        self, lm: AbstractLanguageModel, prompts: "list[list[str]]"
    ) -> list[list[dict]]:
        """
        Safely run batch processing with optimized event loop management.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompts: List of prompts to process.
        :type prompts: "list[list[str]]"
        :return: List of responses.
        :rtype: list[list[dict]]
        :raises RuntimeError: If called from within an async context
        """

        async def _run_batch():
            return await self._run_batch_async(lm, prompts)

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # If we reach here, we're in an async context
            raise RuntimeError(
                "Cannot call _run_async_batch_safely from within an async context. "
                "Use 'await _run_batch_async()' instead or call from a synchronous context."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No event loop running, safe to use asyncio.run
                return asyncio.run(_run_batch())
            else:
                # Re-raise the error about being in async context
                raise

    async def _run_batch_async(
        self, lm: AbstractLanguageModel, prompts: "list[list[str]]"
    ) -> list[list[dict]]:
        """
        Run batch processing asynchronously for scoring.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompts: List of prompts to process.
        :type prompts: "list[list[str]]"
        :return: List of responses.
        :rtype: list[list[dict]]
        """
        async with lm:  # Use async context manager
            return await lm.query_batch(
                prompts, max_concurrent=self.max_concurrent, batch_size=self.batch_size
            )

    def _execute_sequential_scoring(
        self,
        lm: AbstractLanguageModel,
        prompter: Prompter,
        parser: Parser,
        previous_thoughts: "list[list[Thought]]",
    ) -> None:
        """
        Execute scoring using sequential processing (fallback method).

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param previous_thoughts: List of thoughts to score.
        :type previous_thoughts: "list[list[Thought]]"
        """
        self.logger.info(
            f"Using sequential scoring for {len(previous_thoughts)} thoughts"
        )

        for thought in previous_thoughts:
            new_thought = Thought.from_thought(thought)
            if self.scoring_function is not None:
                self.logger.debug(
                    "Using scoring function %s to score state",
                    self.scoring_function,
                )
                score = self.scoring_function(thought.state)
            else:
                prompt = prompter.score_prompt([thought.state])
                self.logger.debug("Prompt for LM: %s", prompt)

                responses = lm.get_response_texts(
                    lm.query(prompt, num_responses=self.num_samples)
                )
                self.logger.debug("Responses from LM: %s", responses)
                score = parser.parse_score_answer([thought.state], responses)[0]

            new_thought.score = score
            self.thoughts.append(new_thought)


class BatchAggregate(Operation):
    """
    Batch-aware operation to aggregate thoughts efficiently using concurrent processing.

    This operation optimizes the aggregation process by:
    - Processing multiple aggregation requests concurrently
    - Using batch processing for language model queries
    - Supporting multiple response generation for better results
    - Implementing configurable concurrency limits
    """

    operation_type: OperationType = OperationType.batch_aggregate

    def __init__(
        self,
        num_responses: int = 1,
        max_concurrent: Optional[int] = None,
        batch_size: Optional[int] = None,
        enable_batch_processing: bool = True,
    ) -> None:
        """
        Initializes a new BatchAggregate operation.

        :param num_responses: Number of responses to use for aggregation. Defaults to 1.
        :type num_responses: int
        :param max_concurrent: Maximum number of concurrent requests. Uses LM default if None.
        :type max_concurrent: Optional[int]
        :param batch_size: Maximum batch size for processing. Uses LM default if None.
        :type batch_size: Optional[int]
        :param enable_batch_processing: Whether to use batch processing optimizations. Defaults to True.
        :type enable_batch_processing: bool
        """
        super().__init__()
        self.num_responses: int = num_responses
        self.max_concurrent: Optional[int] = max_concurrent
        self.batch_size: Optional[int] = batch_size
        self.enable_batch_processing: bool = enable_batch_processing
        self.thoughts: "list[list[Thought]]" = []

    def get_thoughts(self) -> "list[list[Thought]]":
        """
        Returns the thoughts associated with the operation after aggregation.

        :return: List of aggregated thoughts.
        :rtype: "list[list[Thought]]"
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the BatchAggregate operation using efficient batch processing.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        assert (
            len(self.predecessors) >= 1
        ), "BatchAggregate operation must have at least one predecessor"

        previous_thoughts: "list[list[Thought]]" = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            return

        # applied in order of score
        base_state: dict = {}
        for thought in sorted(previous_thoughts, key=lambda thought: thought.score):
            base_state = {**base_state, **thought.state}

        previous_thought_states = [thought.state for thought in previous_thoughts]

        # Check if the language model supports batch processing and we have multiple responses
        if (
            self.enable_batch_processing
            and hasattr(lm, "query_batch")
            and self.num_responses > 1
        ):
            self._execute_batch_aggregation(
                lm, prompter, parser, previous_thought_states, base_state
            )
        else:
            self._execute_sequential_aggregation(
                lm, prompter, parser, previous_thought_states, base_state
            )

        self.logger.info(
            "BatchAggregate operation %d created %d aggregated thoughts",
            self.id,
            len(self.thoughts),
        )

    def _execute_batch_aggregation(
        self,
        lm: AbstractLanguageModel,
        prompter: Prompter,
        parser: Parser,
        previous_thought_states: list[dict],
        base_state: dict,
    ) -> None:
        """
        Execute aggregation using batch processing for improved performance.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param previous_thought_states: List of thought states to aggregate.
        :type previous_thought_states: list[dict]
        :param base_state: Base state for new thoughts.
        :type base_state: dict
        """
        self.logger.info(f"Using batch aggregation for {self.num_responses} responses")

        # Prepare prompts for batch processing
        prompt = prompter.aggregation_prompt(previous_thought_states)
        prompts = [prompt] * self.num_responses

        self.logger.debug(
            f"Prepared {len(prompts)} aggregation prompts for batch processing"
        )

        # Use optimized async execution for batch processing
        responses = self._run_async_batch_safely(lm, prompts)

        # Process responses and create aggregated thoughts
        response_texts = []
        for response in responses:
            try:
                response_texts.extend(lm.get_response_texts([response]))
            except Exception as e:
                self.logger.error(
                    f"Error extracting text from aggregation response: {e}"
                )
                response_texts.append(f"Error: {str(e)}")

        self.logger.debug("Aggregation responses from LM: %s", response_texts)

        try:
            parsed = parser.parse_aggregation_answer(
                previous_thought_states, response_texts
            )

            if isinstance(parsed, dict):
                parsed = [parsed]
            for new_state in parsed:
                self.thoughts.append(Thought({**base_state, **new_state}))
        except Exception as e:
            self.logger.error(f"Error parsing aggregation responses: {e}")
            # Create a fallback aggregated thought
            fallback_state = {
                **base_state,
                "aggregation_error": str(e),
                "batch_processing_failed": True,
            }
            self.thoughts.append(Thought(fallback_state))

    def _run_async_batch_safely(
        self, lm: AbstractLanguageModel, prompts: "list[list[str]]"
    ) -> list[list[dict]]:
        """
        Safely run batch processing with optimized event loop management.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompts: List of prompts to process.
        :type prompts: "list[list[str]]"
        :return: List of responses.
        :rtype: list[list[dict]]
        :raises RuntimeError: If called from within an async context
        """

        async def _run_batch():
            return await self._run_batch_async(lm, prompts)

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # If we reach here, we're in an async context
            raise RuntimeError(
                "Cannot call _run_async_batch_safely from within an async context. "
                "Use 'await _run_batch_async()' instead or call from a synchronous context."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No event loop running, safe to use asyncio.run
                return asyncio.run(_run_batch())
            else:
                # Re-raise the error about being in async context
                raise

    async def _run_batch_async(
        self, lm: AbstractLanguageModel, prompts: "list[list[str]]"
    ) -> list[list[dict]]:
        """
        Run batch processing asynchronously for aggregation.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompts: List of prompts to process.
        :type prompts: "list[list[str]]"
        :return: List of responses.
        :rtype: list[list[dict]]
        """
        async with lm:  # Use async context manager
            return await lm.query_batch(
                prompts, max_concurrent=self.max_concurrent, batch_size=self.batch_size
            )

    def _execute_sequential_aggregation(
        self,
        lm: AbstractLanguageModel,
        prompter: Prompter,
        parser: Parser,
        previous_thought_states: list[dict],
        base_state: dict,
    ) -> None:
        """
        Execute aggregation using sequential processing (fallback method).

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param previous_thought_states: List of thought states to aggregate.
        :type previous_thought_states: list[dict]
        :param base_state: Base state for new thoughts.
        :type base_state: dict
        """
        self.logger.info(
            f"Using sequential aggregation for {self.num_responses} responses"
        )

        prompt = prompter.aggregation_prompt(previous_thought_states)
        self.logger.debug("Prompt for LM: %s", prompt)

        responses = lm.get_response_texts(
            lm.query(prompt, num_responses=self.num_responses)
        )

        self.logger.debug("Responses from LM: %s", responses)

        parsed = parser.parse_aggregation_answer(previous_thought_states, responses)

        if isinstance(parsed, dict):
            parsed = [parsed]
        for new_state in parsed:
            self.thoughts.append(Thought({**base_state, **new_state}))
