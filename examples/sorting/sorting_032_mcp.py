# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Example demonstrating how to use Graph of Thoughts with MCP (Model Context Protocol) language models.
This example shows how to connect to MCP hosts like Claude Desktop, VSCode, or Cursor.
"""

import csv
import datetime
import json
import logging
import os
# Type annotations use built-in types for Python 3.12+ compatibility
from typing import Callable

from graph_of_thoughts import controller, language_models, operations, parser, prompter

# Import the original sorting example components
try:
    from .sorting_032 import (
        SortingParser,
        SortingPrompter,
        cot,
        got,
        io,
        tot,
        tot2,
        utils,
    )
except ImportError:
    from sorting_032 import (
        SortingParser,
        SortingPrompter,
        cot,
        got,
        io,
        tot,
        tot2,
        utils,
    )


def run_mcp(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    mcp_model_name: str = "mcp_claude_desktop",
) -> float:
    """
    Controller function that executes each specified method for each specified
    sample using MCP language models while the budget is not exhausted.

    :param data_ids: Indices of the sample to be run.
    :type data_ids: List[int]
    :param methods: List of functions to generate Graphs of Operations.
    :type methods: Each function generates a Graph of Operation.
    :param budget: Language model budget for the execution in dollars.
    :type budget: float
    :param mcp_model_name: Name of the MCP model configuration to use.
    :type mcp_model_name: str
    :return: Spent budget in dollars.
    :rtype: float
    """

    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "sorting_032.csv")
    data = []
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([int(row[0]), row[1], row[2]])

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results_mcp")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{mcp_model_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)

    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "mcp_model": mcp_model_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__))

    for data in selected_data:
        logging.info(f"Running data {data[0]}: {data[1]}")
        if budget <= 0.0:
            logging.error(
                f"Budget has been depleted, stopping. Data {data[0]} has not been run."
            )
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            logging.info(f"Budget left: {budget}")
            if budget <= 0.0:
                logging.error(
                    f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                )
                break

            # Use MCP language model instead of ChatGPT
            mcp_config_path = os.path.join(
                os.path.dirname(__file__),
                "../../graph_of_thoughts/language_models/mcp_config.json",
            )

            # Fallback to template if config doesn't exist
            if not os.path.exists(mcp_config_path):
                mcp_config_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../graph_of_thoughts/language_models/mcp_config_template.json",
                )

            lm = language_models.MCPLanguageModel(
                mcp_config_path,
                model_name=mcp_model_name,
                cache=True,
            )

            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                SortingPrompter(),
                SortingParser(),
                {
                    "original": data[1],
                    "current": "",
                    "phase": 0,
                    "method": method.__name__,
                },
            )
            try:
                executor.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{data[0]}.json",
            )
            executor.output_graph(path)
            budget -= lm.cost

    return orig_budget - budget


def demo_mcp_connection() -> None:
    """
    Demonstrate basic MCP connection and usage.
    """
    print("Graph of Thoughts MCP Demo")
    print("=" * 40)

    # Test data
    to_be_sorted = "[3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9]"

    print(f"Input to sort: {to_be_sorted}")
    print()

    # Available MCP configurations
    mcp_configs = ["mcp_claude_desktop", "mcp_vscode", "mcp_cursor", "mcp_http_server"]

    print("Available MCP configurations:")
    for i, config in enumerate(mcp_configs, 1):
        print(f"  {i}. {config}")
    print()

    # For demo purposes, we'll use Claude Desktop
    selected_config = "mcp_claude_desktop"
    print(f"Using configuration: {selected_config}")
    print()

    try:
        # Create MCP language model
        mcp_config_path = os.path.join(
            os.path.dirname(__file__),
            "../../graph_of_thoughts/language_models/mcp_config_template.json",
        )

        lm = language_models.MCPLanguageModel(
            mcp_config_path,
            model_name=selected_config,
            cache=False,
        )

        print("MCP Language Model created successfully!")
        print(f"Transport type: {lm.transport_type}")
        print(f"Client info: {lm.client_info}")
        print(f"Capabilities: {lm.capabilities}")
        print("Note: This example uses the new MCP protocol-compliant implementation")
        print()

        # Create a simple Graph of Operations for demonstration
        gop = operations.GraphOfOperations()
        gop.append_operation(operations.Generate(1, 1))
        gop.append_operation(operations.Score(1, False, utils.num_errors))
        gop.append_operation(operations.GroundTruth(utils.test_sorting))

        # Create the Controller
        ctrl = controller.Controller(
            lm,
            gop,
            SortingPrompter(),
            SortingParser(),
            {"original": to_be_sorted, "current": "", "method": "io"},
        )

        print("Running Graph of Thoughts with MCP...")
        ctrl.run()

        print(f"Execution completed!")
        print(f"Estimated cost: ${lm.cost:.4f}")

        # Output results
        output_path = "mcp_demo_output.json"
        ctrl.output_graph(output_path)
        print(f"Results saved to: {output_path}")

    except Exception as e:
        print(f"Error during MCP demo: {e}")
        print("This is expected if no MCP host is running.")
        print(
            "To use MCP, ensure you have Claude Desktop, VSCode, or Cursor running with MCP support."
        )


if __name__ == "__main__":
    """
    MCP version of the sorting example.

    Input (x)   : an unordered list of 32 numbers between 0 and 9 (inclusive)
    Output (y)  : a sorted list of 32 numbers between 0 and 9 (inclusive)
    Correct     : y == sorted(x)
    Input Example:
        [0, 1, 9, 4, 2, 2, 0, 5, 1...]
    Output Example:
        [0, 0, 0, 0, 1, 1, 1, 1, 2...]
    """

    # Run the demo
    demo_mcp_connection()

    # Uncomment the following to run the full experiment with MCP
    # budget = 30
    # samples = [item for item in range(0, 10)]  # Smaller sample for demo
    # approaches = [io, cot]  # Simpler approaches for demo
    #
    # spent = run_mcp(samples, approaches, budget, "mcp_claude_desktop")
    # print(f"Spent {spent} out of {budget} budget using MCP.")
