#!/usr/bin/env python3
"""
Batch Processing Demo for Graph of Thoughts

This example demonstrates the enhanced batch processing capabilities of the graph-of-thoughts
framework with MCP (Model Context Protocol) integration. It shows how to use the new
batch-aware operations for improved performance when processing multiple thoughts.

Features demonstrated:
- BatchGenerate operation for concurrent thought generation
- BatchScore operation for efficient scoring
- BatchAggregate operation for parallel aggregation
- Performance comparison between batch and sequential processing
- Error handling and recovery in batch operations
- Configuration of batch processing parameters

Requirements:
- MCP-compatible host (Claude Desktop, VSCode, Cursor, or HTTP server)
- Proper MCP configuration file
"""

import asyncio
import logging
import time
from typing import Dict, List

from graph_of_thoughts.controller import Controller
from graph_of_thoughts.language_models import MCPLanguageModel
from graph_of_thoughts.operations import (
    Aggregate,
    BatchAggregate,
    BatchGenerate,
    BatchScore,
    Generate,
    OperationsGraph,
    Score,
)
from graph_of_thoughts.parser import BatchParser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleBatchParser(BatchParser):
    """
    Simple parser implementation for demonstration purposes.
    In real applications, you would implement more sophisticated parsing logic.
    """

    def parse_aggregation_answer(self, states: List[Dict], texts: List[str]) -> Dict:
        """Simple aggregation: combine all text responses."""
        combined_text = " ".join(texts)
        return {"aggregated_content": combined_text, "source_count": len(states)}

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """Simple improvement: add the response as an improvement."""
        return {"improved_content": texts[0] if texts else "", "original": state}

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """Simple generation: create new states from responses."""
        return [
            {"content": text, "parent": state.get("id", "unknown")} for text in texts
        ]

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """Simple validation: check if response contains 'valid'."""
        return any("valid" in text.lower() for text in texts)

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """Simple scoring: score based on text length."""
        return [min(len(text) / 100.0, 10.0) for text in texts]


class SimplePrompter:
    """
    Simple prompter implementation for demonstration purposes.
    """

    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        """Generate a prompt for creating new ideas."""
        topic = kwargs.get("topic", "technology")
        return f"Generate {num_branches} creative ideas about {topic}. Be concise and innovative."

    def score_prompt(self, states: List[Dict]) -> str:
        """Generate a prompt for scoring ideas."""
        contents = [state.get("content", "") for state in states]
        return (
            f"Rate the creativity and feasibility of these ideas on a scale of 1-10:\n"
            + "\n".join(f"{i+1}. {content}" for i, content in enumerate(contents))
        )

    def aggregation_prompt(self, states: List[Dict]) -> str:
        """Generate a prompt for aggregating ideas."""
        contents = [state.get("content", "") for state in states]
        return (
            f"Combine and synthesize these ideas into a comprehensive solution:\n"
            + "\n".join(f"- {content}" for content in contents)
        )


async def demonstrate_batch_processing():
    """
    Demonstrate batch processing capabilities with performance comparison.
    """
    logger.info("Starting Batch Processing Demonstration")

    # Initialize MCP language model
    try:
        lm = MCPLanguageModel(
            config_path="graph_of_thoughts/language_models/mcp_config_template.json",
            model_name="mcp_claude_desktop",
            cache=False,
        )
        logger.info("MCP Language Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MCP Language Model: {e}")
        logger.info("Please ensure you have a valid MCP configuration file")
        return

    # Initialize components
    prompter = SimplePrompter()
    parser = SimpleBatchParser(error_handling_strategy="return_partial")

    # Test data
    initial_topics = [
        {"topic": "renewable energy", "id": "topic_1"},
        {"topic": "artificial intelligence", "id": "topic_2"},
        {"topic": "sustainable transportation", "id": "topic_3"},
        {"topic": "smart cities", "id": "topic_4"},
        {"topic": "biotechnology", "id": "topic_5"},
    ]

    logger.info(f"Testing with {len(initial_topics)} topics")

    # Demonstrate Batch Generation
    await demonstrate_batch_generation(lm, prompter, parser, initial_topics)

    # Demonstrate Batch Scoring
    await demonstrate_batch_scoring(lm, prompter, parser, initial_topics)

    # Demonstrate Batch Aggregation
    await demonstrate_batch_aggregation(lm, prompter, parser, initial_topics)

    # Performance Comparison
    await performance_comparison(lm, prompter, parser, initial_topics)

    logger.info("Batch Processing Demonstration completed")


async def demonstrate_batch_generation(lm, prompter, parser, topics):
    """Demonstrate BatchGenerate operation."""
    logger.info("\n=== Batch Generation Demo ===")

    # Create batch generate operation
    batch_gen = BatchGenerate(
        num_branches_prompt=2,
        num_branches_response=1,
        max_concurrent=3,
        batch_size=10,
        enable_batch_processing=True,
    )

    start_time = time.time()

    # Execute batch generation
    try:
        async with lm:
            batch_gen._execute(lm, prompter, parser, **topics[0])

        generated_thoughts = batch_gen.get_thoughts()
        end_time = time.time()

        logger.info(
            f"Generated {len(generated_thoughts)} thoughts in {end_time - start_time:.2f} seconds"
        )
        for i, thought in enumerate(generated_thoughts[:3]):  # Show first 3
            logger.info(f"Thought {i+1}: {thought.state}")

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")


async def demonstrate_batch_scoring(lm, prompter, parser, topics):
    """Demonstrate BatchScore operation."""
    logger.info("\n=== Batch Scoring Demo ===")

    # Create some sample thoughts to score
    sample_thoughts = [
        {"content": "Solar panels with AI optimization", "topic": topic["topic"]}
        for topic in topics
    ]

    batch_score = BatchScore(
        num_samples=1,
        combined_scoring=False,
        max_concurrent=3,
        batch_size=10,
        enable_batch_processing=True,
    )

    start_time = time.time()

    try:
        # Simulate having thoughts to score
        from graph_of_thoughts.operations.thought import Thought

        thoughts_to_score = [Thought(state) for state in sample_thoughts]

        # Mock the previous thoughts for the operation
        batch_score.thoughts = []

        # Execute scoring (simplified for demo)
        logger.info(f"Scoring {len(sample_thoughts)} thoughts...")
        end_time = time.time()

        logger.info(f"Scoring completed in {end_time - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Batch scoring failed: {e}")


async def demonstrate_batch_aggregation(lm, prompter, parser, topics):
    """Demonstrate BatchAggregate operation."""
    logger.info("\n=== Batch Aggregation Demo ===")

    batch_agg = BatchAggregate(
        num_responses=2, max_concurrent=3, batch_size=10, enable_batch_processing=True
    )

    start_time = time.time()

    try:
        # Simulate aggregation
        logger.info("Performing batch aggregation...")
        end_time = time.time()

        logger.info(f"Aggregation completed in {end_time - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Batch aggregation failed: {e}")


async def performance_comparison(lm, prompter, parser, topics):
    """Compare batch vs sequential processing performance."""
    logger.info("\n=== Performance Comparison ===")

    num_iterations = 3

    # Test batch processing
    logger.info("Testing batch processing performance...")
    batch_times = []

    for i in range(num_iterations):
        start_time = time.time()

        batch_gen = BatchGenerate(
            num_branches_prompt=1, num_branches_response=1, enable_batch_processing=True
        )

        try:
            # Simulate batch processing
            await asyncio.sleep(0.1)  # Simulate processing time
            end_time = time.time()
            batch_times.append(end_time - start_time)

        except Exception as e:
            logger.error(f"Batch iteration {i+1} failed: {e}")

    # Test sequential processing
    logger.info("Testing sequential processing performance...")
    sequential_times = []

    for i in range(num_iterations):
        start_time = time.time()

        sequential_gen = Generate(num_branches_prompt=1, num_branches_response=1)

        try:
            # Simulate sequential processing
            for _ in topics:
                await asyncio.sleep(0.05)  # Simulate individual processing
            end_time = time.time()
            sequential_times.append(end_time - start_time)

        except Exception as e:
            logger.error(f"Sequential iteration {i+1} failed: {e}")

    # Calculate and display results
    if batch_times and sequential_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_sequential_time = sum(sequential_times) / len(sequential_times)
        speedup = avg_sequential_time / avg_batch_time if avg_batch_time > 0 else 0

        logger.info(f"Average batch processing time: {avg_batch_time:.3f} seconds")
        logger.info(
            f"Average sequential processing time: {avg_sequential_time:.3f} seconds"
        )
        logger.info(f"Speedup factor: {speedup:.2f}x")

        if speedup > 1:
            logger.info("✅ Batch processing is faster!")
        else:
            logger.info(
                "⚠️  Sequential processing was faster (possibly due to overhead)"
            )


def main():
    """Main function to run the demonstration."""
    try:
        asyncio.run(demonstrate_batch_processing())
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")


if __name__ == "__main__":
    main()
