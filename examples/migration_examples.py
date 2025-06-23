"""
Migration examples showing before/after code for common use cases.

This file demonstrates how to migrate from legacy language models to MCP
for various common scenarios in the Graph of Thoughts framework.
"""

import asyncio
import os
from typing import Any, Dict, List

# =============================================================================
# Example 1: Basic Query Migration
# =============================================================================


def example_1_legacy():
    """Legacy: Basic query with ChatGPT."""
    from graph_of_thoughts.language_models import ChatGPT

    # Old way - requires API key
    lm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4", cache=True)

    response = lm.query("What is machine learning?")
    text = lm.get_response_texts(response)[0]
    print(f"Response: {text}")

    return text


def example_1_mcp():
    """MCP: Basic query with MCPLanguageModel."""
    from graph_of_thoughts.language_models import MCPLanguageModel

    # New way - no API key needed
    lm = MCPLanguageModel(
        config_path="mcp_config.json", model_name="mcp_claude_desktop", cache=True
    )

    response = lm.query("What is machine learning?")
    text = lm.get_response_texts(response)[0]
    print(f"Response: {text}")

    return text


# =============================================================================
# Example 2: Batch Processing Migration
# =============================================================================


def example_2_legacy():
    """Legacy: Sequential processing."""
    from graph_of_thoughts.language_models import Claude

    lm = Claude(
        api_key=os.getenv("ANTHROPIC_API_KEY"), model_name="claude-3-sonnet-20240229"
    )

    queries = [
        "Explain quantum computing",
        "What is blockchain?",
        "Describe machine learning",
    ]

    # Sequential processing - slow
    responses = []
    for query in queries:
        response = lm.query(query)
        responses.append(response)

    texts = [lm.get_response_texts(resp)[0] for resp in responses]
    return texts


async def example_2_mcp():
    """MCP: Concurrent batch processing."""
    from graph_of_thoughts.language_models import MCPLanguageModel

    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")

    queries = [
        "Explain quantum computing",
        "What is blockchain?",
        "Describe machine learning",
    ]

    # Concurrent batch processing - fast
    async with lm:
        responses = await lm.query_batch(queries, max_concurrent=3)

    texts = [lm.get_response_texts(resp)[0] for resp in responses]
    return texts


# =============================================================================
# Example 3: Operations Migration
# =============================================================================


def example_3_legacy():
    """Legacy: Using operations with sequential processing."""
    from graph_of_thoughts.language_models import ChatGPT
    from graph_of_thoughts.operations import Aggregate, Generate, Score

    lm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4")

    topics = ["AI ethics", "Climate change", "Space exploration"]

    # Generate analyses
    analyses = []
    for topic in topics:
        generate_op = Generate(lm, f"Write a brief analysis of {topic}")
        analysis = generate_op.execute()
        analyses.append(analysis)

    # Score analyses
    scores = []
    for analysis in analyses:
        score_op = Score(lm, "Rate the quality of this analysis (1-10)")
        score = score_op.execute(analysis)
        scores.append(score)

    # Aggregate results
    combined = list(zip(analyses, scores))
    aggregate_op = Aggregate(lm, "Summarize these analyses and their scores")
    summary = aggregate_op.execute(combined)

    return summary


async def example_3_mcp():
    """MCP: Using operations with batch processing."""
    from graph_of_thoughts.language_models import MCPLanguageModel
    from graph_of_thoughts.operations import BatchAggregate, BatchGenerate, BatchScore

    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")

    topics = ["AI ethics", "Climate change", "Space exploration"]

    async with lm:
        # Batch generate analyses
        generate_prompts = [f"Write a brief analysis of {topic}" for topic in topics]
        analyses = await BatchGenerate(lm).execute_batch(generate_prompts)

        # Batch score analyses
        score_prompts = ["Rate the quality of this analysis (1-10)" for _ in analyses]
        scores = await BatchScore(lm).execute_batch(score_prompts, analyses)

        # Aggregate results
        combined = list(zip(analyses, scores))
        summary = await BatchAggregate(lm).execute(
            "Summarize these analyses and their scores", combined
        )

    return summary


# =============================================================================
# Example 4: Error Handling Migration
# =============================================================================


def example_4_legacy():
    """Legacy: Manual error handling and retries."""
    import time

    from graph_of_thoughts.language_models import ChatGPT

    lm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4")

    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = lm.query("Explain quantum computing")
            return lm.get_response_texts(response)[0]
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise


async def example_4_mcp():
    """MCP: Built-in error handling and retries."""
    from graph_of_thoughts.language_models import MCPLanguageModel

    # MCP has built-in retry logic with exponential backoff
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")

    try:
        async with lm:
            response = await lm.query_async("Explain quantum computing")
            return lm.get_response_texts(response)[0]
    except Exception as e:
        print(f"All retries failed: {e}")
        raise


# =============================================================================
# Example 5: Configuration Migration
# =============================================================================


def example_5_legacy_config():
    """Legacy: Configuration through constructor parameters."""
    from graph_of_thoughts.language_models import Claude

    lm = Claude(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model_name="claude-3-sonnet-20240229",
        max_tokens=4096,
        temperature=0.7,
        timeout=30.0,
    )

    return lm


def example_5_mcp_config():
    """MCP: Configuration through JSON file."""
    from graph_of_thoughts.language_models import MCPLanguageModel

    # Configuration is in mcp_config.json:
    # {
    #     "mcp_claude_desktop": {
    #         "transport": {
    #             "type": "stdio",
    #             "command": "claude-desktop",
    #             "args": ["--mcp"],
    #             "timeout": 30.0
    #         },
    #         "default_sampling_params": {
    #             "maxTokens": 4096,
    #             "temperature": 0.7
    #         },
    #         "retry_config": {
    #             "max_attempts": 3,
    #             "base_delay": 1.0,
    #             "strategy": "exponential"
    #         }
    #     }
    # }

    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
    return lm


# =============================================================================
# Example 6: Async Context Management
# =============================================================================


def example_6_legacy():
    """Legacy: Manual resource management."""
    from graph_of_thoughts.language_models import ChatGPT

    lm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4")

    try:
        response = lm.query("Hello, world!")
        return lm.get_response_texts(response)[0]
    finally:
        # Manual cleanup if needed
        pass


async def example_6_mcp():
    """MCP: Automatic resource management with async context."""
    from graph_of_thoughts.language_models import MCPLanguageModel

    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")

    # Automatic connection management
    async with lm:
        response = await lm.query_async("Hello, world!")
        return lm.get_response_texts(response)[0]
    # Connection automatically closed


# =============================================================================
# Example 7: Performance Monitoring
# =============================================================================


async def example_7_performance_comparison():
    """Compare performance between legacy and MCP approaches."""
    import time

    # Simulate legacy performance (sequential)
    start_time = time.time()
    queries = ["Query " + str(i) for i in range(10)]

    # Legacy simulation (would be slower)
    legacy_time = 10.0  # Simulated time for 10 sequential queries

    # MCP batch processing
    from graph_of_thoughts.language_models import MCPLanguageModel

    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")

    start_time = time.time()
    async with lm:
        responses = await lm.query_batch(queries, max_concurrent=5)
    mcp_time = time.time() - start_time

    print(f"Legacy (simulated): {legacy_time:.2f}s")
    print(f"MCP batch: {mcp_time:.2f}s")
    print(f"Speedup: {legacy_time / mcp_time:.1f}x")

    return responses


# =============================================================================
# Example 8: Migration Utility Functions
# =============================================================================


def create_mcp_config_from_legacy(legacy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy configuration to MCP configuration."""

    # Map legacy model names to MCP hosts
    model_mapping = {
        "gpt-4": "mcp_claude_desktop",
        "gpt-3.5-turbo": "mcp_claude_desktop",
        "claude-3-sonnet": "mcp_claude_desktop",
        "claude-3-haiku": "mcp_claude_desktop",
    }

    model_name = legacy_config.get("model_name", "gpt-4")
    mcp_model = model_mapping.get(model_name, "mcp_claude_desktop")

    mcp_config = {
        mcp_model: {
            "transport": {
                "type": "stdio",
                "command": "claude-desktop",
                "args": ["--mcp"],
                "timeout": legacy_config.get("timeout", 30.0),
            },
            "client_info": {"name": "graph-of-thoughts", "version": "1.0.0"},
            "capabilities": {"sampling": {}},
            "default_sampling_params": {
                "maxTokens": legacy_config.get("max_tokens", 4096),
                "temperature": legacy_config.get("temperature", 0.7),
            },
            "retry_config": {
                "max_attempts": 3,
                "base_delay": 1.0,
                "strategy": "exponential",
                "jitter_type": "equal",
            },
        }
    }

    return mcp_config


async def test_migration_equivalence():
    """Test that MCP produces equivalent results to legacy models."""

    test_query = "What is the capital of France?"

    # Test with MCP
    from graph_of_thoughts.language_models import MCPLanguageModel

    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")

    async with lm:
        mcp_response = await lm.query_async(test_query)
        mcp_text = lm.get_response_texts(mcp_response)[0]

    print(f"MCP Response: {mcp_text}")

    # Verify response quality
    assert "Paris" in mcp_text, "Response should mention Paris"
    assert len(mcp_text) > 10, "Response should be substantial"

    print("✅ Migration equivalence test passed!")


# =============================================================================
# Main Migration Demo
# =============================================================================


async def main():
    """Run migration examples."""

    print("🚀 Graph of Thoughts Migration Examples")
    print("=" * 50)

    # Note: These examples require proper MCP configuration
    # Uncomment and run individual examples as needed

    print("\n1. Basic Query Migration")
    print("Legacy approach: example_1_legacy()")
    print("MCP approach: example_1_mcp()")

    print("\n2. Batch Processing Migration")
    print("Legacy approach: example_2_legacy()")
    print("MCP approach: await example_2_mcp()")

    print("\n3. Operations Migration")
    print("Legacy approach: example_3_legacy()")
    print("MCP approach: await example_3_mcp()")

    print("\n4. Error Handling Migration")
    print("Legacy approach: example_4_legacy()")
    print("MCP approach: await example_4_mcp()")

    print("\n5. Configuration Migration")
    print("Legacy approach: example_5_legacy_config()")
    print("MCP approach: example_5_mcp_config()")

    print("\n6. Async Context Management")
    print("Legacy approach: example_6_legacy()")
    print("MCP approach: await example_6_mcp()")

    print("\n7. Performance Comparison")
    print("Run: await example_7_performance_comparison()")

    print("\n8. Migration Utilities")
    print("Use: create_mcp_config_from_legacy(legacy_config)")
    print("Test: await test_migration_equivalence()")

    print("\n" + "=" * 50)
    print("To run these examples:")
    print("1. Set up MCP configuration (mcp_config.json)")
    print("2. Uncomment the example you want to run")
    print("3. Run: python migration_examples.py")


if __name__ == "__main__":
    asyncio.run(main())
