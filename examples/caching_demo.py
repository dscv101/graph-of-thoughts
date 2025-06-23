#!/usr/bin/env python3
"""
Intelligent Caching Demo for Graph of Thoughts MCP Client.

This script demonstrates the intelligent caching capabilities of the MCP client,
showing how caching improves performance for repeated queries and operations.

Features demonstrated:
- Response caching with parameter awareness
- Cache hit rate monitoring
- Performance comparison with/without caching
- Cache statistics and monitoring
- Configuration caching benefits
"""

import time
import json
import tempfile
import sys
import os
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MCP language model
from graph_of_thoughts.language_models import MCPLanguageModel
from graph_of_thoughts.language_models.caching import get_global_cache_stats, clear_global_cache


def create_demo_config() -> str:
    """Create a demo configuration file for testing."""
    config = {
        "mcp_demo": {
            "transport": {
                "type": "stdio",
                "command": "echo",
                "args": [],
                "env": {}
            },
            "client_info": {
                "name": "caching-demo",
                "version": "1.0.0"
            },
            "capabilities": {
                "sampling": {}
            },
            "default_sampling_params": {
                "temperature": 0.7,
                "maxTokens": 1000
            },
            "connection_config": {
                "timeout": 10.0,
                "retry_attempts": 1
            },
            "cost_tracking": {
                "prompt_token_cost": 0.001,
                "response_token_cost": 0.002
            },
            "caching": {
                "max_size": 100,
                "default_ttl": 300.0,
                "response_cache_size": 50,
                "config_cache_size": 10,
                "metadata_cache_size": 25,
                "response_ttl": 180.0,
                "config_ttl": 600.0,
                "metadata_ttl": 300.0
            }
        }
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        return f.name


def demo_response_caching():
    """Demonstrate response caching capabilities."""
    print("=== Response Caching Demo ===")

    # Test the caching system directly without MCP connections
    from graph_of_thoughts.language_models.caching import MultiLevelCacheManager, CacheConfig

    config = CacheConfig(
        response_cache_size=10,
        response_ttl=300.0
    )
    cache_manager = MultiLevelCacheManager(config)

    # Simulate queries
    queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "What is artificial intelligence?",  # Duplicate - should hit cache
        "Describe neural networks",
        "What is artificial intelligence?",  # Another duplicate
    ]

    print("Simulating queries with caching...")

    for i, query in enumerate(queries, 1):
        start_time = time.time()

        # Check cache first
        cached_response = cache_manager.get_response(query, temperature=0.7)

        if cached_response is not None:
            # Cache hit
            elapsed = time.time() - start_time
            print(f"Query {i}: '{query[:30]}...' - {elapsed*1000:.1f}ms (CACHE HIT)")
        else:
            # Cache miss - simulate processing time
            time.sleep(0.1)  # Simulate network/processing delay
            response = f"Response to: {query}"
            cache_manager.put_response(query, response, temperature=0.7)
            elapsed = time.time() - start_time
            print(f"Query {i}: '{query[:30]}...' - {elapsed*1000:.1f}ms (CACHE MISS)")

    # Show cache statistics
    stats = cache_manager.get_all_stats()
    if stats:
        response_stats = stats.get('response_cache', {})
        print(f"\nCache Statistics:")
        print(f"  Hits: {response_stats.get('hits', 0)}")
        print(f"  Misses: {response_stats.get('misses', 0)}")
        print(f"  Hit Rate: {response_stats.get('hit_rate', 0):.1%}")
        print(f"  Cache Size: {response_stats.get('size', 0)}")

    print()


def demo_parameter_awareness():
    """Demonstrate parameter-aware caching."""
    print("=== Parameter-Aware Caching Demo ===")

    from graph_of_thoughts.language_models.caching import MultiLevelCacheManager, CacheConfig

    config = CacheConfig(response_cache_size=10)
    cache_manager = MultiLevelCacheManager(config)

    # Same query with different parameters should create different cache entries
    base_query = "Generate a creative story"

    print("Testing parameter-aware caching...")

    # These should all be cache misses (different parameters)
    test_cases = [
        (base_query, {"num_responses": 1}, "Single response"),
        (base_query, {"num_responses": 3}, "Multiple responses"),
        (base_query, {"num_responses": 1}, "Single response again"),  # Should be cache hit
        (base_query, {"temperature": 0.7}, "Different temperature"),
        (base_query, {"num_responses": 1}, "Single response repeat"),  # Should be cache hit
    ]

    for query, params, description in test_cases:
        start_time = time.time()

        # Check cache
        cached_response = cache_manager.get_response(query, **params)

        if cached_response is not None:
            elapsed = time.time() - start_time
            print(f"  {description}: {elapsed*1000:.1f}ms (CACHE HIT)")
        else:
            time.sleep(0.05)  # Simulate processing
            response = f"Response to: {query} with {params}"
            cache_manager.put_response(query, response, **params)
            elapsed = time.time() - start_time
            print(f"  {description}: {elapsed*1000:.1f}ms (CACHE MISS)")

    # Show updated statistics
    stats = cache_manager.get_all_stats()
    if stats:
        response_stats = stats.get('response_cache', {})
        print(f"\nUpdated Cache Statistics:")
        print(f"  Hit Rate: {response_stats.get('hit_rate', 0):.1%}")
        print(f"  Total Entries: {response_stats.get('size', 0)}")

    print()


def demo_configuration_caching():
    """Demonstrate configuration caching benefits."""
    print("=== Configuration Caching Demo ===")

    from graph_of_thoughts.language_models.caching import MultiLevelCacheManager, CacheConfig

    config = CacheConfig(config_cache_size=10)
    cache_manager = MultiLevelCacheManager(config)

    print("Simulating configuration loading...")

    config_path = "/path/to/config.json"
    model_name = "test_model"

    # First load - cache miss
    start_time = time.time()
    cached_config = cache_manager.get_config(config_path, model_name)
    if cached_config is None:
        time.sleep(0.1)  # Simulate file I/O
        test_config = {"model": "test", "params": {"temperature": 0.7}}
        cache_manager.put_config(config_path, model_name, test_config)
        first_load_time = time.time() - start_time
        print(f"First load (file I/O): {first_load_time*1000:.1f}ms")

    # Second load - cache hit
    start_time = time.time()
    cached_config = cache_manager.get_config(config_path, model_name)
    second_load_time = time.time() - start_time
    print(f"Second load (cache hit): {second_load_time*1000:.1f}ms")

    # Third load - also cache hit
    start_time = time.time()
    cached_config = cache_manager.get_config(config_path, model_name)
    third_load_time = time.time() - start_time
    print(f"Third load (cache hit): {third_load_time*1000:.1f}ms")

    # Show configuration cache statistics
    stats = cache_manager.get_all_stats()
    if stats:
        config_stats = stats.get('config_cache', {})
        print(f"\nConfiguration Cache Statistics:")
        print(f"  Hits: {config_stats.get('hits', 0)}")
        print(f"  Misses: {config_stats.get('misses', 0)}")
        print(f"  Hit Rate: {config_stats.get('hit_rate', 0):.1%}")

    print()


def demo_cache_monitoring():
    """Demonstrate cache monitoring and statistics."""
    print("=== Cache Monitoring Demo ===")

    from graph_of_thoughts.language_models.caching import MultiLevelCacheManager, CacheConfig

    config = CacheConfig(response_cache_size=10)
    cache_manager = MultiLevelCacheManager(config)

    # Simulate various operations
    queries = [
        "What is Python?",
        "What is Java?",
        "What is Python?",  # Cache hit
        "What is C++?",
        "What is Java?",    # Cache hit
    ]

    print("Performing operations and monitoring cache...")

    for i, query in enumerate(queries, 1):
        # Check cache
        cached_response = cache_manager.get_response(query)

        if cached_response is None:
            # Cache miss
            time.sleep(0.05)  # Simulate processing
            response = f"Response to: {query}"
            cache_manager.put_response(query, response)
            status = "MISS"
        else:
            status = "HIT"

        # Get current statistics
        stats = cache_manager.get_all_stats()
        if stats:
            response_stats = stats.get('response_cache', {})
            hit_rate = response_stats.get('hit_rate', 0)
            size = response_stats.get('size', 0)
            print(f"Operation {i}: {status} - Hit rate {hit_rate:.1%}, Cache size {size}")

    # Final statistics summary
    print("\nFinal Cache Summary:")
    stats = cache_manager.get_all_stats()
    if stats:
        for cache_type, cache_stats in stats.items():
            print(f"  {cache_type.replace('_', ' ').title()}:")
            print(f"    Hits: {cache_stats.get('hits', 0)}")
            print(f"    Misses: {cache_stats.get('misses', 0)}")
            print(f"    Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
            print(f"    Size: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}")

    print()


def demo_cache_management():
    """Demonstrate cache management operations."""
    print("=== Cache Management Demo ===")

    from graph_of_thoughts.language_models.caching import MultiLevelCacheManager, CacheConfig

    config = CacheConfig(response_cache_size=10)
    cache_manager = MultiLevelCacheManager(config)

    # Add some entries to cache
    queries = ["Query 1", "Query 2", "Query 3"]
    for query in queries:
        cache_manager.put_response(query, f"Response to {query}")

    # Show cache state before clearing
    stats_before = cache_manager.get_all_stats()
    if stats_before:
        response_stats = stats_before.get('response_cache', {})
        print(f"Before clearing - Cache size: {response_stats.get('size', 0)}")

    # Clear cache
    print("Clearing cache...")
    cache_manager.clear_all()

    # Show cache state after clearing
    stats_after = cache_manager.get_all_stats()
    if stats_after:
        response_stats = stats_after.get('response_cache', {})
        print(f"After clearing - Cache size: {response_stats.get('size', 0)}")

    # Test global cache operations
    print("\nTesting global cache operations...")
    global_stats = get_global_cache_stats()
    print(f"Global cache stats available: {bool(global_stats)}")

    clear_global_cache()
    print("Global cache cleared")

    print()


def main():
    """Run all caching demos."""
    print("🚀 Intelligent Caching System Demo")
    print("=" * 50)
    print()
    
    demos = [
        demo_response_caching,
        demo_parameter_awareness,
        demo_configuration_caching,
        demo_cache_monitoring,
        demo_cache_management,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"Demo {demo.__name__} failed: {e}")
            print()
    
    print("✅ Caching demo completed!")
    print("\nKey Benefits Demonstrated:")
    print("• Response caching reduces latency for repeated queries")
    print("• Parameter-aware keys ensure cache accuracy")
    print("• Configuration caching speeds up initialization")
    print("• Comprehensive monitoring enables optimization")
    print("• Easy cache management and cleanup")


if __name__ == "__main__":
    main()
