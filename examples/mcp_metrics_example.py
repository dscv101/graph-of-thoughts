#!/usr/bin/env python3
"""
MCP Metrics and Monitoring Example.

This example demonstrates how to use the comprehensive metrics and monitoring
system with the MCP implementation. It shows how to:

1. Enable metrics collection in configuration
2. Track request performance and errors
3. Monitor circuit breaker health
4. Export metrics in different formats
5. Set up custom monitoring callbacks

Requirements:
    - MCP server (Claude Desktop, VSCode, or remote server)
    - Configuration file with metrics enabled
    - Optional: Prometheus or other monitoring system
"""

import asyncio
import json

# Add the parent directory to the path to import graph_of_thoughts
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from graph_of_thoughts.language_models import MCPLanguageModel


def setup_custom_metrics_callbacks(lm):
    """Set up custom metrics export callbacks."""

    def prometheus_export_callback(metrics):
        """Export metrics to Prometheus format."""
        prometheus_data = lm.export_metrics("prometheus")
        print("\n=== Prometheus Metrics ===")
        print(
            prometheus_data[:500] + "..."
            if len(prometheus_data) > 500
            else prometheus_data
        )

    def health_check_callback(metrics):
        """Perform health checks based on metrics."""
        health = lm.get_health_status()
        if health["overall_status"] != "healthy":
            print(f"\n⚠️  Health Alert: System status is {health['overall_status']}")
            for component, status in health["components"].items():
                if status.get("status") != "healthy":
                    print(f"   - {component}: {status.get('status', 'unknown')}")

    # Add callbacks to the metrics collector
    if lm.metrics_collector:
        lm.metrics_collector.add_export_callback(prometheus_export_callback)
        lm.metrics_collector.add_export_callback(health_check_callback)


async def demonstrate_metrics_collection():
    """Demonstrate comprehensive metrics collection."""

    print("🔍 MCP Metrics and Monitoring Demonstration")
    print("=" * 50)

    # Initialize MCP client with metrics enabled
    config_path = (
        Path(__file__).parent.parent
        / "graph_of_thoughts"
        / "language_models"
        / "mcp_config_with_metrics_template.json"
    )

    try:
        lm = MCPLanguageModel(
            config_path=str(config_path), model_name="mcp_claude_desktop", cache=True
        )

        # Set up custom metrics callbacks
        setup_custom_metrics_callbacks(lm)

        print(
            f"✅ Initialized MCP client with metrics: {lm.metrics_collector is not None}"
        )

    except Exception as e:
        print(f"❌ Failed to initialize MCP client: {e}")
        print("💡 Make sure you have a valid MCP configuration file")
        return

    async with lm:
        print("\n📊 Starting metrics collection...")

        # Perform various operations to generate metrics
        test_queries = [
            "What is machine learning?",
            "Explain quantum computing in simple terms",
            "Write a short poem about technology",
            "What are the benefits of renewable energy?",
            "How does artificial intelligence work?",
        ]

        print(f"\n🚀 Executing {len(test_queries)} test queries...")

        for i, query in enumerate(test_queries, 1):
            try:
                print(f"   Query {i}/{len(test_queries)}: {query[:30]}...")

                start_time = time.time()
                response = await lm._query_async(query)
                duration = time.time() - start_time

                text = lm.get_response_texts(response)[0]
                print(f"   ✅ Response received in {duration:.2f}s ({len(text)} chars)")

                # Add some delay between requests
                await asyncio.sleep(1)

            except Exception as e:
                print(f"   ❌ Query failed: {e}")

        # Demonstrate metrics retrieval
        print("\n📈 Current Metrics Summary:")
        print("-" * 30)

        metrics = lm.get_metrics()
        if metrics:
            req_metrics = metrics["requests"]
            print(f"Total Requests: {req_metrics['total']}")
            print(f"Successful: {req_metrics['successful']}")
            print(f"Failed: {req_metrics['failed']}")
            print(f"Success Rate: {100 - req_metrics['error_rate']:.1f}%")
            print(f"Average Latency: {req_metrics['avg_latency_ms']:.2f}ms")
            print(f"95th Percentile: {req_metrics['p95_latency_ms']:.2f}ms")
            print(f"Total Tokens: {metrics['tokens']['total']}")

            # Method-specific metrics
            if metrics["methods"]:
                print(f"\n📋 Method-Specific Metrics:")
                for method, method_data in metrics["methods"].items():
                    print(f"   {method}:")
                    print(f"     Requests: {method_data['total_requests']}")
                    print(f"     Avg Latency: {method_data['avg_latency_ms']:.2f}ms")
                    print(f"     Error Rate: {method_data['error_rate']:.1f}%")

        # Circuit breaker status
        print(f"\n🔧 Circuit Breaker Status:")
        print("-" * 30)
        cb_status = lm.get_circuit_breaker_status()
        if cb_status:
            print(f"State: {cb_status['state']}")
            print(f"Healthy: {cb_status['is_healthy']}")
            print(f"Total Requests: {cb_status['total_requests']}")
            print(f"Failed Requests: {cb_status['failed_requests']}")
            if cb_status["total_requests"] > 0:
                error_rate = (
                    cb_status["failed_requests"] / cb_status["total_requests"]
                ) * 100
                print(f"Error Rate: {error_rate:.1f}%")
        else:
            print("Circuit breaker not enabled")

        # Overall health status
        print(f"\n🏥 Health Status:")
        print("-" * 30)
        health = lm.get_health_status()
        print(f"Overall Status: {health['overall_status']}")
        for component, status in health["components"].items():
            component_status = status.get("status", "unknown")
            print(f"   {component}: {component_status}")

        # Export metrics in different formats
        print(f"\n📤 Metrics Export Examples:")
        print("-" * 30)

        # JSON export
        json_metrics = lm.export_metrics("json")
        print(f"JSON Export: {len(json_metrics)} characters")

        # Prometheus export
        prometheus_metrics = lm.export_metrics("prometheus")
        print(f"Prometheus Export: {len(prometheus_metrics)} characters")

        # CSV export
        csv_metrics = lm.export_metrics("csv")
        print(f"CSV Export: {len(csv_metrics)} characters")

        # Trigger metrics export (calls configured callbacks)
        print(f"\n🔄 Triggering metrics export...")
        lm.trigger_metrics_export()

        # Demonstrate error tracking
        print(f"\n❌ Error Summary:")
        print("-" * 30)
        error_summary = lm.get_error_summary()
        if error_summary and error_summary["total_errors"] > 0:
            print(f"Total Errors: {error_summary['total_errors']}")
            print(f"Error Rate: {error_summary['error_rate']:.1f}%")
            print(f"Error Types: {list(error_summary['error_counts'].keys())}")
        else:
            print("No errors recorded")

        print(f"\n✨ Metrics demonstration completed!")
        print(f"💡 Check the exported metrics files for detailed data")


def demonstrate_metrics_configuration():
    """Show how to configure metrics in different scenarios."""

    print("\n⚙️  Metrics Configuration Examples:")
    print("=" * 40)

    # Basic metrics configuration
    basic_config = {
        "metrics": {"enabled": True, "export_interval": 60.0, "export_format": "json"}
    }

    # Advanced metrics configuration
    advanced_config = {
        "metrics": {
            "enabled": True,
            "export_interval": 30.0,
            "export_format": "prometheus",
            "max_history_size": 2000,
            "include_detailed_timings": True,
            "export_file": "mcp_metrics.txt",
            "export_to_console": True,
        }
    }

    # Production monitoring configuration
    production_config = {
        "metrics": {
            "enabled": True,
            "export_interval": 15.0,
            "export_format": "prometheus",
            "max_history_size": 5000,
            "include_detailed_timings": False,
            "export_file": "/var/log/mcp/metrics.prom",
        },
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 3,
            "recovery_timeout": 60.0,
            "monitoring_window": 300.0,
        },
    }

    print("Basic Configuration:")
    print(json.dumps(basic_config, indent=2))

    print("\nAdvanced Configuration:")
    print(json.dumps(advanced_config, indent=2))

    print("\nProduction Configuration:")
    print(json.dumps(production_config, indent=2))


async def main():
    """Main demonstration function."""

    print("🎯 MCP Metrics and Monitoring System")
    print("=" * 50)
    print("This example demonstrates comprehensive metrics collection")
    print("and monitoring capabilities for MCP implementations.")
    print()

    # Show configuration examples
    demonstrate_metrics_configuration()

    # Run the main demonstration
    await demonstrate_metrics_collection()

    print("\n🎉 Demonstration completed!")
    print("📚 For more information, see the metrics documentation")


if __name__ == "__main__":
    asyncio.run(main())
