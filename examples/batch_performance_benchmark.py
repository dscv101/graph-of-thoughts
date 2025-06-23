#!/usr/bin/env python3
"""
Batch Processing Performance Benchmark

This script benchmarks the performance improvements of batch processing
compared to sequential processing in the graph-of-thoughts framework.

It measures:
- Throughput (requests per second)
- Latency (time per request)
- Resource utilization
- Error rates
- Scalability characteristics
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from graph_of_thoughts.language_models import MCPLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    method: str
    total_requests: int
    total_time: float
    successful_requests: int
    failed_requests: int
    throughput: float  # requests per second
    avg_latency: float  # seconds per request
    min_latency: float
    max_latency: float
    median_latency: float
    error_rate: float


class PerformanceBenchmark:
    """Performance benchmark for batch vs sequential processing."""

    def __init__(self, lm: MCPLanguageModel):
        self.lm = lm
        self.test_queries = [
            "Explain quantum computing in simple terms",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Describe the process of photosynthesis",
            "What is blockchain technology?",
            "How do neural networks learn?",
            "What causes climate change?",
            "Explain the theory of relativity",
            "How do vaccines work?",
            "What is artificial intelligence?",
        ]

    async def benchmark_sequential_processing(
        self, num_requests: int
    ) -> BenchmarkResult:
        """Benchmark sequential processing."""
        logger.info(f"Benchmarking sequential processing with {num_requests} requests")

        start_time = time.time()
        latencies = []
        successful = 0
        failed = 0

        async with self.lm:
            for i in range(num_requests):
                query = self.test_queries[i % len(self.test_queries)]

                request_start = time.time()
                try:
                    response = await self.lm._query_async(query, num_responses=1)
                    request_end = time.time()

                    latencies.append(request_end - request_start)
                    successful += 1

                except Exception as e:
                    request_end = time.time()
                    latencies.append(request_end - request_start)
                    failed += 1
                    logger.debug(f"Request {i+1} failed: {e}")

        total_time = time.time() - start_time

        return BenchmarkResult(
            method="Sequential",
            total_requests=num_requests,
            total_time=total_time,
            successful_requests=successful,
            failed_requests=failed,
            throughput=num_requests / total_time,
            avg_latency=statistics.mean(latencies) if latencies else 0,
            min_latency=min(latencies) if latencies else 0,
            max_latency=max(latencies) if latencies else 0,
            median_latency=statistics.median(latencies) if latencies else 0,
            error_rate=failed / num_requests if num_requests > 0 else 0,
        )

    async def benchmark_batch_processing(
        self, num_requests: int, max_concurrent: int = 10, batch_size: int = 50
    ) -> BenchmarkResult:
        """Benchmark batch processing."""
        logger.info(
            f"Benchmarking batch processing with {num_requests} requests "
            f"(max_concurrent={max_concurrent}, batch_size={batch_size})"
        )

        # Prepare queries
        queries = [
            self.test_queries[i % len(self.test_queries)] for i in range(num_requests)
        ]

        start_time = time.time()

        try:
            async with self.lm:
                responses = await self.lm.query_batch(
                    queries, max_concurrent=max_concurrent, batch_size=batch_size
                )

            total_time = time.time() - start_time

            # Count successful vs failed responses
            successful = 0
            failed = 0

            for response in responses:
                if response.get("metadata", {}).get("error", False):
                    failed += 1
                else:
                    successful += 1

            # For batch processing, we estimate latency as total_time / num_requests
            # since individual request timing is not available
            estimated_latency = total_time / num_requests if num_requests > 0 else 0

            return BenchmarkResult(
                method="Batch",
                total_requests=num_requests,
                total_time=total_time,
                successful_requests=successful,
                failed_requests=failed,
                throughput=num_requests / total_time,
                avg_latency=estimated_latency,
                min_latency=estimated_latency,  # Approximation
                max_latency=estimated_latency,  # Approximation
                median_latency=estimated_latency,  # Approximation
                error_rate=failed / num_requests if num_requests > 0 else 0,
            )

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Batch processing failed: {e}")

            return BenchmarkResult(
                method="Batch",
                total_requests=num_requests,
                total_time=total_time,
                successful_requests=0,
                failed_requests=num_requests,
                throughput=0,
                avg_latency=0,
                min_latency=0,
                max_latency=0,
                median_latency=0,
                error_rate=1.0,
            )

    def print_results(self, results: List[BenchmarkResult]):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)

        # Header
        print(
            f"{'Method':<12} {'Requests':<10} {'Time(s)':<8} {'Success':<8} {'Failed':<8} "
            f"{'RPS':<8} {'Latency(s)':<12} {'Error%':<8}"
        )
        print("-" * 80)

        # Results
        for result in results:
            print(
                f"{result.method:<12} {result.total_requests:<10} {result.total_time:<8.2f} "
                f"{result.successful_requests:<8} {result.failed_requests:<8} "
                f"{result.throughput:<8.1f} {result.avg_latency:<12.3f} "
                f"{result.error_rate*100:<8.1f}"
            )

        print("-" * 80)

        # Calculate improvements
        if len(results) >= 2:
            sequential = next((r for r in results if r.method == "Sequential"), None)
            batch = next((r for r in results if r.method == "Batch"), None)

            if sequential and batch and sequential.throughput > 0:
                throughput_improvement = batch.throughput / sequential.throughput
                latency_improvement = (
                    sequential.avg_latency / batch.avg_latency
                    if batch.avg_latency > 0
                    else 0
                )

                print(f"\nIMPROVEMENTS:")
                print(f"Throughput improvement: {throughput_improvement:.2f}x")
                print(f"Latency improvement: {latency_improvement:.2f}x")

                if throughput_improvement > 1.2:
                    print(
                        "✅ Batch processing shows significant performance improvement!"
                    )
                elif throughput_improvement > 1.0:
                    print("✅ Batch processing shows modest performance improvement")
                else:
                    print(
                        "⚠️  Sequential processing was faster (possibly due to overhead)"
                    )

        print("=" * 80)

    async def run_scalability_test(self):
        """Test how batch processing scales with different request counts."""
        logger.info("Running scalability test...")

        request_counts = [5, 10, 20, 50]
        results = []

        for count in request_counts:
            logger.info(f"\nTesting with {count} requests...")

            # Test sequential
            seq_result = await self.benchmark_sequential_processing(count)
            results.append(seq_result)

            # Test batch
            batch_result = await self.benchmark_batch_processing(
                count, max_concurrent=5
            )
            results.append(batch_result)

            # Brief pause between tests
            await asyncio.sleep(1)

        # Print scalability results
        print("\n" + "=" * 60)
        print("SCALABILITY TEST RESULTS")
        print("=" * 60)
        print(
            f"{'Requests':<10} {'Sequential RPS':<15} {'Batch RPS':<12} {'Speedup':<10}"
        )
        print("-" * 60)

        for i in range(0, len(results), 2):
            if i + 1 < len(results):
                seq = results[i]
                batch = results[i + 1]
                speedup = batch.throughput / seq.throughput if seq.throughput > 0 else 0

                print(
                    f"{seq.total_requests:<10} {seq.throughput:<15.1f} "
                    f"{batch.throughput:<12.1f} {speedup:<10.2f}x"
                )

        print("=" * 60)


async def main():
    """Main benchmark function."""
    logger.info("Starting Performance Benchmark")

    # Initialize MCP language model
    try:
        lm = MCPLanguageModel(
            config_path="graph_of_thoughts/language_models/mcp_config_template.json",
            model_name="mcp_claude_desktop",
        )
        logger.info("✅ MCP Language Model initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize MCP Language Model: {e}")
        logger.info("Please ensure you have a valid MCP configuration file")
        return

    benchmark = PerformanceBenchmark(lm)

    # Run basic benchmark
    num_requests = 20
    results = []

    logger.info(f"\nRunning benchmark with {num_requests} requests...")

    # Test sequential processing
    seq_result = await benchmark.benchmark_sequential_processing(num_requests)
    results.append(seq_result)

    # Brief pause
    await asyncio.sleep(2)

    # Test batch processing
    batch_result = await benchmark.benchmark_batch_processing(
        num_requests, max_concurrent=5, batch_size=10
    )
    results.append(batch_result)

    # Print results
    benchmark.print_results(results)

    # Run scalability test
    await benchmark.run_scalability_test()

    logger.info("✅ Performance benchmark completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
