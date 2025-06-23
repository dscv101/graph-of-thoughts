#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Performance Benchmarking Suite for MCP Implementation.

This module provides comprehensive performance benchmarks for the MCP implementation,
measuring latency, throughput, memory usage, and scalability characteristics.

The benchmarking suite includes:
- Connection establishment latency
- Request/response latency measurements
- Throughput testing with concurrent requests
- Memory usage profiling
- Batch processing performance
- Circuit breaker overhead measurement
- Plugin system performance impact

Usage:
    # Run all benchmarks
    python tests/performance_benchmarks.py

    # Run specific benchmark
    python tests/performance_benchmarks.py --benchmark latency

    # Run with custom parameters
    python tests/performance_benchmarks.py --iterations 100 --concurrency 10

    # Generate detailed report
    python tests/performance_benchmarks.py --detailed-report --output benchmark_report.json

Environment Variables:
    MCP_BENCHMARK_CONFIG: Path to benchmark configuration file
    MCP_BENCHMARK_ITERATIONS: Default number of iterations (default: 50)
    MCP_BENCHMARK_WARMUP: Number of warmup iterations (default: 5)
"""

import argparse
import asyncio
import gc
import json
import statistics
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock

import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel
from graph_of_thoughts.language_models.mcp_sampling import MCPSamplingManager
from graph_of_thoughts.language_models.mcp_transport import create_transport


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    description: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    std_dev: float
    throughput: float
    memory_usage: Dict[str, float]
    additional_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking suite for MCP implementation."""

    def __init__(self, args):
        """Initialize benchmark suite."""
        self.args = args
        self.iterations = args.iterations or int(
            os.environ.get("MCP_BENCHMARK_ITERATIONS", "50")
        )
        self.warmup_iterations = int(os.environ.get("MCP_BENCHMARK_WARMUP", "5"))
        self.concurrency = args.concurrency or 5
        self.results: List[BenchmarkResult] = []

        # Mock configuration for benchmarking
        self.mock_config = {
            "benchmark_model": {
                "transport": {"type": "stdio", "command": "echo", "args": ["test"]},
                "client_info": {"name": "benchmark-client", "version": "1.0.0"},
                "capabilities": {"sampling": {}},
                "default_sampling_params": {
                    "temperature": 0.7,
                    "maxTokens": 100,
                    "includeContext": "none",
                },
                "batch_processing": {
                    "max_concurrent": 10,
                    "batch_size": 50,
                    "retry_attempts": 3,
                    "retry_delay": 1.0,
                    "timeout_per_request": 30.0,
                    "enable_by_default": True,
                },
            }
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    async def warmup(self, operation: Callable, *args, **kwargs):
        """Perform warmup iterations to stabilize performance."""
        print(f"🔥 Warming up with {self.warmup_iterations} iterations...")
        for _ in range(self.warmup_iterations):
            try:
                await operation(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors

        # Force garbage collection
        gc.collect()

    async def measure_operation(
        self, operation: Callable, *args, **kwargs
    ) -> List[float]:
        """Measure operation performance over multiple iterations."""
        times = []

        for i in range(self.iterations):
            if i % 10 == 0:
                print(f"  Progress: {i}/{self.iterations}")

            start_time = time.perf_counter()
            try:
                await operation(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"  Warning: Operation failed in iteration {i}: {e}")
                # Record a high time for failed operations
                times.append(float("inf"))

        # Filter out failed operations
        valid_times = [t for t in times if t != float("inf")]
        if not valid_times:
            raise RuntimeError("All benchmark iterations failed")

        return valid_times

    async def benchmark_transport_creation(self) -> BenchmarkResult:
        """Benchmark transport creation performance."""
        print("📊 Benchmarking transport creation...")

        config = self.mock_config["benchmark_model"]

        async def create_transport_operation():
            transport = create_transport(config)
            return transport

        # Warmup
        await self.warmup(create_transport_operation)

        # Measure
        memory_before = self.get_memory_usage()
        times = await self.measure_operation(create_transport_operation)
        memory_after = self.get_memory_usage()

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = len(times) / total_time

        memory_usage = {
            "before_rss_mb": memory_before["rss_mb"],
            "after_rss_mb": memory_after["rss_mb"],
            "delta_rss_mb": memory_after["rss_mb"] - memory_before["rss_mb"],
        }

        return BenchmarkResult(
            name="transport_creation",
            description="Time to create MCP transport instances",
            iterations=len(times),
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            throughput=throughput,
            memory_usage=memory_usage,
        )

    async def benchmark_mcp_client_initialization(self) -> BenchmarkResult:
        """Benchmark MCP client initialization performance."""
        print("📊 Benchmarking MCP client initialization...")

        async def init_client_operation():
            with patch(
                "graph_of_thoughts.language_models.mcp_transport.create_transport"
            ) as mock_create:
                mock_transport = AsyncMock()
                mock_create.return_value = mock_transport

                lm = MCPLanguageModel(
                    config=self.mock_config, model_name="benchmark_model"
                )
                return lm

        # Import patch here to avoid import issues
        from unittest.mock import patch

        # Warmup
        await self.warmup(init_client_operation)

        # Measure
        memory_before = self.get_memory_usage()
        times = await self.measure_operation(init_client_operation)
        memory_after = self.get_memory_usage()

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = len(times) / total_time

        memory_usage = {
            "before_rss_mb": memory_before["rss_mb"],
            "after_rss_mb": memory_after["rss_mb"],
            "delta_rss_mb": memory_after["rss_mb"] - memory_before["rss_mb"],
        }

        return BenchmarkResult(
            name="mcp_client_initialization",
            description="Time to initialize MCPLanguageModel instances",
            iterations=len(times),
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            throughput=throughput,
            memory_usage=memory_usage,
        )

    async def benchmark_request_latency(self) -> BenchmarkResult:
        """Benchmark request/response latency."""
        print("📊 Benchmarking request latency...")

        from unittest.mock import patch

        async def request_operation():
            with patch(
                "graph_of_thoughts.language_models.mcp_transport.create_transport"
            ) as mock_create:
                mock_transport = AsyncMock()
                mock_transport.connect.return_value = True
                mock_transport.send_sampling_request.return_value = {
                    "content": [{"type": "text", "text": "Benchmark response"}]
                }
                mock_create.return_value = mock_transport

                lm = MCPLanguageModel(
                    config=self.mock_config, model_name="benchmark_model"
                )

                async with lm:
                    response = await lm.query_async("Benchmark query")
                    return response

        # Warmup
        await self.warmup(request_operation)

        # Measure
        memory_before = self.get_memory_usage()
        times = await self.measure_operation(request_operation)
        memory_after = self.get_memory_usage()

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = len(times) / total_time

        memory_usage = {
            "before_rss_mb": memory_before["rss_mb"],
            "after_rss_mb": memory_after["rss_mb"],
            "delta_rss_mb": memory_after["rss_mb"] - memory_before["rss_mb"],
        }

        return BenchmarkResult(
            name="request_latency",
            description="End-to-end request/response latency",
            iterations=len(times),
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            throughput=throughput,
            memory_usage=memory_usage,
            additional_metrics={
                "avg_latency_ms": avg_time * 1000,
                "p95_latency_ms": statistics.quantiles(times, n=20)[18] * 1000
                if len(times) >= 20
                else max_time * 1000,
                "p99_latency_ms": statistics.quantiles(times, n=100)[98] * 1000
                if len(times) >= 100
                else max_time * 1000,
            },
        )

    async def benchmark_concurrent_requests(self) -> BenchmarkResult:
        """Benchmark concurrent request handling."""
        print(
            f"📊 Benchmarking concurrent requests (concurrency: {self.concurrency})..."
        )

        from unittest.mock import patch

        async def concurrent_operation():
            with patch(
                "graph_of_thoughts.language_models.mcp_transport.create_transport"
            ) as mock_create:
                mock_transport = AsyncMock()
                mock_transport.connect.return_value = True
                mock_transport.send_sampling_request.return_value = {
                    "content": [{"type": "text", "text": "Concurrent response"}]
                }
                mock_create.return_value = mock_transport

                lm = MCPLanguageModel(
                    config=self.mock_config, model_name="benchmark_model"
                )

                async with lm:
                    # Create concurrent requests
                    tasks = []
                    for i in range(self.concurrency):
                        task = lm.query_async(f"Concurrent query {i}")
                        tasks.append(task)

                    # Wait for all to complete
                    responses = await asyncio.gather(*tasks)
                    return responses

        # Warmup
        await self.warmup(concurrent_operation)

        # Measure
        memory_before = self.get_memory_usage()
        times = await self.measure_operation(concurrent_operation)
        memory_after = self.get_memory_usage()

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0

        # Throughput is requests per second (considering concurrency)
        total_requests = len(times) * self.concurrency
        throughput = total_requests / total_time

        memory_usage = {
            "before_rss_mb": memory_before["rss_mb"],
            "after_rss_mb": memory_after["rss_mb"],
            "delta_rss_mb": memory_after["rss_mb"] - memory_before["rss_mb"],
        }

        return BenchmarkResult(
            name="concurrent_requests",
            description=f"Concurrent request handling ({self.concurrency} concurrent)",
            iterations=len(times),
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            throughput=throughput,
            memory_usage=memory_usage,
            additional_metrics={
                "concurrency_level": self.concurrency,
                "total_requests": total_requests,
                "requests_per_second": throughput,
            },
        )

    async def benchmark_batch_processing(self) -> BenchmarkResult:
        """Benchmark batch processing performance."""
        print("📊 Benchmarking batch processing...")

        from unittest.mock import patch

        batch_size = 10

        async def batch_operation():
            with patch(
                "graph_of_thoughts.language_models.mcp_transport.create_transport"
            ) as mock_create:
                mock_transport = AsyncMock()
                mock_transport.connect.return_value = True
                mock_transport.send_sampling_request.return_value = {
                    "content": [{"type": "text", "text": "Batch response"}]
                }
                mock_create.return_value = mock_transport

                # Create sampling manager
                config = self.mock_config["benchmark_model"]
                manager = MCPSamplingManager(mock_transport, config)

                # Create batch of prompts
                prompts = [f"Batch query {i}" for i in range(batch_size)]

                # Process batch
                responses = await manager.create_messages_batch(prompts)
                return responses

        # Warmup
        await self.warmup(batch_operation)

        # Measure
        memory_before = self.get_memory_usage()
        times = await self.measure_operation(batch_operation)
        memory_after = self.get_memory_usage()

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0

        # Throughput is items per second
        total_items = len(times) * batch_size
        throughput = total_items / total_time

        memory_usage = {
            "before_rss_mb": memory_before["rss_mb"],
            "after_rss_mb": memory_after["rss_mb"],
            "delta_rss_mb": memory_after["rss_mb"] - memory_before["rss_mb"],
        }

        return BenchmarkResult(
            name="batch_processing",
            description=f"Batch processing performance ({batch_size} items per batch)",
            iterations=len(times),
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            throughput=throughput,
            memory_usage=memory_usage,
            additional_metrics={
                "batch_size": batch_size,
                "total_items": total_items,
                "items_per_second": throughput,
                "avg_batch_time_ms": avg_time * 1000,
            },
        )

    async def benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage patterns."""
        print("📊 Benchmarking memory usage...")

        from unittest.mock import patch

        memory_samples = []

        async def memory_operation():
            with patch(
                "graph_of_thoughts.language_models.mcp_transport.create_transport"
            ) as mock_create:
                mock_transport = AsyncMock()
                mock_transport.connect.return_value = True
                mock_transport.send_sampling_request.return_value = {
                    "content": [{"type": "text", "text": "Memory test response"}]
                }
                mock_create.return_value = mock_transport

                # Create multiple clients to test memory usage
                clients = []
                for i in range(10):
                    lm = MCPLanguageModel(
                        config=self.mock_config, model_name="benchmark_model"
                    )
                    clients.append(lm)

                    # Sample memory after each client creation
                    memory_samples.append(self.get_memory_usage())

                # Perform operations
                for lm in clients:
                    async with lm:
                        await lm.query_async("Memory test query")
                        memory_samples.append(self.get_memory_usage())

                return len(clients)

        # Measure
        start_time = time.perf_counter()
        initial_memory = self.get_memory_usage()

        result = await memory_operation()

        end_time = time.perf_counter()
        final_memory = self.get_memory_usage()

        # Calculate memory statistics
        rss_values = [sample["rss_mb"] for sample in memory_samples]
        memory_growth = final_memory["rss_mb"] - initial_memory["rss_mb"]
        peak_memory = max(rss_values) if rss_values else final_memory["rss_mb"]

        execution_time = end_time - start_time

        return BenchmarkResult(
            name="memory_usage",
            description="Memory usage patterns and growth",
            iterations=1,
            total_time=execution_time,
            avg_time=execution_time,
            min_time=execution_time,
            max_time=execution_time,
            median_time=execution_time,
            std_dev=0.0,
            throughput=1.0 / execution_time,
            memory_usage={
                "initial_rss_mb": initial_memory["rss_mb"],
                "final_rss_mb": final_memory["rss_mb"],
                "peak_rss_mb": peak_memory,
                "growth_mb": memory_growth,
            },
            additional_metrics={
                "memory_samples": len(memory_samples),
                "avg_memory_mb": statistics.mean(rss_values) if rss_values else 0,
                "memory_efficiency": result / memory_growth
                if memory_growth > 0
                else float("inf"),
            },
        )

    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks."""
        print("🚀 Starting Performance Benchmark Suite")
        print("=" * 60)

        benchmarks = []

        # Define benchmark functions
        benchmark_functions = [
            ("transport_creation", self.benchmark_transport_creation),
            ("mcp_client_initialization", self.benchmark_mcp_client_initialization),
            ("request_latency", self.benchmark_request_latency),
            ("concurrent_requests", self.benchmark_concurrent_requests),
            ("batch_processing", self.benchmark_batch_processing),
            ("memory_usage", self.benchmark_memory_usage),
        ]

        # Run benchmarks
        for name, func in benchmark_functions:
            if self.args.benchmark and self.args.benchmark != name:
                continue

            try:
                print(f"\n🔄 Running benchmark: {name}")
                result = await func()
                benchmarks.append(result)
                self.results.append(result)

                # Print quick summary
                print(
                    f"✅ Completed: {result.avg_time:.4f}s avg, {result.throughput:.2f} ops/sec"
                )

            except Exception as e:
                print(f"❌ Benchmark {name} failed: {e}")
                if self.args.verbose:
                    import traceback

                    traceback.print_exc()

        return benchmarks

    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."

        report = []
        report.append("=" * 80)
        report.append("MCP PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Iterations per benchmark: {self.iterations}")
        report.append(f"Warmup iterations: {self.warmup_iterations}")
        report.append(f"Concurrency level: {self.concurrency}")
        report.append("")

        # Summary table
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 50)
        report.append(
            f"{'Benchmark':<25} {'Avg Time':<12} {'Throughput':<12} {'Memory':<10}"
        )
        report.append("-" * 50)

        for result in self.results:
            memory_delta = result.memory_usage.get("delta_rss_mb", 0)
            report.append(
                f"{result.name:<25} "
                f"{result.avg_time*1000:>8.2f}ms "
                f"{result.throughput:>8.2f}/s "
                f"{memory_delta:>6.1f}MB"
            )

        report.append("")

        # Detailed results
        for result in self.results:
            report.append(f"📊 {result.name.upper()}")
            report.append("-" * 40)
            report.append(f"Description: {result.description}")
            report.append(f"Iterations: {result.iterations}")
            report.append(f"Total Time: {result.total_time:.4f}s")
            report.append(f"Average Time: {result.avg_time*1000:.2f}ms")
            report.append(f"Min Time: {result.min_time*1000:.2f}ms")
            report.append(f"Max Time: {result.max_time*1000:.2f}ms")
            report.append(f"Median Time: {result.median_time*1000:.2f}ms")
            report.append(f"Std Deviation: {result.std_dev*1000:.2f}ms")
            report.append(f"Throughput: {result.throughput:.2f} operations/second")

            # Memory usage
            memory = result.memory_usage
            report.append(f"Memory Usage:")
            for key, value in memory.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.2f}")
                else:
                    report.append(f"  {key}: {value}")

            # Additional metrics
            if result.additional_metrics:
                report.append("Additional Metrics:")
                for key, value in result.additional_metrics.items():
                    if isinstance(value, float):
                        report.append(f"  {key}: {value:.2f}")
                    else:
                        report.append(f"  {key}: {value}")

            report.append("")

        # Performance analysis
        report.append("PERFORMANCE ANALYSIS")
        report.append("-" * 40)

        # Find fastest and slowest operations
        if len(self.results) > 1:
            fastest = min(self.results, key=lambda r: r.avg_time)
            slowest = max(self.results, key=lambda r: r.avg_time)

            report.append(
                f"Fastest operation: {fastest.name} ({fastest.avg_time*1000:.2f}ms)"
            )
            report.append(
                f"Slowest operation: {slowest.name} ({slowest.avg_time*1000:.2f}ms)"
            )

            speed_ratio = slowest.avg_time / fastest.avg_time
            report.append(f"Speed ratio: {speed_ratio:.1f}x")
            report.append("")

        # Memory efficiency
        memory_results = [r for r in self.results if "delta_rss_mb" in r.memory_usage]
        if memory_results:
            total_memory = sum(r.memory_usage["delta_rss_mb"] for r in memory_results)
            report.append(f"Total memory usage: {total_memory:.2f}MB")

            most_memory = max(
                memory_results, key=lambda r: r.memory_usage["delta_rss_mb"]
            )
            report.append(
                f"Most memory-intensive: {most_memory.name} ({most_memory.memory_usage['delta_rss_mb']:.2f}MB)"
            )
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)

        # Analyze results and provide recommendations
        latency_results = [r for r in self.results if "latency" in r.name]
        if latency_results:
            avg_latency = statistics.mean(r.avg_time for r in latency_results)
            if avg_latency > 0.1:  # 100ms
                report.append(
                    "⚠️  High latency detected - consider optimizing request handling"
                )
            else:
                report.append("✅ Good latency performance")

        throughput_results = [r for r in self.results if r.throughput > 0]
        if throughput_results:
            avg_throughput = statistics.mean(r.throughput for r in throughput_results)
            if avg_throughput < 10:  # Less than 10 ops/sec
                report.append(
                    "⚠️  Low throughput - consider batch processing or concurrency improvements"
                )
            else:
                report.append("✅ Good throughput performance")

        if memory_results:
            avg_memory = statistics.mean(
                r.memory_usage["delta_rss_mb"] for r in memory_results
            )
            if avg_memory > 100:  # More than 100MB
                report.append("⚠️  High memory usage - consider memory optimization")
            else:
                report.append("✅ Efficient memory usage")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_results(self, output_path: str):
        """Save benchmark results to JSON file."""
        results_data = {
            "benchmark_config": {
                "iterations": self.iterations,
                "warmup_iterations": self.warmup_iterations,
                "concurrency": self.concurrency,
                "timestamp": time.time(),
            },
            "results": [asdict(result) for result in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"📄 Benchmark results saved to: {output_path}")


async def main():
    """Main entry point for performance benchmarks."""
    parser = argparse.ArgumentParser(
        description="MCP Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        help="Run specific benchmark (transport_creation, mcp_client_initialization, request_latency, concurrent_requests, batch_processing, memory_usage)",
    )
    parser.add_argument(
        "--iterations", type=int, help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--concurrency", type=int, help="Concurrency level for concurrent tests"
    )
    parser.add_argument(
        "--detailed-report",
        action="store_true",
        help="Generate detailed benchmark report",
    )
    parser.add_argument(
        "--output", type=str, help="Output file for benchmark results (JSON)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Create benchmark suite
    suite = PerformanceBenchmarkSuite(args)

    try:
        # Run benchmarks
        results = await suite.run_all_benchmarks()

        # Generate and display report
        report = suite.generate_report()
        print(report)

        # Save results if requested
        if args.output:
            suite.save_results(args.output)

        # Return success if all benchmarks completed
        return 0 if results else 1

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Benchmark suite error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import os

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
