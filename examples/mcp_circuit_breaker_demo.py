#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Circuit Breaker Demonstration.

This script demonstrates the circuit breaker pattern implementation for MCP connections,
showing how it provides resilience against failing services and automatic recovery.

The circuit breaker helps prevent cascading failures by:
- Detecting when services are failing
- Failing fast when services are down
- Automatically attempting recovery
- Providing detailed metrics and monitoring

Usage:
    python examples/mcp_circuit_breaker_demo.py

Features Demonstrated:
    - Circuit breaker configuration and setup
    - Failure detection and circuit opening
    - Fast failure responses when circuit is open
    - Automatic recovery testing (half-open state)
    - Service recovery and circuit closing
    - Metrics collection and monitoring
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.language_models.mcp_circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    MCPCircuitBreaker,
    create_circuit_breaker_from_config,
)


class MockFailingService:
    """Mock service that can be configured to fail for demonstration."""

    def __init__(self):
        self.call_count = 0
        self.failure_mode = False
        self.failure_rate = 0.0
        self.recovery_after_calls = None

    def set_failure_mode(self, enabled: bool, failure_rate: float = 1.0):
        """Set failure mode and rate."""
        self.failure_mode = enabled
        self.failure_rate = failure_rate
        print(
            f"🔧 Service failure mode: {'ON' if enabled else 'OFF'} (rate: {failure_rate})"
        )

    def set_recovery_after_calls(self, calls: int):
        """Set service to recover after a number of calls."""
        self.recovery_after_calls = calls
        print(f"🔧 Service will recover after {calls} calls")

    async def call_service(self, operation: str = "test") -> str:
        """Simulate a service call that may fail."""
        self.call_count += 1

        # Check if service should recover
        if self.recovery_after_calls and self.call_count >= self.recovery_after_calls:
            self.failure_mode = False
            self.recovery_after_calls = None
            print(f"✅ Service recovered after {self.call_count} calls")

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Determine if this call should fail
        if self.failure_mode:
            import random

            if random.random() < self.failure_rate:
                raise Exception(f"Service failure on call {self.call_count}")

        return f"Success: {operation} completed (call #{self.call_count})"


async def demonstrate_basic_circuit_breaker():
    """Demonstrate basic circuit breaker functionality."""
    print("🔄 Basic Circuit Breaker Demonstration")
    print("=" * 50)

    # Create circuit breaker with aggressive settings for demo
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=5.0,
        half_open_max_calls=2,
        success_threshold=2,
        expected_exceptions=(Exception,),
    )

    circuit_breaker = MCPCircuitBreaker(config)
    service = MockFailingService()

    print(f"📊 Initial state: {circuit_breaker.get_state().value}")

    # Phase 1: Normal operation
    print("\n🟢 Phase 1: Normal Operation")
    for i in range(3):
        try:
            async with circuit_breaker:
                result = await service.call_service(f"operation_{i+1}")
                print(f"  ✅ {result}")
        except Exception as e:
            print(f"  ❌ {e}")

    metrics = circuit_breaker.get_metrics()
    print(
        f"📊 Metrics: {metrics.successful_requests} success, {metrics.failed_requests} failures"
    )

    # Phase 2: Service starts failing
    print("\n🔴 Phase 2: Service Failures")
    service.set_failure_mode(True, 1.0)  # 100% failure rate

    for i in range(5):
        try:
            async with circuit_breaker:
                result = await service.call_service(f"failing_operation_{i+1}")
                print(f"  ✅ {result}")
        except CircuitBreakerOpenError as e:
            print(f"  🚫 Circuit breaker open: {e}")
        except Exception as e:
            print(f"  ❌ Service error: {e}")

        print(f"     State: {circuit_breaker.get_state().value}")

    metrics = circuit_breaker.get_metrics()
    print(
        f"📊 Metrics: {metrics.successful_requests} success, {metrics.failed_requests} failures, {metrics.circuit_open_count} opens"
    )

    # Phase 3: Wait for recovery attempt
    print(f"\n⏳ Phase 3: Waiting for Recovery (timeout: {config.recovery_timeout}s)")
    print("Circuit breaker will attempt recovery after timeout...")

    # Try calls during open period
    for i in range(3):
        try:
            async with circuit_breaker:
                result = await service.call_service("during_open")
                print(f"  ✅ {result}")
        except CircuitBreakerOpenError as e:
            print(f"  🚫 Fast failure: Circuit breaker is open")
        await asyncio.sleep(1)

    # Wait for recovery timeout
    await asyncio.sleep(config.recovery_timeout)

    # Phase 4: Recovery testing (half-open)
    print("\n🟡 Phase 4: Recovery Testing (Half-Open)")
    service.set_recovery_after_calls(
        service.call_count + 2
    )  # Recover after 2 more calls

    for i in range(4):
        try:
            async with circuit_breaker:
                result = await service.call_service(f"recovery_test_{i+1}")
                print(f"  ✅ {result}")
        except CircuitBreakerOpenError as e:
            print(f"  🚫 Recovery failed, circuit re-opened")
        except Exception as e:
            print(f"  ❌ Service error during recovery: {e}")

        print(f"     State: {circuit_breaker.get_state().value}")
        await asyncio.sleep(0.5)

    final_metrics = circuit_breaker.get_metrics()
    print(f"\n📊 Final Metrics:")
    print(f"   Total requests: {final_metrics.total_requests}")
    print(f"   Successful: {final_metrics.successful_requests}")
    print(f"   Failed: {final_metrics.failed_requests}")
    print(f"   Circuit opened: {final_metrics.circuit_open_count} times")
    print(f"   Final state: {circuit_breaker.get_state().value}")


async def demonstrate_config_based_circuit_breaker():
    """Demonstrate circuit breaker creation from configuration."""
    print("\n🔧 Configuration-Based Circuit Breaker")
    print("=" * 50)

    # Configuration with circuit breaker enabled
    config = {
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 2,
            "recovery_timeout": 3.0,
            "half_open_max_calls": 1,
            "success_threshold": 1,
            "monitoring_window": 30.0,
            "minimum_throughput": 2,
        }
    }

    circuit_breaker = create_circuit_breaker_from_config(config)

    if circuit_breaker:
        print("✅ Circuit breaker created from configuration")
        service = MockFailingService()

        # Test with failures
        service.set_failure_mode(True, 1.0)

        for i in range(4):
            try:
                result = await circuit_breaker.call(
                    service.call_service, f"config_test_{i+1}"
                )
                print(f"  ✅ {result}")
            except CircuitBreakerOpenError:
                print(f"  🚫 Circuit breaker blocked call {i+1}")
            except Exception as e:
                print(f"  ❌ Service error: {e}")
    else:
        print("❌ Circuit breaker not created (disabled in config)")


async def demonstrate_metrics_monitoring():
    """Demonstrate circuit breaker metrics and monitoring."""
    print("\n📊 Metrics and Monitoring Demonstration")
    print("=" * 50)

    config = CircuitBreakerConfig(
        failure_threshold=3, recovery_timeout=2.0, monitoring_window=10.0
    )

    circuit_breaker = MCPCircuitBreaker(config)
    service = MockFailingService()

    # Generate mixed success/failure pattern
    service.set_failure_mode(True, 0.6)  # 60% failure rate

    print("Generating mixed success/failure pattern...")
    for i in range(10):
        try:
            async with circuit_breaker:
                result = await service.call_service(f"mixed_test_{i+1}")
                print(f"  ✅ Call {i+1}: Success")
        except CircuitBreakerOpenError:
            print(f"  🚫 Call {i+1}: Circuit open")
        except Exception:
            print(f"  ❌ Call {i+1}: Service failure")

        # Show metrics every few calls
        if (i + 1) % 3 == 0:
            metrics = circuit_breaker.get_metrics()
            success_rate = (
                metrics.successful_requests / max(metrics.total_requests, 1) * 100
            )
            print(
                f"     📊 Success rate: {success_rate:.1f}% ({metrics.successful_requests}/{metrics.total_requests})"
            )

        await asyncio.sleep(0.2)

    # Final metrics summary
    final_metrics = circuit_breaker.get_metrics()
    print(f"\n📈 Final Metrics Summary:")
    print(f"   State: {circuit_breaker.get_state().value}")
    print(f"   Total requests: {final_metrics.total_requests}")
    print(
        f"   Success rate: {final_metrics.successful_requests / max(final_metrics.total_requests, 1) * 100:.1f}%"
    )
    print(f"   Circuit opened: {final_metrics.circuit_open_count} times")
    if final_metrics.last_failure_time:
        time_since_failure = time.time() - final_metrics.last_failure_time
        print(f"   Time since last failure: {time_since_failure:.1f}s")


async def main():
    """Main demonstration function."""
    print("🚀 MCP Circuit Breaker Pattern Demonstration")
    print("=" * 60)
    print()

    try:
        await demonstrate_basic_circuit_breaker()
        await demonstrate_config_based_circuit_breaker()
        await demonstrate_metrics_monitoring()

        print("\n✨ Circuit breaker demonstration completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("  • Fast failure when services are down")
        print("  • Automatic recovery testing")
        print("  • Detailed metrics and monitoring")
        print("  • Configurable thresholds and timeouts")
        print("  • Prevention of cascading failures")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
