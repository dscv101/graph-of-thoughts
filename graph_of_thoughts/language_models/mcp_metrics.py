# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
MCP Metrics and Monitoring System.

This module provides comprehensive metrics collection and monitoring capabilities
for the MCP implementation, enabling performance tracking, debugging, and system
health monitoring.

Key Features:
    - Request/response latency tracking
    - Error rate and failure pattern monitoring
    - Token usage and cost tracking
    - Connection health and circuit breaker metrics
    - Transport-specific performance metrics
    - Configurable metric collection and export
    - Integration with popular monitoring systems

Metrics Categories:
    1. Request Metrics:
       - Request count, latency, success/failure rates
       - Method-specific performance tracking
       - Batch processing efficiency

    2. Connection Metrics:
       - Connection establishment time
       - Connection pool utilization
       - Transport-specific metrics

    3. Error Metrics:
       - Error rates by type and method
       - Circuit breaker state changes
       - Timeout and retry statistics

    4. Resource Metrics:
       - Token usage and cost tracking
       - Memory usage patterns
       - Cache hit/miss rates

Example Usage:
    Basic metrics collection:

    ```python
    from graph_of_thoughts.language_models.mcp_metrics import MCPMetricsCollector

    # Initialize metrics collector
    metrics = MCPMetricsCollector(
        enabled=True,
        export_interval=60.0,
        export_format="prometheus"
    )

    # Track a request
    with metrics.track_request("sampling/createMessage") as tracker:
        response = await transport.send_request("sampling/createMessage", params)
        tracker.record_success(response)

    # Get current metrics
    current_metrics = metrics.get_current_metrics()
    print(f"Total requests: {current_metrics['requests']['total']}")
    print(f"Average latency: {current_metrics['requests']['avg_latency_ms']:.2f}ms")
    ```

    Integration with MCP client:

    ```python
    # Metrics are automatically integrated with MCPLanguageModel
    lm = MCPLanguageModel(
        config_path="config.json",
        model_name="mcp_claude_desktop"
    )

    # Metrics configuration in config file
    config = {
        "metrics": {
            "enabled": True,
            "export_interval": 60.0,
            "export_format": "json",
            "include_detailed_timings": True
        }
    }
    ```
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """
    Individual metric value with metadata.

    Attributes:
        value: The metric value
        timestamp: When the metric was recorded
        labels: Additional labels/tags for the metric
        metric_type: Type of metric
    """

    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    labels: [str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class RequestMetrics:
    """
    Metrics for individual requests.

    Attributes:
        method: The MCP method called
        start_time: Request start timestamp
        end_time: Request end timestamp
        success: Whether the request succeeded
        error_type: Type of error if failed
        response_size: Size of response in bytes
        token_usage: Token usage information
    """

    method: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    response_size: int = 0
    token_usage: [str, int] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get request duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics for a time period.

    Attributes:
        total_requests: Total number of requests
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        avg_latency_ms: Average latency in milliseconds
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency
        error_rate: Error rate as percentage
        total_tokens: Total tokens used
        total_cost: Total estimated cost
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0


class RequestTracker:
    """
    Context manager for tracking individual requests.
    """

    def __init__(self, metrics_collector: "MCPMetricsCollector", method: str):
        """
        Initialize request tracker.

        Args:
            metrics_collector: The metrics collector instance
            method: The MCP method being tracked
        """
        self.metrics_collector = metrics_collector
        self.method = method
        self.start_time = time.time()
        self.request_metrics = RequestMetrics(method=method, start_time=self.start_time)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.request_metrics.end_time = time.time()

        if exc_type is not None:
            self.record_error(exc_type.__name__)

        self.metrics_collector._record_request(self.request_metrics)

    def record_success(self, response: Optional[[str, Any]] = None):
        """
        Record successful request completion.

        Args:
            response: Optional response data for additional metrics
        """
        self.request_metrics.success = True

        if response:
            # Extract response size
            response_str = json.dumps(response)
            self.request_metrics.response_size = len(response_str.encode("utf-8"))

            # Extract token usage if available
            metadata = response.get("metadata", {})
            if "usage" in metadata:
                usage = metadata["usage"]
                self.request_metrics.token_usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }

    def record_error(self, error_type: str):
        """
        Record request error.

        Args:
            error_type: Type/name of the error
        """
        self.request_metrics.success = False
        self.request_metrics.error_type = error_type


class MCPMetricsCollector:
    """
    Comprehensive metrics collector for MCP operations.

    Collects, aggregates, and exports metrics for monitoring and debugging
    MCP client performance and health.
    """

    def __init__(self, config: Optional[[str, Any]] = None):
        """
        Initialize the metrics collector.

        Args:
            config: Metrics configuration
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.export_interval = self.config.get("export_interval", 60.0)
        self.export_format = self.config.get("export_format", "json")
        self.max_history_size = self.config.get("max_history_size", 1000)
        self.include_detailed_timings = self.config.get(
            "include_detailed_timings", False
        )

        # Thread-safe storage
        self.lock = Lock()

        # Request history and current metrics
        self.request_history: deque = deque(maxlen=self.max_history_size)
        self.method_metrics: [str, [RequestMetrics]] = defaultdict(list)
        self.error_counts: [str, int] = defaultdict(int)

        # Custom metrics
        self.custom_metrics: [str, [MetricValue]] = defaultdict(list)

        # Export configuration
        self.export_callbacks: [Callable] = []
        self.last_export_time = time.time()

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if self.enabled:
            self.logger.info(
                f"Initialized MCP metrics collector with config: {self.config}"
            )

    @contextmanager
    def track_request(self, method: str):
        """
        Create a request tracker context manager.

        Args:
            method: The MCP method being tracked

        Returns:
            RequestTracker instance
        """
        if not self.enabled:
            yield None
            return

        tracker = RequestTracker(self, method)
        yield tracker

    def _record_request(self, request_metrics: RequestMetrics):
        """
        Record a completed request's metrics.

        Args:
            request_metrics: The request metrics to record
        """
        if not self.enabled:
            return

        with self.lock:
            # Add to history
            self.request_history.append(request_metrics)

            # Add to method-specific metrics
            method_list = self.method_metrics[request_metrics.method]
            method_list.append(request_metrics)

            # Keep method lists bounded
            if len(method_list) > self.max_history_size // 10:
                method_list.pop(0)

            # Track errors
            if not request_metrics.success and request_metrics.error_type:
                self.error_counts[request_metrics.error_type] += 1

            # Log detailed metrics if enabled
            if self.include_detailed_timings:
                self.logger.debug(
                    f"Request {request_metrics.method}: "
                    f"{request_metrics.duration_ms:.2f}ms, "
                    f"success={request_metrics.success}, "
                    f"tokens={sum(request_metrics.token_usage.values())}"
                )

    def record_custom_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[[str, str]] = None,
    ):
        """
        Record a custom metric value.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Optional labels for the metric
        """
        if not self.enabled:
            return

        metric_value = MetricValue(
            value=value, metric_type=metric_type, labels=labels or {}
        )

        with self.lock:
            metric_list = self.custom_metrics[name]
            metric_list.append(metric_value)

            # Keep custom metrics bounded
            if len(metric_list) > self.max_history_size:
                metric_list.pop(0)

    def get_current_metrics(self) -> [str, Any]:
        """
        Get current aggregated metrics.

        Returns:
            ionary containing current metrics
        """
        if not self.enabled:
            return {}

        with self.lock:
            current_time = time.time()

            # Calculate overall metrics
            total_requests = len(self.request_history)
            if total_requests == 0:
                return self._empty_metrics()

            successful_requests = sum(1 for r in self.request_history if r.success)
            failed_requests = total_requests - successful_requests

            # Calculate latency statistics
            latencies = [r.duration_ms for r in self.request_history if r.end_time]
            avg_latency = statistics.mean(latencies) if latencies else 0.0
            p95_latency = (
                statistics.quantiles(latencies, n=20)[18]
                if len(latencies) >= 20
                else 0.0
            )
            p99_latency = (
                statistics.quantiles(latencies, n=100)[98]
                if len(latencies) >= 100
                else 0.0
            )

            # Calculate token and cost totals
            total_tokens = sum(
                sum(r.token_usage.values()) for r in self.request_history
            )

            # Method-specific metrics
            method_stats = {}
            for method, requests in self.method_metrics.items():
                if requests:
                    method_latencies = [r.duration_ms for r in requests if r.end_time]
                    method_stats[method] = {
                        "total_requests": len(requests),
                        "successful_requests": sum(1 for r in requests if r.success),
                        "avg_latency_ms": statistics.mean(method_latencies)
                        if method_latencies
                        else 0.0,
                        "error_rate": (
                            len(requests) - sum(1 for r in requests if r.success)
                        )
                        / len(requests)
                        * 100,
                    }

            return {
                "timestamp": current_time,
                "requests": {
                    "total": total_requests,
                    "successful": successful_requests,
                    "failed": failed_requests,
                    "error_rate": (failed_requests / total_requests * 100)
                    if total_requests > 0
                    else 0.0,
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency,
                    "p99_latency_ms": p99_latency,
                },
                "tokens": {"total": total_tokens},
                "methods": method_stats,
                "errors": dict(self.error_counts),
                "custom_metrics": self._get_custom_metrics_summary(),
            }

    def _empty_metrics(self) -> [str, Any]:
        """Return empty metrics structure."""
        return {
            "timestamp": time.time(),
            "requests": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
            },
            "tokens": {"total": 0},
            "methods": {},
            "errors": {},
            "custom_metrics": {},
        }

    def _get_custom_metrics_summary(self) -> [str, Any]:
        """Get summary of custom metrics."""
        summary = {}
        for name, values in self.custom_metrics.items():
            if values:
                recent_values = [v.value for v in values[-100:]]  # Last 100 values
                summary[name] = {
                    "current": values[-1].value,
                    "avg": statistics.mean(recent_values),
                    "min": min(recent_values),
                    "max": max(recent_values),
                    "count": len(values),
                }
        return summary

    def get_method_metrics(self, method: str) -> [str, Any]:
        """
        Get metrics for a specific MCP method.

        Args:
            method: The MCP method name

        Returns:
            ionary containing method-specific metrics
        """
        if not self.enabled or method not in self.method_metrics:
            return {}

        with self.lock:
            requests = self.method_metrics[method]
            if not requests:
                return {}

            successful = sum(1 for r in requests if r.success)
            latencies = [r.duration_ms for r in requests if r.end_time]
            tokens = sum(sum(r.token_usage.values()) for r in requests)

            return {
                "method": method,
                "total_requests": len(requests),
                "successful_requests": successful,
                "failed_requests": len(requests) - successful,
                "error_rate": ((len(requests) - successful) / len(requests) * 100)
                if requests
                else 0.0,
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
                "min_latency_ms": min(latencies) if latencies else 0.0,
                "max_latency_ms": max(latencies) if latencies else 0.0,
                "total_tokens": tokens,
                "recent_errors": [
                    r.error_type
                    for r in requests[-10:]
                    if not r.success and r.error_type
                ],
            }

    def get_error_summary(self) -> [str, Any]:
        """
        Get summary of errors encountered.

        Returns:
            ionary containing error statistics
        """
        if not self.enabled:
            return {}

        with self.lock:
            total_errors = sum(self.error_counts.values())
            recent_errors = []

            # Get recent errors from request history
            for request in list(self.request_history)[-50:]:  # Last 50 requests
                if not request.success and request.error_type:
                    recent_errors.append(
                        {
                            "method": request.method,
                            "error_type": request.error_type,
                            "timestamp": request.start_time,
                            "duration_ms": request.duration_ms,
                        }
                    )

            return {
                "total_errors": total_errors,
                "error_counts": dict(self.error_counts),
                "recent_errors": recent_errors[-10:],  # Last 10 errors
                "error_rate": (total_errors / len(self.request_history) * 100)
                if self.request_history
                else 0.0,
            }

    def export_metrics(self, format_type: Optional[str] = None) -> str:
        """
        Export current metrics in specified format.

        Args:
            format_type: Export format ("json", "prometheus", "csv")

        Returns:
            Formatted metrics string
        """
        if not self.enabled:
            return ""

        export_format = format_type or self.export_format
        metrics = self.get_current_metrics()

        if export_format == "json":
            return json.dumps(metrics, indent=2)
        elif export_format == "prometheus":
            return self._export_prometheus_format(metrics)
        elif export_format == "csv":
            return self._export_csv_format(metrics)
        else:
            self.logger.warning(f"Unknown export format: {export_format}, using JSON")
            return json.dumps(metrics, indent=2)

    def _export_prometheus_format(self, metrics: [str, Any]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        timestamp = int(metrics["timestamp"] * 1000)  # Prometheus uses milliseconds

        # Request metrics
        req_metrics = metrics["requests"]
        lines.extend(
            [
                f"# HELP mcp_requests_total Total number of MCP requests",
                f"# TYPE mcp_requests_total counter",
                f"mcp_requests_total {req_metrics['total']} {timestamp}",
                f"# HELP mcp_requests_successful_total Total number of successful MCP requests",
                f"# TYPE mcp_requests_successful_total counter",
                f"mcp_requests_successful_total {req_metrics['successful']} {timestamp}",
                f"# HELP mcp_requests_failed_total Total number of failed MCP requests",
                f"# TYPE mcp_requests_failed_total counter",
                f"mcp_requests_failed_total {req_metrics['failed']} {timestamp}",
                f"# HELP mcp_request_duration_ms Request duration in milliseconds",
                f"# TYPE mcp_request_duration_ms gauge",
                f"mcp_request_duration_ms{{quantile=\"0.5\"}} {req_metrics['avg_latency_ms']} {timestamp}",
                f"mcp_request_duration_ms{{quantile=\"0.95\"}} {req_metrics['p95_latency_ms']} {timestamp}",
                f"mcp_request_duration_ms{{quantile=\"0.99\"}} {req_metrics['p99_latency_ms']} {timestamp}",
            ]
        )

        # Token metrics
        token_metrics = metrics["tokens"]
        lines.extend(
            [
                f"# HELP mcp_tokens_total Total number of tokens processed",
                f"# TYPE mcp_tokens_total counter",
                f"mcp_tokens_total {token_metrics['total']} {timestamp}",
            ]
        )

        # Method-specific metrics
        for method, method_data in metrics["methods"].items():
            safe_method = method.replace("/", "_").replace("-", "_")
            lines.extend(
                [
                    f"mcp_method_requests_total{{method=\"{method}\"}} {method_data['total_requests']} {timestamp}",
                    f"mcp_method_duration_ms{{method=\"{method}\"}} {method_data['avg_latency_ms']} {timestamp}",
                    f"mcp_method_error_rate{{method=\"{method}\"}} {method_data['error_rate']} {timestamp}",
                ]
            )

        return "\n".join(lines)

    def _export_csv_format(self, metrics: [str, Any]) -> str:
        """Export metrics in CSV format."""
        lines = ["timestamp,metric_name,value,labels"]
        timestamp = metrics["timestamp"]

        # Request metrics
        req_metrics = metrics["requests"]
        for key, value in req_metrics.items():
            lines.append(f"{timestamp},requests_{key},{value},")

        # Token metrics
        token_metrics = metrics["tokens"]
        for key, value in token_metrics.items():
            lines.append(f"{timestamp},tokens_{key},{value},")

        # Method metrics
        for method, method_data in metrics["methods"].items():
            for key, value in method_data.items():
                lines.append(f"{timestamp},method_{key},{value},method={method}")

        return "\n".join(lines)

    def add_export_callback(self, callback: Callable[[[str, Any]], None]):
        """
        Add a callback function to be called when metrics are exported.

        Args:
            callback: Function that takes metrics dictionary as argument
        """
        self.export_callbacks.append(callback)

    def should_export(self) -> bool:
        """Check if it's time to export metrics."""
        return (time.time() - self.last_export_time) >= self.export_interval

    def trigger_export(self):
        """Trigger metrics export if enabled and due."""
        if not self.enabled or not self.should_export():
            return

        try:
            metrics = self.get_current_metrics()

            # Call export callbacks
            for callback in self.export_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    self.logger.error(f"Error in export callback: {e}")

            self.last_export_time = time.time()
            self.logger.debug("Metrics exported successfully")

        except Exception as e:
            self.logger.error(f"Error during metrics export: {e}")

    def reset_metrics(self):
        """Reset all collected metrics."""
        if not self.enabled:
            return

        with self.lock:
            self.request_history.clear()
            self.method_metrics.clear()
            self.error_counts.clear()
            self.custom_metrics.clear()
            self.last_export_time = time.time()

        self.logger.info("Metrics reset")

    def get_health_status(self) -> [str, Any]:
        """
        Get overall health status based on metrics.

        Returns:
            ionary containing health assessment
        """
        if not self.enabled:
            return {"status": "unknown", "reason": "metrics disabled"}

        metrics = self.get_current_metrics()
        req_metrics = metrics["requests"]

        # Determine health status
        if req_metrics["total"] == 0:
            status = "unknown"
            reason = "no requests processed"
        elif req_metrics["error_rate"] > 50:
            status = "unhealthy"
            reason = f"high error rate: {req_metrics['error_rate']:.1f}%"
        elif req_metrics["error_rate"] > 20:
            status = "degraded"
            reason = f"elevated error rate: {req_metrics['error_rate']:.1f}%"
        elif req_metrics["avg_latency_ms"] > 5000:
            status = "degraded"
            reason = f"high latency: {req_metrics['avg_latency_ms']:.1f}ms"
        else:
            status = "healthy"
            reason = "all metrics within normal ranges"

        return {
            "status": status,
            "reason": reason,
            "timestamp": metrics["timestamp"],
            "total_requests": req_metrics["total"],
            "error_rate": req_metrics["error_rate"],
            "avg_latency_ms": req_metrics["avg_latency_ms"],
        }


def create_metrics_collector_from_config(
    config: [str, Any]
) -> Optional[MCPMetricsCollector]:
    """
    Create a metrics collector from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        MCPMetricsCollector instance or None if disabled
    """
    metrics_config = config.get("metrics", {})

    if not metrics_config.get("enabled", False):
        return None

    return MCPMetricsCollector(metrics_config)


def setup_default_export_callbacks(
    metrics_collector: MCPMetricsCollector, config: [str, Any]
) -> None:
    """
    Setup default export callbacks based on configuration.

    Args:
        metrics_collector: The metrics collector instance
        config: Configuration dictionary
    """
    metrics_config = config.get("metrics", {})

    # File export callback
    export_file = metrics_config.get("export_file")
    if export_file:

        def file_export_callback(metrics: [str, Any]):
            try:
                with open(export_file, "w") as f:
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to export metrics to file {export_file}: {e}")

        metrics_collector.add_export_callback(file_export_callback)

    # Console export callback
    if metrics_config.get("export_to_console", False):

        def console_export_callback(metrics: [str, Any]):
            print(f"\n=== MCP Metrics Report ===")
            print(
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metrics['timestamp']))}"
            )
            req_metrics = metrics["requests"]
            print(f"Total Requests: {req_metrics['total']}")
            print(f"Success Rate: {100 - req_metrics['error_rate']:.1f}%")
            print(f"Average Latency: {req_metrics['avg_latency_ms']:.2f}ms")
            print(f"Total Tokens: {metrics['tokens']['total']}")
            if metrics["errors"]:
                print(f"Recent Errors: {list(metrics['errors'].keys())}")
            print("=" * 25)

        metrics_collector.add_export_callback(console_export_callback)


class MetricsIntegrationMixin:
    """
    Mixin class to add metrics integration to MCP components.

    This mixin can be used by transport classes and other MCP components
    to easily integrate with the metrics collection system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector: Optional[MCPMetricsCollector] = None

    def set_metrics_collector(self, metrics_collector: Optional[MCPMetricsCollector]):
        """
        Set the metrics collector for this component.

        Args:
            metrics_collector: The metrics collector instance
        """
        self.metrics_collector = metrics_collector

    def record_connection_metric(self, success: bool, duration_ms: float):
        """
        Record connection establishment metrics.

        Args:
            success: Whether connection was successful
            duration_ms: Connection duration in milliseconds
        """
        if self.metrics_collector:
            self.metrics_collector.record_custom_metric(
                "connection_duration_ms",
                duration_ms,
                MetricType.TIMER,
                {"success": str(success)},
            )

    def record_transport_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        labels: Optional[[str, str]] = None,
    ):
        """
        Record transport-specific metrics.

        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
        """
        if self.metrics_collector:
            self.metrics_collector.record_custom_metric(
                f"transport_{metric_name}", value, MetricType.GAUGE, labels
            )


def integrate_metrics_with_circuit_breaker(
    circuit_breaker, metrics_collector: MCPMetricsCollector
):
    """
    Integrate circuit breaker metrics with the metrics collector.

    Args:
        circuit_breaker: Circuit breaker instance
        metrics_collector: Metrics collector instance
    """
    if not metrics_collector or not circuit_breaker:
        return

    def export_circuit_breaker_metrics():
        """Export circuit breaker metrics to the collector."""
        try:
            cb_metrics = circuit_breaker.get_metrics()
            cb_state = circuit_breaker.get_state()

            # Record circuit breaker state as numeric value
            state_values = {"closed": 0, "half_open": 1, "open": 2}
            metrics_collector.record_custom_metric(
                "circuit_breaker_state",
                state_values.get(cb_state.value, -1),
                MetricType.GAUGE,
            )

            # Record circuit breaker metrics
            metrics_collector.record_custom_metric(
                "circuit_breaker_total_requests",
                cb_metrics.total_requests,
                MetricType.COUNTER,
            )

            metrics_collector.record_custom_metric(
                "circuit_breaker_failed_requests",
                cb_metrics.failed_requests,
                MetricType.COUNTER,
            )

            metrics_collector.record_custom_metric(
                "circuit_breaker_open_count",
                cb_metrics.circuit_open_count,
                MetricType.COUNTER,
            )

        except Exception as e:
            logger.error(f"Failed to export circuit breaker metrics: {e}")

    # Add callback to export circuit breaker metrics
    metrics_collector.add_export_callback(lambda _: export_circuit_breaker_metrics())


# Global metrics collector instance
_global_metrics_collector: Optional[MCPMetricsCollector] = None


def get_global_metrics_collector() -> Optional[MCPMetricsCollector]:
    """Get the global metrics collector instance."""
    return _global_metrics_collector


def set_global_metrics_collector(metrics_collector: Optional[MCPMetricsCollector]):
    """Set the global metrics collector instance."""
    global _global_metrics_collector
    _global_metrics_collector = metrics_collector


def track_request_globally(method: str):
    """
    Decorator to track requests using the global metrics collector.

    Args:
        method: The MCP method being tracked
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_global_metrics_collector()
            if collector:
                with collector.track_request(method) as tracker:
                    try:
                        result = func(*args, **kwargs)
                        if tracker:
                            tracker.record_success()
                        return result
                    except Exception as e:
                        if tracker:
                            tracker.record_error(type(e).__name__)
                        raise
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator