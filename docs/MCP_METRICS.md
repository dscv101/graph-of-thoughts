# MCP Metrics and Monitoring System

The MCP implementation includes a comprehensive metrics and monitoring system that provides detailed insights into performance, reliability, and health of your MCP connections.

## Overview

The metrics system collects, aggregates, and exports various performance and operational metrics including:

- **Request Metrics**: Latency, throughput, success/failure rates
- **Connection Metrics**: Connection establishment time, pool utilization
- **Error Metrics**: Error rates, failure patterns, timeout statistics
- **Resource Metrics**: Token usage, cost tracking, memory patterns
- **Circuit Breaker Metrics**: State changes, failure thresholds, recovery patterns

## Configuration

### Basic Configuration

Enable metrics collection by adding a `metrics` section to your MCP configuration:

```json
{
  "mcp_claude_desktop": {
    "transport": {
      "type": "stdio",
      "command": "claude-desktop"
    },
    "metrics": {
      "enabled": true,
      "export_interval": 60.0,
      "export_format": "json"
    }
  }
}
```

### Advanced Configuration

For production environments, use more detailed configuration:

```json
{
  "metrics": {
    "enabled": true,
    "export_interval": 30.0,
    "export_format": "prometheus",
    "max_history_size": 2000,
    "include_detailed_timings": true,
    "export_file": "mcp_metrics.prom",
    "export_to_console": false
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable/disable metrics collection |
| `export_interval` | number | `60.0` | Metrics export interval in seconds |
| `export_format` | string | `"json"` | Export format: "json", "prometheus", "csv" |
| `max_history_size` | number | `1000` | Maximum number of requests to keep in history |
| `include_detailed_timings` | boolean | `false` | Include detailed timing information in logs |
| `export_file` | string | `null` | File path for metrics export |
| `export_to_console` | boolean | `false` | Print metrics to console |

## Usage

### Basic Usage

```python
from graph_of_thoughts.language_models import MCPLanguageModel

# Initialize with metrics enabled
lm = MCPLanguageModel(
    config_path="config.json",
    model_name="mcp_claude_desktop"
)

# Use normally - metrics are collected automatically
response = lm.query("What is machine learning?")

# Get current metrics
metrics = lm.get_metrics()
print(f"Total requests: {metrics['requests']['total']}")
print(f"Average latency: {metrics['requests']['avg_latency_ms']:.2f}ms")
```

### Advanced Usage

```python
# Get method-specific metrics
sampling_metrics = lm.get_method_metrics("sampling/createMessage")
print(f"Sampling requests: {sampling_metrics['total_requests']}")

# Get error summary
errors = lm.get_error_summary()
print(f"Error rate: {errors['error_rate']:.1f}%")

# Get health status
health = lm.get_health_status()
print(f"Overall status: {health['overall_status']}")

# Export metrics in different formats
json_metrics = lm.export_metrics("json")
prometheus_metrics = lm.export_metrics("prometheus")
csv_metrics = lm.export_metrics("csv")
```

## Metrics Types

### Request Metrics

- **Total Requests**: Count of all requests made
- **Successful Requests**: Count of successful requests
- **Failed Requests**: Count of failed requests
- **Error Rate**: Percentage of failed requests
- **Average Latency**: Mean response time in milliseconds
- **P95/P99 Latency**: 95th and 99th percentile response times
- **Token Usage**: Total tokens processed

### Connection Metrics

- **Connection Duration**: Time to establish connection
- **Connection Pool Stats**: Active and keepalive connections (HTTP)
- **Connection Success Rate**: Percentage of successful connections

### Error Metrics

- **Error Counts**: Count by error type
- **Recent Errors**: List of recent error occurrences
- **Error Patterns**: Analysis of error frequency and timing

### Circuit Breaker Metrics

- **State**: Current circuit breaker state (closed/open/half-open)
- **State Changes**: Number of state transitions
- **Failure Threshold**: Configured failure limits
- **Recovery Metrics**: Success rate during recovery attempts

## Export Formats

### JSON Format

Standard JSON structure with nested metrics:

```json
{
  "timestamp": 1703123456.789,
  "requests": {
    "total": 150,
    "successful": 145,
    "failed": 5,
    "error_rate": 3.33,
    "avg_latency_ms": 245.67
  },
  "methods": {
    "sampling/createMessage": {
      "total_requests": 150,
      "avg_latency_ms": 245.67,
      "error_rate": 3.33
    }
  }
}
```

### Prometheus Format

Compatible with Prometheus monitoring:

```
# HELP mcp_requests_total Total number of MCP requests
# TYPE mcp_requests_total counter
mcp_requests_total 150 1703123456789

# HELP mcp_request_duration_ms Request duration in milliseconds
# TYPE mcp_request_duration_ms gauge
mcp_request_duration_ms{quantile="0.95"} 456.78 1703123456789
```

### CSV Format

Tabular format for analysis:

```csv
timestamp,metric_name,value,labels
1703123456.789,requests_total,150,
1703123456.789,requests_avg_latency_ms,245.67,
1703123456.789,method_total_requests,150,method=sampling/createMessage
```

## Custom Metrics

### Recording Custom Metrics

```python
from graph_of_thoughts.language_models.mcp_metrics import MetricType

# Access the metrics collector
collector = lm.metrics_collector

# Record custom metrics
collector.record_custom_metric(
    "custom_processing_time", 
    123.45, 
    MetricType.TIMER,
    labels={"operation": "data_processing"}
)

collector.record_custom_metric(
    "cache_hit_rate", 
    85.5, 
    MetricType.GAUGE,
    labels={"cache_type": "response"}
)
```

### Custom Export Callbacks

```python
def custom_export_callback(metrics):
    """Custom metrics export function."""
    # Send to external monitoring system
    send_to_datadog(metrics)
    
    # Log critical metrics
    if metrics["requests"]["error_rate"] > 10:
        logger.warning(f"High error rate: {metrics['requests']['error_rate']:.1f}%")

# Add callback to metrics collector
lm.metrics_collector.add_export_callback(custom_export_callback)
```

## Health Monitoring

### Health Status

The system provides comprehensive health assessment:

```python
health = lm.get_health_status()
# Returns:
{
  "overall_status": "healthy",  # healthy, degraded, unhealthy, disconnected
  "timestamp": 1703123456.789,
  "components": {
    "metrics": {
      "status": "healthy",
      "reason": "all metrics within normal ranges"
    },
    "circuit_breaker": {
      "status": "healthy",
      "state": "closed",
      "error_rate": 2.1
    },
    "connection": {
      "status": "healthy",
      "connected": true
    }
  }
}
```

### Health Thresholds

Default health thresholds:

- **Healthy**: Error rate < 20%, latency < 5000ms
- **Degraded**: Error rate 20-50%, latency 5000-10000ms
- **Unhealthy**: Error rate > 50%, latency > 10000ms

## Integration with Monitoring Systems

### Prometheus Integration

```python
# Export metrics for Prometheus scraping
prometheus_data = lm.export_metrics("prometheus")

# Write to file for Prometheus file discovery
with open("/var/lib/prometheus/mcp_metrics.prom", "w") as f:
    f.write(prometheus_data)
```

### Grafana Dashboard

Use the Prometheus metrics to create Grafana dashboards:

- Request rate and latency graphs
- Error rate monitoring
- Circuit breaker state visualization
- Connection pool utilization

### Custom Monitoring

```python
def setup_monitoring(lm):
    """Set up comprehensive monitoring."""
    
    def alert_callback(metrics):
        if metrics["requests"]["error_rate"] > 25:
            send_alert("High MCP error rate", metrics)
        
        if metrics["requests"]["avg_latency_ms"] > 5000:
            send_alert("High MCP latency", metrics)
    
    def metrics_to_datadog(metrics):
        # Send metrics to DataDog
        statsd.gauge("mcp.requests.total", metrics["requests"]["total"])
        statsd.gauge("mcp.requests.latency", metrics["requests"]["avg_latency_ms"])
        statsd.gauge("mcp.requests.error_rate", metrics["requests"]["error_rate"])
    
    lm.metrics_collector.add_export_callback(alert_callback)
    lm.metrics_collector.add_export_callback(metrics_to_datadog)
```

## Performance Impact

The metrics system is designed for minimal performance impact:

- **Memory Usage**: Bounded by `max_history_size` setting
- **CPU Overhead**: < 1% for typical workloads
- **Storage**: Configurable export intervals and formats
- **Network**: No network overhead unless using remote export

## Best Practices

1. **Production Settings**:
   - Set appropriate `max_history_size` for your memory constraints
   - Use `export_interval` of 15-60 seconds
   - Disable `include_detailed_timings` for performance

2. **Development Settings**:
   - Enable `include_detailed_timings` for debugging
   - Use `export_to_console` for immediate feedback
   - Set shorter `export_interval` for rapid iteration

3. **Monitoring**:
   - Set up alerts for error rates > 20%
   - Monitor latency trends over time
   - Track circuit breaker state changes

4. **Storage**:
   - Rotate metrics files regularly
   - Use appropriate export formats for your monitoring stack
   - Consider compression for long-term storage

## Troubleshooting

### Common Issues

1. **Metrics Not Collected**:
   - Verify `enabled: true` in configuration
   - Check that metrics collector is initialized
   - Ensure proper configuration format

2. **High Memory Usage**:
   - Reduce `max_history_size`
   - Increase `export_interval`
   - Enable regular metrics export

3. **Export Failures**:
   - Check file permissions for `export_file`
   - Verify export format is supported
   - Review callback function errors

### Debug Mode

Enable debug logging for metrics:

```python
import logging
logging.getLogger('graph_of_thoughts.language_models.mcp_metrics').setLevel(logging.DEBUG)
```

This provides detailed information about metrics collection, export, and any issues encountered.
