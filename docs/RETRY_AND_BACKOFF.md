# Advanced Retry and Backoff Strategies

The Graph of Thoughts MCP client provides sophisticated retry and backoff mechanisms to handle transient failures gracefully and improve overall system reliability.

## Overview

The retry system includes:

- **Multiple retry strategies**: Exponential, linear, fixed, and adaptive backoff
- **Jitter support**: Prevents thundering herd problems
- **Error-specific retry policies**: Different retry behavior for different error types
- **Adaptive strategies**: Learns from success/failure patterns
- **Circuit breaker integration**: Works with circuit breaker patterns
- **Configurable timeouts**: Adjustable timeout multipliers for retries

## Retry Strategies

### 1. Exponential Backoff (Default)

Delays increase exponentially with each retry attempt:

```python
delay = base_delay * (backoff_multiplier ^ attempt)
```

**Best for**: General purpose, API rate limiting, temporary server overload

### 2. Linear Backoff

Delays increase linearly with each attempt:

```python
delay = base_delay * (attempt + 1)
```

**Best for**: Predictable delay patterns, resource contention

### 3. Fixed Backoff

Constant delay between retries:

```python
delay = base_delay
```

**Best for**: Simple scenarios, testing, known fixed recovery times

### 4. Adaptive Backoff

Adjusts delay based on recent success/failure patterns:

- Reduces delays after consecutive successes
- Increases delays after consecutive failures
- Learns optimal timing for your specific environment

**Best for**: Production environments, varying load conditions

## Jitter Types

### 1. No Jitter (`none`)
No randomization applied to delays.

### 2. Full Jitter (`full`)
Randomizes delay between 0 and calculated delay:
```python
actual_delay = random.uniform(0, calculated_delay)
```

### 3. Equal Jitter (`equal`) - Default
Adds small random variation (±10%):
```python
jitter = calculated_delay * 0.1 * random.uniform(-1, 1)
actual_delay = calculated_delay + jitter
```

### 4. Decorrelated Jitter (`decorrelated`)
AWS-style decorrelated jitter to prevent synchronization:
```python
actual_delay = random.uniform(base_delay, previous_delay * 3)
```

## Configuration

### Basic Configuration

Add retry configuration to your MCP config file:

```json
{
    "mcp_claude_desktop": {
        "retry_config": {
            "max_attempts": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "backoff_multiplier": 2.0,
            "strategy": "exponential",
            "jitter_type": "equal",
            "timeout_multiplier": 1.0,
            "circuit_breaker_integration": true
        }
    }
}
```

### Error-Specific Configuration

Configure different retry behavior for different error types:

```json
{
    "retry_config": {
        "max_attempts": 3,
        "connection_error_max_attempts": 5,
        "timeout_error_max_attempts": 2,
        "server_error_max_attempts": 1,
        "base_delay": 1.0,
        "strategy": "exponential"
    }
}
```

### Adaptive Strategy Configuration

Fine-tune adaptive behavior:

```json
{
    "retry_config": {
        "strategy": "adaptive",
        "success_threshold_for_reduction": 5,
        "failure_threshold_for_increase": 3,
        "base_delay": 1.0,
        "max_delay": 120.0
    }
}
```

## Usage Examples

### Basic Usage

The retry system works automatically with all MCP operations:

```python
from graph_of_thoughts.language_models import MCPLanguageModel

# Retry configuration is loaded from config file
lm = MCPLanguageModel("config.json", "mcp_claude_desktop")

# Automatic retry on failures
response = lm.query("What is machine learning?")
```

### Programmatic Configuration

Override retry settings programmatically:

```python
from graph_of_thoughts.language_models.mcp_client import (
    RetryConfig, RetryStrategy, BackoffJitterType
)

# Create custom retry configuration
retry_config = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=60.0,
    strategy=RetryStrategy.ADAPTIVE,
    jitter_type=BackoffJitterType.DECORRELATED,
    connection_error_max_attempts=7,
    timeout_error_max_attempts=3
)

# Apply to language model (requires custom initialization)
```

### Batch Processing with Retry

Batch operations use the same retry configuration:

```python
async def batch_example():
    lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
    
    queries = [
        "Explain quantum computing",
        "What is machine learning?",
        "Describe blockchain technology"
    ]
    
    async with lm:
        # Each query in the batch uses the configured retry strategy
        responses = await lm.query_batch(queries)
        return responses
```

## Best Practices

### 1. Choose Appropriate Strategies

- **Development/Testing**: Use `fixed` or `linear` for predictable behavior
- **Production**: Use `exponential` or `adaptive` for better resilience
- **High-traffic**: Use `adaptive` with decorrelated jitter

### 2. Configure Error-Specific Retries

```json
{
    "retry_config": {
        "connection_error_max_attempts": 5,  // Network issues
        "timeout_error_max_attempts": 2,     // Don't retry timeouts too much
        "server_error_max_attempts": 1       // Server errors often aren't transient
    }
}
```

### 3. Use Appropriate Jitter

- **Single client**: `equal` jitter is sufficient
- **Multiple clients**: Use `decorrelated` jitter
- **Testing**: Use `none` for predictable behavior

### 4. Set Reasonable Limits

```json
{
    "retry_config": {
        "max_attempts": 3,      // Don't retry forever
        "max_delay": 60.0,      // Cap maximum delay
        "base_delay": 1.0       // Start with reasonable delay
    }
}
```

### 5. Monitor and Adjust

Use adaptive strategy in production and monitor metrics:

```python
# Check retry statistics
status = lm.get_circuit_breaker_status()
if status:
    print(f"Success rate: {status['successful_requests'] / status['total_requests']}")
```

## Integration with Circuit Breaker

The retry system integrates with circuit breaker patterns:

```json
{
    "retry_config": {
        "circuit_breaker_integration": true,
        "strategy": "adaptive"
    },
    "circuit_breaker": {
        "enabled": true,
        "failure_threshold": 5,
        "recovery_timeout": 30.0
    }
}
```

When circuit breaker integration is enabled:
- Retry attempts respect circuit breaker state
- Failed retries contribute to circuit breaker failure count
- Successful retries help close the circuit breaker

## Performance Considerations

### Memory Usage
- Retry managers maintain minimal state
- Adaptive strategy tracks recent success/failure patterns
- No significant memory overhead

### CPU Usage
- Jitter calculations are lightweight
- Strategy calculations are O(1)
- Minimal CPU overhead per retry

### Network Usage
- Retries only occur on actual failures
- Jitter prevents synchronized retry storms
- Adaptive strategy reduces unnecessary retries over time

## Troubleshooting

### High Retry Rates

If you see many retries:

1. Check network connectivity
2. Verify MCP server health
3. Consider increasing timeouts
4. Review error-specific retry limits

### Slow Response Times

If responses are slow:

1. Reduce `max_attempts`
2. Lower `max_delay`
3. Use `linear` instead of `exponential` strategy
4. Disable jitter for testing

### Thundering Herd

If multiple clients cause server overload:

1. Enable `decorrelated` jitter
2. Use `adaptive` strategy
3. Implement circuit breaker
4. Stagger client startup times

## Migration from Legacy Retry

The new retry system is backward compatible:

```python
# Legacy batch processing parameters still work
responses = await lm.query_batch(
    queries,
    retry_attempts=3,      # Maps to max_attempts
    retry_delay=1.0        # Maps to base_delay
)
```

But new configuration provides more control:

```json
{
    "retry_config": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "strategy": "exponential",
        "jitter_type": "equal"
    }
}
```
