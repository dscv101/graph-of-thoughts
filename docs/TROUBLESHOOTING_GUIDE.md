# Troubleshooting Guide: MCP Integration Issues

This guide helps you diagnose and resolve common issues when using the Graph of Thoughts MCP integration.

## Quick Diagnostics

### 1. Test MCP Connection

```python
import asyncio
from graph_of_thoughts.language_models import MCPLanguageModel

async def test_connection():
    try:
        lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
        async with lm:
            response = await lm.query_async("Hello, world!")
            print("✅ Connection successful!")
            print(f"Response: {lm.get_response_texts(response)[0]}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    return True

# Run the test
asyncio.run(test_connection())
```

### 2. Validate Configuration

```python
import json
from graph_of_thoughts.language_models.mcp_config_validator import validate_config

def validate_mcp_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Basic validation
        required_fields = ['transport', 'client_info', 'capabilities']
        for model_name, model_config in config.items():
            for field in required_fields:
                if field not in model_config:
                    print(f"❌ Missing required field '{field}' in {model_name}")
                    return False
        
        print("✅ Configuration validation passed!")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

# Validate your config
validate_mcp_config("mcp_config.json")
```

## Common Issues and Solutions

### Issue 1: Connection Timeout

**Symptoms:**
```
MCPTimeoutError: Request timeout after 30s
```

**Causes:**
- MCP host is not responding
- Network connectivity issues
- Host is overloaded

**Solutions:**

1. **Increase timeout:**
```json
{
    "mcp_claude_desktop": {
        "transport": {
            "timeout": 60.0
        }
    }
}
```

2. **Check host status:**
```bash
# For Claude Desktop
ps aux | grep claude-desktop

# For VSCode MCP
ps aux | grep code
```

3. **Test network connectivity:**
```bash
# Test local connectivity
netstat -an | grep LISTEN

# Test remote connectivity (if using HTTP transport)
curl -v http://your-mcp-server:port/health
```

### Issue 2: Authentication Failures

**Symptoms:**
```
MCPServerError: Authentication failed
MCPConnectionError: Access forbidden
```

**Solutions:**

1. **Verify MCP host authentication:**
```json
{
    "mcp_claude_desktop": {
        "transport": {
            "type": "stdio",
            "command": "claude-desktop",
            "args": ["--mcp", "--auth-token", "your-token"]
        }
    }
}
```

2. **Check environment variables:**
```bash
echo $CLAUDE_API_KEY
echo $MCP_AUTH_TOKEN
```

3. **Verify host permissions:**
```bash
# Check if command is executable
which claude-desktop
ls -la $(which claude-desktop)
```

### Issue 3: Protocol Version Mismatch

**Symptoms:**
```
MCPProtocolError: Unsupported protocol version
```

**Solutions:**

1. **Update MCP SDK:**
```bash
pip install --upgrade mcp
```

2. **Check protocol version:**
```python
from graph_of_thoughts.language_models.mcp_transport import get_supported_versions
print(f"Supported versions: {get_supported_versions()}")
```

3. **Force specific version:**
```json
{
    "mcp_claude_desktop": {
        "protocol_version": "2024-11-05"
    }
}
```

### Issue 4: Memory Issues with Batch Processing

**Symptoms:**
```
MemoryError: Unable to allocate memory
Process killed (OOM)
```

**Solutions:**

1. **Reduce batch size:**
```json
{
    "mcp_claude_desktop": {
        "batch_processing": {
            "batch_size": 10,
            "max_concurrent": 3
        }
    }
}
```

2. **Monitor memory usage:**
```python
import psutil
import asyncio

async def monitor_batch_processing():
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
    async with lm:
        responses = await lm.query_batch(queries, max_concurrent=5)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
    print(f"Memory increase: {final_memory - initial_memory:.1f}MB")
```

3. **Use streaming for large datasets:**
```python
async def process_large_dataset(queries):
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
    
    # Process in smaller chunks
    chunk_size = 50
    all_responses = []
    
    async with lm:
        for i in range(0, len(queries), chunk_size):
            chunk = queries[i:i + chunk_size]
            chunk_responses = await lm.query_batch(chunk)
            all_responses.extend(chunk_responses)
            
            # Optional: clear cache between chunks
            if hasattr(lm, 'cache_manager'):
                lm.cache_manager.clear()
    
    return all_responses
```

### Issue 5: Circuit Breaker Triggering

**Symptoms:**
```
CircuitBreakerOpenError: Circuit breaker is open
```

**Solutions:**

1. **Check circuit breaker status:**
```python
lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
status = lm.get_circuit_breaker_status()
if status:
    print(f"Circuit state: {status['state']}")
    print(f"Failure rate: {status['failed_requests'] / status['total_requests']:.2%}")
```

2. **Adjust circuit breaker settings:**
```json
{
    "mcp_claude_desktop": {
        "circuit_breaker": {
            "failure_threshold": 10,
            "recovery_timeout": 60.0,
            "half_open_max_calls": 5
        }
    }
}
```

3. **Reset circuit breaker:**
```python
# Wait for recovery timeout or restart application
import time
time.sleep(60)  # Wait for recovery timeout
```

### Issue 6: Async Context Issues

**Symptoms:**
```
RuntimeError: Cannot call _run_async_query from within an async context
RuntimeError: asyncio.run() cannot be called from a running event loop
```

**Solutions:**

1. **Use appropriate methods:**
```python
# In async context
async def async_function():
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
    async with lm:
        response = await lm.query_async("Hello")  # Use query_async
    return response

# In sync context
def sync_function():
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
    response = lm.query("Hello")  # Use query
    return response
```

2. **Check event loop:**
```python
import asyncio

def safe_async_call(coro):
    try:
        loop = asyncio.get_running_loop()
        # Already in async context, create task
        return asyncio.create_task(coro)
    except RuntimeError:
        # No running loop, use asyncio.run
        return asyncio.run(coro)
```

## Performance Troubleshooting

### Slow Response Times

**Diagnosis:**
```python
import time
import asyncio

async def benchmark_performance():
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
    
    # Test single query
    start_time = time.time()
    async with lm:
        response = await lm.query_async("Hello, world!")
    single_time = time.time() - start_time
    
    # Test batch query
    queries = ["Hello"] * 10
    start_time = time.time()
    async with lm:
        responses = await lm.query_batch(queries)
    batch_time = time.time() - start_time
    
    print(f"Single query: {single_time:.2f}s")
    print(f"Batch queries: {batch_time:.2f}s ({batch_time/10:.2f}s per query)")
    print(f"Speedup: {single_time / (batch_time/10):.1f}x")

asyncio.run(benchmark_performance())
```

**Optimizations:**

1. **Enable caching:**
```json
{
    "mcp_claude_desktop": {
        "caching": {
            "enabled": true,
            "max_size": 1000,
            "ttl": 3600
        }
    }
}
```

2. **Optimize retry settings:**
```json
{
    "mcp_claude_desktop": {
        "retry_config": {
            "max_attempts": 2,
            "base_delay": 0.5,
            "strategy": "fixed"
        }
    }
}
```

3. **Use connection pooling:**
```json
{
    "mcp_claude_desktop": {
        "transport": {
            "type": "http",
            "connection_pool_size": 10,
            "keep_alive": true
        }
    }
}
```

### High Error Rates

**Diagnosis:**
```python
async def analyze_error_patterns():
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
    
    test_queries = ["Hello"] * 100
    errors = []
    successes = 0
    
    async with lm:
        for i, query in enumerate(test_queries):
            try:
                await lm.query_async(query)
                successes += 1
            except Exception as e:
                errors.append((i, type(e).__name__, str(e)))
    
    print(f"Success rate: {successes}/{len(test_queries)} ({successes/len(test_queries):.1%})")
    
    # Analyze error types
    error_types = {}
    for _, error_type, _ in errors:
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    print("Error breakdown:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")

asyncio.run(analyze_error_patterns())
```

## Debugging Tools

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('graph_of_thoughts.language_models')
logger.setLevel(logging.DEBUG)

# Create handler with detailed format
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
```

### MCP Protocol Inspector

```python
class MCPProtocolInspector:
    def __init__(self, lm):
        self.lm = lm
        self.requests = []
        self.responses = []
    
    async def inspect_request(self, method, params):
        print(f"📤 Sending: {method}")
        print(f"   Params: {params}")
        self.requests.append((method, params))
        
        try:
            response = await self.lm.transport.send_request(method, params)
            print(f"📥 Received: {response}")
            self.responses.append(response)
            return response
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def get_stats(self):
        return {
            'total_requests': len(self.requests),
            'total_responses': len(self.responses),
            'success_rate': len(self.responses) / len(self.requests) if self.requests else 0
        }

# Usage
inspector = MCPProtocolInspector(lm)
```

### Configuration Validator

```python
def validate_mcp_configuration(config_path):
    """Comprehensive configuration validation."""
    issues = []
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        return [f"Failed to load config: {e}"]
    
    for model_name, model_config in config.items():
        # Check required fields
        required = ['transport', 'client_info', 'capabilities']
        for field in required:
            if field not in model_config:
                issues.append(f"{model_name}: Missing required field '{field}'")
        
        # Validate transport
        transport = model_config.get('transport', {})
        if 'type' not in transport:
            issues.append(f"{model_name}: Transport type not specified")
        elif transport['type'] == 'stdio':
            if 'command' not in transport:
                issues.append(f"{model_name}: stdio transport missing 'command'")
        elif transport['type'] == 'http':
            if 'url' not in transport:
                issues.append(f"{model_name}: http transport missing 'url'")
        
        # Validate retry config
        retry_config = model_config.get('retry_config', {})
        if 'max_attempts' in retry_config:
            if not isinstance(retry_config['max_attempts'], int) or retry_config['max_attempts'] < 1:
                issues.append(f"{model_name}: Invalid max_attempts in retry_config")
    
    return issues

# Validate configuration
issues = validate_mcp_configuration("mcp_config.json")
if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  ❌ {issue}")
else:
    print("✅ Configuration is valid")
```

## Getting Help

### Collecting Debug Information

When reporting issues, include:

1. **System Information:**
```python
import sys
import platform
import pkg_resources

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Graph of Thoughts: {pkg_resources.get_distribution('graph-of-thoughts').version}")
print(f"MCP SDK: {pkg_resources.get_distribution('mcp').version}")
```

2. **Configuration (sanitized):**
```python
import json

def sanitize_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Remove sensitive information
    for model_config in config.values():
        if 'auth_token' in model_config.get('transport', {}):
            model_config['transport']['auth_token'] = '***REDACTED***'
    
    return json.dumps(config, indent=2)

print(sanitize_config("mcp_config.json"))
```

3. **Error Logs:**
```python
# Enable comprehensive logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='mcp_debug.log'
)
```

### Community Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check latest documentation for updates
- **Examples**: Review example configurations and code
- **Discord/Slack**: Join community discussions
