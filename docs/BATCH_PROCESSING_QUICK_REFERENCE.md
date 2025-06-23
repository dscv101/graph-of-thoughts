# Batch Processing Quick Reference

## Quick Setup

1. **Install/Update Graph of Thoughts**
   ```bash
   pip install -e .  # If installing from source
   ```

2. **Configure MCP**
   ```bash
   cp graph_of_thoughts/language_models/mcp_config_template.json config.json
   # Edit config.json with your MCP host settings
   ```

3. **Basic Usage**
   ```python
   from graph_of_thoughts.language_models import MCPLanguageModel
   
   lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
   
   # Batch processing
   async with lm:
       responses = await lm.query_batch([
           "Query 1", "Query 2", "Query 3"
       ])
   ```

## Key Classes

### MCPLanguageModel
```python
# Enhanced with batch processing
lm = MCPLanguageModel("config.json", "mcp_claude_desktop")

# Batch query method
await lm.query_batch(
    queries,                    # List[str]
    max_concurrent=10,          # Optional[int]
    batch_size=50,             # Optional[int]
    retry_attempts=3,          # Optional[int]
    retry_delay=1.0            # Optional[float]
)
```

### Batch Operations
```python
from graph_of_thoughts.operations import (
    BatchGenerate, BatchScore, BatchAggregate
)

# Batch generation
batch_gen = BatchGenerate(
    num_branches_prompt=2,
    num_branches_response=1,
    max_concurrent=5,
    enable_batch_processing=True
)

# Batch scoring
batch_score = BatchScore(
    num_samples=1,
    max_concurrent=3,
    enable_batch_processing=True
)

# Batch aggregation
batch_agg = BatchAggregate(
    num_responses=2,
    max_concurrent=5,
    enable_batch_processing=True
)
```

### BatchParser
```python
from graph_of_thoughts.parser import BatchParser

parser = BatchParser(error_handling_strategy="return_partial")

# Batch parsing methods
results, errors = parser.parse_batch_generate_answers(states, batch_texts)
results, errors = parser.parse_batch_score_answers(states, batch_texts)
results, errors = parser.parse_batch_aggregation_answers(states, batch_texts)
```

## Configuration Options

### MCP Config File
```json
{
    "mcp_claude_desktop": {
        "batch_processing": {
            "max_concurrent": 10,
            "batch_size": 50,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "timeout_per_request": 30.0,
            "enable_by_default": true
        }
    }
}
```

### Error Handling Strategies
- `"skip_errors"`: Skip failed items, continue processing
- `"raise_on_error"`: Stop on first error
- `"return_partial"`: Return default values for failed items

## Performance Guidelines

### Recommended Settings

| Use Case | max_concurrent | batch_size | Notes |
|----------|----------------|------------|-------|
| Real-time | 3-5 | 10-20 | Low latency |
| Balanced | 8-12 | 30-50 | Good performance |
| Batch jobs | 15-20 | 100+ | Maximum throughput |

### Optimization Tips

1. **Start Conservative**
   ```python
   # Begin with safe settings
   responses = await lm.query_batch(queries, max_concurrent=5, batch_size=20)
   ```

2. **Monitor Performance**
   ```python
   import time
   start = time.time()
   responses = await lm.query_batch(queries)
   throughput = len(queries) / (time.time() - start)
   print(f"Throughput: {throughput:.1f} queries/second")
   ```

3. **Handle Large Batches**
   ```python
   # Process in chunks for very large datasets
   chunk_size = 100
   for i in range(0, len(large_query_list), chunk_size):
       chunk = large_query_list[i:i + chunk_size]
       responses = await lm.query_batch(chunk)
   ```

## Common Patterns

### Simple Batch Query
```python
async def simple_batch():
    lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
    queries = ["What is AI?", "Explain ML", "Define NLP"]
    
    async with lm:
        responses = await lm.query_batch(queries)
        texts = lm.get_response_texts(responses)
        
        for q, a in zip(queries, texts):
            print(f"Q: {q}\nA: {a}\n")
```

### Batch Operations in Graph
```python
from graph_of_thoughts.operations import BatchGenerate
from graph_of_thoughts.controller import Controller
from graph_of_thoughts.operations.graph_of_operations import GraphOfOperations

# Create graph with batch operations
graph = GraphOfOperations()
batch_gen = BatchGenerate(max_concurrent=5)
graph.add_operation(batch_gen)

# Execute
controller = Controller(lm, graph, prompter, parser, initial_state)
controller.run()
```

### Error Handling
```python
try:
    responses = await lm.query_batch(queries, max_concurrent=10)
except Exception as e:
    print(f"Batch failed: {e}")
    # Fallback to sequential
    responses = []
    for query in queries:
        try:
            response = await lm._query_async(query)
            responses.append(response)
        except Exception as query_error:
            print(f"Query failed: {query_error}")
            responses.append(None)
```

### Performance Comparison
```python
import time

# Sequential timing
start = time.time()
for query in queries:
    response = lm.query(query)
sequential_time = time.time() - start

# Batch timing
start = time.time()
async with lm:
    responses = await lm.query_batch(queries)
batch_time = time.time() - start

speedup = sequential_time / batch_time
print(f"Speedup: {speedup:.2f}x")
```

## Troubleshooting

### Common Issues

1. **Connection Timeouts**
   - Reduce `max_concurrent`
   - Increase timeout in config
   - Check MCP host capacity

2. **Memory Issues**
   - Reduce `batch_size`
   - Process in smaller chunks
   - Monitor memory usage

3. **Rate Limiting**
   - Add delays between batches
   - Reduce concurrency
   - Implement exponential backoff

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed batch processing logs
lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
```

## Migration Checklist

- [ ] Update to latest graph-of-thoughts version
- [ ] Add batch processing config to MCP config file
- [ ] Replace high-volume operations with batch equivalents
- [ ] Test with small batches first
- [ ] Optimize concurrency settings
- [ ] Implement error handling
- [ ] Monitor performance improvements

## Examples

Run these examples to see batch processing in action:

```bash
# Basic usage
python examples/simple_batch_example.py

# Performance demo
python examples/batch_processing_demo.py

# Benchmark comparison
python examples/batch_performance_benchmark.py
```

## Support

- 📖 Full documentation: [docs/BATCH_PROCESSING.md](BATCH_PROCESSING.md)
- 💡 Examples: [examples/](../examples/)
- 🐛 Issues: [GitHub Issues](https://github.com/spcl/graph-of-thoughts/issues)
- 📧 Contact: [nils.blach@inf.ethz.ch](mailto:nils.blach@inf.ethz.ch)
