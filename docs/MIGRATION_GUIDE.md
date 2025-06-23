# Migration Guide: From Legacy Language Models to MCP

This guide helps you migrate from legacy language model implementations to the new Model Context Protocol (MCP) based system in Graph of Thoughts.

## Overview

The Graph of Thoughts framework has been refactored to use the Model Context Protocol (MCP) instead of direct API keys. This provides:

- **Better Integration**: Works with Claude Desktop, VSCode, Cursor, and other MCP hosts
- **Enhanced Security**: No need to manage API keys directly
- **Improved Reliability**: Built-in retry, circuit breaker, and error handling
- **Better Performance**: Connection pooling, caching, and batch processing

## Migration Scenarios

### Scenario 1: From OpenAI API to MCP Claude Desktop

**Before (Legacy):**
```python
from graph_of_thoughts.language_models import ChatGPT

# Old way with API keys
lm = ChatGPT(
    api_key="sk-...",
    model_name="gpt-4",
    cache=True
)

response = lm.query("What is machine learning?")
text = lm.get_response_texts(response)[0]
```

**After (MCP):**
```python
from graph_of_thoughts.language_models import MCPLanguageModel

# New way with MCP configuration
lm = MCPLanguageModel(
    config_path="mcp_config.json",
    model_name="mcp_claude_desktop",
    cache=True
)

response = lm.query("What is machine learning?")
text = lm.get_response_texts(response)[0]
```

**Configuration File (`mcp_config.json`):**
```json
{
    "mcp_claude_desktop": {
        "transport": {
            "type": "stdio",
            "command": "claude-desktop",
            "args": ["--mcp"]
        },
        "client_info": {
            "name": "graph-of-thoughts",
            "version": "1.0.0"
        },
        "capabilities": {
            "sampling": {}
        }
    }
}
```

### Scenario 2: From Anthropic API to MCP

**Before:**
```python
from graph_of_thoughts.language_models import Claude

lm = Claude(
    api_key="sk-ant-...",
    model_name="claude-3-sonnet-20240229",
    max_tokens=4096
)
```

**After:**
```python
from graph_of_thoughts.language_models import MCPLanguageModel

lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
```

### Scenario 3: Batch Processing Migration

**Before:**
```python
# Sequential processing
responses = []
for query in queries:
    response = lm.query(query)
    responses.append(response)
```

**After:**
```python
# Concurrent batch processing
async def process_batch():
    async with lm:
        responses = await lm.query_batch(queries, max_concurrent=10)
    return responses

responses = asyncio.run(process_batch())
```

## Step-by-Step Migration

### Step 1: Install MCP Dependencies

```bash
# Install MCP SDK
pip install mcp

# Or if using the Graph of Thoughts package
pip install graph-of-thoughts[mcp]
```

### Step 2: Set Up MCP Host

Choose your preferred MCP host:

#### Option A: Claude Desktop
1. Install Claude Desktop
2. Configure MCP server in Claude Desktop settings
3. Use stdio transport in configuration

#### Option B: VSCode with MCP Extension
1. Install VSCode MCP extension
2. Configure MCP servers in VSCode settings
3. Use stdio transport

#### Option C: Remote MCP Server
1. Deploy MCP server to remote host
2. Use HTTP transport in configuration

### Step 3: Create MCP Configuration

Create `mcp_config.json`:

```json
{
    "mcp_claude_desktop": {
        "transport": {
            "type": "stdio",
            "command": "claude-desktop",
            "args": ["--mcp"],
            "timeout": 30.0
        },
        "client_info": {
            "name": "graph-of-thoughts",
            "version": "1.0.0"
        },
        "capabilities": {
            "sampling": {}
        },
        "default_sampling_params": {
            "maxTokens": 4096,
            "temperature": 0.7
        }
    }
}
```

### Step 4: Update Code

Replace legacy language model imports:

```python
# Remove these imports
# from graph_of_thoughts.language_models import ChatGPT, Claude, HuggingFace

# Add this import
from graph_of_thoughts.language_models import MCPLanguageModel
```

Update initialization:

```python
# Old
lm = ChatGPT(api_key="...", model_name="gpt-4")

# New
lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
```

### Step 5: Test Migration

Create a test script to verify functionality:

```python
import asyncio
from graph_of_thoughts.language_models import MCPLanguageModel

async def test_migration():
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
    
    # Test basic query
    async with lm:
        response = await lm.query_async("Hello, world!")
        print(f"Response: {lm.get_response_texts(response)[0]}")
        
        # Test batch processing
        queries = ["What is AI?", "Explain quantum computing", "Define blockchain"]
        batch_responses = await lm.query_batch(queries)
        
        for i, resp in enumerate(batch_responses):
            text = lm.get_response_texts(resp)[0]
            print(f"Query {i+1}: {text[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_migration())
```

## Configuration Migration

### Legacy Configuration Mapping

| Legacy Parameter | MCP Configuration | Notes |
|------------------|-------------------|-------|
| `api_key` | Not needed | Handled by MCP host |
| `model_name` | `model_name` parameter | Use MCP model identifier |
| `max_tokens` | `default_sampling_params.maxTokens` | In config file |
| `temperature` | `default_sampling_params.temperature` | In config file |
| `timeout` | `transport.timeout` | Connection timeout |

### Environment Variables

**Before:**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

**After:**
```bash
# No API keys needed - MCP host handles authentication
export MCP_CONFIG_PATH="/path/to/mcp_config.json"
```

## Advanced Migration Examples

### Custom Retry Configuration

```json
{
    "mcp_claude_desktop": {
        "retry_config": {
            "max_attempts": 5,
            "base_delay": 1.0,
            "strategy": "exponential",
            "jitter_type": "equal",
            "connection_error_max_attempts": 7,
            "timeout_error_max_attempts": 3
        }
    }
}
```

### Circuit Breaker Configuration

```json
{
    "mcp_claude_desktop": {
        "circuit_breaker": {
            "enabled": true,
            "failure_threshold": 5,
            "recovery_timeout": 30.0,
            "half_open_max_calls": 3
        }
    }
}
```

### Batch Processing Configuration

```json
{
    "mcp_claude_desktop": {
        "batch_processing": {
            "max_concurrent": 10,
            "batch_size": 50,
            "retry_attempts": 3,
            "timeout_per_request": 30.0
        }
    }
}
```

## Common Migration Issues

### Issue 1: Import Errors

**Problem:**
```python
ImportError: cannot import name 'ChatGPT' from 'graph_of_thoughts.language_models'
```

**Solution:**
Update imports to use `MCPLanguageModel`:
```python
from graph_of_thoughts.language_models import MCPLanguageModel
```

### Issue 2: Configuration Not Found

**Problem:**
```python
FileNotFoundError: [Errno 2] No such file or directory: 'mcp_config.json'
```

**Solution:**
1. Create configuration file in correct location
2. Use absolute path: `MCPLanguageModel("/full/path/to/mcp_config.json", "model_name")`
3. Set environment variable: `export MCP_CONFIG_PATH="/path/to/config"`

### Issue 3: MCP Host Connection Failed

**Problem:**
```python
MCPConnectionError: Failed to connect to MCP host
```

**Solution:**
1. Verify MCP host is running
2. Check transport configuration
3. Verify command and arguments
4. Test connection manually

### Issue 4: Async Context Issues

**Problem:**
```python
RuntimeError: Cannot call _run_async_query from within an async context
```

**Solution:**
Use async methods in async contexts:
```python
# In async function
async with lm:
    response = await lm.query_async("Hello")

# In sync context
response = lm.query("Hello")
```

## Performance Optimization

### Before Migration Benchmark

```python
import time

start_time = time.time()
responses = []
for query in queries:
    response = lm.query(query)
    responses.append(response)
end_time = time.time()

print(f"Sequential processing: {end_time - start_time:.2f}s")
```

### After Migration Benchmark

```python
import asyncio
import time

async def benchmark_batch():
    start_time = time.time()
    async with lm:
        responses = await lm.query_batch(queries, max_concurrent=10)
    end_time = time.time()
    
    print(f"Batch processing: {end_time - start_time:.2f}s")
    return responses

responses = asyncio.run(benchmark_batch())
```

## Rollback Strategy

If you need to rollback to legacy implementation:

1. **Keep Legacy Dependencies:**
   ```bash
   pip install openai anthropic  # Keep these installed
   ```

2. **Maintain Dual Configuration:**
   ```python
   USE_MCP = os.getenv("USE_MCP", "false").lower() == "true"
   
   if USE_MCP:
       from graph_of_thoughts.language_models import MCPLanguageModel
       lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
   else:
       from graph_of_thoughts.language_models import ChatGPT
       lm = ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
   ```

3. **Feature Flag Approach:**
   ```python
   class LanguageModelFactory:
       @staticmethod
       def create(use_mcp=True):
           if use_mcp:
               return MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
           else:
               return ChatGPT(api_key=os.getenv("OPENAI_API_KEY"))
   ```

## Next Steps

After successful migration:

1. **Remove Legacy Dependencies:**
   ```bash
   pip uninstall openai anthropic
   ```

2. **Update Documentation:**
   - Update README files
   - Update deployment scripts
   - Update CI/CD pipelines

3. **Monitor Performance:**
   - Set up metrics collection
   - Monitor error rates
   - Track response times

4. **Optimize Configuration:**
   - Tune retry strategies
   - Adjust batch sizes
   - Configure circuit breakers

## Getting Help

If you encounter issues during migration:

1. **Check Logs:** Enable debug logging to see detailed error messages
2. **Review Configuration:** Validate your MCP configuration against examples
3. **Test Connectivity:** Use MCP client tools to test host connectivity
4. **Consult Documentation:** Review MCP protocol documentation
5. **Community Support:** Ask questions in the Graph of Thoughts community

For specific migration assistance, provide:
- Current language model configuration
- Target MCP host setup
- Error messages and logs
- Expected vs. actual behavior

## Migration Examples by Use Case

### Research and Analysis Workflows

**Before:**
```python
# Legacy research workflow
from graph_of_thoughts.language_models import ChatGPT
from graph_of_thoughts.operations import Generate, Score, Aggregate

lm = ChatGPT(api_key="sk-...", model_name="gpt-4")

# Sequential processing
topics = ["AI ethics", "Quantum computing", "Climate change"]
analyses = []

for topic in topics:
    # Generate analysis
    generate_op = Generate(lm, "Analyze the current state of " + topic)
    analysis = generate_op.execute()

    # Score relevance
    score_op = Score(lm, "Rate the relevance of this analysis (1-10)")
    score = score_op.execute(analysis)

    analyses.append((analysis, score))

# Aggregate results
aggregate_op = Aggregate(lm, "Summarize these analyses")
summary = aggregate_op.execute(analyses)
```

**After:**
```python
# Modern MCP workflow with batch processing
from graph_of_thoughts.language_models import MCPLanguageModel
from graph_of_thoughts.operations import BatchGenerate, BatchScore, BatchAggregate

async def research_workflow():
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")

    topics = ["AI ethics", "Quantum computing", "Climate change"]

    async with lm:
        # Batch generate analyses
        generate_prompts = [f"Analyze the current state of {topic}" for topic in topics]
        analyses = await BatchGenerate(lm).execute_batch(generate_prompts)

        # Batch score relevance
        score_prompts = ["Rate the relevance of this analysis (1-10)" for _ in analyses]
        scores = await BatchScore(lm).execute_batch(score_prompts, analyses)

        # Aggregate results
        combined_data = list(zip(analyses, scores))
        summary = await BatchAggregate(lm).execute("Summarize these analyses", combined_data)

        return summary

# Run the workflow
import asyncio
result = asyncio.run(research_workflow())
```

### Educational Content Generation

**Before:**
```python
# Legacy educational workflow
from graph_of_thoughts.language_models import Claude

lm = Claude(api_key="sk-ant-...", model_name="claude-3-sonnet")

subjects = ["Mathematics", "Physics", "Chemistry", "Biology"]
grade_levels = ["elementary", "middle", "high school"]

lesson_plans = {}
for subject in subjects:
    lesson_plans[subject] = {}
    for grade in grade_levels:
        prompt = f"Create a lesson plan for {grade} level {subject}"
        response = lm.query(prompt)
        lesson_plans[subject][grade] = lm.get_response_texts(response)[0]
```

**After:**
```python
# Modern MCP workflow
from graph_of_thoughts.language_models import MCPLanguageModel
import asyncio

async def generate_lesson_plans():
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")

    subjects = ["Mathematics", "Physics", "Chemistry", "Biology"]
    grade_levels = ["elementary", "middle", "high school"]

    # Create all prompts
    prompts = []
    keys = []
    for subject in subjects:
        for grade in grade_levels:
            prompts.append(f"Create a lesson plan for {grade} level {subject}")
            keys.append((subject, grade))

    async with lm:
        # Batch process all lesson plans
        responses = await lm.query_batch(prompts, max_concurrent=8)

        # Organize results
        lesson_plans = {}
        for (subject, grade), response in zip(keys, responses):
            if subject not in lesson_plans:
                lesson_plans[subject] = {}
            lesson_plans[subject][grade] = lm.get_response_texts(response)[0]

        return lesson_plans

# Generate all lesson plans efficiently
lesson_plans = asyncio.run(generate_lesson_plans())
```

### Code Generation and Review

**Before:**
```python
# Legacy code generation
from graph_of_thoughts.language_models import ChatGPT

lm = ChatGPT(api_key="sk-...", model_name="gpt-4")

functions_to_generate = [
    "binary search algorithm",
    "quicksort implementation",
    "fibonacci sequence generator",
    "prime number checker"
]

generated_code = []
for func_desc in functions_to_generate:
    prompt = f"Generate Python code for {func_desc} with docstrings and type hints"
    response = lm.query(prompt)
    code = lm.get_response_texts(response)[0]

    # Review the code
    review_prompt = f"Review this Python code for bugs and improvements:\n{code}"
    review_response = lm.query(review_prompt)
    review = lm.get_response_texts(review_response)[0]

    generated_code.append({
        'description': func_desc,
        'code': code,
        'review': review
    })
```

**After:**
```python
# Modern MCP workflow with parallel processing
from graph_of_thoughts.language_models import MCPLanguageModel
import asyncio

async def generate_and_review_code():
    lm = MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")

    functions_to_generate = [
        "binary search algorithm",
        "quicksort implementation",
        "fibonacci sequence generator",
        "prime number checker"
    ]

    async with lm:
        # Generate all code first
        generation_prompts = [
            f"Generate Python code for {func_desc} with docstrings and type hints"
            for func_desc in functions_to_generate
        ]

        code_responses = await lm.query_batch(generation_prompts, max_concurrent=4)
        generated_codes = [lm.get_response_texts(resp)[0] for resp in code_responses]

        # Review all code
        review_prompts = [
            f"Review this Python code for bugs and improvements:\n{code}"
            for code in generated_codes
        ]

        review_responses = await lm.query_batch(review_prompts, max_concurrent=4)
        reviews = [lm.get_response_texts(resp)[0] for resp in review_responses]

        # Combine results
        results = []
        for desc, code, review in zip(functions_to_generate, generated_codes, reviews):
            results.append({
                'description': desc,
                'code': code,
                'review': review
            })

        return results

# Generate and review code efficiently
code_results = asyncio.run(generate_and_review_code())
```
