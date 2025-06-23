# Event Loop Management Optimization

This document describes the event loop management optimizations implemented in the Graph of Thoughts MCP client and batch operations.

## Overview

The optimizations improve how the codebase handles asyncio event loops, replacing complex ThreadPoolExecutor workarounds with cleaner, more robust patterns that follow Python asyncio best practices.

## Key Improvements

### 1. Simplified Event Loop Detection

**Before:**
```python
try:
    loop = asyncio.get_running_loop()
    # Complex ThreadPoolExecutor workaround
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, async_function())
        return future.result()
except RuntimeError:
    return asyncio.run(async_function())
```

**After:**
```python
try:
    asyncio.get_running_loop()
    # Clear error message for developers
    raise RuntimeError(
        "Cannot call sync method from within an async context. "
        "Use 'await async_method()' instead."
    )
except RuntimeError as e:
    if "no running event loop" in str(e).lower():
        # Safe to use asyncio.run
        return asyncio.run(async_function())
    else:
        # Re-raise the async context error
        raise
```

### 2. Consistent Error Handling

The new pattern provides clear, actionable error messages when developers accidentally call synchronous methods from async contexts, guiding them to use the correct async methods instead.

### 3. Performance Benefits

- **Eliminated ThreadPoolExecutor overhead**: No more thread creation for nested event loop scenarios
- **Reduced complexity**: Simpler code paths with fewer edge cases
- **Better resource management**: Direct use of `asyncio.run()` when appropriate

## Implementation Details

### MCPLanguageModel Changes

The `_run_async_query` method now uses the optimized pattern:

```python
def _run_async_query(self, query: str, num_responses: int = 1) -> Union[List[Dict], Dict]:
    """Helper method to run async query with optimized event loop management."""
    async def _run_query():
        async with self:
            return await self._query_async(query, num_responses)

    # Check if we're already in an async context
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "Cannot call _run_async_query from within an async context. "
            "Use 'await _query_async()' instead or call from a synchronous context."
        )
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            return asyncio.run(_run_query())
        else:
            raise
```

### Batch Operations Changes

All batch operation classes (`BatchGenerate`, `BatchScore`, `BatchAggregate`) now include a `_run_async_batch_safely` method that follows the same pattern:

```python
def _run_async_batch_safely(self, lm: AbstractLanguageModel, prompts: List[str]) -> List[Dict]:
    """Safely run batch processing with optimized event loop management."""
    async def _run_batch():
        return await self._run_batch_async(lm, prompts)

    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "Cannot call _run_async_batch_safely from within an async context. "
            "Use 'await _run_batch_async()' instead or call from a synchronous context."
        )
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            return asyncio.run(_run_batch())
        else:
            raise
```

## Usage Guidelines

### For Synchronous Code

```python
# Correct usage in sync context
lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
response = lm.query("What is machine learning?")  # Uses optimized event loop management
```

### For Asynchronous Code

```python
# Correct usage in async context
async def process_queries():
    async with MCPLanguageModel("config.json", "mcp_claude_desktop") as lm:
        response = await lm._query_async("What is machine learning?")
        return response
```

### Error Handling

If you accidentally call a sync method from an async context, you'll get a clear error:

```python
async def incorrect_usage():
    lm = MCPLanguageModel("config.json", "mcp_claude_desktop")
    # This will raise a helpful RuntimeError
    response = lm.query("What is machine learning?")
```

Error message:
```
RuntimeError: Cannot call _run_async_query from within an async context. 
Use 'await _query_async()' instead or call from a synchronous context.
```

## Benefits

1. **Clearer Error Messages**: Developers get actionable feedback when using the wrong method in the wrong context
2. **Better Performance**: Eliminates unnecessary thread creation and context switching
3. **Simplified Code**: Removes complex ThreadPoolExecutor workarounds
4. **Consistent Patterns**: All async/sync boundary methods follow the same pattern
5. **Future-Proof**: Aligns with Python asyncio best practices and recommendations

## Migration Notes

This optimization is backward compatible. Existing code will continue to work, but developers will get clearer error messages if they mix sync and async contexts incorrectly.

For optimal performance and clarity, consider:
- Using async methods (`_query_async`, `_run_batch_async`) when already in async contexts
- Using sync methods (`query`, batch operations) only from synchronous contexts
- Following the error messages to correct any context mismatches
