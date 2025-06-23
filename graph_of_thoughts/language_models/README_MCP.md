# Graph of Thoughts MCP Implementation

This directory contains the Model Context Protocol (MCP) implementation for the Graph of Thoughts framework. The MCP implementation enables seamless integration with MCP hosts like Claude Desktop, VSCode, Cursor, and remote MCP servers.

## Overview

The MCP implementation provides a complete, protocol-compliant client that can communicate with any MCP host. It abstracts away the complexity of the MCP protocol while providing a simple, familiar interface for language model interactions.

### Key Features

- **Full MCP Protocol Compliance**: Implements the complete MCP specification with JSON-RPC 2.0 messaging
- **Multiple Transport Support**: Both stdio (local) and HTTP (remote) transports
- **Robust Error Handling**: Comprehensive error handling with custom exception hierarchy
- **Async Support**: Full async/await support with proper resource management
- **Configuration Migration**: Automatic migration from legacy configuration formats
- **Cost Tracking**: Built-in token usage and cost estimation
- **Response Caching**: Optional response caching for improved performance

## Architecture

```
┌─────────────────────┐
│   MCPLanguageModel │  ← Main interface (inherits from AbstractLanguageModel)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    MCPTransport    │  ← Abstract transport layer
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌─────────┐ ┌─────────┐
│ Stdio   │ │  HTTP   │  ← Concrete transport implementations
│Transport│ │Transport│
└─────────┘ └─────────┘
```

### Core Components

1. **MCPLanguageModel** (`mcp_client.py`): Main interface for language model interactions
2. **MCPTransport** (`mcp_transport.py`): Transport layer abstraction with stdio and HTTP implementations
3. **MCPSamplingManager** (`mcp_sampling.py`): Advanced sampling features and conversation management
4. **MCPProtocolValidator** (`mcp_protocol.py`): Protocol validation and message formatting utilities

## Quick Start

### Basic Usage

```python
from graph_of_thoughts.language_models import MCPLanguageModel

# Initialize with Claude Desktop
lm = MCPLanguageModel(
    config_path="mcp_config.json",
    model_name="mcp_claude_desktop",
    cache=True
)

# Query the model
response = lm.query("What is machine learning?")
text = lm.get_response_texts(response)[0]
print(text)
```

### Async Usage

```python
async def example():
    async with MCPLanguageModel("config.json", "mcp_claude_desktop") as lm:
        response = await lm._query_async("Explain quantum computing")
        return lm.get_response_texts(response)[0]

import asyncio
result = asyncio.run(example())
```

## Configuration

### Configuration File Format

Create a JSON configuration file with your MCP host settings:

```json
{
    "mcp_claude_desktop": {
        "transport": {
            "type": "stdio",
            "command": "claude-desktop",
            "args": ["--mcp-server"],
            "env": {}
        },
        "client_info": {
            "name": "graph-of-thoughts",
            "version": "0.0.3"
        },
        "capabilities": {
            "sampling": {}
        },
        "default_sampling_params": {
            "temperature": 1.0,
            "maxTokens": 4096,
            "includeContext": "thisServer"
        },
        "connection_config": {
            "timeout": 30.0,
            "retry_attempts": 3
        },
        "cost_tracking": {
            "prompt_token_cost": 0.003,
            "response_token_cost": 0.015
        }
    }
}
```

### Supported MCP Hosts

#### Claude Desktop (Stdio)
```json
{
    "transport": {
        "type": "stdio",
        "command": "claude-desktop",
        "args": ["--mcp-server"]
    }
}
```

#### VSCode with MCP Extension (Stdio)
```json
{
    "transport": {
        "type": "stdio",
        "command": "code",
        "args": ["--mcp-server"]
    }
}
```

#### Remote MCP Server (HTTP)
```json
{
    "transport": {
        "type": "http",
        "url": "https://api.example.com/mcp",
        "headers": {
            "Authorization": "Bearer your-token"
        }
    }
}
```

## Advanced Features

### Multiple Responses

```python
# Generate multiple creative responses
responses = lm.query("Write a haiku about technology", num_responses=3)
for i, haiku in enumerate(lm.get_response_texts(responses)):
    print(f"Haiku {i+1}:\n{haiku}\n")
```

### Conversation Management

```python
from graph_of_thoughts.language_models.mcp_sampling import MCPSamplingManager
from graph_of_thoughts.language_models.mcp_transport import create_transport

transport = create_transport(config)
sampling_manager = MCPSamplingManager(transport, config)

async with transport:
    # Start conversation
    response1 = await sampling_manager.create_conversation_completion(
        "Hello, I'm learning about Python."
    )
    
    # Continue with history
    response2 = await sampling_manager.create_conversation_completion(
        "What are Python's main advantages?",
        use_history=True
    )
```

### Batch Processing

```python
prompts = [
    "Explain photosynthesis",
    "Describe quantum mechanics", 
    "What is machine learning?"
]

async with transport:
    responses = await sampling_manager.create_batch_completions(prompts)
    for response in responses:
        print(response)
```

## Error Handling

The MCP implementation provides comprehensive error handling:

```python
from graph_of_thoughts.language_models.mcp_transport import (
    MCPConnectionError, MCPTimeoutError, MCPServerError
)

try:
    response = lm.query("Complex query")
except MCPConnectionError as e:
    print(f"Connection failed: {e}")
except MCPTimeoutError as e:
    print(f"Request timed out: {e}")
except MCPServerError as e:
    print(f"Server error {e.error_code}: {e}")
```

## Testing

Run the integration tests to verify your setup:

```bash
python test_mcp_integration.py
```

Run the example to test functionality:

```bash
python examples/sorting/sorting_032_mcp.py
```

## Migration from Legacy Configurations

The implementation automatically migrates old configuration formats. Legacy configurations with `transport_type` and `host_type` fields are automatically converted to the new MCP-compliant format.

## Performance Considerations

- **Connection Reuse**: Connections are maintained and reused across requests
- **Caching**: Enable response caching for repeated queries
- **Batch Processing**: Use batch operations for multiple requests
- **Async Operations**: Use async methods for better concurrency

## Troubleshooting

### Common Issues

1. **Connection Failures**: Check that the MCP host is installed and accessible
2. **Timeout Errors**: Increase timeout values in connection_config
3. **Protocol Errors**: Verify configuration format and MCP host compatibility
4. **Permission Errors**: Ensure proper permissions for stdio commands

### Debug Logging

Enable debug logging to see detailed protocol communication:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing to the MCP implementation:

1. Follow the existing code style and documentation patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure protocol compliance with the official MCP specification

## License

This implementation is part of the Graph of Thoughts project and follows the same BSD-style license.
