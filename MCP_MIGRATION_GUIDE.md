# Migration Guide: From API Keys to Model Context Protocol (MCP)

This guide helps you migrate your Graph of Thoughts implementation from using direct API key-based language models to the Model Context Protocol (MCP) for enhanced security, flexibility, and integration with modern AI development environments.

## Overview

The Model Context Protocol (MCP) provides a standardized way to connect AI applications to language models through various hosts like Claude Desktop, VSCode, Cursor, or remote MCP servers. This eliminates the need to manage API keys directly in your code and provides better integration with your development environment.

## Benefits of MCP Migration

- **Enhanced Security**: No need to store API keys in configuration files
- **Better Integration**: Seamless connection with AI-enabled IDEs and applications
- **Flexibility**: Switch between different language models and providers easily
- **Human-in-the-Loop**: Built-in support for user oversight and control
- **Context Sharing**: Automatic context sharing between tools and applications

## Prerequisites

Before migrating to MCP, ensure you have:

1. **Python 3.8+** with the updated Graph of Thoughts package
2. **MCP-enabled host** (one of the following):
   - Claude Desktop with MCP support
   - VSCode with MCP extension
   - Cursor with MCP integration
   - A remote MCP server
3. **Updated dependencies** (automatically installed with the new package)

## Migration Steps

### Step 1: Update Dependencies

The new version of Graph of Thoughts includes MCP support. Update your installation:

```bash
pip install --upgrade graph_of_thoughts
```

Or if installing from source:
```bash
pip install -e .
```

### Step 2: Create MCP Configuration

Create an MCP configuration file by copying the template:

```bash
cp graph_of_thoughts/language_models/mcp_config_template.json graph_of_thoughts/language_models/mcp_config.json
```

Edit the configuration file to match your setup:

```json
{
    "mcp_claude_desktop": {
        "transport_type": "stdio",
        "host_type": "claude_desktop",
        "model_preferences": {
            "hints": [
                {"name": "claude-3-5-sonnet"},
                {"name": "claude-3-haiku"}
            ],
            "costPriority": 0.3,
            "speedPriority": 0.4,
            "intelligencePriority": 0.8
        },
        "sampling_config": {
            "temperature": 1.0,
            "max_tokens": 4096,
            "stop_sequences": [],
            "include_context": "thisServer"
        },
        "connection_config": {
            "timeout": 30.0,
            "retry_attempts": 3,
            "retry_delay": 1.0
        },
        "prompt_token_cost": 0.003,
        "response_token_cost": 0.015
    }
}
```

### Step 3: Update Your Code

#### Before (API Key-based):
```python
from graph_of_thoughts import controller, language_models, operations

# Old way - requires API key
lm = language_models.ChatGPT(
    "config.json",
    model_name="chatgpt",
    cache=True
)
```

#### After (MCP-based):
```python
from graph_of_thoughts import controller, language_models, operations

# New way - uses MCP
lm = language_models.MCPLanguageModel(
    "graph_of_thoughts/language_models/mcp_config.json",
    model_name="mcp_claude_desktop",
    cache=True
)
```

### Step 4: Configure Your MCP Host

#### For Claude Desktop:
1. Install Claude Desktop with MCP support
2. Configure Claude Desktop to expose MCP server
3. Ensure the MCP server is running when you execute your code

#### For VSCode:
1. Install the MCP extension for VSCode
2. Configure the extension settings
3. Start VSCode with MCP server enabled

#### For Cursor:
1. Ensure Cursor has MCP support enabled
2. Configure MCP settings in Cursor preferences
3. Start Cursor with MCP server running

#### For Remote MCP Server:
1. Set up your remote MCP server
2. Update the configuration to use HTTP transport:
```json
{
    "mcp_http_server": {
        "transport_type": "http",
        "server_url": "http://your-server:8000/mcp",
        ...
    }
}
```

## Configuration Options

### Transport Types

- **stdio**: For local MCP hosts (Claude Desktop, VSCode, Cursor)
- **http**: For remote MCP servers

### Model Preferences

Configure model selection preferences:
- `hints`: Preferred model names or families
- `costPriority`: Importance of minimizing costs (0-1)
- `speedPriority`: Importance of low latency (0-1)
- `intelligencePriority`: Importance of model capabilities (0-1)

### Sampling Configuration

Control how the language model generates responses:
- `temperature`: Randomness of responses (0.0-1.0)
- `max_tokens`: Maximum response length
- `stop_sequences`: Sequences that stop generation
- `include_context`: Context inclusion ("none", "thisServer", "allServers")

## Example Migration

Here's a complete example showing the migration of a sorting task:

### Before (API Key):
```python
import os
from graph_of_thoughts import controller, language_models, operations
from examples.sorting.sorting_032 import SortingPrompter, SortingParser

# Required API key in environment or config
lm = language_models.ChatGPT(
    "config.json",
    model_name="chatgpt",
    cache=True
)

# Create operations graph
gop = operations.GraphOfOperations()
gop.append_operation(operations.Generate(1, 1))
gop.append_operation(operations.Score(1, False, utils.num_errors))
gop.append_operation(operations.GroundTruth(utils.test_sorting))

# Run controller
ctrl = controller.Controller(
    lm, gop, SortingPrompter(), SortingParser(),
    {"original": "[3, 1, 4, 1, 5, 9, 2, 6]", "current": "", "method": "io"}
)
ctrl.run()
```

### After (MCP):
```python
import os
from graph_of_thoughts import controller, language_models, operations
from examples.sorting.sorting_032 import SortingPrompter, SortingParser

# No API key needed - uses MCP host
lm = language_models.MCPLanguageModel(
    "graph_of_thoughts/language_models/mcp_config.json",
    model_name="mcp_claude_desktop",
    cache=True
)

# Same operations graph
gop = operations.GraphOfOperations()
gop.append_operation(operations.Generate(1, 1))
gop.append_operation(operations.Score(1, False, utils.num_errors))
gop.append_operation(operations.GroundTruth(utils.test_sorting))

# Same controller usage
ctrl = controller.Controller(
    lm, gop, SortingPrompter(), SortingParser(),
    {"original": "[3, 1, 4, 1, 5, 9, 2, 6]", "current": "", "method": "io"}
)
ctrl.run()
```

## Troubleshooting

### Common Issues

1. **Connection Failed**: Ensure your MCP host is running and properly configured
2. **Model Not Available**: Check that your model preferences match available models
3. **Timeout Errors**: Increase timeout values in connection_config
4. **Permission Denied**: Verify MCP host permissions and access rights

### Debugging

Enable debug logging to troubleshoot issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Fallback Strategy

You can maintain both API key and MCP configurations for fallback:
```python
try:
    lm = language_models.MCPLanguageModel("mcp_config.json", "mcp_claude_desktop")
except Exception as e:
    print(f"MCP connection failed: {e}, falling back to API key")
    lm = language_models.ChatGPT("config.json", "chatgpt")
```

## Best Practices

1. **Test Connection**: Always test your MCP connection before running large experiments
2. **Monitor Costs**: MCP still incurs costs - monitor usage through your host
3. **Use Caching**: Enable caching for repeated experiments
4. **Configure Timeouts**: Set appropriate timeouts for your use case
5. **Handle Errors**: Implement proper error handling for connection issues

## Support

For issues with MCP integration:
- Check the MCP documentation for your specific host
- Review the Graph of Thoughts logs for detailed error messages
- Ensure your MCP host supports the required protocol version
- Test with simple examples before complex operations

## Next Steps

After successful migration:
1. Explore advanced MCP features like context sharing
2. Experiment with different model preferences
3. Set up multiple MCP configurations for different use cases
4. Consider implementing custom MCP servers for specialized needs
