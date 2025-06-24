# Augment Code MCP Integration Guide

## Overview

Augment Code has **native MCP support** for both local and remote agents, making it an excellent host for the Graph of Thoughts MCP server. This integration enables enhanced context through external sources and tools.

## 🚀 Quick Setup

### Step 1: Find Your Python Executable Path

**IMPORTANT**: Always use the full Python executable path to avoid import issues.

```bash
# Find your Python path
python -c "import sys; print(sys.executable)"
```

Example output: `C:\Program Files\Anaconda3\python.exe`

### Step 2: Ensure Graph of Thoughts MCP Server is Ready

Test your server using the full Python path:

```bash
# Test the server (replace with your actual Python path)
"C:\Program Files\Anaconda3\python.exe" -m graph_of_thoughts --info

# Use the automated configuration generator
python generate_mcp_configs.py
```

### Step 2: Configure Augment Code MCP Settings

Augment Code supports MCP through its settings configuration. Add the Graph of Thoughts server to your MCP configuration.

## 📋 Configuration Options

### Option 1: VS Code Extension Settings (Recommended)

If using Augment Code in VS Code, add to your `settings.json`:

```json
{
  "augment.mcp.servers": {
    "graph-of-thoughts": {
      "command": "C:\\Program Files\\Anaconda3\\python.exe",
      "args": ["-m", "graph_of_thoughts"],
      "description": "Graph of Thoughts reasoning server",
      "autoApprove": [
        "break_down_task",
        "generate_thoughts",
        "score_thoughts",
        "validate_and_improve",
        "aggregate_results",
        "create_reasoning_chain"
      ]
    }
  }
}
```

**⚠️ Important**: Replace `C:\\Program Files\\Anaconda3\\python.exe` with your actual Python path from Step 1.

### Option 2: JetBrains IDE Configuration

For JetBrains IDEs with Augment Code plugin:

1. **Open Settings** → **Tools** → **Augment Code** → **MCP Servers**
2. **Add New Server**:
   - **Name**: `graph-of-thoughts`
   - **Command**: `C:\Program Files\Anaconda3\python.exe`
   - **Arguments**: `-m graph_of_thoughts`
   - **Description**: `Graph of Thoughts reasoning server`

### Option 3: Augment Code Configuration File

Create or edit `~/.augment/mcp_config.json`:

```json
{
  "servers": {
    "graph-of-thoughts": {
      "transport": {
        "type": "stdio",
        "command": "C:\\Program Files\\Anaconda3\\python.exe",
        "args": ["-m", "graph_of_thoughts"]
      },
      "capabilities": {
        "tools": true,
        "resources": true,
        "prompts": true
      },
      "autoApprove": [
        "break_down_task",
        "generate_thoughts", 
        "score_thoughts",
        "validate_and_improve",
        "aggregate_results",
        "create_reasoning_chain"
      ],
      "description": "Advanced reasoning workflows using Graph of Thoughts methodology"
    }
  }
}
```

## 🔧 Platform-Specific Configurations

### Windows
```json
{
  "command": "C:\\Program Files\\Anaconda3\\python.exe",
  "args": ["-m", "graph_of_thoughts"]
}
```

### macOS
```json
{
  "command": "/usr/local/bin/python3",
  "args": ["-m", "graph_of_thoughts"]
}
```

### Linux
```json
{
  "command": "/usr/bin/python3",
  "args": ["-m", "graph_of_thoughts"]
}
```

## 🎯 Using Graph of Thoughts in Augment Code

Once configured, you can use Graph of Thoughts tools in Augment Code:

### 1. **Task Decomposition**
```
@graph-of-thoughts break down this complex coding task into manageable subtasks:
"Build a REST API with authentication, database integration, and real-time notifications"
```

### 2. **Solution Generation**
```
@graph-of-thoughts generate multiple approaches for:
"Optimizing database query performance in a high-traffic application"
```

### 3. **Code Review and Improvement**
```
@graph-of-thoughts validate and improve this solution:
[paste your code here]
```

### 4. **Complete Reasoning Workflows**
```
@graph-of-thoughts create a reasoning chain for:
"Designing a scalable microservices architecture"
```

## 🛠️ Advanced Configuration

### Enable Debug Logging
```json
{
  "command": "C:\\Program Files\\Anaconda3\\python.exe",
  "args": ["-m", "graph_of_thoughts", "--debug"],
  "env": {
    "PYTHONPATH": "C:\\path\\to\\graph-of-thoughts"
  }
}
```

### Custom Working Directory
```json
{
  "command": "C:\\Program Files\\Anaconda3\\python.exe",
  "args": ["-m", "graph_of_thoughts"],
  "cwd": "${workspaceFolder}",
  "env": {
    "GOT_LOG_LEVEL": "INFO"
  }
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Server Not Found**
   - Verify Python path is correct
   - Check that `graph_of_thoughts` package is installed
   - Run `python validate_mcp_setup.py` for diagnostics

2. **Permission Errors**
   - Ensure Augment Code has permission to execute Python
   - Try running with administrator privileges

3. **Connection Timeout**
   - Increase timeout in Augment Code settings
   - Check firewall/antivirus blocking

### Verification Steps

1. **Check Server Status**:
   ```bash
   python -m graph_of_thoughts --info
   ```

2. **Test in Augment Code**:
   - Look for "graph-of-thoughts" in available MCP tools
   - Try a simple command like `break_down_task`

3. **Check Logs**:
   - Enable debug logging in both Augment Code and the server
   - Review logs for connection errors

## 🎉 Benefits of Integration

- **Enhanced Reasoning**: Access to Graph of Thoughts methodology directly in your IDE
- **Seamless Workflow**: No context switching between tools
- **Automated Analysis**: Break down complex coding tasks systematically
- **Multiple Perspectives**: Generate diverse solution approaches
- **Iterative Improvement**: Validate and refine solutions automatically

## 📚 Next Steps

1. **Configure the server** using the appropriate method for your setup
2. **Restart Augment Code** to load the new MCP server
3. **Test the integration** with a simple Graph of Thoughts command
4. **Explore advanced features** like reasoning chains and solution validation

The Graph of Thoughts MCP server is now ready to enhance your coding workflow in Augment Code!
