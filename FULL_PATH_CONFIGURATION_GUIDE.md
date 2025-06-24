# 🔧 Full Path Configuration Guide for Graph of Thoughts MCP Server

## 🚨 **Critical Issue Resolved**

**Problem**: Using relative `"python"` commands in MCP configurations causes import failures when Graph of Thoughts isn't in the system PATH or when using different Python environments.

**Solution**: Always use the **full Python executable path** in all MCP client configurations.

## 🔍 **Step 1: Find Your Python Path**

Run this command to find your exact Python executable path:

```bash
python -c "import sys; print(sys.executable)"
```

**Example outputs:**
- Windows: `C:\Program Files\Anaconda3\python.exe`
- macOS: `/usr/local/bin/python3`
- Linux: `/usr/bin/python3`

## ✅ **Step 2: Use the Automated Configuration Generator**

We've created a tool that automatically generates correct configurations with full paths:

```bash
python generate_mcp_configs.py
```

This will:
- ✅ Detect your Python path automatically
- ✅ Test Graph of Thoughts installation
- ✅ Generate configurations for all MCP clients
- ✅ Use proper path quoting for spaces

## 📋 **Correct Configurations (Examples)**

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "graph-of-thoughts": {
      "command": "C:\\Program Files\\Anaconda3\\python.exe",
      "args": ["-m", "graph_of_thoughts"],
      "env": {}
    }
  }
}
```

### VS Code (`settings.json`)

```json
{
  "mcp.servers": {
    "graph-of-thoughts": {
      "command": "C:\\Program Files\\Anaconda3\\python.exe",
      "args": ["-m", "graph_of_thoughts"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

### Augment Code (`settings.json`)

```json
{
  "augment.mcp.servers": {
    "graph-of-thoughts": {
      "command": "C:\\Program Files\\Anaconda3\\python.exe",
      "args": ["-m", "graph_of_thoughts"],
      "description": "Graph of Thoughts reasoning server"
    }
  }
}
```

### Cursor

```json
{
  "mcp": {
    "servers": {
      "graph-of-thoughts": {
        "command": "\"C:\\Program Files\\Anaconda3\\python.exe\" -m graph_of_thoughts"
      }
    }
  }
}
```

## ❌ **What NOT to Do**

```json
// ❌ WRONG - Will fail if Graph of Thoughts not in PATH
{
  "command": "python -m graph_of_thoughts"
}

// ❌ WRONG - Relative python command
{
  "command": "python",
  "args": ["-m", "graph_of_thoughts"]
}

// ❌ WRONG - Unquoted path with spaces
{
  "command": "C:\\Program Files\\Anaconda3\\python.exe -m graph_of_thoughts"
}
```

## 🛠️ **Available Tools**

### 1. **Configuration Generator**
```bash
python generate_mcp_configs.py
```
- Generates all client configurations
- Uses correct Python paths automatically
- Tests server functionality

### 2. **Setup Validator**
```bash
python validate_mcp_setup.py
```
- Validates your setup
- Tests all components
- Provides troubleshooting info

### 3. **Augment Code Setup**
```bash
python setup_augment_code_mcp.py
```
- Specific setup for Augment Code
- Multiple configuration options
- Automated VS Code settings update

## 🔧 **Troubleshooting**

### Issue: "Graph of Thoughts not found"
**Solution**: Use the full Python path where Graph of Thoughts is installed.

### Issue: "Command not recognized"
**Solution**: Quote paths with spaces properly.

### Issue: "Module not found"
**Solution**: Ensure Graph of Thoughts is installed in the Python environment you're using.

### Issue: "Permission denied"
**Solution**: Check file permissions and run with appropriate privileges.

## ✅ **Verification Steps**

1. **Test your Python path**:
   ```bash
   "C:\Program Files\Anaconda3\python.exe" -m graph_of_thoughts --info
   ```

2. **Verify Graph of Thoughts installation**:
   ```bash
   "C:\Program Files\Anaconda3\python.exe" -c "import graph_of_thoughts; print('✅ Installed')"
   ```

3. **Test MCP server startup**:
   ```bash
   "C:\Program Files\Anaconda3\python.exe" -m graph_of_thoughts --debug
   ```

## 🎯 **Key Takeaways**

- ✅ **Always use full Python executable paths**
- ✅ **Use the configuration generator for accuracy**
- ✅ **Test configurations before deploying**
- ✅ **Quote paths with spaces properly**
- ✅ **Verify Graph of Thoughts installation in the correct Python environment**

## 🚀 **Next Steps**

1. **Run the configuration generator**: `python generate_mcp_configs.py`
2. **Copy the appropriate configuration** to your MCP client
3. **Restart your MCP client**
4. **Test the integration** with a Graph of Thoughts tool
5. **Enjoy enhanced reasoning capabilities!**

---

**The Graph of Thoughts MCP server is fully ready for deployment with proper full-path configurations!** 🎉
