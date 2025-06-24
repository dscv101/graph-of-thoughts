# MCP Server "Client is closing" Error - Troubleshooting Guide

## 🔍 Problem Analysis

The error `"Client is closing"` indicates that the **MCP client** (Claude Desktop, VSCode, Cursor, etc.) is terminating the connection to the MCP server, not that the server itself is failing.

## ✅ Server Status: HEALTHY

Our diagnostic tests confirm:

- ✅ Server creation works perfectly
- ✅ All MCP handlers are registered correctly
- ✅ Stdio transport imports successfully
- ✅ No import or dependency issues

## 🛠️ Common Causes & Solutions

### 1. **Spaces in Python Path (MOST COMMON)**

**Problem**: Python installed in "Program Files" causes command parsing errors.

**Error**: `'C:\Program' is not recognized as an internal or external command`

**Solutions**:

```json
// ❌ WRONG - This will fail with spaces in path:
{
  "command": "C:\\Program Files\\Anaconda3\\python.exe -m graph_of_thoughts"
}

// ✅ CORRECT - Use separate command and args:
{
  "command": "C:\\Program Files\\Anaconda3\\python.exe",
  "args": ["-m", "graph_of_thoughts"]
}

// ✅ ALTERNATIVE - Use short path (Windows):
{
  "command": "C:\\PROGRA~1\\Anaconda3\\python.exe",
  "args": ["-m", "graph_of_thoughts"]
}
```

### 2. **Incorrect Command Path**

**Problem**: The MCP client can't find or execute the Python command.

**Solutions**:

```json
// Use full path with proper args separation:
{
  "command": "C:\\Python312\\python.exe",
  "args": ["-m", "graph_of_thoughts"]
}

// Or use python3 if available:
{
  "command": "python3",
  "args": ["-m", "graph_of_thoughts"]
}
```

### 3. **Python Environment Issues**

**Problem**: Wrong Python environment or missing dependencies.

**Solutions**:

```bash
# Check Python version (3.12+ required)
python --version

# Verify package installation
python -c "import graph_of_thoughts; print('✅ Package installed')"

# Check MCP dependency
python -c "import mcp; print('✅ MCP SDK available')"

# Install if missing
pip install graph_of_thoughts
```

### 4. **Working Directory Issues**

**Problem**: MCP client starts server in wrong directory.

**Solutions**:

```json
// Add explicit working directory
{
  "command": "python",
  "args": ["-m", "graph_of_thoughts"],
  "cwd": "C:\\path\\to\\your\\project"
}
```

### 5. **Permission Issues**

**Problem**: Insufficient permissions to execute Python or access files.

**Solutions**:

- Run MCP client as administrator (Windows)
- Check file permissions on Python executable
- Verify antivirus isn't blocking execution

### 6. **Environment Variables**

**Problem**: Missing PYTHONPATH or other environment variables.

**Solutions**:

```json
{
  "command": "python",
  "args": ["-m", "graph_of_thoughts"],
  "env": {
    "PYTHONPATH": "C:\\path\\to\\graph-of-thoughts",
    "PATH": "C:\\Python312;C:\\Python312\\Scripts"
  }
}
```

## 🔧 Debugging Steps

### Step 1: Test Server Manually

```bash
# Test server startup
python -m graph_of_thoughts --info

# Test with debug logging
python -m graph_of_thoughts --debug
```

### Step 2: Check MCP Client Logs

- **Claude Desktop**: Check `%APPDATA%\Claude\logs`
- **VSCode**: Check Output panel → MCP
- **Cursor**: Check developer console

### Step 3: Test with Minimal Config

```json
{
  "mcpServers": {
    "graph-of-thoughts": {
      "command": "python",
      "args": ["-m", "graph_of_thoughts", "--debug"]
    }
  }
}
```

## 📋 Platform-Specific Configurations

### Windows (Claude Desktop)

```json
{
  "mcpServers": {
    "graph-of-thoughts": {
      "command": "python.exe",
      "args": ["-m", "graph_of_thoughts"],
      "env": {
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

### macOS/Linux (Claude Desktop)

```json
{
  "mcpServers": {
    "graph-of-thoughts": {
      "command": "/usr/bin/python3",
      "args": ["-m", "graph_of_thoughts"]
    }
  }
}
```

### VSCode

```json
{
  "mcp.servers": {
    "graph-of-thoughts": {
      "command": "python",
      "args": ["-m", "graph_of_thoughts"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

## 🚀 Quick Fix Commands

### For Windows

```bash
# Find Python path
where python

# Test with full path
"C:\Users\YourUser\AppData\Local\Programs\Python\Python312\python.exe" -m graph_of_thoughts --info
```

### For macOS/Linux

```bash
# Find Python path
which python3

# Test with full path
/usr/bin/python3 -m graph_of_thoughts --info
```

## ✅ Verification

After applying fixes, verify the server works:

1. **Test server startup**:

   ```bash
   python -m graph_of_thoughts --info
   ```

2. **Check MCP client connection**:
   - Restart your MCP client
   - Look for "graph-of-thoughts" in available tools
   - Try using a tool like `break_down_task`

## 📞 Still Having Issues?

If the problem persists:

1. **Collect debug information**:

   ```bash
   python -m graph_of_thoughts --debug > server_debug.log 2>&1
   ```

2. **Check system requirements**:
   - Python 3.12+
   - All dependencies installed
   - Sufficient disk space and memory

3. **Try alternative approaches**:
   - Use virtual environment
   - Install in different location
   - Try different Python version

The MCP server itself is **ready for deployment** - the issue is with client configuration, not server functionality.
