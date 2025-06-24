# 🚨 QUICK FIX: "Program is not recognized" Error

## Problem
You're getting this error:
```
'C:\Program' is not recognized as an internal or external command,
operable program or batch file.
```

## Root Cause
**Spaces in the Python path** are causing command parsing issues. The path `C:\Program Files\Anaconda3\python.exe` is being split at the space.

## ✅ IMMEDIATE SOLUTION

### For Claude Desktop
Replace your current configuration with this **EXACT** configuration:

```json
{
  "mcpServers": {
    "graph-of-thoughts": {
      "command": "C:\\Program Files\\Anaconda3\\python.exe",
      "args": ["-m", "graph_of_thoughts"]
    }
  }
}
```

**Key Points:**
- ✅ Use `"command"` and `"args"` **separately**
- ✅ Do **NOT** put the module name in the command string
- ✅ Use double backslashes `\\` in the path

### For VSCode
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

### For Cursor
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

## ❌ What NOT to Do

```json
// ❌ WRONG - This will fail:
{
  "command": "C:\\Program Files\\Anaconda3\\python.exe -m graph_of_thoughts"
}

// ❌ WRONG - Missing args separation:
{
  "command": "C:\\Program Files\\Anaconda3\\python.exe -m graph_of_thoughts",
  "args": []
}
```

## 🔍 How to Find Your Python Path

Run this command to find your exact Python path:
```bash
python -c "import sys; print(sys.executable)"
```

Then use that **exact path** in your configuration.

## 🚀 Alternative Solutions

### Option 1: Use Short Path (Windows)
```json
{
  "command": "C:\\PROGRA~1\\Anaconda3\\python.exe",
  "args": ["-m", "graph_of_thoughts"]
}
```

### Option 2: Use python from PATH (if available)
```json
{
  "command": "python",
  "args": ["-m", "graph_of_thoughts"]
}
```

### Option 3: Install Python without spaces
Install Python in a path without spaces like:
- `C:\Python312\python.exe`
- `C:\Anaconda3\python.exe`

## ✅ Verification Steps

1. **Apply the correct configuration**
2. **Restart your MCP client** (Claude Desktop, VSCode, etc.)
3. **Test the connection** by trying to use a Graph of Thoughts tool
4. **Check for "graph-of-thoughts" in your available tools**

## 🎉 Success Indicators

You'll know it's working when:
- ✅ No more "Program is not recognized" errors
- ✅ "graph-of-thoughts" appears in your MCP tools list
- ✅ You can use tools like `break_down_task`

## 📞 Still Having Issues?

If this doesn't work:
1. Run our validator: `python validate_mcp_setup.py`
2. Use the **exact configuration** it generates
3. Check the troubleshooting guide: `MCP_CLIENT_TROUBLESHOOTING.md`

**The MCP server is working perfectly - this is purely a configuration issue!**
