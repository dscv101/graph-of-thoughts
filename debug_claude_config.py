#!/usr/bin/env python3
"""
Debug Claude Desktop configuration for Graph of Thoughts MCP server.
"""

import json
import os
import platform
from pathlib import Path

def find_claude_config():
    """Find Claude Desktop configuration file."""
    system = platform.system()
    
    if system == "Windows":
        # Windows paths
        paths = [
            Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",
            Path.home() / ".claude" / "claude_desktop_config.json",
            Path("claude_desktop_config.json"),  # Current directory
        ]
    elif system == "Darwin":  # macOS
        paths = [
            Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
            Path.home() / ".claude" / "claude_desktop_config.json",
        ]
    else:  # Linux
        paths = [
            Path.home() / ".config" / "claude" / "claude_desktop_config.json",
            Path.home() / ".claude" / "claude_desktop_config.json",
        ]
    
    for path in paths:
        if path.exists():
            print(f"✅ Found config file: {path}")
            return path
    
    print("❌ No Claude config file found in standard locations:")
    for path in paths:
        print(f"   - {path}")
    return None

def check_config_content(config_path):
    """Check the content of the configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\n📄 Current config content:")
        print("-" * 50)
        print(content)
        print("-" * 50)
        
        # Try to parse as JSON
        try:
            config = json.loads(content)
            print("\n✅ JSON is valid")
            
            # Check for graph-of-thoughts server
            if "mcpServers" in config:
                if "graph-of-thoughts" in config["mcpServers"]:
                    server_config = config["mcpServers"]["graph-of-thoughts"]
                    print(f"\n📋 Graph of Thoughts server config:")
                    print(f"   Command: {server_config.get('command', 'NOT SET')}")
                    print(f"   Args: {server_config.get('args', 'NOT SET')}")
                    
                    # Check if command and args are properly separated
                    command = server_config.get('command', '')
                    if '-m graph_of_thoughts' in command:
                        print("\n❌ PROBLEM FOUND: '-m graph_of_thoughts' is in the command field!")
                        print("   This should be moved to the args field.")
                        return False
                    else:
                        print("\n✅ Command and args appear to be properly separated")
                        return True
                else:
                    print("\n❌ 'graph-of-thoughts' server not found in mcpServers")
            else:
                print("\n❌ 'mcpServers' section not found in config")
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON parsing error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        return False

def generate_correct_config():
    """Generate the correct configuration."""
    import sys
    
    correct_config = {
        "mcpServers": {
            "graph-of-thoughts": {
                "command": sys.executable,
                "args": ["-m", "graph_of_thoughts"]
            }
        }
    }
    
    print(f"\n🔧 CORRECT configuration:")
    print(json.dumps(correct_config, indent=2))
    
    return correct_config

def main():
    """Main debugging function."""
    print("🔍 Claude Desktop Configuration Debugger")
    print("=" * 50)
    
    # Find config file
    config_path = find_claude_config()
    if not config_path:
        print("\n💡 You may need to create the config file manually.")
        print("   Typical location on Windows:")
        print(f"   {Path.home() / 'AppData' / 'Roaming' / 'Claude' / 'claude_desktop_config.json'}")
        generate_correct_config()
        return
    
    # Check current config
    is_correct = check_config_content(config_path)
    
    if not is_correct:
        print(f"\n🔧 To fix this, replace the content of:")
        print(f"   {config_path}")
        print(f"\nWith this EXACT configuration:")
        generate_correct_config()
        
        print(f"\n📝 Steps to fix:")
        print(f"1. Close Claude Desktop completely")
        print(f"2. Edit the file: {config_path}")
        print(f"3. Replace ALL content with the correct config above")
        print(f"4. Save the file")
        print(f"5. Restart Claude Desktop")
    else:
        print(f"\n✅ Configuration looks correct!")
        print(f"   If you're still getting errors, try:")
        print(f"   1. Restart Claude Desktop completely")
        print(f"   2. Check Claude Desktop logs for more details")

if __name__ == "__main__":
    main()
