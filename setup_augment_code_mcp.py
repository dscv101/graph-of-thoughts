#!/usr/bin/env python3
"""
Quick setup script for Augment Code MCP integration with Graph of Thoughts.

This script generates the correct configuration for Augment Code to use
the Graph of Thoughts MCP server.
"""

import json
import os
import platform
import sys
from pathlib import Path

def get_python_executable():
    """Get the current Python executable path."""
    return sys.executable

def generate_vscode_settings():
    """Generate VS Code settings for Augment Code."""
    python_exe = get_python_executable()
    
    config = {
        "augment.mcp.servers": {
            "graph-of-thoughts": {
                "command": python_exe,
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
    
    return config

def generate_augment_config_file():
    """Generate Augment Code configuration file."""
    python_exe = get_python_executable()
    
    config = {
        "servers": {
            "graph-of-thoughts": {
                "transport": {
                    "type": "stdio",
                    "command": python_exe,
                    "args": ["-m", "graph_of_thoughts"]
                },
                "capabilities": {
                    "tools": True,
                    "resources": True,
                    "prompts": True
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
    
    return config

def get_vscode_settings_path():
    """Get VS Code settings.json path."""
    system = platform.system()
    
    if system == "Windows":
        return Path.home() / "AppData" / "Roaming" / "Code" / "User" / "settings.json"
    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Code" / "User" / "settings.json"
    else:  # Linux
        return Path.home() / ".config" / "Code" / "User" / "settings.json"

def get_augment_config_path():
    """Get Augment Code configuration path."""
    return Path.home() / ".augment" / "mcp_config.json"

def update_vscode_settings():
    """Update VS Code settings with Augment Code MCP configuration."""
    settings_path = get_vscode_settings_path()
    augment_config = generate_vscode_settings()
    
    # Read existing settings
    existing_settings = {}
    if settings_path.exists():
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                existing_settings = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_settings = {}
    
    # Merge with Augment Code settings
    existing_settings.update(augment_config)
    
    # Ensure directory exists
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write updated settings
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(existing_settings, f, indent=2)
    
    return settings_path

def create_augment_config_file():
    """Create Augment Code configuration file."""
    config_path = get_augment_config_path()
    config = generate_augment_config_file()
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def print_manual_instructions():
    """Print manual setup instructions."""
    python_exe = get_python_executable()
    
    print("\n" + "="*60)
    print("📋 MANUAL SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n🔧 Option 1: VS Code Extension Settings")
    print("Add this to your VS Code settings.json:")
    print("-" * 40)
    vscode_config = generate_vscode_settings()
    print(json.dumps(vscode_config, indent=2))
    
    print("\n🔧 Option 2: JetBrains IDE Configuration")
    print("1. Open Settings → Tools → Augment Code → MCP Servers")
    print("2. Add New Server:")
    print(f"   - Name: graph-of-thoughts")
    print(f"   - Command: {python_exe}")
    print(f"   - Arguments: -m graph_of_thoughts")
    print(f"   - Description: Graph of Thoughts reasoning server")
    
    print("\n🔧 Option 3: Augment Code Configuration File")
    print(f"Create file: {get_augment_config_path()}")
    print("-" * 40)
    augment_config = generate_augment_config_file()
    print(json.dumps(augment_config, indent=2))

def main():
    """Main setup function."""
    print("🚀 Augment Code MCP Setup for Graph of Thoughts")
    print("="*60)

    python_exe = get_python_executable()
    print(f"🔍 Using Python: {python_exe}")

    # Validate Graph of Thoughts installation
    try:
        import graph_of_thoughts
        print("✅ Graph of Thoughts package found")
    except ImportError:
        print("❌ Graph of Thoughts package not found!")
        print(f"   Install with: {python_exe} -m pip install graph_of_thoughts")
        return False
    
    # Test server startup
    try:
        import subprocess
        result = subprocess.run(
            [python_exe, "-m", "graph_of_thoughts", "--info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✅ Graph of Thoughts MCP server is working")
        else:
            print("❌ Graph of Thoughts MCP server test failed")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False
    
    # Ask user for setup preference
    print("\n📋 Setup Options:")
    print("1. Auto-update VS Code settings")
    print("2. Create Augment Code config file")
    print("3. Show manual instructions only")
    
    try:
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == "1":
            try:
                settings_path = update_vscode_settings()
                print(f"\n✅ Updated VS Code settings: {settings_path}")
                print("   Restart VS Code to apply changes")
            except Exception as e:
                print(f"❌ Failed to update VS Code settings: {e}")
                print_manual_instructions()
        
        elif choice == "2":
            try:
                config_path = create_augment_config_file()
                print(f"\n✅ Created Augment Code config: {config_path}")
                print("   Restart Augment Code to apply changes")
            except Exception as e:
                print(f"❌ Failed to create config file: {e}")
                print_manual_instructions()
        
        else:
            print_manual_instructions()
    
    except KeyboardInterrupt:
        print("\n\n📋 Manual setup instructions:")
        print_manual_instructions()
    
    print("\n" + "="*60)
    print("🎉 Setup complete!")
    print("\n📝 Next steps:")
    print("1. Restart Augment Code")
    print("2. Look for 'graph-of-thoughts' in available MCP tools")
    print("3. Try: @graph-of-thoughts break down this task: 'Build a web API'")
    print("="*60)
    
    return True

if __name__ == "__main__":
    main()
