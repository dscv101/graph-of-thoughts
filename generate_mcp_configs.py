#!/usr/bin/env python3
"""
Generate MCP client configurations with full Python paths.

This script generates correct MCP configurations for all supported clients,
ensuring that the full Python executable path is used to avoid import issues.
"""

import json
import os
import platform
import sys
from pathlib import Path

def get_python_executable():
    """Get the current Python executable path."""
    return sys.executable

def get_safe_python_path():
    """Get Python path with proper quoting for command line usage."""
    python_path = get_python_executable()
    if " " in python_path and not python_path.startswith('"'):
        return f'"{python_path}"'
    return python_path

def generate_claude_desktop_config():
    """Generate Claude Desktop configuration."""
    python_exe = get_python_executable()
    
    return {
        "mcpServers": {
            "graph-of-thoughts": {
                "command": python_exe,
                "args": ["-m", "graph_of_thoughts"],
                "env": {},
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

def generate_vscode_config():
    """Generate VS Code configuration."""
    python_exe = get_python_executable()
    
    return {
        "mcp.servers": {
            "graph-of-thoughts": {
                "command": python_exe,
                "args": ["-m", "graph_of_thoughts"],
                "cwd": "${workspaceFolder}",
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

def generate_cursor_config():
    """Generate Cursor configuration."""
    python_exe = get_safe_python_path()
    
    return {
        "mcp": {
            "servers": {
                "graph-of-thoughts": {
                    "command": f"{python_exe} -m graph_of_thoughts",
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
    }

def generate_augment_code_vscode_config():
    """Generate Augment Code VS Code configuration."""
    python_exe = get_python_executable()
    
    return {
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

def generate_augment_code_config_file():
    """Generate Augment Code configuration file."""
    python_exe = get_python_executable()
    
    return {
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

def save_config_file(config, filename, description):
    """Save configuration to a file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Generated {description}: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to generate {description}: {e}")
        return False

def print_config(config, title):
    """Print configuration to console."""
    print(f"\n📋 {title}")
    print("-" * 50)
    print(json.dumps(config, indent=2))

def main():
    """Main function to generate all configurations."""
    print("🚀 MCP Configuration Generator for Graph of Thoughts")
    print("=" * 60)
    
    python_exe = get_python_executable()
    print(f"🔍 Using Python: {python_exe}")
    
    # Test Graph of Thoughts installation
    try:
        import graph_of_thoughts
        print("✅ Graph of Thoughts package found")
    except ImportError:
        print("❌ Graph of Thoughts package not found!")
        print(f"   Install with: {python_exe} -m pip install graph_of_thoughts")
        return False
    
    # Test server functionality
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
            return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False
    
    print("\n🔧 Generating configurations...")
    
    # Generate all configurations
    configs = {
        "Claude Desktop": generate_claude_desktop_config(),
        "VS Code": generate_vscode_config(),
        "Cursor": generate_cursor_config(),
        "Augment Code (VS Code)": generate_augment_code_vscode_config(),
        "Augment Code (Config File)": generate_augment_code_config_file(),
    }
    
    # Print all configurations
    for title, config in configs.items():
        print_config(config, title)
    
    # Ask if user wants to save files
    print(f"\n💾 Save configuration files?")
    save_files = input("Save files? (y/N): ").strip().lower()
    
    if save_files in ['y', 'yes']:
        success_count = 0
        
        # Save individual config files
        if save_config_file(configs["Claude Desktop"], "claude_desktop_config.json", "Claude Desktop config"):
            success_count += 1
        
        if save_config_file(configs["VS Code"], "vscode_mcp_config.json", "VS Code config"):
            success_count += 1
        
        if save_config_file(configs["Cursor"], "cursor_mcp_config.json", "Cursor config"):
            success_count += 1
        
        if save_config_file(configs["Augment Code (Config File)"], "augment_mcp_config.json", "Augment Code config"):
            success_count += 1
        
        print(f"\n✅ Generated {success_count} configuration files")
        print("\n📝 Next steps:")
        print("1. Copy the appropriate config to your MCP client")
        print("2. Restart your MCP client")
        print("3. Look for 'graph-of-thoughts' in available tools")
    
    print("\n" + "=" * 60)
    print("🎉 Configuration generation complete!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main()
