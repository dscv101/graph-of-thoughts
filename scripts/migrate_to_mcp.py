#!/usr/bin/env python3
"""
Migration script to help users transition from legacy language models to MCP.

This script:
1. Analyzes existing code for legacy language model usage
2. Generates MCP configuration files
3. Provides migration recommendations
4. Tests MCP connectivity

Usage:
    python migrate_to_mcp.py --analyze /path/to/project
    python migrate_to_mcp.py --generate-config --host claude-desktop
    python migrate_to_mcp.py --test-connection mcp_config.json
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import asyncio


class LegacyCodeAnalyzer:
    """Analyzes Python code for legacy language model usage."""
    
    LEGACY_IMPORTS = {
        'ChatGPT': 'graph_of_thoughts.language_models.ChatGPT',
        'Claude': 'graph_of_thoughts.language_models.Claude',
        'HuggingFace': 'graph_of_thoughts.language_models.HuggingFace',
        'OpenAI': 'graph_of_thoughts.language_models.OpenAI'
    }
    
    def __init__(self):
        self.findings = []
    
    def analyze_file(self, file_path: Path) -> List[Dict]:
        """Analyze a single Python file for legacy usage."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Find legacy imports
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and 'graph_of_thoughts.language_models' in node.module:
                        for alias in node.names:
                            if alias.name in self.LEGACY_IMPORTS:
                                findings.append({
                                    'type': 'legacy_import',
                                    'file': str(file_path),
                                    'line': node.lineno,
                                    'class': alias.name,
                                    'suggestion': 'Replace with MCPLanguageModel'
                                })
                
                # Find legacy instantiations
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.LEGACY_IMPORTS:
                            findings.append({
                                'type': 'legacy_instantiation',
                                'file': str(file_path),
                                'line': node.lineno,
                                'class': node.func.id,
                                'suggestion': 'Replace with MCPLanguageModel(config_path, model_name)'
                            })
            
            # Find API key usage
            api_key_patterns = [
                r'api_key\s*=\s*["\']sk-[^"\']+["\']',
                r'OPENAI_API_KEY',
                r'ANTHROPIC_API_KEY',
                r'HUGGINGFACE_API_KEY'
            ]
            
            for pattern in api_key_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    findings.append({
                        'type': 'api_key_usage',
                        'file': str(file_path),
                        'line': line_num,
                        'pattern': match.group(),
                        'suggestion': 'Remove API key - MCP handles authentication'
                    })
        
        except Exception as e:
            findings.append({
                'type': 'analysis_error',
                'file': str(file_path),
                'error': str(e)
            })
        
        return findings
    
    def analyze_directory(self, directory: Path) -> List[Dict]:
        """Analyze all Python files in a directory."""
        all_findings = []
        
        for py_file in directory.rglob('*.py'):
            if py_file.name.startswith('.'):
                continue
            
            findings = self.analyze_file(py_file)
            all_findings.extend(findings)
        
        return all_findings


class MCPConfigGenerator:
    """Generates MCP configuration files."""
    
    TEMPLATES = {
        'claude-desktop': {
            'transport': {
                'type': 'stdio',
                'command': 'claude-desktop',
                'args': ['--mcp'],
                'timeout': 30.0
            },
            'client_info': {
                'name': 'graph-of-thoughts',
                'version': '1.0.0'
            },
            'capabilities': {
                'sampling': {}
            },
            'default_sampling_params': {
                'maxTokens': 4096,
                'temperature': 0.7
            },
            'retry_config': {
                'max_attempts': 3,
                'base_delay': 1.0,
                'strategy': 'exponential',
                'jitter_type': 'equal'
            }
        },
        'vscode': {
            'transport': {
                'type': 'stdio',
                'command': 'code',
                'args': ['--mcp-server'],
                'timeout': 30.0
            },
            'client_info': {
                'name': 'graph-of-thoughts',
                'version': '1.0.0'
            },
            'capabilities': {
                'sampling': {}
            }
        },
        'remote-http': {
            'transport': {
                'type': 'http',
                'url': 'http://localhost:8000/mcp',
                'timeout': 30.0,
                'headers': {
                    'Content-Type': 'application/json'
                }
            },
            'client_info': {
                'name': 'graph-of-thoughts',
                'version': '1.0.0'
            },
            'capabilities': {
                'sampling': {}
            }
        }
    }
    
    def generate_config(self, host_type: str, output_path: str = 'mcp_config.json') -> bool:
        """Generate MCP configuration file."""
        if host_type not in self.TEMPLATES:
            print(f"❌ Unknown host type: {host_type}")
            print(f"Available types: {', '.join(self.TEMPLATES.keys())}")
            return False
        
        config = {
            f'mcp_{host_type.replace("-", "_")}': self.TEMPLATES[host_type]
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Generated MCP configuration: {output_path}")
            print(f"   Host type: {host_type}")
            print(f"   Model name: mcp_{host_type.replace('-', '_')}")
            return True
        
        except Exception as e:
            print(f"❌ Failed to generate config: {e}")
            return False


class MCPConnectionTester:
    """Tests MCP connectivity."""
    
    async def test_connection(self, config_path: str) -> bool:
        """Test MCP connection with given configuration."""
        try:
            # Import here to avoid circular imports
            from graph_of_thoughts.language_models import MCPLanguageModel
            
            # Load config to get model name
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_names = list(config.keys())
            if not model_names:
                print("❌ No models found in configuration")
                return False
            
            model_name = model_names[0]
            print(f"🔍 Testing connection to {model_name}...")
            
            # Test connection
            lm = MCPLanguageModel(config_path, model_name)
            
            async with lm:
                response = await lm.query_async("Hello, world!")
                response_text = lm.get_response_texts(response)[0]
                
                print("✅ Connection successful!")
                print(f"   Response: {response_text[:100]}...")
                return True
        
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("   Make sure graph-of-thoughts is installed with MCP support")
            return False
        
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {config_path}")
            return False
        
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("   Check your MCP host configuration and ensure it's running")
            return False


def print_analysis_report(findings: List[Dict]):
    """Print analysis report."""
    if not findings:
        print("✅ No legacy language model usage found!")
        return
    
    print(f"📊 Found {len(findings)} items requiring migration:")
    print()
    
    # Group by file
    by_file = {}
    for finding in findings:
        file_path = finding.get('file', 'unknown')
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(finding)
    
    for file_path, file_findings in by_file.items():
        print(f"📄 {file_path}")
        
        for finding in file_findings:
            if finding['type'] == 'legacy_import':
                print(f"   Line {finding['line']}: Legacy import '{finding['class']}'")
                print(f"   💡 {finding['suggestion']}")
            
            elif finding['type'] == 'legacy_instantiation':
                print(f"   Line {finding['line']}: Legacy instantiation '{finding['class']}'")
                print(f"   💡 {finding['suggestion']}")
            
            elif finding['type'] == 'api_key_usage':
                print(f"   Line {finding['line']}: API key usage")
                print(f"   💡 {finding['suggestion']}")
            
            elif finding['type'] == 'analysis_error':
                print(f"   ❌ Analysis error: {finding['error']}")
        
        print()


def print_migration_recommendations(findings: List[Dict]):
    """Print migration recommendations."""
    if not findings:
        return
    
    print("🚀 Migration Recommendations:")
    print()
    
    # Count different types
    legacy_classes = set()
    has_api_keys = False
    
    for finding in findings:
        if finding['type'] in ['legacy_import', 'legacy_instantiation']:
            legacy_classes.add(finding['class'])
        elif finding['type'] == 'api_key_usage':
            has_api_keys = True
    
    print("1. Update imports:")
    print("   Replace:")
    for cls in legacy_classes:
        print(f"     from graph_of_thoughts.language_models import {cls}")
    print("   With:")
    print("     from graph_of_thoughts.language_models import MCPLanguageModel")
    print()
    
    print("2. Update instantiation:")
    for cls in legacy_classes:
        print(f"   Replace: {cls}(api_key='...', model_name='...')")
    print("   With: MCPLanguageModel('mcp_config.json', 'mcp_model_name')")
    print()
    
    if has_api_keys:
        print("3. Remove API keys:")
        print("   - Delete API key variables and environment variables")
        print("   - MCP handles authentication through the host")
        print()
    
    print("4. Generate MCP configuration:")
    print("   python migrate_to_mcp.py --generate-config --host claude-desktop")
    print()
    
    print("5. Test connection:")
    print("   python migrate_to_mcp.py --test-connection mcp_config.json")


def main():
    parser = argparse.ArgumentParser(description='Migrate from legacy language models to MCP')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--analyze', metavar='PATH', help='Analyze directory for legacy usage')
    group.add_argument('--generate-config', action='store_true', help='Generate MCP configuration')
    group.add_argument('--test-connection', metavar='CONFIG', help='Test MCP connection')
    
    parser.add_argument('--host', choices=['claude-desktop', 'vscode', 'remote-http'],
                       help='MCP host type for config generation')
    parser.add_argument('--output', default='mcp_config.json',
                       help='Output path for generated config')
    
    args = parser.parse_args()
    
    if args.analyze:
        print("🔍 Analyzing code for legacy language model usage...")
        print()
        
        analyzer = LegacyCodeAnalyzer()
        findings = analyzer.analyze_directory(Path(args.analyze))
        
        print_analysis_report(findings)
        print_migration_recommendations(findings)
    
    elif args.generate_config:
        if not args.host:
            print("❌ --host is required for config generation")
            print("Available hosts: claude-desktop, vscode, remote-http")
            sys.exit(1)
        
        print(f"🔧 Generating MCP configuration for {args.host}...")
        print()
        
        generator = MCPConfigGenerator()
        success = generator.generate_config(args.host, args.output)
        
        if success:
            print()
            print("Next steps:")
            print(f"1. Review and customize {args.output}")
            print("2. Test connection: python migrate_to_mcp.py --test-connection " + args.output)
            print("3. Update your code to use MCPLanguageModel")
    
    elif args.test_connection:
        print(f"🔌 Testing MCP connection with {args.test_connection}...")
        print()
        
        tester = MCPConnectionTester()
        success = asyncio.run(tester.test_connection(args.test_connection))
        
        if success:
            print()
            print("🎉 Your MCP setup is working correctly!")
            print("You can now use MCPLanguageModel in your code.")
        else:
            print()
            print("❌ Connection test failed. Check the troubleshooting guide:")
            print("   docs/TROUBLESHOOTING_GUIDE.md")


if __name__ == '__main__':
    main()
