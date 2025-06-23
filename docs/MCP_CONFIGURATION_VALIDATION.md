# MCP Configuration Validation System

The MCP Configuration Validation System provides comprehensive validation for Model Context Protocol (MCP) configurations, ensuring protocol compliance, security, and proper setup before runtime.

## Features

### 🔍 Comprehensive Validation
- **Startup Configuration Validation**: Validates configuration files at application startup
- **Runtime Configuration Validation**: Validates configurations during runtime with additional checks
- **Protocol Compliance**: Ensures configurations follow MCP specification requirements
- **Security Validation**: Identifies potential security issues and dangerous configurations
- **Detailed Error Reporting**: Provides clear error messages with field paths and context

### 🛡️ Security Checks
- **Command Safety**: Detects potentially dangerous commands in stdio transport
- **Environment Variable Security**: Identifies sensitive data in environment variables
- **HTTP Security**: Warns about insecure HTTP connections and authentication issues
- **Path Validation**: Validates file paths and permissions

### 📋 Configuration Templates
- **Template Generation**: Creates valid configuration templates for different transport types
- **Customizable Templates**: Supports custom model names and transport configurations
- **Best Practices**: Templates include recommended settings and security practices

## Usage

### Python API

#### Basic Validation

```python
from graph_of_thoughts.language_models.mcp_protocol import (
    MCPConfigurationValidator,
    MCPConfigurationError
)

# Create validator
validator = MCPConfigurationValidator(
    strict_mode=True,
    enable_security_checks=True
)

# Validate configuration file at startup
try:
    is_valid = validator.validate_startup_configuration("config.json")
    print(f"Configuration is valid: {is_valid}")
    
    # Check for warnings
    if validator.validation_warnings:
        for warning in validator.validation_warnings:
            print(f"Warning: {warning}")
            
except MCPConfigurationError as e:
    print(f"Configuration error: {e}")
    print(f"Field path: {e.field_path}")
```

#### Runtime Validation

```python
# Validate configuration at runtime
config = {
    "transport": {"type": "stdio", "command": "claude-desktop"},
    "client_info": {"name": "my-app", "version": "1.0.0"},
    "capabilities": {"sampling": {}}
}

try:
    validator.validate_runtime_configuration(config, "my_model")
    print("Runtime validation successful")
except MCPConfigurationError as e:
    print(f"Runtime validation failed: {e}")
```

#### Generate Configuration Templates

```python
# Generate stdio template
stdio_template = validator.generate_configuration_template(
    transport_type="stdio",
    model_name="mcp_claude_desktop"
)
print(stdio_template)

# Generate HTTP template
http_template = validator.generate_configuration_template(
    transport_type="http",
    model_name="mcp_remote_server"
)
print(http_template)
```

#### Configuration Summary

```python
# Get configuration summary
summary = validator.get_configuration_summary("config.json")
print(f"Models: {summary['model_count']}")
for model_name, info in summary['models'].items():
    print(f"  {model_name}: {info['transport_type']}")
```

### Command Line Interface

#### Validate Configuration

```bash
# Basic validation
python -m graph_of_thoughts.language_models.mcp_protocol validate config.json

# Strict validation with verbose output
python -m graph_of_thoughts.language_models.mcp_protocol validate config.json --strict --verbose

# Validation without security checks
python -m graph_of_thoughts.language_models.mcp_protocol validate config.json --no-security
```

#### Generate Templates

```bash
# Generate stdio template
python -m graph_of_thoughts.language_models.mcp_protocol generate-template stdio

# Generate HTTP template with custom name
python -m graph_of_thoughts.language_models.mcp_protocol generate-template http --model-name my_server

# Save template to file
python -m graph_of_thoughts.language_models.mcp_protocol generate-template stdio -o config.json
```

#### Configuration Summary

```bash
# Get human-readable summary
python -m graph_of_thoughts.language_models.mcp_protocol summary config.json

# Get JSON summary
python -m graph_of_thoughts.language_models.mcp_protocol summary config.json --json
```

### Standalone Function

```python
from graph_of_thoughts.language_models.mcp_protocol import validate_mcp_config_file

# Simple validation
is_valid = validate_mcp_config_file("config.json")

# Validation with options
is_valid = validate_mcp_config_file(
    "config.json",
    strict_mode=True,
    enable_security_checks=True
)
```

## Configuration Structure

### Required Sections

Every MCP configuration must include:

```json
{
  "model_name": {
    "transport": {
      "type": "stdio|http",
      // Transport-specific fields
    },
    "client_info": {
      "name": "application-name",
      "version": "1.0.0"
    }
  }
}
```

### Transport Types

#### Stdio Transport

```json
{
  "transport": {
    "type": "stdio",
    "command": "executable-name",
    "args": ["--arg1", "--arg2"],
    "env": {
      "ENV_VAR": "value"
    }
  }
}
```

#### HTTP Transport

```json
{
  "transport": {
    "type": "http",
    "url": "https://api.example.com/mcp",
    "headers": {
      "Content-Type": "application/json",
      "Authorization": "Bearer token"
    },
    "session_management": true
  }
}
```

### Optional Sections

#### Capabilities

```json
{
  "capabilities": {
    "sampling": {},
    "roots": {"listChanged": true},
    "experimental": {}
  }
}
```

#### Default Sampling Parameters

```json
{
  "default_sampling_params": {
    "temperature": 1.0,
    "maxTokens": 4096,
    "includeContext": "thisServer",
    "modelPreferences": {
      "hints": [{"name": "claude-3-5-sonnet"}],
      "costPriority": 0.3,
      "speedPriority": 0.4,
      "intelligencePriority": 0.8
    },
    "stopSequences": ["END"]
  }
}
```

#### Connection Configuration

```json
{
  "connection_config": {
    "timeout": 30.0,
    "request_timeout": 60.0,
    "retry_attempts": 3,
    "retry_delay": 1.0
  }
}
```

#### Cost Tracking

```json
{
  "cost_tracking": {
    "prompt_token_cost": 0.003,
    "response_token_cost": 0.015
  }
}
```

#### Metrics

```json
{
  "metrics": {
    "enabled": true,
    "export_interval": 60.0,
    "export_format": "json",
    "max_history_size": 1000,
    "export_file": "metrics.json"
  }
}
```

## Validation Rules

### Transport Validation
- **Type**: Must be "stdio" or "http"
- **Stdio**: Requires "command" field, validates args and env
- **HTTP**: Requires "url" field, validates URL format and headers

### Client Info Validation
- **Name**: Must be non-empty string
- **Version**: Must be non-empty string (warns if not semantic versioning)
- **Title**: Optional string field

### Sampling Parameters Validation
- **Temperature**: Number between 0 and 2
- **MaxTokens**: Positive integer
- **IncludeContext**: Must be "none", "thisServer", or "allServers"
- **Model Preferences**: Validates hints and priority values (0-1)
- **Stop Sequences**: Array of strings

### Connection Config Validation
- **Timeout Values**: Must be positive numbers
- **Retry Attempts**: Non-negative integer (warns if > 10)

### Security Validation
- **Dangerous Commands**: Detects potentially harmful commands
- **Sensitive Environment Variables**: Identifies variables that may contain secrets
- **HTTP Security**: Warns about insecure connections and weak authentication

## Error Handling

### MCPConfigurationError

The validation system uses `MCPConfigurationError` for detailed error reporting:

```python
try:
    validator.validate_startup_configuration("config.json")
except MCPConfigurationError as e:
    print(f"Error: {e}")
    print(f"Field: {e.field_path}")
    print(f"Validation errors: {e.validation_errors}")
```

### Error Context

Errors include:
- **Message**: Human-readable error description
- **Field Path**: Dot-notation path to the problematic field
- **Validation Errors**: List of specific validation failures

## Integration

### MCP Client Integration

The validation system is automatically integrated into `MCPLanguageModel`:

```python
from graph_of_thoughts.language_models import MCPLanguageModel

# Validation happens automatically during initialization
lm = MCPLanguageModel("config.json", "model_name")
```

### Custom Integration

```python
from graph_of_thoughts.language_models.mcp_protocol import MCPConfigurationValidator

class MyMCPClient:
    def __init__(self, config_path):
        validator = MCPConfigurationValidator()
        validator.validate_startup_configuration(config_path)
        # Continue with initialization
```

## Best Practices

1. **Always validate configurations at startup**
2. **Enable security checks in production**
3. **Use strict mode for development**
4. **Review and address all warnings**
5. **Use HTTPS for remote connections**
6. **Avoid hardcoding sensitive data in configurations**
7. **Use semantic versioning for client info**
8. **Set appropriate timeouts and retry limits**

## Examples

See `examples/mcp_config_validation_example.py` for comprehensive usage examples and `tests/test_mcp_configuration_validation.py` for test cases demonstrating all validation features.
