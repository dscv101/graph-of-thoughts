# MCP Implementation API Reference

This document provides comprehensive API documentation for the Graph of Thoughts MCP (Model Context Protocol) implementation.

## Table of Contents

- [Overview](#overview)
- [Core Classes](#core-classes)
- [Configuration](#configuration)
- [Transport Layer](#transport-layer)
- [Sampling and Messaging](#sampling-and-messaging)
- [Plugin System](#plugin-system)
- [Circuit Breaker](#circuit-breaker)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

The MCP implementation provides a robust interface for communicating with MCP-compatible language models and services. It supports multiple transport protocols, advanced features like circuit breakers and plugin systems, and comprehensive error handling.

### Key Features

- **Multiple Transport Protocols**: stdio and HTTP transports
- **Plugin Architecture**: Extensible host-specific optimizations
- **Circuit Breaker Pattern**: Resilience against failing services
- **Batch Processing**: Efficient handling of multiple requests
- **Comprehensive Error Handling**: Detailed error types and recovery mechanisms
- **Performance Monitoring**: Built-in metrics and benchmarking

## Core Classes

### MCPLanguageModel

The main interface for interacting with MCP services.

```python
class MCPLanguageModel:
    """
    Main interface for MCP language model interactions.
    
    Provides both synchronous and asynchronous interfaces for querying
    MCP-compatible language models with comprehensive error handling,
    performance monitoring, and advanced features.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        model_name: str = "mcp_claude_desktop"
    ):
        """
        Initialize MCP language model.
        
        Args:
            config_path: Path to JSON configuration file
            config: Configuration dictionary (alternative to config_path)
            model_name: Name of the model configuration to use
            
        Raises:
            ValueError: If configuration is invalid or model not found
            FileNotFoundError: If config_path doesn't exist
        """
```

#### Methods

##### query_async(prompt, **kwargs)

Asynchronously query the MCP service.

```python
async def query_async(
    self,
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    num_responses: int = 1,
    include_context: Optional[str] = None,
    model_preferences: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Asynchronously query the MCP service.
    
    Args:
        prompt: The input prompt/question
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        stop_sequences: List of stop sequences
        num_responses: Number of responses to generate
        include_context: Context inclusion mode ("none", "thisServer", "allServers")
        model_preferences: Model selection preferences
        
    Returns:
        List of response strings
        
    Raises:
        MCPConnectionError: If connection fails
        MCPTimeoutError: If request times out
        MCPServerError: If server returns error
        MCPValidationError: If request parameters are invalid
    """
```

**Example:**

```python
import asyncio
from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel

async def example_query():
    lm = MCPLanguageModel(
        config_path="mcp_config.json",
        model_name="mcp_claude_desktop"
    )
    
    async with lm:
        response = await lm.query_async(
            "Explain quantum computing in simple terms",
            temperature=0.7,
            max_tokens=500
        )
        print(response[0])

asyncio.run(example_query())
```

##### query(prompt, **kwargs)

Synchronously query the MCP service.

```python
def query(
    self,
    prompt: str,
    **kwargs
) -> List[str]:
    """
    Synchronously query the MCP service.
    
    This is a convenience wrapper around query_async() for synchronous usage.
    
    Args:
        prompt: The input prompt/question
        **kwargs: Same arguments as query_async()
        
    Returns:
        List of response strings
        
    Raises:
        Same exceptions as query_async()
    """
```

**Example:**

```python
lm = MCPLanguageModel(config_path="mcp_config.json")
response = lm.query("What is artificial intelligence?")
print(response[0])
```

##### get_response_texts(response)

Extract text content from MCP responses.

```python
def get_response_texts(self, response: Union[List[Dict], Dict]) -> List[str]:
    """
    Extract text content from MCP response(s).
    
    Args:
        response: Single response dict or list of response dicts
        
    Returns:
        List of extracted text strings
    """
```

##### get_circuit_breaker_status()

Get circuit breaker status and metrics.

```python
def get_circuit_breaker_status(self) -> Optional[Dict[str, Any]]:
    """
    Get circuit breaker status and metrics if available.
    
    Returns:
        Dict with circuit breaker status or None if not enabled
        
    Example return value:
    {
        "state": "closed",
        "is_healthy": True,
        "total_requests": 100,
        "successful_requests": 95,
        "failed_requests": 5,
        "circuit_open_count": 1,
        "last_failure_time": 1234567890.0,
        "state_change_time": 1234567890.0
    }
    """
```

##### is_service_healthy()

Check if the MCP service is healthy.

```python
def is_service_healthy(self) -> bool:
    """
    Check if the MCP service is healthy based on circuit breaker state.
    
    Returns:
        True if service is healthy or circuit breaker not enabled
    """
```

#### Context Manager Support

The MCPLanguageModel supports async context manager protocol:

```python
async with MCPLanguageModel(config_path="config.json") as lm:
    response = await lm.query_async("Hello, world!")
    print(response[0])
# Connection automatically closed
```

## Configuration

### Configuration File Format

MCP configurations are stored in JSON format with the following structure:

```json
{
    "mcp_claude_desktop": {
        "transport": {
            "type": "stdio",
            "command": "claude-desktop-mcp-server",
            "args": [],
            "env": {}
        },
        "client_info": {
            "name": "graph-of-thoughts",
            "version": "0.0.3"
        },
        "capabilities": {
            "sampling": {},
            "tools": {},
            "resources": {},
            "prompts": {}
        },
        "default_sampling_params": {
            "modelPreferences": {
                "hints": [{"name": "claude-3-5-sonnet"}],
                "costPriority": 0.3,
                "speedPriority": 0.4,
                "intelligencePriority": 0.8
            },
            "temperature": 1.0,
            "maxTokens": 4096,
            "stopSequences": [],
            "includeContext": "thisServer"
        },
        "connection_config": {
            "timeout": 30.0,
            "retry_attempts": 3,
            "retry_delay": 1.0
        },
        "cost_tracking": {
            "prompt_token_cost": 0.003,
            "response_token_cost": 0.015
        },
        "batch_processing": {
            "max_concurrent": 10,
            "batch_size": 50,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "timeout_per_request": 30.0,
            "enable_by_default": true
        },
        "circuit_breaker": {
            "enabled": false,
            "failure_threshold": 5,
            "recovery_timeout": 30.0,
            "half_open_max_calls": 3,
            "success_threshold": 2,
            "monitoring_window": 60.0,
            "minimum_throughput": 10
        }
    }
}
```

### Configuration Parameters

#### Transport Configuration

- **type**: Transport protocol ("stdio" or "http")
- **command**: Command to execute (stdio only)
- **args**: Command arguments (stdio only)
- **env**: Environment variables (stdio only)
- **url**: Server URL (HTTP only)
- **headers**: HTTP headers (HTTP only)

#### Client Information

- **name**: Client application name
- **version**: Client version

#### Capabilities

Defines what MCP capabilities the client supports:
- **sampling**: Text generation capabilities
- **tools**: Tool execution capabilities
- **resources**: Resource access capabilities
- **prompts**: Prompt template capabilities

#### Default Sampling Parameters

- **modelPreferences**: Model selection preferences
- **temperature**: Default sampling temperature
- **maxTokens**: Default maximum tokens
- **stopSequences**: Default stop sequences
- **includeContext**: Default context inclusion mode

#### Connection Configuration

- **timeout**: Connection timeout in seconds
- **retry_attempts**: Number of retry attempts
- **retry_delay**: Delay between retries in seconds

#### Cost Tracking

- **prompt_token_cost**: Cost per prompt token
- **response_token_cost**: Cost per response token

#### Batch Processing

- **max_concurrent**: Maximum concurrent requests
- **batch_size**: Maximum batch size
- **retry_attempts**: Retry attempts for failed requests
- **retry_delay**: Delay between retries
- **timeout_per_request**: Timeout per individual request
- **enable_by_default**: Enable batch processing by default

#### Circuit Breaker

- **enabled**: Enable circuit breaker protection
- **failure_threshold**: Number of failures before opening circuit
- **recovery_timeout**: Time to wait before testing recovery
- **half_open_max_calls**: Maximum calls in half-open state
- **success_threshold**: Successes needed to close circuit
- **monitoring_window**: Time window for failure rate calculation
- **minimum_throughput**: Minimum requests before considering failure rate

## Transport Layer

### MCPTransport (Abstract Base Class)

```python
class MCPTransport(ABC):
    """
    Abstract base class for MCP transport implementations.
    
    Defines the interface that all transport implementations must follow.
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to MCP server."""
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        
    @abstractmethod
    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request and wait for response."""
        
    @abstractmethod
    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send notification (no response expected)."""
```

### StdioMCPTransport

Transport implementation for stdio-based MCP servers.

```python
class StdioMCPTransport(MCPTransport):
    """
    Stdio transport for MCP communication.
    
    Communicates with MCP servers via stdin/stdout using subprocess.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stdio transport.
        
        Args:
            config: Transport configuration including command and args
        """
```

### HTTPMCPTransport

Transport implementation for HTTP-based MCP servers.

```python
class HTTPMCPTransport(MCPTransport):
    """
    HTTP transport for MCP communication.
    
    Communicates with MCP servers via HTTP/HTTPS requests.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HTTP transport.
        
        Args:
            config: Transport configuration including URL and headers
        """
```

### Transport Factory

```python
def create_transport(config: Dict[str, Any]) -> MCPTransport:
    """
    Factory function to create appropriate transport based on configuration.
    
    Args:
        config: Complete MCP configuration
        
    Returns:
        MCPTransport instance (StdioMCPTransport or HTTPMCPTransport)
        
    Raises:
        ValueError: If transport type is not supported
    """
```

**Example:**

```python
from graph_of_thoughts.language_models.mcp_transport import create_transport

config = {
    "transport": {
        "type": "stdio",
        "command": "claude-desktop",
        "args": ["--mcp-server"]
    },
    "client_info": {"name": "my-app", "version": "1.0.0"}
}

transport = create_transport(config)
```

## Sampling and Messaging

### MCPSamplingManager

Manages MCP sampling requests and batch processing.

```python
class MCPSamplingManager:
    """
    Manager for MCP sampling operations with batch processing support.
    """

    def __init__(self, transport: MCPTransport, config: Dict[str, Any]):
        """
        Initialize sampling manager.

        Args:
            transport: MCP transport instance
            config: Configuration dictionary
        """

    async def create_message(
        self,
        prompt: str,
        role: str = "user",
        **sampling_params
    ) -> Dict[str, Any]:
        """
        Create a single message using MCP sampling.

        Args:
            prompt: The input prompt
            role: Message role ("user", "assistant", "system")
            **sampling_params: Additional sampling parameters

        Returns:
            MCP response dictionary
        """

    async def create_messages_batch(
        self,
        prompts: List[str],
        **sampling_params
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Create multiple messages using batch processing.

        Args:
            prompts: List of input prompts
            **sampling_params: Additional sampling parameters

        Returns:
            List of MCP responses (None for failed requests)
        """
```

### Message Creation Functions

```python
def create_text_message(text: str, role: str = "user") -> Dict[str, Any]:
    """
    Create a text message for MCP sampling.

    Args:
        text: Message text content
        role: Message role

    Returns:
        Formatted message dictionary
    """

def create_image_message(
    image_data: str,
    mime_type: str,
    role: str = "user"
) -> Dict[str, Any]:
    """
    Create an image message for MCP sampling.

    Args:
        image_data: Base64-encoded image data
        mime_type: Image MIME type
        role: Message role

    Returns:
        Formatted message dictionary
    """

def create_sampling_request(
    messages: List[Dict[str, Any]],
    temperature: float = 1.0,
    max_tokens: int = 4096,
    stop_sequences: Optional[List[str]] = None,
    include_context: Optional[MCPIncludeContext] = None,
    model_preferences: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a complete MCP sampling request.

    Args:
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        stop_sequences: Stop sequences
        include_context: Context inclusion mode
        model_preferences: Model selection preferences

    Returns:
        Complete MCP sampling request
    """
```

### MCPIncludeContext Enum

```python
class MCPIncludeContext(Enum):
    """Context inclusion modes for MCP sampling."""
    NONE = "none"
    THIS_SERVER = "thisServer"
    ALL_SERVERS = "allServers"
```

**Example:**

```python
from graph_of_thoughts.language_models.mcp_sampling import (
    create_text_message, create_sampling_request, MCPIncludeContext
)

# Create messages
messages = [
    create_text_message("Hello, how are you?", role="user"),
    create_text_message("I'm doing well, thank you!", role="assistant"),
    create_text_message("What can you help me with?", role="user")
]

# Create sampling request
request = create_sampling_request(
    messages=messages,
    temperature=0.7,
    max_tokens=1000,
    include_context=MCPIncludeContext.THIS_SERVER
)
```

## Plugin System

### MCPHostPlugin (Abstract Base Class)

```python
class MCPHostPlugin(ABC):
    """
    Abstract base class for MCP host plugins.

    Each MCP host should implement this interface to provide
    host-specific configuration, validation, and capabilities.
    """

    @abstractmethod
    def get_host_name(self) -> str:
        """Get unique identifier for this MCP host."""

    @abstractmethod
    def get_display_name(self) -> str:
        """Get human-readable display name for this MCP host."""

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this MCP host."""

    @abstractmethod
    def get_capabilities(self) -> HostCapabilities:
        """Get capabilities supported by this MCP host."""

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration for this MCP host."""

    def customize_transport_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Customize transport configuration for this host."""
```

### HostCapabilities

```python
@dataclass
class HostCapabilities:
    """Represents the capabilities of an MCP host."""
    supports_resources: bool = False
    supports_prompts: bool = False
    supports_tools: bool = False
    supports_sampling: bool = False
    supports_roots: bool = False
    supports_discovery: bool = False
    transport_types: List[str] = None
    authentication_methods: List[str] = None
```

### MCPPluginManager

```python
class MCPPluginManager:
    """High-level manager for MCP host plugins."""

    def list_hosts(self) -> List[str]:
        """Get list of all available MCP host names."""

    def get_host_info(self, host_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific MCP host."""

    def register_plugin(self, plugin: MCPHostPlugin) -> bool:
        """Register a new MCP host plugin."""

    def generate_config_template(self, host_name: str) -> Optional[str]:
        """Generate JSON configuration template for a specific host."""

    def validate_config(self, host_name: str, config: Dict[str, Any]) -> bool:
        """Validate configuration for a specific MCP host."""

    def create_transport(self, config: Dict[str, Any]) -> Any:
        """Create MCP transport using the plugin system."""
```

### Built-in Plugins

#### ClaudeDesktopPlugin
- **Host Name**: `claude_desktop`
- **Transport**: stdio
- **Capabilities**: Tools, Resources, Prompts, Sampling

#### VSCodePlugin
- **Host Name**: `vscode`
- **Transport**: stdio
- **Capabilities**: Tools, Resources, Prompts, Sampling, Roots, Discovery

#### CursorPlugin
- **Host Name**: `cursor`
- **Transport**: stdio
- **Capabilities**: Tools, Sampling

#### HTTPServerPlugin
- **Host Name**: `http_server`
- **Transport**: HTTP
- **Capabilities**: Tools, Resources, Prompts, Sampling, Discovery

**Example:**

```python
from graph_of_thoughts.language_models.mcp_plugin_manager import MCPPluginManager

manager = MCPPluginManager()

# List available hosts
hosts = manager.list_hosts()
print(f"Available hosts: {hosts}")

# Get host capabilities
info = manager.get_host_info("claude_desktop")
print(f"Claude Desktop capabilities: {info['capabilities']}")

# Generate configuration template
template = manager.generate_config_template("vscode")
print(template)
```

## Circuit Breaker

### MCPCircuitBreaker

```python
class MCPCircuitBreaker:
    """Circuit breaker implementation for MCP operations."""

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker with configuration."""

    async def __aenter__(self):
        """Async context manager entry."""

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""

    async def call(self, operation: Callable, *args, **kwargs):
        """Call an operation through the circuit breaker."""

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""

    def is_closed(self) -> bool:
        """Check if circuit breaker is closed (normal operation)."""

    def is_open(self) -> bool:
        """Check if circuit breaker is open (failing fast)."""

    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open (testing recovery)."""
```

### CircuitBreakerConfig

```python
@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    expected_exceptions: tuple = field(default_factory=lambda: (Exception,))
    success_threshold: int = 2
    monitoring_window: float = 60.0
    minimum_throughput: int = 10
```

### CircuitBreakerState

```python
class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open" # Testing recovery
```

**Example:**

```python
from graph_of_thoughts.language_models.mcp_circuit_breaker import (
    MCPCircuitBreaker, CircuitBreakerConfig
)

# Create circuit breaker
config = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=15.0
)
circuit_breaker = MCPCircuitBreaker(config)

# Use with operations
async def protected_operation():
    async with circuit_breaker:
        return await some_mcp_operation()

try:
    result = await protected_operation()
except CircuitBreakerOpenError:
    print("Service is currently unavailable")
```

## Error Handling

### Exception Hierarchy

```python
MCPTransportError                    # Base exception for all MCP errors
├── MCPConnectionError              # Connection-related errors
├── MCPTimeoutError                 # Request timeout errors
├── MCPProtocolError                # Protocol violation errors
├── MCPServerError                  # Server-side errors
├── MCPValidationError              # Request validation errors
└── CircuitBreakerOpenError         # Circuit breaker is open
```

### Exception Details

#### MCPTransportError
Base exception for all MCP-related errors.

```python
class MCPTransportError(Exception):
    """Base exception for MCP transport errors."""
    pass
```

#### MCPConnectionError
Raised when connection to MCP server fails.

```python
class MCPConnectionError(MCPTransportError):
    """Exception raised when MCP connection fails."""
    pass
```

#### MCPTimeoutError
Raised when MCP requests timeout.

```python
class MCPTimeoutError(MCPTransportError):
    """Exception raised when MCP request times out."""
    pass
```

#### MCPServerError
Raised when MCP server returns an error.

```python
class MCPServerError(MCPTransportError):
    """Exception raised when MCP server returns an error."""
    pass
```

#### CircuitBreakerOpenError
Raised when circuit breaker is open.

```python
class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass
```

### Error Handling Best Practices

```python
from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel
from graph_of_thoughts.language_models.mcp_transport import (
    MCPConnectionError, MCPTimeoutError, MCPServerError
)
from graph_of_thoughts.language_models.mcp_circuit_breaker import CircuitBreakerOpenError

async def robust_mcp_query(prompt: str) -> Optional[str]:
    """Example of robust error handling for MCP queries."""
    lm = MCPLanguageModel(config_path="config.json")

    try:
        async with lm:
            response = await lm.query_async(prompt)
            return response[0] if response else None

    except MCPConnectionError as e:
        print(f"Connection failed: {e}")
        # Maybe try alternative configuration or fallback
        return None

    except MCPTimeoutError as e:
        print(f"Request timed out: {e}")
        # Maybe retry with shorter prompt or different parameters
        return None

    except MCPServerError as e:
        print(f"Server error: {e}")
        # Server-side issue, may need to wait and retry
        return None

    except CircuitBreakerOpenError as e:
        print(f"Service unavailable: {e}")
        # Circuit breaker is protecting against failures
        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        # Log for debugging
        return None
```

## Examples

### Basic Usage

```python
import asyncio
from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel

async def basic_example():
    """Basic MCP usage example."""
    # Initialize with configuration file
    lm = MCPLanguageModel(
        config_path="mcp_config.json",
        model_name="mcp_claude_desktop"
    )

    # Simple query
    response = await lm.query_async("What is machine learning?")
    print(response[0])

# Run the example
asyncio.run(basic_example())
```

### Advanced Configuration

```python
import asyncio
from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel

async def advanced_example():
    """Advanced MCP usage with custom parameters."""
    # Configuration dictionary
    config = {
        "mcp_custom": {
            "transport": {
                "type": "stdio",
                "command": "my-mcp-server",
                "args": ["--mode", "production"]
            },
            "client_info": {
                "name": "my-application",
                "version": "2.0.0"
            },
            "capabilities": {
                "sampling": {},
                "tools": {}
            },
            "default_sampling_params": {
                "temperature": 0.8,
                "maxTokens": 2048
            },
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 3,
                "recovery_timeout": 20.0
            }
        }
    }

    lm = MCPLanguageModel(config=config, model_name="mcp_custom")

    async with lm:
        # Query with custom parameters
        response = await lm.query_async(
            "Explain neural networks",
            temperature=0.5,
            max_tokens=1000,
            stop_sequences=["END", "STOP"]
        )

        print(f"Response: {response[0]}")

        # Check service health
        if lm.is_service_healthy():
            print("Service is healthy")

        # Get circuit breaker status
        status = lm.get_circuit_breaker_status()
        if status:
            print(f"Circuit breaker state: {status['state']}")

asyncio.run(advanced_example())
```

### Plugin System Usage

```python
from graph_of_thoughts.language_models.mcp_plugin_manager import MCPPluginManager
from graph_of_thoughts.language_models.mcp_host_plugins import MCPHostPlugin, HostCapabilities

# Create custom plugin
class MyCustomPlugin(MCPHostPlugin):
    def get_host_name(self) -> str:
        return "my_custom_host"

    def get_display_name(self) -> str:
        return "My Custom MCP Host"

    def get_default_config(self) -> dict:
        return {
            "transport": {
                "type": "stdio",
                "command": "my-custom-mcp-server"
            },
            "client_info": {
                "name": "graph-of-thoughts",
                "version": "0.0.3"
            },
            "capabilities": {
                "sampling": {},
                "tools": {}
            }
        }

    def get_capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            supports_tools=True,
            supports_sampling=True,
            transport_types=["stdio"]
        )

# Use plugin system
manager = MCPPluginManager()

# Register custom plugin
manager.register_plugin(MyCustomPlugin())

# Generate configuration
template = manager.generate_config_template("my_custom_host")
print(template)

# Create transport using plugin system
config = manager.create_host_config("my_custom_host")
transport = manager.create_transport(config)
```

### Batch Processing

```python
import asyncio
from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel

async def batch_example():
    """Example of batch processing multiple queries."""
    lm = MCPLanguageModel(config_path="config.json")

    # List of prompts to process
    prompts = [
        "What is Python?",
        "Explain machine learning",
        "What is cloud computing?",
        "Define artificial intelligence",
        "What is data science?"
    ]

    async with lm:
        # Process prompts sequentially
        results = []
        for prompt in prompts:
            try:
                response = await lm.query_async(prompt, max_tokens=200)
                results.append(response[0])
            except Exception as e:
                print(f"Failed to process '{prompt}': {e}")
                results.append(None)

        # Display results
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"\nQuery {i+1}: {prompt}")
            print(f"Response: {result[:100]}..." if result else "Failed")

asyncio.run(batch_example())
```

### Performance Monitoring

```python
import asyncio
import time
from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel

async def performance_example():
    """Example of performance monitoring."""
    lm = MCPLanguageModel(config_path="config.json")

    async with lm:
        # Measure query performance
        start_time = time.time()

        response = await lm.query_async(
            "Write a short story about a robot",
            max_tokens=500
        )

        end_time = time.time()

        print(f"Query completed in {end_time - start_time:.2f} seconds")
        print(f"Response length: {len(response[0])} characters")

        # Get circuit breaker metrics
        cb_status = lm.get_circuit_breaker_status()
        if cb_status:
            print(f"Total requests: {cb_status['total_requests']}")
            print(f"Success rate: {cb_status['successful_requests'] / cb_status['total_requests'] * 100:.1f}%")

asyncio.run(performance_example())
```

### Error Recovery

```python
import asyncio
from graph_of_thoughts.language_models.mcp_client import MCPLanguageModel
from graph_of_thoughts.language_models.mcp_transport import MCPConnectionError, MCPTimeoutError

async def error_recovery_example():
    """Example of error recovery strategies."""
    lm = MCPLanguageModel(config_path="config.json")

    max_retries = 3
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            async with lm:
                response = await lm.query_async("Hello, world!")
                print(f"Success on attempt {attempt + 1}: {response[0]}")
                break

        except MCPConnectionError as e:
            print(f"Connection failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("All retry attempts failed")

        except MCPTimeoutError as e:
            print(f"Timeout on attempt {attempt + 1}: {e}")
            # Maybe reduce max_tokens or simplify prompt
            break

        except Exception as e:
            print(f"Unexpected error: {e}")
            break

asyncio.run(error_recovery_example())
```

## Best Practices

### Configuration Management

1. **Use Configuration Files**: Store configurations in JSON files for easy management
2. **Environment-Specific Configs**: Use different configurations for development, testing, and production
3. **Validate Configurations**: Always validate configurations before use
4. **Secure Credentials**: Store sensitive information like API keys securely

### Error Handling

1. **Specific Exception Handling**: Catch specific exceptions rather than generic ones
2. **Graceful Degradation**: Implement fallback mechanisms for service failures
3. **Retry Logic**: Implement exponential backoff for transient failures
4. **Circuit Breaker**: Use circuit breakers for external service dependencies

### Performance Optimization

1. **Connection Reuse**: Use async context managers to reuse connections
2. **Batch Processing**: Process multiple requests efficiently
3. **Appropriate Timeouts**: Set reasonable timeouts for different operations
4. **Monitor Performance**: Track response times and success rates

### Security

1. **Input Validation**: Validate all inputs before sending to MCP services
2. **Output Sanitization**: Sanitize responses before displaying to users
3. **Secure Transport**: Use HTTPS for HTTP transports when possible
4. **Access Control**: Implement proper authentication and authorization

---

For more examples and detailed usage patterns, see the [examples](../examples/) directory in the repository.
```
```
