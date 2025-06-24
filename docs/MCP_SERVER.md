# Graph of Thoughts MCP Server

The Graph of Thoughts MCP Server exposes Graph of Thoughts reasoning capabilities through the Model Context Protocol (MCP), allowing LLM hosts like Claude Desktop, VSCode, and Cursor to leverage advanced reasoning workflows.

## Overview

The MCP server provides:

- **Tools**: Executable functions for Graph of Thoughts operations
- **Resources**: Access to operation results, templates, and configurations
- **Prompts**: Reusable prompt templates for common reasoning workflows

## Quick Start

### Installation

The MCP server is included with the Graph of Thoughts package:

```bash
pip install graph_of_thoughts
```

### Running the Server

#### Stdio Transport (Recommended)

For use with Claude Desktop, VSCode, or Cursor:

```bash
python -m graph_of_thoughts
```

#### Command Line Options

```bash
# Show server information
python -m graph_of_thoughts --info

# Enable debug logging
python -m graph_of_thoughts --debug

# Show version
python -m graph_of_thoughts --version
```

## Integration with MCP Hosts

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "graph-of-thoughts": {
      "command": "python",
      "args": ["-m", "graph_of_thoughts"],
      "env": {}
    }
  }
}
```

### VSCode

Configure in your VSCode settings:

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

### Cursor

Add to your Cursor configuration:

```json
{
  "mcp": {
    "servers": {
      "graph-of-thoughts": {
        "command": "python -m graph_of_thoughts"
      }
    }
  }
}
```

### Augment Code

Augment Code has native MCP support for both VS Code and JetBrains IDEs.

#### VS Code Extension Settings

Add to your VS Code `settings.json`:

```json
{
  "augment.mcp.servers": {
    "graph-of-thoughts": {
      "command": "python",
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

#### JetBrains IDE Configuration

1. Open **Settings** → **Tools** → **Augment Code** → **MCP Servers**
2. **Add New Server**:
   - **Name**: `graph-of-thoughts`
   - **Command**: `python`
   - **Arguments**: `-m graph_of_thoughts`
   - **Description**: `Graph of Thoughts reasoning server`

#### Quick Setup

Use the automated setup script:

```bash
python setup_augment_code_mcp.py
```

## Available Tools

### break_down_task

Decompose complex tasks into manageable subtasks.

**Parameters:**

- `task` (required): The complex task to break down
- `max_subtasks` (optional): Maximum number of subtasks (default: 5)
- `domain` (optional): Task domain for context (default: "general")

**Example:**

```json
{
  "task": "Build a web application for task management",
  "max_subtasks": 5,
  "domain": "programming"
}
```

### generate_thoughts

Generate multiple thought approaches for problem solving.

**Parameters:**

- `problem` (required): The problem to generate thoughts for
- `num_thoughts` (optional): Number of thoughts to generate (default: 3)
- `approach_type` (optional): Type of approach (default: "analytical")

**Example:**

```json
{
  "problem": "Optimize database query performance",
  "num_thoughts": 3,
  "approach_type": "analytical"
}
```

### score_thoughts

Evaluate and score different thought approaches.

**Parameters:**

- `thoughts` (required): Array of thoughts to score
- `criteria` (optional): Scoring criteria (default: "feasibility, effectiveness, and clarity")

**Example:**

```json
{
  "thoughts": [
    "Use indexing to improve performance",
    "Implement query caching",
    "Optimize database schema"
  ],
  "criteria": "implementation difficulty and impact"
}
```

### validate_and_improve

Validate solutions and improve them iteratively.

**Parameters:**

- `solution` (required): The solution to validate and improve
- `validation_criteria` (optional): Criteria for validation
- `max_iterations` (optional): Maximum improvement iterations (default: 3)

**Example:**

```json
{
  "solution": "Use bubble sort to sort the array",
  "validation_criteria": "time complexity and space efficiency",
  "max_iterations": 2
}
```

### aggregate_results

Combine multiple results into a comprehensive solution.

**Parameters:**

- `results` (required): Array of results to aggregate
- `aggregation_method` (optional): Method for aggregation (default: "synthesis")

**Example:**

```json
{
  "results": [
    "Solution A: Fast but memory-intensive",
    "Solution B: Memory-efficient but slower"
  ],
  "aggregation_method": "synthesis"
}
```

### create_reasoning_chain

Build complete reasoning workflows with multiple operations.

**Parameters:**

- `problem` (required): The problem to solve with a reasoning chain
- `workflow_type` (optional): Type of workflow (default: "generate_score_select")
- `num_branches` (optional): Number of parallel reasoning branches (default: 3)

**Example:**

```json
{
  "problem": "Design a recommendation system",
  "workflow_type": "generate_score_select",
  "num_branches": 3
}
```

## Available Resources

### got://operations/results

Access to operation execution results and history.

**Format:** JSON
**Content:** Dictionary of operation IDs and their results

### got://templates/prompts

Reusable prompt templates for different domains.

**Format:** JSON
**Content:** Collection of prompt templates with placeholders

### got://configs/examples

Example configurations and workflow patterns.

**Format:** JSON
**Content:** Example workflows and configuration patterns

### got://logs/execution

Execution logs and debugging information.

**Format:** Plain text
**Content:** Timestamped log entries for all operations

## Available Prompts

### analyze-problem

Structured problem analysis using Graph of Thoughts methodology.

**Arguments:**

- `problem` (required): The problem statement to analyze
- `domain` (optional): Problem domain for context

### generate-solutions

Generate multiple solution approaches with different perspectives.

**Arguments:**

- `problem` (required): The problem to solve
- `num_approaches` (optional): Number of approaches to generate

### evaluate-options

Systematic evaluation of different solution options.

**Arguments:**

- `options` (required): List of options to evaluate
- `criteria` (optional): Evaluation criteria

## Workflow Examples

### Basic Problem Solving

1. Use `break_down_task` to decompose the problem
2. Use `generate_thoughts` for each subtask
3. Use `score_thoughts` to evaluate approaches
4. Use `validate_and_improve` on the best approach

### Complex Reasoning Chain

1. Use `create_reasoning_chain` with "generate_score_select" workflow
2. Review results in `got://operations/results` resource
3. Use `aggregate_results` if multiple solutions are needed

### Iterative Improvement

1. Start with `generate_thoughts` for initial approaches
2. Use `validate_and_improve` iteratively
3. Use `aggregate_results` to combine improved solutions

## Configuration

The server can be configured using the `mcp_server_config.json` file:

```json
{
  "server": {
    "name": "graph-of-thoughts",
    "version": "0.0.3"
  },
  "performance": {
    "max_concurrent_operations": 10,
    "operation_timeout": 300
  },
  "logging": {
    "level": "INFO"
  }
}
```

## Troubleshooting

### Server Won't Start

1. Check Python version (3.12+ required)
2. Verify Graph of Thoughts package is installed
3. Check for import errors with `python -c "import graph_of_thoughts"`

### Connection Issues

1. Verify the command path in your MCP host configuration
2. Check that the server starts with `python -m graph_of_thoughts --info`
3. Review logs for error messages

### Performance Issues

1. Reduce `num_thoughts` and `num_branches` parameters
2. Check system resources and memory usage
3. Enable debug logging to identify bottlenecks

## Development

### Running Tests

```bash
# Run all MCP server tests
python -m pytest tests/test_mcp_server.py -v

# Run integration tests
python -m pytest tests/test_mcp_server_integration.py -v

# Run all tests
python tests/run_all_tests.py --pattern "*mcp_server*"
```

### Adding New Tools

1. Add tool definition to `_register_tools()` method
2. Implement the tool handler method
3. Add tests in `test_mcp_server.py`
4. Update documentation

## Support

For issues and questions:

- GitHub Issues: [graph-of-thoughts repository](https://github.com/spcl/graph-of-thoughts)
- Documentation: See `docs/` directory
- Examples: See `examples/` directory
