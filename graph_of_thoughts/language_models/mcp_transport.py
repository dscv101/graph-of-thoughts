# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

import asyncio
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPTransport(ABC):
    """
    Abstract base class for MCP transport implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MCP transport.

        :param config: Transport configuration
        :type config: Dict[str, Any]
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session: Optional[ClientSession] = None
        self.connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the MCP host.

        :return: True if connection successful, False otherwise
        :rtype: bool
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close the connection to the MCP host.
        """
        pass

    @abstractmethod
    async def send_sampling_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a sampling request to the MCP host.

        :param request: The sampling request
        :type request: Dict[str, Any]
        :return: The response from the host
        :rtype: Dict[str, Any]
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class StdioMCPTransport(MCPTransport):
    """
    MCP transport implementation using stdio communication.
    This is used for connecting to local MCP hosts like Claude Desktop, VSCode, or Cursor.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the stdio MCP transport.

        :param config: Transport configuration
        :type config: Dict[str, Any]
        """
        super().__init__(config)
        self.process: Optional[subprocess.Popen] = None
        self.stdio_transport = None
        self.write_stream = None

    async def connect(self) -> bool:
        """
        Establish stdio connection to the MCP host.

        :return: True if connection successful, False otherwise
        :rtype: bool
        """
        try:
            host_type = self.config.get("host_type", "claude_desktop")
            
            # For now, we'll use a simple approach where the host is expected to be running
            # In a real implementation, you might need to start the host process or connect differently
            if host_type == "claude_desktop":
                # Claude Desktop typically runs as a separate process
                # We would need to connect to its MCP server
                self.logger.info("Attempting to connect to Claude Desktop MCP server")
                # This is a placeholder - actual implementation would depend on how Claude Desktop exposes MCP
                server_params = StdioServerParameters(
                    command="claude-desktop-mcp-server",  # Hypothetical command
                    args=[],
                    env=None
                )
            elif host_type == "vscode":
                self.logger.info("Attempting to connect to VSCode MCP server")
                server_params = StdioServerParameters(
                    command="code",  # VSCode command
                    args=["--mcp-server"],  # Hypothetical MCP server flag
                    env=None
                )
            elif host_type == "cursor":
                self.logger.info("Attempting to connect to Cursor MCP server")
                server_params = StdioServerParameters(
                    command="cursor",  # Cursor command
                    args=["--mcp-server"],  # Hypothetical MCP server flag
                    env=None
                )
            else:
                raise ValueError(f"Unsupported host type: {host_type}")

            # For now, we'll simulate a connection since the actual MCP server commands
            # would depend on the specific implementation of each host
            self.logger.warning("Simulating MCP connection - actual implementation would connect to real MCP server")
            self.connected = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect via stdio: {e}")
            return False

    async def disconnect(self) -> None:
        """
        Close the stdio connection to the MCP host.
        """
        try:
            if self.session:
                await self.session.close()
            if self.stdio_transport:
                await self.stdio_transport.close()
            if self.process:
                self.process.terminate()
                self.process.wait()
            self.connected = False
            self.logger.info("Disconnected from MCP host")
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

    async def send_sampling_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a sampling request via stdio.

        :param request: The sampling request
        :type request: Dict[str, Any]
        :return: The response from the host
        :rtype: Dict[str, Any]
        """
        if not self.connected:
            raise RuntimeError("Not connected to MCP host")

        try:
            # This is a placeholder implementation
            # In a real implementation, this would use the MCP protocol to send sampling requests
            self.logger.info(f"Sending sampling request: {request}")
            
            # Simulate a response for now
            response = {
                "model": "claude-3-5-sonnet",
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": "This is a simulated response from the MCP host."
                },
                "stopReason": "endTurn"
            }
            
            return response

        except Exception as e:
            self.logger.error(f"Failed to send sampling request: {e}")
            raise


class HTTPMCPTransport(MCPTransport):
    """
    MCP transport implementation using HTTP communication.
    This is used for connecting to remote MCP servers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HTTP MCP transport.

        :param config: Transport configuration
        :type config: Dict[str, Any]
        """
        super().__init__(config)
        self.client: Optional[httpx.AsyncClient] = None
        self.server_url = config.get("server_url", "http://localhost:8000/mcp")

    async def connect(self) -> bool:
        """
        Establish HTTP connection to the MCP server.

        :return: True if connection successful, False otherwise
        :rtype: bool
        """
        try:
            timeout = self.config.get("connection_config", {}).get("timeout", 30.0)
            self.client = httpx.AsyncClient(timeout=timeout)
            
            # Test connection with a ping or initialization request
            response = await self.client.post(
                f"{self.server_url}/initialize",
                json={"protocolVersion": "2024-11-05", "capabilities": {}}
            )
            
            if response.status_code == 200:
                self.connected = True
                self.logger.info(f"Connected to MCP server at {self.server_url}")
                return True
            else:
                self.logger.error(f"Failed to connect: HTTP {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to connect via HTTP: {e}")
            return False

    async def disconnect(self) -> None:
        """
        Close the HTTP connection to the MCP server.
        """
        try:
            if self.client:
                await self.client.aclose()
            self.connected = False
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

    async def send_sampling_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a sampling request via HTTP.

        :param request: The sampling request
        :type request: Dict[str, Any]
        :return: The response from the server
        :rtype: Dict[str, Any]
        """
        if not self.connected or not self.client:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self.client.post(
                f"{self.server_url}/sampling/createMessage",
                json=request
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(f"Sampling request failed: HTTP {response.status_code}")

        except Exception as e:
            self.logger.error(f"Failed to send sampling request: {e}")
            raise


def create_transport(config: Dict[str, Any]) -> MCPTransport:
    """
    Factory function to create the appropriate MCP transport based on configuration.

    :param config: Transport configuration
    :type config: Dict[str, Any]
    :return: MCP transport instance
    :rtype: MCPTransport
    """
    transport_type = config.get("transport_type", "stdio")
    
    if transport_type == "stdio":
        return StdioMCPTransport(config)
    elif transport_type == "http":
        return HTTPMCPTransport(config)
    else:
        raise ValueError(f"Unsupported transport type: {transport_type}")
