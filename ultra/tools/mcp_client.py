"""MCP client — connect ARIA to external tool servers via Model Context Protocol.

MCP is a standard protocol for exposing tools, resources, and prompts
from external servers. This client discovers tools from MCP servers
and makes them available to ARIA's agent loop.

Requires: pip install mcp
Environment: ARIA_MCP_SERVERS (comma-separated list of server commands)

Usage:
    from ultra.tools.mcp_client import MCPManager
    manager = MCPManager()
    manager.add_server("npx -y @modelcontextprotocol/server-filesystem /tmp")
    await manager.start()
    tools = manager.available_tools()
    result = await manager.call_tool("read_file", {"path": "/tmp/test.txt"})
    await manager.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("aria.mcp")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""

    def to_prompt(self) -> str:
        """Format as a tool description for LLM prompts."""
        params = ""
        if self.input_schema and self.input_schema.get("properties"):
            props = self.input_schema["properties"]
            required = self.input_schema.get("required", [])
            parts = []
            for k, v in props.items():
                desc = v.get("description", v.get("type", ""))
                req = " (required)" if k in required else ""
                parts.append(f"    {k}: {desc}{req}")
            params = "\n" + "\n".join(parts)
        return f"- {self.name}: {self.description}{params}"


@dataclass
class MCPServer:
    """Configuration for an MCP server."""
    name: str
    command: str  # e.g. "npx -y @modelcontextprotocol/server-filesystem /tmp"
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # runtime state
    session: Any = None
    tools: list[MCPTool] = field(default_factory=list)
    connected: bool = False


class MCPManager:
    """Manages connections to multiple MCP servers."""

    def __init__(self):
        self.servers: dict[str, MCPServer] = {}
        self._read_env_servers()

    def _read_env_servers(self) -> None:
        """Read server configs from ARIA_MCP_SERVERS env var.

        Format: "name:command args,name2:command2 args2"
        Example: "filesystem:npx -y @modelcontextprotocol/server-filesystem /tmp"
        """
        raw = os.getenv("ARIA_MCP_SERVERS", "")
        if not raw:
            return
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                name, cmd = entry.split(":", 1)
                self.add_server(cmd.strip(), name=name.strip())

    def add_server(self, command: str, name: str | None = None,
                   env: dict[str, str] | None = None) -> None:
        """Register an MCP server."""
        parts = command.split()
        if not name:
            name = parts[0].split("/")[-1].replace("@", "").replace("-", "_")
        self.servers[name] = MCPServer(
            name=name, command=parts[0], args=parts[1:],
            env=env or {},
        )
        logger.info(f"MCP server registered: {name} ({command})")

    async def start(self) -> None:
        """Connect to all registered MCP servers."""
        if not _HAS_MCP:
            logger.warning("MCP not installed. Run: pip install mcp")
            return
        for name, server in self.servers.items():
            try:
                await self._connect_server(server)
            except Exception as e:
                logger.error(f"Failed to connect MCP server '{name}': {e}")

    async def stop(self) -> None:
        """Disconnect from all servers."""
        for server in self.servers.values():
            if server.session:
                try:
                    await server.session.__aexit__(None, None, None)
                except Exception:
                    pass
                server.session = None
                server.connected = False

    async def _connect_server(self, server: MCPServer) -> None:
        """Connect to a single MCP server and discover its tools."""
        params = StdioServerParameters(
            command=server.command,
            args=server.args,
            env=server.env or None,
        )
        # store the context manager for later use
        server._cm = stdio_client(params)
        read_stream, write_stream = await server._cm.__aenter__()
        server.session = ClientSession(read_stream, write_stream)
        await server.session.__aenter__()
        await server.session.initialize()

        # discover tools
        tools_result = await server.session.list_tools()
        server.tools = [
            MCPTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {},
                server_name=server.name,
            )
            for t in tools_result.tools
        ]
        server.connected = True
        logger.info(
            f"MCP server '{server.name}' connected: "
            f"{len(server.tools)} tools discovered")

    # ── tool access ──────────────────────────────────────────────

    def available_tools(self) -> list[MCPTool]:
        """Get all tools from all connected servers."""
        tools = []
        for server in self.servers.values():
            if server.connected:
                tools.extend(server.tools)
        return tools

    def tools_for_prompt(self) -> str:
        """Format all available MCP tools as a prompt section."""
        tools = self.available_tools()
        if not tools:
            return ""
        lines = ["[MCP Tools — external tool servers]"]
        for t in tools:
            lines.append(t.to_prompt())
        return "\n".join(lines)

    async def call_tool(self, tool_name: str,
                        arguments: dict[str, Any]) -> str:
        """Call an MCP tool by name. Returns the result as a string."""
        # find which server has this tool
        for server in self.servers.values():
            if not server.connected:
                continue
            for tool in server.tools:
                if tool.name == tool_name:
                    try:
                        result = await server.session.call_tool(
                            tool_name, arguments)
                        # extract text content
                        if hasattr(result, "content"):
                            parts = []
                            for block in result.content:
                                if hasattr(block, "text"):
                                    parts.append(block.text)
                                else:
                                    parts.append(str(block))
                            return "\n".join(parts)
                        return str(result)
                    except Exception as e:
                        return f"MCP tool error: {e}"
        return f"MCP tool '{tool_name}' not found"

    def get_status(self) -> dict[str, dict]:
        """Get status of all registered servers."""
        return {
            name: {
                "connected": server.connected,
                "tools": len(server.tools),
                "tool_names": [t.name for t in server.tools],
            }
            for name, server in self.servers.items()
        }
