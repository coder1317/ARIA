"""Tool Registry — unified interface for every ARIA capability.

Every tool (terminal, browser, research, editor, MCP, memory, etc.)
is registered as a ToolDefinition with a JSON Schema, risk level,
and handler callable. The registry:

1. Generates structured tool definitions for LLM prompts
2. Validates arguments against schemas before execution
3. Routes tool calls to the right handler
4. Tracks execution history for the runtime

This replaces the current pattern where the LLM's response is
parsed with fragile keyword matching. Instead, the model receives
structured tool definitions and responds with structured tool calls.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("aria.tools")


class RiskLevel(Enum):
    """Permission tiers for tool execution."""
    READ_ONLY = "read_only"       # no side effects
    LOW = "low"                   # minor side effects
    MODERATE = "moderate"         # file changes, package installs
    HIGH = "high"                 # git operations, deployments
    CRITICAL = "critical"         # destructive, irreversible


@dataclass
class ToolParam:
    """A single parameter for a tool."""
    name: str
    type: str                    # "string", "integer", "boolean", "array", "object"
    description: str = ""
    required: bool = False
    default: Any = None
    enum: list[str] | None = None

    def to_schema(self) -> dict:
        s: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            s["enum"] = self.enum
        if self.default is not None:
            s["default"] = self.default
        return s


@dataclass
class ToolDefinition:
    """A registered tool with schema and handler."""
    name: str
    description: str
    handler: Callable[..., Any]
    params: list[ToolParam] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    category: str = "general"       # filesystem, terminal, web, memory, etc.
    enabled: bool = True

    def to_prompt_schema(self) -> str:
        """Format as a tool description for LLM system prompts."""
        lines = [f"## {self.name}", self.description, ""]
        if self.params:
            lines.append("Parameters:")
            for p in self.params:
                req = " [required]" if p.required else ""
                enum = f" (one of: {', '.join(p.enum)})" if p.enum else ""
                lines.append(f"  - {p.name} ({p.type}{req}): {p.description}{enum}")
        lines.append(f"Risk: {self.risk_level.value}")
        lines.append(f"Category: {self.category}")
        return "\n".join(lines)

    def to_json_schema(self) -> dict:
        """Full JSON Schema for programmatic use."""
        properties = {}
        required = []
        for p in self.params:
            properties[p.name] = p.to_schema()
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolCall:
    """A structured tool call from the LLM."""
    tool: str
    args: dict[str, Any]
    call_id: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(
            tool=d.get("tool", d.get("name", "")),
            args=d.get("args", d.get("parameters", {})),
            call_id=d.get("call_id", d.get("id", "")),
        )


@dataclass
class ToolResult:
    """Result of executing a tool.

    Enhanced with observation semantics for better agent reasoning.
    """
    tool: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    # Phase 1: enhanced semantics
    artifacts: list[str] = field(default_factory=list)  # files/paths produced
    metadata: dict[str, Any] = field(default_factory=dict)  # extra info
    retryable: bool = True  # can this be retried?
    requires_approval: bool = False  # needs human approval?
    source: str = ""  # which provider/tool produced this

    def to_dict(self) -> dict:
        d = {"tool": self.tool, "success": self.success, "output": self.output}
        if self.error:
            d["error"] = self.error
        d["duration_ms"] = round(self.duration_ms, 1)
        if self.artifacts:
            d["artifacts"] = self.artifacts
        if self.metadata:
            d["metadata"] = self.metadata
        if self.requires_approval:
            d["requires_approval"] = True
        return d

    def summary(self) -> str:
        """Compact summary for inclusion in runtime context."""
        if self.error:
            return f"[{self.tool} FAILED] {self.error}"
        out = str(self.output) if self.output else "ok"
        if len(out) > 500:
            out = out[:500] + "..."
        parts = [f"[{self.tool}] {out}"]
        if self.artifacts:
            parts.append(f"  artifacts: {', '.join(self.artifacts[:3])}")
        return "\n".join(parts)


class ToolRegistry:
    """Central registry for all ARIA tools.

    Usage:
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="terminal.execute",
            description="Run a shell command",
            handler=my_handler,
            params=[ToolParam("command", "string", required=True)],
            risk_level=RiskLevel.MODERATE,
            category="terminal",
        ))

        # Generate LLM prompt
        prompt = registry.to_prompt()

        # Execute a tool call
        result = registry.execute(ToolCall("terminal.execute", {"command": "ls"}))
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._history: list[ToolResult] = []
        self._mcp_manager = None  # Phase 2: MCP integration
        self._security = None  # Phase 5: Execution security

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool. Overwrites if name already exists."""
        self._tools[tool.name] = tool
        logger.debug("registered tool: %s (risk=%s)", tool.name, tool.risk_level.value)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self, enabled_only: bool = True,
                   category: str | None = None) -> list[ToolDefinition]:
        """List all registered tools, optionally filtered."""
        tools = list(self._tools.values())
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def to_prompt(self, category: str | None = None) -> str:
        """Generate tool descriptions for inclusion in LLM system prompts.

        This is the key integration point: the LLM receives these
        structured definitions and responds with structured tool calls
        instead of relying on intent classification.
        """
        tools = self.list_tools(category=category)
        if not tools:
            return "No tools available."
        lines = [
            "# Available Tools",
            "",
            "You have access to these tools. To use a tool, respond with:",
            "```json",
            '{"tool_call": {"tool": "<name>", "args": {<parameters>}}}',
            "```",
            "",
            "You may call multiple tools sequentially. After each tool call, "
            "you will receive the result. Evaluate whether to continue, "
            "replan, or stop.",
            "",
        ]
        for tool in tools:
            lines.append(tool.to_prompt_schema())
            lines.append("")
        return "\n".join(lines)

    def to_json_schemas(self) -> list[dict]:
        """All tools as JSON Schema (for programmatic use)."""
        return [t.to_json_schema() for t in self._tools.values() if t.enabled]

    def to_ollama_tools(self) -> list[dict]:
        """Generate Ollama-native tool definitions.

        Ollama uses OpenAI-compatible tool format:
        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]

        This enables native structured tool calling in the LLM.
        """
        return self.to_json_schemas()

    def set_security(self, security) -> None:
        """Set execution security policy."""
        self._security = security

    def execute(self, call: ToolCall, auto_approve_moderate: bool = True) -> ToolResult:
        """Execute a tool call with validation + risk enforcement + security policy.

        1. Check tool exists and is enabled
        2. Phase 5: Run security policy checks
        3. Validate required args
        4. Enforce risk-level policy (CRITICAL always blocked, HIGH configurable)
        5. Call handler
        6. Phase 5: Redact sensitive output
        7. Track in history
        """
        start = time.time()

        tool = self._tools.get(call.tool)
        if tool is None:
            return ToolResult(call.tool, False, error=f"unknown tool: {call.tool}")
        if not tool.enabled:
            return ToolResult(call.tool, False, error=f"tool disabled: {call.tool}")

        # ── Phase 5: Security policy check ──────────────────────
        if self._security:
            policy_result = self._security.check_tool_call(call, tool.risk_level)
            if policy_result is not None:
                self._history.append(policy_result)
                return policy_result

        # ── Risk-level enforcement ──────────────────────────────
        if tool.risk_level == RiskLevel.CRITICAL:
            return ToolResult(
                call.tool, False,
                error=f"[SECURITY] CRITICAL tool '{call.tool}' requires explicit "
                      "human approval. Cannot execute autonomously.",
            )
        if tool.risk_level == RiskLevel.HIGH and not auto_approve_moderate:
            return ToolResult(
                call.tool, False,
                error=f"[SECURITY] HIGH-risk tool '{call.tool}' requires approval."
            )

        # Validate required params
        required = {p.name for p in tool.params if p.required}
        provided = set(call.args.keys())
        missing = required - provided
        if missing:
            return ToolResult(
                call.tool, False,
                error=f"missing required args: {', '.join(sorted(missing))}",
            )

        # Validate enum params
        for p in tool.params:
            if p.enum and p.name in call.args:
                val = str(call.args[p.name])
                if val not in p.enum:
                    return ToolResult(
                        call.tool, False,
                        error=f"invalid value for {p.name}: '{val}' "
                              f"(expected one of: {', '.join(p.enum)})",
                    )

        # Execute
        try:
            output = tool.handler(**call.args)
            duration = (time.time() - start) * 1000
            # Phase 5: Redact sensitive output
            if self._security and isinstance(output, str):
                output = self._security.redact_output(output)
            result = ToolResult(call.tool, True, output=output, duration_ms=duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            result = ToolResult(call.tool, False, error=f"{type(e).__name__}: {e}",
                                duration_ms=duration)

        self._history.append(result)
        return result

    @property
    def history(self) -> list[ToolResult]:
        return list(self._history)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def categories(self) -> list[str]:
        return sorted(set(t.category for t in self._tools.values()))

    def register_mcp_server(self, mcp_manager, server_name: str | None = None) -> None:
        """Register tools from an MCP server into the unified registry.

        MCP tools get namespaced: mcp.<server>.<tool_name>
        This makes them indistinguishable from native tools to the LLM.
        """
        self._mcp_manager = mcp_manager
        tools = mcp_manager.available_tools()
        count = 0
        for tool in tools:
            if server_name and tool.server_name != server_name:
                continue
            # Namespace: mcp.<server>.<tool>
            ns_name = f"mcp.{tool.server_name}.{tool.name}"
            # Convert MCP input schema to ToolParams
            params = []
            schema_props = tool.input_schema.get("properties", {}) if tool.input_schema else {}
            required_set = set(tool.input_schema.get("required", []) if tool.input_schema else [])
            for pname, pdef in schema_props.items():
                params.append(ToolParam(
                    name=pname,
                    type=pdef.get("type", "string"),
                    description=pdef.get("description", ""),
                    required=pname in required_set,
                ))

            # Create async handler wrapper
            def _make_handler(tn=tool.name):
                def _handler(**kwargs):
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Already in async context — use thread
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                result = pool.submit(
                                    asyncio.run,
                                    mcp_manager.call_tool(tn, kwargs)
                                ).result(timeout=30)
                        else:
                            result = asyncio.run(mcp_manager.call_tool(tn, kwargs))
                    except Exception as e:
                        result = f"MCP error: {e}"
                    return result
                return _handler

            self.register(ToolDefinition(
                name=ns_name,
                description=tool.description or f"MCP tool: {tool.name}",
                handler=_make_handler(),
                params=params,
                risk_level=RiskLevel.LOW,
                category="mcp",
            ))
            count += 1
        logger.info("registered %d MCP tools from '%s'", count,
                     server_name or "all servers")

    def unregister_mcp(self, server_name: str | None = None) -> None:
        """Remove all MCP tools (optionally from a specific server)."""
        to_remove = [
            name for name in self._tools
            if name.startswith("mcp.") and (server_name is None or server_name in name)
        ]
        for name in to_remove:
            del self._tools[name]
        logger.info("unregistered %d MCP tools", len(to_remove))

    def dynamic_add(self, name: str, description: str,
                    handler: Callable, params: list[ToolParam] | None = None,
                    risk_level: RiskLevel = RiskLevel.LOW,
                    category: str = "dynamic") -> None:
        """Add a tool dynamically at runtime."""
        self.register(ToolDefinition(
            name=name, description=description, handler=handler,
            params=params or [], risk_level=risk_level, category=category,
        ))

    def dynamic_remove(self, name: str) -> bool:
        """Remove a tool at runtime."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={self.tool_count} categories={self.categories}>"
