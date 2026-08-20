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
    """Result of executing a tool."""
    tool: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        d = {"tool": self.tool, "success": self.success, "output": self.output}
        if self.error:
            d["error"] = self.error
        d["duration_ms"] = round(self.duration_ms, 1)
        return d

    def summary(self) -> str:
        """Compact summary for inclusion in runtime context."""
        if self.error:
            return f"[{self.tool} FAILED] {self.error}"
        out = str(self.output) if self.output else "ok"
        if len(out) > 500:
            out = out[:500] + "..."
        return f"[{self.tool}] {out}"


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

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call with validation.

        1. Check tool exists
        2. Validate required args
        3. Call handler
        4. Track in history
        """
        start = time.time()

        tool = self._tools.get(call.tool)
        if tool is None:
            return ToolResult(call.tool, False, error=f"unknown tool: {call.tool}")
        if not tool.enabled:
            return ToolResult(call.tool, False, error=f"tool disabled: {call.tool}")

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

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={self.tool_count} categories={self.categories}>"
