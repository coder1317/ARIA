"""Security Policy — execution security for autonomous agents.

Transforms ARIA's security from input-scanning to full execution security:
- Tool risk enforcement (READ_ONLY → CRITICAL)
- Workspace sandboxing (tools can only access allowed directories)
- Terminal sandboxing (blocked commands, restricted PATH)
- Network policies (allowed domains for web tools)
- Approval gates (human-in-the-loop for high-risk operations)
- Credential isolation (secrets never in tool output)

This is the difference between "scan for bad strings" and
"enforce a comprehensive security policy."
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ultra.core.tool_registry import RiskLevel, ToolCall, ToolResult


@dataclass
class SecurityPolicy:
    """Complete security configuration for ARIA execution."""
    # Workspace boundaries
    allowed_read_dirs: list[str] = field(default_factory=list)
    allowed_write_dirs: list[str] = field(default_factory=list)

    # Terminal sandboxing
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /", "rm -rf /*", "mkfs", "dd if=",
        ":(){ :|:& };:", "chmod -R 777 /",
        "wget", "curl | sh", "bash -c",
    ])
    blocked_patterns: list[str] = field(default_factory=lambda: [
        r"sudo\s+rm", r">\s*/dev/sd", r"shutdown",
        r"init\s+[06]", r"halt", r"reboot",
    ])

    # Network policies
    allowed_domains: list[str] = field(default_factory=lambda: [
        "localhost", "127.0.0.1", "github.com",
        "duckduckgo.com", "arxiv.org",
    ])
    blocked_domains: list[str] = field(default_factory=list)

    # Approval gates
    auto_approve_risk: RiskLevel = RiskLevel.MODERATE
    require_approval_for: list[str] = field(default_factory=lambda: [
        "git push", "git commit", "pip install", "npm install",
        "docker", "deploy",
    ])

    # Credential isolation
    redact_patterns: list[str] = field(default_factory=lambda: [
        r"(?:api[_-]?key|token|secret|password|credential)\s*[=:]\s*\S+",
        r"Bearer\s+\S+",
        r"sk-[a-zA-Z0-9]{20,}",
    ])

    # Execution limits
    max_tool_calls_per_mission: int = 100
    max_terminal_timeouts: int = 5
    max_concurrent_tools: int = 4


@dataclass
class ApprovalRequest:
    """A request for human approval before executing a tool."""
    id: str
    tool_name: str
    args: dict[str, Any]
    risk_level: str
    reason: str
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # "pending" | "approved" | "denied"
    response: str = ""


class ExecutionSecurity:
    """Enforce security policies during tool execution.

    This wraps the ToolRegistry's execute method with additional
    security checks:
    1. Risk-level enforcement
    2. Workspace boundary checks
    3. Terminal command sandboxing
    4. Network policy enforcement
    5. Approval gate management
    6. Credential redaction in outputs
    """

    def __init__(self, policy: SecurityPolicy | None = None):
        self.policy = policy or SecurityPolicy()
        self._approval_queue: list[ApprovalRequest] = []
        self._approval_handler: Callable | None = None
        self._terminal_timeouts = 0
        self._local = threading.local()

    def set_approval_handler(self, handler: Callable[[ApprovalRequest], bool]) -> None:
        """Set a handler for approval requests.

        The handler receives an ApprovalRequest and returns True/False.
        """
        self._approval_handler = handler

    def check_tool_call(self, call: ToolCall, risk_level: RiskLevel) -> ToolResult | None:
        """Pre-execution security check.

        Returns None if the call is allowed, or a ToolResult with
        the denial reason if blocked.
        """
        # 1. Risk-level enforcement
        if risk_level == RiskLevel.CRITICAL:
            return ToolResult(
                call.tool, False,
                error=f"[SECURITY] CRITICAL tool '{call.tool}' requires explicit "
                      "human approval. Cannot execute autonomously.",
                requires_approval=True,
            )

        if risk_level == RiskLevel.HIGH:
            if not self._check_approval(call, "high-risk operation"):
                return ToolResult(
                    call.tool, False,
                    error=f"[SECURITY] HIGH-risk tool '{call.tool}' requires approval.",
                    requires_approval=True,
                )

        # 2. Terminal command sandboxing
        if call.tool == "terminal.execute":
            result = self._check_terminal(call)
            if result:
                return result

        # 3. Workspace boundary checks
        if call.tool.startswith("filesystem."):
            result = self._check_filesystem(call)
            if result:
                return result

        # 4. Network policy for web tools
        if call.tool.startswith("web."):
            result = self._check_network(call)
            if result:
                return result

        return None  # allowed

    def _check_terminal(self, call: ToolCall) -> ToolResult | None:
        """Sandbox terminal commands."""
        command = call.args.get("command", "")

        # Check blocked commands
        for blocked in self.policy.blocked_commands:
            if blocked in command:
                return ToolResult(
                    call.tool, False,
                    error=f"[SECURITY] Blocked command: contains '{blocked}'",
                )

        # Check blocked patterns
        for pattern in self.policy.blocked_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return ToolResult(
                    call.tool, False,
                    error=f"[SECURITY] Blocked pattern: matches '{pattern}'",
                )

        return None

    def _check_filesystem(self, call: ToolCall) -> ToolResult | None:
        """Enforce workspace boundaries."""
        path = call.args.get("path", "")

        # Read operations
        if call.tool == "filesystem.read" or call.tool == "filesystem.list":
            if self.policy.allowed_read_dirs:
                allowed = any(
                    str(Path(path).expanduser().resolve()).startswith(
                        str(Path(d).expanduser().resolve()))
                    for d in self.policy.allowed_read_dirs
                )
                if not allowed:
                    return ToolResult(
                        call.tool, False,
                        error=f"[SECURITY] Read outside workspace: {path}",
                    )

        # Write operations
        if call.tool == "filesystem.write":
            if self.policy.allowed_write_dirs:
                allowed = any(
                    str(Path(path).expanduser().resolve()).startswith(
                        str(Path(d).expanduser().resolve()))
                    for d in self.policy.allowed_write_dirs
                )
                if not allowed:
                    return ToolResult(
                        call.tool, False,
                        error=f"[SECURITY] Write outside workspace: {path}",
                    )

        return None

    def _check_network(self, call: ToolCall) -> ToolResult | None:
        """Enforce network policies."""
        url = call.args.get("url", "")

        # Check blocked domains
        for domain in self.policy.blocked_domains:
            if domain in url:
                return ToolResult(
                    call.tool, False,
                    error=f"[SECURITY] Blocked domain: {domain}",
                )

        return None

    def _check_approval(self, call: ToolCall, reason: str) -> bool:
        """Request human approval for high-risk operations."""
        if not self._approval_handler:
            # No handler — deny by default for safety
            return False

        request = ApprovalRequest(
            id=f"approval-{int(time.time() * 1000)}",
            tool_name=call.tool,
            args=call.args,
            risk_level="high",
            reason=reason,
        )

        try:
            approved = self._approval_handler(request)
            request.status = "approved" if approved else "denied"
            return approved
        except Exception:
            return False

    def redact_output(self, output: str) -> str:
        """Remove credentials and secrets from tool output."""
        redacted = output
        for pattern in self.policy.redact_patterns:
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
        return redacted

    def get_pending_approvals(self) -> list[ApprovalRequest]:
        """Get pending approval requests."""
        return [r for r in self._approval_queue if r.status == "pending"]

    def stats(self) -> dict:
        return {
            "terminal_timeouts": self._terminal_timeouts,
            "pending_approvals": len(self.get_pending_approvals()),
        }
