"""BaseAgent — audit + performance wrapper around agent functions (spec §5.1).

ARIA keeps the proven v4 agent functions (module-level, client-based)
and wraps them so every run gets: timing, audit-log entry, provider name,
and a uniform AgentResult. The Agent class gives the spec's registry shape
(Brain.agents["research"].execute(...)) without rewriting working code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ultra.audit import AuditLog


@dataclass
class AgentResult:
    success: bool = False
    data: dict[str, Any] | None = None
    error: str | None = None
    duration_sec: float = 0.0
    provider: str | None = None

    def get(self, key: str, default=None):
        return self.data.get(key, default) if self.data else default


def run_agent(agent_name: str, task_type: str, fn: Callable,
              audit: AuditLog | None = None, **kwargs) -> AgentResult:
    """Run an agent function with timing, audit logging, and error capture."""
    start = time.time()
    provider: str | None = None
    try:
        result = fn(**kwargs)
        if isinstance(result, tuple) and len(result) == 2 and \
                isinstance(result[1], str) and result[0] is not None:
            # (data, provider_name) tuples from pool-aware functions
            data, provider = result
        else:
            data = result
        return AgentResult(success=True, data=_as_dict(data),
                           duration_sec=time.time() - start, provider=provider)
    except Exception as e:
        return AgentResult(success=False, error=f"{type(e).__name__}: {e}",
                           duration_sec=time.time() - start)
    finally:
        if audit is not None:
            audit.log(actor=f"agent:{agent_name}", action="execute",
                      task_type=task_type,
                      detail={"kwargs": {k: str(v)[:200] for k, v in kwargs.items()}},
                      duration_ms=(time.time() - start) * 1000,
                      provider=provider)


def _as_dict(data: Any) -> dict | None:
    if isinstance(data, dict):
        return data
    if data is None:
        return None
    return {"result": data}


class Agent:
    """Spec-style agent: name + task_type + callable with .execute()."""

    def __init__(self, name: str, task_type: str, fn: Callable,
                 audit: AuditLog | None = None):
        self.name = name
        self.task_type = task_type
        self.fn = fn
        self.audit = audit

    def execute(self, **kwargs) -> AgentResult:
        return run_agent(self.name, self.task_type, self.fn,
                         audit=self.audit, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Agent {self.name} ({self.task_type})>"
