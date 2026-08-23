"""Policy Engine — Event → Agent action mapping.

Connects the EventBus to autonomous agent actions. When events occur
(file changes, test failures, goal staleness, etc.), the policy engine
decides whether ARIA should act and what to do.

This transforms ARIA from reactive (user prompts) to proactive
(events trigger actions automatically).

Usage:
    engine = PolicyEngine(event_bus, runtime, goals)
    engine.load_default_policies()
    # Events now automatically trigger appropriate agent actions
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ultra.core.events import EventBus, Event, EventType

logger = logging.getLogger("aria.policy")


@dataclass
class Policy:
    """A rule that maps events to agent actions."""
    name: str
    event_type: str
    condition: Callable[[Event], bool] = lambda e: True
    action_type: str = "task"  # "task" | "notify" | "log" | "custom"
    action_config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 2  # 0=highest
    cooldown_seconds: float = 60.0  # min time between triggers
    last_triggered: float = 0.0

    def should_trigger(self, event: Event) -> bool:
        """Check if this policy should trigger for the given event."""
        if not self.enabled:
            return False
        if event.type != self.event_type:
            return False
        if not self.condition(event):
            return False
        if time.time() - self.last_triggered < self.cooldown_seconds:
            return False
        return True


class PolicyEngine:
    """Event-driven action engine.

    Monitors the EventBus and triggers autonomous actions based on
    configured policies. This is how ARIA becomes proactive — it
    reacts to events without waiting for user prompts.
    """

    def __init__(self, event_bus: EventBus, runtime=None, goals=None,
                 task_manager=None, notify=None):
        self.event_bus = event_bus
        self.runtime = runtime
        self.goals = goals
        self.task_manager = task_manager
        self.notify = notify
        self.policies: list[Policy] = []
        self._action_log: list[dict] = []
        self._running = False

        # Subscribe to all events
        self.event_bus.on("*", self._on_event, priority=100)

    def add_policy(self, policy: Policy) -> None:
        """Register a policy."""
        self.policies.append(policy)
        self.policies.sort(key=lambda p: p.priority)

    def remove_policy(self, name: str) -> bool:
        """Remove a policy by name."""
        before = len(self.policies)
        self.policies = [p for p in self.policies if p.name != name]
        return len(self.policies) < before

    def _on_event(self, event: Event) -> None:
        """Handle an incoming event by checking all policies."""
        for policy in self.policies:
            if policy.should_trigger(event):
                try:
                    self._execute_policy(policy, event)
                    policy.last_triggered = time.time()
                except Exception as e:
                    logger.warning("policy '%s' failed: %s", policy.name, e)

    def _execute_policy(self, policy: Policy, event: Event) -> None:
        """Execute a triggered policy."""
        logger.info("policy triggered: %s (event: %s)", policy.name, event.type)

        if policy.action_type == "task" and self.task_manager:
            task_type = policy.action_config.get("task_type", "chat")
            payload = policy.action_config.get("payload", {})
            # Inject event data into payload
            if not payload:
                payload = self._build_payload(policy, event)
            task_id = self.task_manager.submit(task_type, payload)
            self._log_action(policy.name, event.type, f"task:{task_id}")

        elif policy.action_type == "notify" and self.notify:
            msg = policy.action_config.get("message", f"Event: {event.type}")
            self.notify.send(msg, level=policy.action_config.get("level", "info"))
            self._log_action(policy.name, event.type, f"notify:{msg[:50]}")

        elif policy.action_type == "log":
            self._log_action(policy.name, event.type, str(event.payload)[:200])

        elif policy.action_type == "custom":
            handler = policy.action_config.get("handler")
            if callable(handler):
                handler(event)
                self._log_action(policy.name, event.type, "custom handler")

    def _build_payload(self, policy: Policy, event: Event) -> dict:
        """Build task payload from event data based on policy config."""
        config = policy.action_config
        task_type = config.get("task_type", "chat")

        if task_type == "research":
            topic = event.payload.get("topic", event.payload.get("text", "latest developments"))
            return {"topic": topic, "mode": config.get("mode", "deep")}
        elif task_type == "build":
            desc = event.payload.get("description", event.payload.get("text", ""))
            return {"description": desc}
        else:
            text = event.payload.get("text", str(event.payload))
            return {"text": text}

    def _log_action(self, policy_name: str, event_type: str, result: str) -> None:
        self._action_log.append({
            "policy": policy_name,
            "event": event_type,
            "result": result,
            "timestamp": time.time(),
        })
        # Keep last 100 entries
        if len(self._action_log) > 100:
            self._action_log = self._action_log[-100:]

    def get_action_log(self, limit: int = 20) -> list[dict]:
        """Recent policy actions."""
        return self._action_log[-limit:]

    def stats(self) -> dict:
        return {
            "policies": len(self.policies),
            "enabled": sum(1 for p in self.policies if p.enabled),
            "actions_taken": len(self._action_log),
        }


def load_default_policies(engine: PolicyEngine) -> None:
    """Load default policies for common ARIA scenarios."""
    from ultra.core.events import EventType

    # Policy 1: When research completes, record it
    engine.add_policy(Policy(
        name="research_completion_log",
        event_type=EventType.RESEARCH_COMPLETED,
        action_type="log",
        action_config={"message": "Research completed"},
        cooldown_seconds=10,
    ))

    # Policy 2: When a build fails, notify user
    engine.add_policy(Policy(
        name="build_failure_notify",
        event_type=EventType.BUILD_FAILED,
        action_type="notify",
        action_config={
            "message": "⚠️ Build failed — check the output for details",
            "level": "warning",
        },
        cooldown_seconds=30,
    ))

    # Policy 3: When a file changes, log it (can be extended to auto-test)
    engine.add_policy(Policy(
        name="file_change_log",
        event_type=EventType.FILE_CHANGED,
        action_type="log",
        action_config={"message": "File changed"},
        cooldown_seconds=5,
    ))

    # Policy 4: When a task fails, log for analysis
    engine.add_policy(Policy(
        name="task_failure_log",
        event_type=EventType.TASK_FAILED,
        action_type="log",
        action_config={"message": "Task failed"},
        cooldown_seconds=10,
    ))

    # Policy 5: When a goal is added, log it
    engine.add_policy(Policy(
        name="goal_added_log",
        event_type=EventType.GOAL_ADDED,
        action_type="log",
        action_config={"message": "Goal added"},
        cooldown_seconds=5,
    ))

    logger.info("loaded %d default policies", len(engine.policies))
