"""EventBus — publish/subscribe system for ARIA's cognitive core.

Every significant state change in ARIA emits an event. Components subscribe
to events they care about and react accordingly. This replaces direct
coupling between agents, memory, scheduler, and UI.

Event flow:
  Component emits event → EventBus → all subscribers notified → handlers run

Design:
  - Thread-safe: handlers run in the emitting thread (not a separate thread)
  - Typed events: every event has a type, payload, timestamp, and source
  - Priority: handlers can be registered with priority (lower = runs first)
  - Wildcard: subscribe to "*" to receive all events
  - History: recent events are kept for debugging/replay
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("aria.events")

# ── Event Types ────────────────────────────────────────────────

class EventType:
    """All event types in the ARIA system."""
    # User interaction
    USER_MESSAGE = "user.message"
    USER_COMMAND = "user.command"

    # Task lifecycle
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"

    # Research
    RESEARCH_STARTED = "research.started"
    RESEARCH_COMPLETED = "research.completed"
    RESEARCH_SOURCES_FOUND = "research.sources_found"

    # Build
    BUILD_STARTED = "build.started"
    BUILD_COMPLETED = "build.completed"
    BUILD_FAILED = "build.failed"
    BUILD_EVALUATED = "build.evaluated"

    # Agent Runtime
    PLAN_CREATED = "plan.created"
    PLAN_COMPLETED = "plan.completed"
    PLAN_FAILED = "plan.failed"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    REPLAN_TRIGGERED = "plan.replan"

    # Memory
    MEMORY_STORED = "memory.stored"
    MEMORY_SEARCHED = "memory.searched"
    EPISODE_RECORDED = "memory.episode"
    PROCEDURE_LEARNED = "memory.procedure"

    # Model / Provider
    MODEL_CHANGED = "model.changed"
    PROVIDER_FAILED = "provider.failed"
    PROVIDER_RECOVERED = "provider.recovered"

    # System
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"

    # Watcher
    FILE_CHANGED = "file.changed"
    FILE_CREATED = "file.created"
    FILE_DELETED = "file.deleted"

    # Notifications
    NOTIFICATION = "notification"

    # Goal
    GOAL_ADDED = "goal.added"
    GOAL_COMPLETED = "goal.completed"
    GOAL_PROGRESS = "goal.progress"


# ── Event Data ─────────────────────────────────────────────────

@dataclass
class Event:
    """A single event in the system."""
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        keys = ", ".join(self.payload.keys()) if self.payload else "none"
        return f"Event({self.type} from={self.source} keys=[{keys}])"


# ── Handler ────────────────────────────────────────────────────

@dataclass
class _Subscription:
    """Internal subscription record."""
    handler: Callable[[Event], None]
    priority: int = 100  # lower = runs first
    event_type: str = "*"  # "*" = all events


# ── EventBus ───────────────────────────────────────────────────

class EventBus:
    """Central publish/subscribe event system.

    Usage:
        bus = EventBus()

        # Subscribe
        bus.on(EventType.TASK_COMPLETED, my_handler)
        bus.on("*", log_all_events)  # receive everything

        # Emit
        bus.emit(EventType.TASK_COMPLETED, {"task_id": "123"}, source="scheduler")
    """

    def __init__(self, history_size: int = 500):
        self._subscribers: dict[str, list[_Subscription]] = defaultdict(list)
        self._wildcards: list[_Subscription] = []
        self._lock = threading.Lock()
        self._history: deque[Event] = deque(maxlen=history_size)
        self._emit_count = 0

    # ── Subscribe ──────────────────────────────────────────────

    def on(self, event_type: str, handler: Callable[[Event], None],
           priority: int = 100) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Event type to listen for, or "*" for all events.
            handler: Callable that receives the Event.
            priority: Lower values run first (default 100).
        """
        sub = _Subscription(handler=handler, priority=priority, event_type=event_type)
        with self._lock:
            if event_type == "*":
                self._wildcards.append(sub)
                self._wildcards.sort(key=lambda s: s.priority)
            else:
                self._subscribers[event_type].append(sub)
                self._subscribers[event_type].sort(key=lambda s: s.priority)

    def off(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe a handler from an event type."""
        with self._lock:
            if event_type == "*":
                self._wildcards = [s for s in self._wildcards if s.handler != handler]
            elif event_type in self._subscribers:
                self._subscribers[event_type] = [
                    s for s in self._subscribers[event_type] if s.handler != handler
                ]

    # ── Emit ───────────────────────────────────────────────────

    def emit(self, event_type: str, payload: dict[str, Any] | None = None,
             source: str = "unknown") -> Event:
        """Emit an event to all subscribers.

        Args:
            event_type: The event type string.
            payload: Optional data dict.
            source: Component that emitted the event.

        Returns:
            The Event object (for chaining).
        """
        event = Event(type=event_type, payload=payload or {}, source=source)
        self._history.append(event)
        self._emit_count += 1

        # Collect matching handlers
        handlers: list[_Subscription] = []
        with self._lock:
            if event_type in self._subscribers:
                handlers.extend(self._subscribers[event_type])
            handlers.extend(self._wildcards)

        # Execute handlers in priority order
        for sub in sorted(handlers, key=lambda s: s.priority):
            try:
                sub.handler(event)
            except Exception as e:
                logger.warning("Event handler error for %s: %s", event_type, e)

        return event

    # ── Convenience emitters ───────────────────────────────────

    def task_created(self, task_id: str, task_type: str, description: str = "") -> Event:
        return self.emit(EventType.TASK_CREATED,
                        {"task_id": task_id, "task_type": task_type, "description": description},
                        source="brain")

    def task_completed(self, task_id: str, result: str = "") -> Event:
        return self.emit(EventType.TASK_COMPLETED,
                        {"task_id": task_id, "result": result[:500]},
                        source="brain")

    def task_failed(self, task_id: str, error: str = "") -> Event:
        return self.emit(EventType.TASK_FAILED,
                        {"task_id": task_id, "error": error[:500]},
                        source="brain")

    def build_started(self, description: str) -> Event:
        return self.emit(EventType.BUILD_STARTED, {"description": description}, source="pipeline")

    def build_completed(self, path: str, score: float) -> Event:
        return self.emit(EventType.BUILD_COMPLETED,
                        {"path": path, "score": score}, source="pipeline")

    def research_completed(self, topic: str, sources: int, confidence: float) -> Event:
        return self.emit(EventType.RESEARCH_COMPLETED,
                        {"topic": topic, "sources": sources, "confidence": confidence},
                        source="research")

    def model_changed(self, old_model: str, new_model: str) -> Event:
        return self.emit(EventType.MODEL_CHANGED,
                        {"old": old_model, "new": new_model}, source="cli")

    def system_startup(self) -> Event:
        return self.emit(EventType.SYSTEM_STARTUP, source="system")

    def notification(self, message: str, level: str = "info") -> Event:
        return self.emit(EventType.NOTIFICATION,
                        {"message": message, "level": level}, source="system")

    # ── Introspection ──────────────────────────────────────────

    @property
    def subscriber_count(self) -> int:
        """Total number of active subscriptions."""
        with self._lock:
            return sum(len(v) for v in self._subscribers.values()) + len(self._wildcards)

    @property
    def emit_count(self) -> int:
        return self._emit_count

    def history(self, limit: int = 20, event_type: str | None = None) -> list[Event]:
        """Get recent events, optionally filtered by type."""
        events = list(self._history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def stats(self) -> dict[str, Any]:
        """EventBus statistics."""
        with self._lock:
            type_counts = {}
            for e in self._history:
                type_counts[e.type] = type_counts.get(e.type, 0) + 1
        return {
            "total_emitted": self._emit_count,
            "subscriber_count": self.subscriber_count,
            "history_size": len(self._history),
            "event_types": type_counts,
        }
