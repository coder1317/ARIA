"""Notification Manager — routes alerts to the right channel.

When ARIA detects something important (build completed, error occurred,
scheduled task finished), the NotificationManager decides where to send
the alert based on what's available and the user's preferences.

Channels:
  - CLI: inline in the terminal (default)
  - File: append to a notification log
  - Telegram: send via the Telegram bot (if configured)

Subscribes to EventBus events and forwards them as notifications.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger("aria.notifications")


@dataclass
class Notification:
    """A single notification."""
    message: str
    level: str = "info"  # "info", "warning", "error", "success"
    source: str = "system"
    timestamp: float = field(default_factory=time.time)
    read: bool = False

    def __str__(self) -> str:
        icon = {"info": "ℹ", "warning": "⚠", "error": "✘", "success": "✓"}.get(
            self.level, "•")
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"[{ts}] {icon} {self.message}"


class NotificationManager:
    """Manages notifications and routes them to available channels.

    Usage:
        notify = NotificationManager()
        notify.subscribe_event_bus(bus)  # auto-notify on events
        notify.info("Build completed!")
        notify.warning("Cloud model slow")
        recent = notify.recent(5)
    """

    def __init__(self, log_path: Path | None = None, max_history: int = 200):
        self._notifications: deque[Notification] = deque(maxlen=max_history)
        self._handlers: list[Callable[[Notification], None]] = []
        self._lock = threading.Lock()
        self._log_path = log_path
        self._callbacks: list[Callable[[str, str], None]] = []  # (level, message) callbacks

        # Stats
        self._counts: dict[str, int] = {"info": 0, "warning": 0, "error": 0, "success": 0}

    # ── Sending ────────────────────────────────────────────────

    def send(self, message: str, level: str = "info", source: str = "system") -> Notification:
        """Send a notification."""
        notif = Notification(message=message, level=level, source=source)
        with self._lock:
            self._notifications.append(notif)
            self._counts[level] = self._counts.get(level, 0) + 1

        # File log
        if self._log_path:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._log_path, "a") as f:
                    f.write(str(notif) + "\n")
            except Exception:
                pass

        # Callbacks
        for cb in self._callbacks:
            try:
                cb(level, message)
            except Exception:
                pass

        # Handlers
        for handler in self._handlers:
            try:
                handler(notif)
            except Exception:
                pass

        return notif

    def info(self, message: str, source: str = "system") -> Notification:
        return self.send(message, level="info", source=source)

    def warning(self, message: str, source: str = "system") -> Notification:
        return self.send(message, level="warning", source=source)

    def error(self, message: str, source: str = "system") -> Notification:
        return self.send(message, level="error", source=source)

    def success(self, message: str, source: str = "system") -> Notification:
        return self.send(message, level="success", source=source)

    # ── Subscription ───────────────────────────────────────────

    def on_notify(self, handler: Callable[[Notification], None]) -> None:
        """Register a handler for all notifications."""
        self._handlers.append(handler)

    def on_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register a (level, message) callback."""
        self._callbacks.append(callback)

    def subscribe_event_bus(self, bus) -> None:
        """Subscribe to EventBus events and auto-notify."""
        from ultra.core.events import EventType

        def on_task_completed(event):
            task_id = event.payload.get("task_id", "?")
            self.success(f"Task {task_id} completed", source=event.source)

        def on_task_failed(event):
            task_id = event.payload.get("task_id", "?")
            error = event.payload.get("error", "unknown")[:100]
            self.error(f"Task {task_id} failed: {error}", source=event.source)

        def on_build_completed(event):
            path = event.payload.get("path", "?")
            score = event.payload.get("score", 0)
            self.success(f"Build complete — score {score:.0%} at {path}", source=event.source)

        def on_build_failed(event):
            desc = event.payload.get("description", "?")[:80]
            self.error(f"Build failed: {desc}", source=event.source)

        def on_research_completed(event):
            topic = event.payload.get("topic", "?")[:60]
            sources = event.payload.get("sources", 0)
            self.info(f"Research done: {topic} ({sources} sources)", source=event.source)

        def on_provider_failed(event):
            provider = event.payload.get("provider", "?")
            self.warning(f"Provider {provider} failed — failover active", source=event.source)

        def on_model_changed(event):
            old = event.payload.get("old", "?")
            new = event.payload.get("new", "?")
            self.info(f"Model switched: {old} → {new}", source=event.source)

        bus.on(EventType.TASK_COMPLETED, on_task_completed)
        bus.on(EventType.TASK_FAILED, on_task_failed)
        bus.on(EventType.BUILD_COMPLETED, on_build_completed)
        bus.on(EventType.BUILD_FAILED, on_build_failed)
        bus.on(EventType.RESEARCH_COMPLETED, on_research_completed)
        bus.on(EventType.PROVIDER_FAILED, on_provider_failed)
        bus.on(EventType.MODEL_CHANGED, on_model_changed)

    # ── Query ──────────────────────────────────────────────────

    def recent(self, limit: int = 10, level: str | None = None) -> list[Notification]:
        """Get recent notifications."""
        with self._lock:
            notifs = list(self._notifications)
        if level:
            notifs = [n for n in notifs if n.level == level]
        return notifs[-limit:]

    def unread(self) -> list[Notification]:
        """Get unread notifications."""
        with self._lock:
            return [n for n in self._notifications if not n.read]

    def mark_all_read(self) -> None:
        with self._lock:
            for n in self._notifications:
                n.read = True

    def stats(self) -> dict[str, int]:
        return dict(self._counts)

    def clear(self) -> None:
        with self._lock:
            self._notifications.clear()
