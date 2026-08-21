"""Tests for Phase 3: EventBus, FileWatcher, NotificationManager, GoalManager."""
import os
import tempfile
import time
from pathlib import Path

import pytest

from ultra.core.events import Event, EventBus, EventType
from ultra.core.watchers import FileSnapshot, FileWatcher
from ultra.core.notifications import NotificationManager, Notification
from ultra.core.goals import GoalManager, Goal


# ── EventBus Tests ──────────────────────────────────────────

class TestEventBus:
    def test_emit_and_receive(self):
        bus = EventBus()
        received = []
        bus.on(EventType.TASK_COMPLETED, lambda e: received.append(e))
        bus.emit(EventType.TASK_COMPLETED, {"task_id": "123"}, source="test")
        assert len(received) == 1
        assert received[0].type == EventType.TASK_COMPLETED
        assert received[0].payload["task_id"] == "123"
        assert received[0].source == "test"

    def test_wildcard_receives_all(self):
        bus = EventBus()
        received = []
        bus.on("*", lambda e: received.append(e.type))
        bus.emit(EventType.TASK_COMPLETED)
        bus.emit(EventType.BUILD_STARTED)
        bus.emit(EventType.USER_MESSAGE)
        assert received == [EventType.TASK_COMPLETED, EventType.BUILD_STARTED, EventType.USER_MESSAGE]

    def test_priority_ordering(self):
        bus = EventBus()
        order = []
        bus.on("test.event", lambda e: order.append("low"), priority=200)
        bus.on("test.event", lambda e: order.append("high"), priority=10)
        bus.on("test.event", lambda e: order.append("mid"), priority=100)
        bus.emit("test.event")
        assert order == ["high", "mid", "low"]

    def test_off_unsubscribes(self):
        bus = EventBus()
        received = []
        handler = lambda e: received.append(1)
        bus.on("test", handler)
        bus.emit("test")
        assert len(received) == 1
        bus.off("test", handler)
        bus.emit("test")
        assert len(received) == 1  # no new event

    def test_history(self):
        bus = EventBus()
        for i in range(5):
            bus.emit(f"type.{i}")
        history = bus.history(limit=3)
        assert len(history) == 3
        assert history[0].type == "type.2"
        assert history[2].type == "type.4"

    def test_history_filter(self):
        bus = EventBus()
        bus.emit(EventType.TASK_COMPLETED)
        bus.emit(EventType.BUILD_STARTED)
        bus.emit(EventType.TASK_COMPLETED)
        history = bus.history(event_type=EventType.TASK_COMPLETED)
        assert len(history) == 2

    def test_stats(self):
        bus = EventBus()
        bus.on("test", lambda e: None)
        bus.emit(EventType.TASK_COMPLETED)
        bus.emit(EventType.BUILD_STARTED)
        stats = bus.stats()
        assert stats["total_emitted"] == 2
        assert stats["subscriber_count"] == 1
        assert stats["event_types"][EventType.TASK_COMPLETED] == 1

    def test_convenience_emitters(self):
        bus = EventBus()
        received = []
        bus.on("*", lambda e: received.append(e.type))
        bus.task_created("t1", "build", "test task")
        bus.task_completed("t1", "done")
        bus.build_started("test")
        bus.build_completed("/path", 0.95)
        bus.research_completed("topic", 5, 0.8)
        bus.model_changed("old", "new")
        bus.system_startup()
        assert EventType.TASK_CREATED in received
        assert EventType.TASK_COMPLETED in received
        assert EventType.BUILD_STARTED in received
        assert EventType.BUILD_COMPLETED in received
        assert EventType.RESEARCH_COMPLETED in received
        assert EventType.MODEL_CHANGED in received
        assert EventType.SYSTEM_STARTUP in received

    def test_handler_error_doesnt_crash(self):
        bus = EventBus()
        def bad_handler(e):
            raise ValueError("oops")
        bus.on("test", bad_handler)
        # Should not raise
        bus.emit("test", {"key": "value"})
        assert bus.emit_count == 1


# ── FileWatcher Tests ───────────────────────────────────────

class TestFileWatcher:
    def test_file_snapshot_diff(self, tmp_path):
        # Create initial files
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        snap1 = FileSnapshot(tmp_path)

        # Ensure different mtime
        import time; time.sleep(0.1)

        # Add a file, modify one, delete one
        (tmp_path / "c.txt").write_text("new")
        (tmp_path / "a.txt").write_text("modified")
        (tmp_path / "b.txt").unlink()
        snap2 = FileSnapshot(tmp_path)

        diff = snap2.diff(snap1)
        assert "c.txt" in diff["created"]
        assert "a.txt" in diff["modified"]
        assert "b.txt" in diff["deleted"]

    def test_watcher_detects_changes(self, tmp_path):
        bus = EventBus()
        watcher = FileWatcher(bus)
        watcher.watch(tmp_path)

        import time; time.sleep(0.1)

        # Create a file
        (tmp_path / "test.txt").write_text("hello")
        results = watcher.check_once()
        assert len(results) == 1
        path, diff = results[0]
        assert "test.txt" in diff["created"]

    def test_watcher_callback(self, tmp_path):
        changes = []
        watcher = FileWatcher()
        watcher.watch(tmp_path)
        watcher.on_change(lambda p, d: changes.append((p, d)))

        import time; time.sleep(0.1)
        (tmp_path / "new.txt").write_text("data")
        watcher.check_once()
        assert len(changes) == 1

    def test_watcher_ignore_patterns(self, tmp_path):
        watcher = FileWatcher()
        watcher.watch(tmp_path)

        # Create files including ignored ones
        (tmp_path / "real.txt").write_text("data")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.pyc").write_text("cache")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git")

        results = watcher.check_once()
        assert len(results) == 1
        _, diff = results[0]
        assert "real.txt" in diff["created"]
        # Ignored files should not appear
        assert not any("__pycache__" in f for f in diff["created"])
        assert not any(".git" in f for f in diff["created"])

    def test_watcher_start_stop(self, tmp_path):
        watcher = FileWatcher()
        watcher.watch(tmp_path)
        assert watcher.watched_count == 1
        watcher.start(interval=1)
        assert watcher._running
        watcher.stop()
        assert not watcher._running

    def test_snapshot_summary(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        watcher = FileWatcher()
        watcher.watch(tmp_path)
        summary = watcher.snapshot_summary()
        assert str(tmp_path) in summary
        assert summary[str(tmp_path)] == 2


# ── NotificationManager Tests ───────────────────────────────

class TestNotificationManager:
    def test_send_and_recent(self):
        notify = NotificationManager()
        notify.info("test info")
        notify.warning("test warning")
        notify.error("test error")
        recent = notify.recent(5)
        assert len(recent) == 3
        assert recent[0].level == "info"
        assert recent[1].level == "warning"
        assert recent[2].level == "error"

    def test_filter_by_level(self):
        notify = NotificationManager()
        notify.info("a")
        notify.info("b")
        notify.warning("c")
        info_only = notify.recent(10, level="info")
        assert len(info_only) == 2

    def test_file_logging(self, tmp_path):
        log_file = tmp_path / "test.log"
        notify = NotificationManager(log_path=log_file)
        notify.info("logged message")
        assert log_file.exists()
        content = log_file.read_text()
        assert "logged message" in content

    def test_callback(self):
        received = []
        notify = NotificationManager()
        notify.on_callback(lambda level, msg: received.append((level, msg)))
        notify.info("hello")
        assert len(received) == 1
        assert received[0] == ("info", "hello")

    def test_stats(self):
        notify = NotificationManager()
        notify.info("a")
        notify.info("b")
        notify.warning("c")
        notify.error("d")
        stats = notify.stats()
        assert stats["info"] == 2
        assert stats["warning"] == 1
        assert stats["error"] == 1

    def test_mark_read(self):
        notify = NotificationManager()
        notify.info("a")
        notify.warning("b")
        unread = notify.unread()
        assert len(unread) == 2
        notify.mark_all_read()
        unread = notify.unread()
        assert len(unread) == 0

    def test_event_bus_subscription(self):
        bus = EventBus()
        notify = NotificationManager()
        notify.subscribe_event_bus(bus)

        bus.task_completed("t1", "done")
        bus.build_completed("/path", 0.9)
        notifs = notify.recent(10)
        assert len(notifs) >= 2


# ── GoalManager Tests ───────────────────────────────────────

class TestGoalManager:
    def test_add_and_list(self, tmp_path):
        goals = GoalManager(tmp_path / "goals.db")
        g1 = goals.add("Build ARIA", priority=1)
        g2 = goals.add("Learn Rust", priority=3)
        active = goals.list_active()
        assert len(active) == 2
        assert active[0].title == "Build ARIA"  # higher priority first

    def test_update_progress(self, tmp_path):
        goals = GoalManager(tmp_path / "goals.db")
        g = goals.add("Test goal")
        goals.update_progress(g.id, 0.5, notes="halfway")
        updated = goals.get(g.id)
        assert updated.progress == 0.5
        assert updated.status == "in_progress"
        assert updated.notes == "halfway"

    def test_complete(self, tmp_path):
        goals = GoalManager(tmp_path / "goals.db")
        g = goals.add("Complete me")
        goals.complete(g.id, notes="all done")
        updated = goals.get(g.id)
        assert updated.status == "completed"
        assert updated.progress == 1.0
        assert updated.completed_at != ""

    def test_abandon(self, tmp_path):
        goals = GoalManager(tmp_path / "goals.db")
        g = goals.add("Abandon me")
        goals.abandon(g.id, reason="no longer needed")
        updated = goals.get(g.id)
        assert updated.status == "abandoned"

    def test_stats(self, tmp_path):
        goals = GoalManager(tmp_path / "goals.db")
        goals.add("a")
        goals.add("b")
        g = goals.add("c")
        goals.complete(g.id)
        stats = goals.stats()
        assert stats["active"] == 2
        assert stats["completed"] == 1

    def test_sub_goals(self, tmp_path):
        goals = GoalManager(tmp_path / "goals.db")
        parent = goals.add("Parent goal")
        child = goals.add("Sub task", parent_id=parent.id)
        assert child.parent_id == parent.id

    def test_list_completed(self, tmp_path):
        goals = GoalManager(tmp_path / "goals.db")
        g = goals.add("Done goal")
        goals.complete(g.id)
        completed = goals.list_completed()
        assert len(completed) == 1
        assert completed[0].title == "Done goal"
