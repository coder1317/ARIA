"""Tests for the Scheduler."""
import tempfile
import time
from pathlib import Path

from ultra.scheduler import Scheduler, ScheduledTask


def _db():
    return Path(tempfile.mktemp(suffix=".db"))


def test_add_and_list():
    s = Scheduler(_db())
    t = s.add("test", "research AI", "daily", daily_time="08:00")
    assert t.id is not None
    assert t.name == "test"
    assert t.schedule_type == "daily"
    tasks = s.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].name == "test"
    s.shutdown()


def test_add_interval():
    s = Scheduler(_db())
    t = s.add("hourly", "check news", "interval", interval_seconds=3600)
    assert t.interval_seconds == 3600
    assert t.next_run != ""
    s.shutdown()


def test_enable_disable():
    s = Scheduler(_db())
    t = s.add("test", "echo hello", "once")
    assert s.disable(t.id)
    tasks = s.list_tasks()
    assert not tasks[0].enabled
    assert s.enable(t.id)
    tasks = s.list_tasks()
    assert tasks[0].enabled
    s.shutdown()


def test_remove():
    s = Scheduler(_db())
    t = s.add("test", "echo hello", "once")
    assert s.remove(t.id)
    assert len(s.list_tasks()) == 0
    assert not s.remove(t.id)  # already removed
    s.shutdown()


def test_run_pending():
    results = []
    s = Scheduler(_db(), dispatch_fn=lambda cmd, ch: f"ran: {cmd}")
    # add a task that's already due (next_run in the past)
    t = s.add("test", "hello world", "once")
    s._conn.execute(
        "UPDATE scheduled_tasks SET next_run = datetime('now', '-1 hour') WHERE id = ?",
        (t.id,))
    s._conn.commit()

    fired = s.run_pending()
    assert len(fired) == 1
    assert fired[0][1] == "test"
    assert "ran: hello world" in fired[0][2]
    # one-shot should be disabled after execution
    task = s.get(t.id)
    assert not task.enabled
    s.shutdown()


def test_run_pending_recurring():
    s = Scheduler(_db(), dispatch_fn=lambda cmd, ch: "ok")
    t = s.add("recurring", "check", "interval", interval_seconds=60)
    # make it due
    s._conn.execute(
        "UPDATE scheduled_tasks SET next_run = datetime('now', '-1 hour') WHERE id = ?",
        (t.id,))
    s._conn.commit()

    fired = s.run_pending()
    assert len(fired) == 1
    # recurring should still be enabled
    task = s.get(t.id)
    assert task.enabled
    assert task.next_run != ""
    s.shutdown()


def test_persistence():
    db = _db()
    s1 = Scheduler(db)
    s1.add("persistent", "test cmd", "once")
    s1.shutdown()

    s2 = Scheduler(db)
    tasks = s2.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].name == "persistent"
    s2.shutdown()
