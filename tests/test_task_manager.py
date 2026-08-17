"""Tests for ultra.task_manager — queue, workers, persistence, retries."""
import time

import pytest

from ultra.task_manager import TaskManager


def _register(tm, results=None):
    calls = {"n": 0}

    def echo(**payload):
        calls["n"] += 1
        return payload.get("text", "done")

    tm.register("echo", echo)
    return calls


def test_submit_and_wait(tmp_path):
    tm = TaskManager(tmp_path / "tasks.db", max_concurrent=2)
    _register(tm)
    tid = tm.submit("echo", {"text": "hello"})
    result = tm.wait_for(tid, timeout=10)
    assert result == "hello"
    assert tm.status(tid) == "completed"
    tm.shutdown()


def test_unknown_type_fails(tmp_path):
    tm = TaskManager(tmp_path / "tasks.db", max_concurrent=1)
    tid = tm.submit("nope", {})
    assert tm.wait_for(tid, timeout=10) is None
    assert tm.status(tid) == "failed"
    tm.shutdown()


def test_cancel_pending(tmp_path):
    import threading
    tm = TaskManager(tmp_path / "tasks.db", max_concurrent=1)
    _register(tm)

    gate = threading.Event()

    def slow(**payload):
        gate.wait(5)
        return "x"

    tm.register("slow", slow)
    tid1 = tm.submit("slow", {})
    time.sleep(0.3)
    # second task queues behind the blocked one → stays pending
    tid2 = tm.submit("echo", {"text": "queued"})
    time.sleep(0.3)
    assert tm.status(tid1) == "running"
    assert tm.cancel(tid2) is True
    gate.set()
    assert tm.wait_for(tid2, timeout=5) is None  # cancelled → no result
    assert tm.status(tid2) == "cancelled"
    tm.shutdown()


def test_retry_on_failure(tmp_path):
    tm = TaskManager(tmp_path / "tasks.db", max_concurrent=1)
    attempts = {"n": 0}

    def flaky(**payload):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient")
        return "recovered"

    tm.register("flaky", flaky)
    tid = tm.submit("flaky", {}, max_retries=3)
    assert tm.wait_for(tid, timeout=10) == "recovered"
    assert attempts["n"] == 3
    tm.shutdown()


def test_persistence_across_restart(tmp_path):
    db = tmp_path / "tasks.db"
    tm = TaskManager(db, max_concurrent=0)  # no workers yet
    _register(tm)
    tid = tm.submit("echo", {"text": "persisted"})
    # new manager with workers picks up the pending task
    tm2 = TaskManager(db, max_concurrent=1)
    _register(tm2)
    assert tm2.wait_for(tid, timeout=10) == "persisted"
    tm.shutdown()
    tm2.shutdown()


def test_list_tasks(tmp_path):
    tm = TaskManager(tmp_path / "tasks.db", max_concurrent=1)
    _register(tm)
    tm.submit("echo", {"text": "a"}, priority="high")
    tm.submit("echo", {"text": "b"}, priority="low")
    tasks = tm.list_tasks()
    assert len(tasks) == 2
    assert tasks[0]["priority"] < tasks[1]["priority"]  # sorted high first
    tm.shutdown()


def test_priority_order_execution(tmp_path):
    tm = TaskManager(tmp_path / "tasks.db", max_concurrent=1)
    order = []

    def track(**payload):
        order.append(payload["tag"])
        return "ok"

    tm.register("track", track)
    tm.submit("track", {"tag": "low"}, priority="low")
    tm.submit("track", {"tag": "high"}, priority="high")
    time.sleep(0.5)
    assert order == ["high", "low"]
    tm.shutdown()
