"""TaskManager — persistent background task queue (spec §4.4).

Tasks are submitted with a task_type that maps to a registered callable
(research, build, market, chat, ...). Worker threads execute them
concurrently (up to max_concurrent_tasks); state is persisted to SQLite
so the queue survives restarts. Supports priorities, dependencies,
timeouts and retries.

Design notes (hardened):
- Dependencies: a task with unmet depends_on stays BLOCKED until its
  dependencies complete successfully, then becomes pending.
- No queue duplication: `_queue` holds each task exactly once. Retries
  flip the status back to pending instead of re-appending.
- Timeout: the timer marks the task timed_out. The underlying thread
  keeps running (threads can't be killed), but the task is never
  re-queued, never retried, and its result is discarded. Long-running
  untrusted work should use subprocess isolation (see Terminal) so the
  process can actually be killed.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# priority: lower number = higher priority
PRIORITY = {"critical": 0, "high": 1, "normal": 2, "low": 3, "background": 4}

STATUS = ("pending", "blocked", "running", "completed", "failed",
          "cancelled", "timed_out")


@dataclass
class Task:
    task_id: str
    task_type: str
    status: str = "pending"
    priority: int = 2
    payload: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 2
    timeout_sec: float = 600.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None


class TaskManager:
    def __init__(self, db_path: str | Path, max_concurrent: int = 2,
                 default_timeout: float = 600.0):
        self.db_path = str(db_path)
        self.max_concurrent = max(1, max_concurrent)
        self.default_timeout = default_timeout
        self.registry: dict[str, Callable] = {}
        self.tasks: dict[str, Task] = {}
        self.lock = threading.RLock()
        self._queue: list[Task] = []   # source of truth for pending work
        self._running: dict[str, threading.Thread] = {}
        self._shutdown = threading.Event()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_pending()
        self._start_workers()

    # ── persistence ─────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    payload TEXT,
                    depends_on TEXT,
                    result TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 2,
                    timeout_sec REAL DEFAULT 600,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL
                )
            """)

    def _persist(self, task: Task) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (task.task_id, task.task_type, task.status, task.priority,
                 json.dumps(task.payload), json.dumps(task.depends_on),
                 json.dumps(task.result) if task.result else None,
                 task.error, task.retry_count, task.max_retries, task.timeout_sec,
                 task.created_at, task.started_at, task.completed_at),
            )

    def _load_pending(self) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status IN ('pending','blocked','running')"
            ).fetchall()
        for r in rows:
            t = Task(
                task_id=r[0], task_type=r[1],
                status="blocked" if (r[5] or "[]") != "[]" else "pending",
                priority=r[3], payload=json.loads(r[4] or "{}"),
                depends_on=json.loads(r[5] or "[]"),
                max_retries=r[9], timeout_sec=r[10], created_at=r[11],
            )
            self.tasks[t.task_id] = t
            self._queue.append(t)

    # ── public API ──────────────────────────────────────────────────

    def register(self, task_type: str, fn: Callable) -> None:
        """Register a callable for a task_type. fn(**payload) -> Any."""
        self.registry[task_type] = fn

    def submit(self, task_type: str, payload: dict | None = None,
               priority: str = "normal", max_retries: int = 2,
               timeout_sec: float | None = None,
               depends_on: list[str] | None = None) -> str:
        task = Task(
            task_id=str(uuid.uuid4())[:12],
            task_type=task_type,
            priority=PRIORITY.get(priority, 2),
            payload=payload or {},
            depends_on=depends_on or [],
            max_retries=max_retries,
            timeout_sec=timeout_sec or self.default_timeout,
        )
        if task.depends_on:
            task.status = "blocked"
        with self.lock:
            self.tasks[task.task_id] = task
            self._queue.append(task)
        self._persist(task)
        return task.task_id

    def get(self, task_id: str) -> Task | None:
        with self.lock:
            return self.tasks.get(task_id)

    def status(self, task_id: str) -> str | None:
        t = self.get(task_id)
        return t.status if t else None

    def wait_for(self, task_id: str, timeout: float = 120.0) -> Any:
        """Block until the task finishes. Returns result or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            t = self.get(task_id)
            if t is None:
                return None
            if t.status in ("completed", "failed", "cancelled", "timed_out"):
                return t.result if t.status == "completed" else None
            time.sleep(0.2)
        return None

    def cancel(self, task_id: str) -> bool:
        with self.lock:
            t = self.tasks.get(task_id)
            if t is None or t.status not in ("pending", "blocked"):
                return False
            t.status = "cancelled"
            self._persist(t)
            return True

    def list_tasks(self, limit: int = 20) -> list[dict]:
        with self.lock:
            tasks = sorted(self.tasks.values(),
                           key=lambda t: (t.priority, -t.created_at))
            out = []
            for t in tasks[:limit]:
                out.append({
                    "task_id": t.task_id, "task_type": t.task_type,
                    "status": t.status, "priority": t.priority,
                    "depends_on": t.depends_on, "error": t.error,
                    "retries": t.retry_count,
                })
            return out

    # ── workers ─────────────────────────────────────────────────────

    def _start_workers(self) -> None:
        for i in range(self.max_concurrent):
            t = threading.Thread(target=self._worker_loop, daemon=True,
                                 name=f"task-worker-{i}")
            t.start()

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            self._refresh_blocked()
            task = self._next_pending()
            if task is None:
                time.sleep(0.2)
                continue
            self._execute(task)

    def _dependencies_met(self, task: Task) -> bool:
        """All depends_on tasks completed successfully?"""
        for dep_id in task.depends_on:
            dep = self.tasks.get(dep_id)
            if dep is None or dep.status != "completed":
                return False
        return True

    def _next_pending(self) -> Task | None:
        with self.lock:
            pending = [t for t in self._queue if t.status == "pending"]
            if not pending:
                return None
            pending.sort(key=lambda t: (t.priority, t.created_at))
            task = pending[0]
            task.status = "running"
            task.started_at = time.time()
            self._persist(task)
            return task

    def _execute(self, task: Task) -> None:
        fn = self.registry.get(task.task_type)
        timer = threading.Timer(task.timeout_sec, self._timeout, args=[task.task_id])
        timer.daemon = True
        timer.start()
        try:
            if fn is None:
                raise RuntimeError(f"no handler registered for '{task.task_type}'")
            task.result = fn(**task.payload)
            if task.status == "running":  # not timed out mid-execution
                task.status = "completed"
                task.completed_at = time.time()
        except Exception as e:
            task.error = f"{type(e).__name__}: {e}"
            if task.retry_count < task.max_retries and task.status == "running":
                task.retry_count += 1
                task.status = "pending"  # re-queue in place — no duplication
                task.started_at = None
                time.sleep(1.0 * task.retry_count)  # simple backoff
            else:
                task.status = "failed"
                task.completed_at = time.time()
        finally:
            timer.cancel()
            self._persist(task)

    def _timeout(self, task_id: str) -> None:
        with self.lock:
            t = self.tasks.get(task_id)
            if t and t.status == "running":
                t.status = "timed_out"
                t.error = f"timeout after {t.timeout_sec}s"
                t.completed_at = time.time()
                self._persist(t)

    def _refresh_blocked(self) -> None:
        """Promote blocked tasks whose dependencies have completed."""
        with self.lock:
            for t in self._queue:
                if t.status == "blocked" and self._dependencies_met(t):
                    t.status = "pending"
                    self._persist(t)

    def shutdown(self) -> None:
        self._shutdown.set()
