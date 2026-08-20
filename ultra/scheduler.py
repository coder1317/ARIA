"""Scheduler — cron-style autonomous task execution.

Stores scheduled tasks in SQLite, runs them in background threads,
and sends results to a configured channel (CLI, Telegram, etc.).

Supports:
  - One-shot (at a specific time)
  - Recurring (every N minutes/hours/days, or cron-like)
  - Persistent across restarts (SQLite-backed)
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("aria.scheduler")


@dataclass
class ScheduledTask:
    id: int | None = None
    name: str = ""
    command: str = ""
    schedule_type: str = "once"  # "once" | "interval" | "daily"
    interval_seconds: int = 0
    daily_time: str = ""  # "HH:MM"
    enabled: bool = True
    last_run: str = ""
    next_run: str = ""
    last_result: str = ""
    channel: str = "cli"  # which channel to send results to
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "command": self.command,
            "schedule_type": self.schedule_type,
            "interval_seconds": self.interval_seconds,
            "daily_time": self.daily_time, "enabled": self.enabled,
            "last_run": self.last_run, "next_run": self.next_run,
            "last_result": self.last_result[:200],
            "channel": self.channel,
        }


class Scheduler:
    """Persistent task scheduler with background execution."""

    def __init__(self, db_path: Path | str, dispatch_fn: Callable | None = None):
        """
        Args:
            db_path: SQLite database path for persistence.
            dispatch_fn: Function to call with (command, channel) when a task fires.
                         Should return the result string. If None, tasks are logged only.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.dispatch_fn = dispatch_fn
        self._running = False
        self._thread: threading.Thread | None = None
        self._conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                command TEXT NOT NULL,
                schedule_type TEXT NOT NULL DEFAULT 'once',
                interval_seconds INTEGER DEFAULT 0,
                daily_time TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1,
                last_run TEXT DEFAULT '',
                next_run TEXT DEFAULT '',
                last_result TEXT DEFAULT '',
                channel TEXT DEFAULT 'cli',
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        self._conn.commit()

    # ── task management ──────────────────────────────────────────

    def add(self, name: str, command: str, schedule_type: str = "once",
            interval_seconds: int = 0, daily_time: str = "",
            channel: str = "cli") -> ScheduledTask:
        """Add a new scheduled task."""
        next_run = self._compute_next_run(schedule_type, interval_seconds, daily_time)
        cur = self._conn.execute(
            """INSERT INTO scheduled_tasks
               (name, command, schedule_type, interval_seconds, daily_time, next_run, channel)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, command, schedule_type, interval_seconds, daily_time, next_run, channel),
        )
        self._conn.commit()
        return ScheduledTask(
            id=cur.lastrowid, name=name, command=command,
            schedule_type=schedule_type, interval_seconds=interval_seconds,
            daily_time=daily_time, next_run=next_run, channel=channel,
        )

    def remove(self, task_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def enable(self, task_id: int) -> bool:
        cur = self._conn.execute(
            "UPDATE scheduled_tasks SET enabled = 1 WHERE id = ?", (task_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def disable(self, task_id: int) -> bool:
        cur = self._conn.execute(
            "UPDATE scheduled_tasks SET enabled = 0 WHERE id = ?", (task_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_tasks(self) -> list[ScheduledTask]:
        rows = self._conn.execute(
            "SELECT * FROM scheduled_tasks ORDER BY next_run").fetchall()
        return [self._row_to_task(r) for r in rows]

    def get(self, task_id: int) -> ScheduledTask | None:
        row = self._conn.execute(
            "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def _row_to_task(self, row: sqlite3.Row) -> ScheduledTask:
        return ScheduledTask(
            id=row["id"], name=row["name"], command=row["command"],
            schedule_type=row["schedule_type"],
            interval_seconds=row["interval_seconds"],
            daily_time=row["daily_time"],
            enabled=bool(row["enabled"]),
            last_run=row["last_run"], next_run=row["next_run"],
            last_result=row["last_result"],
            channel=row["channel"],
            created_at=row["created_at"],
        )

    # ── scheduling logic ─────────────────────────────────────────

    def _compute_next_run(self, schedule_type: str, interval_seconds: int,
                          daily_time: str) -> str:
        now = datetime.now()
        if schedule_type == "once":
            return (now + timedelta(seconds=max(interval_seconds, 60))).isoformat()
        elif schedule_type == "interval":
            return (now + timedelta(seconds=max(interval_seconds, 60))).isoformat()
        elif schedule_type == "daily" and daily_time:
            try:
                h, m = map(int, daily_time.split(":"))
                target = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if target <= now:
                    target += timedelta(days=1)
                return target.isoformat()
            except ValueError:
                return (now + timedelta(hours=1)).isoformat()
        return (now + timedelta(hours=1)).isoformat()

    def _update_next_run(self, task: ScheduledTask) -> None:
        next_run = self._compute_next_run(
            task.schedule_type, task.interval_seconds, task.daily_time)
        self._conn.execute(
            "UPDATE scheduled_tasks SET next_run = ? WHERE id = ?",
            (next_run, task.id))
        self._conn.commit()

    # ── execution ────────────────────────────────────────────────

    def _execute_task(self, task: ScheduledTask) -> str:
        """Execute a task and return the result."""
        logger.info(f"Executing scheduled task: {task.name} ({task.command})")
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE scheduled_tasks SET last_run = ? WHERE id = ?", (now, task.id))
        self._conn.commit()

        result = ""
        try:
            if self.dispatch_fn:
                result = self.dispatch_fn(task.command, task.channel)
            else:
                result = f"[scheduler] Would execute: {task.command}"
        except Exception as e:
            result = f"[scheduler] Error: {e}"
            logger.error(f"Task {task.name} failed: {e}")

        self._conn.execute(
            "UPDATE scheduled_tasks SET last_result = ? WHERE id = ?",
            (result[:1000], task.id))
        self._conn.commit()
        return result

    def run_pending(self) -> list[tuple[int, str, str]]:
        """Check for due tasks and execute them. Returns [(task_id, name, result)]."""
        now = datetime.now().isoformat()
        rows = self._conn.execute(
            "SELECT * FROM scheduled_tasks WHERE enabled = 1 AND next_run <= ?",
            (now,)).fetchall()

        results = []
        for row in rows:
            task = self._row_to_task(row)
            result = self._execute_task(task)
            self._update_next_run(task)
            # for one-shot tasks, disable after execution
            if task.schedule_type == "once":
                self._conn.execute(
                    "UPDATE scheduled_tasks SET enabled = 0 WHERE id = ?", (task.id,))
                self._conn.commit()
            results.append((task.id, task.name, result))
        return results

    # ── background loop ──────────────────────────────────────────

    def start(self, check_interval: int = 30) -> None:
        """Start the background scheduler loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, args=(check_interval,), daemon=True)
        self._thread.start()
        logger.info(f"Scheduler started (checking every {check_interval}s)")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Scheduler stopped")

    def _loop(self, interval: int) -> None:
        while self._running:
            try:
                fired = self.run_pending()
                for task_id, name, result in fired:
                    logger.info(f"Scheduled task '{name}' completed: {result[:100]}")
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
            time.sleep(interval)

    def shutdown(self) -> None:
        self.stop()
        try:
            self._conn.close()
        except Exception:
            pass
