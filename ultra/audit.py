"""AuditLog — append-only, queryable operation log (spec §7.1).

Every significant operation (intent dispatch, agent run, file write,
command execution, LLM inference) is recorded with actor, action,
timestamps, duration, provider and error. Rows are never updated or
deleted — the log is append-only.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    task_type TEXT,
                    detail TEXT,
                    duration_ms REAL,
                    provider TEXT,
                    error TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actor ON audit_log(actor)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON audit_log(ts)")

    def log(self, actor: str, action: str, task_type: str | None = None,
            detail: dict | str | None = None, duration_ms: float | None = None,
            provider: str | None = None, error: str | None = None) -> None:
        """Append one entry."""
        if isinstance(detail, dict):
            detail = json.dumps(detail, default=str)[:4000]
        with self.lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO audit_log (ts, actor, action, task_type, detail,"
                    " duration_ms, provider, error) VALUES (?,?,?,?,?,?,?,?)",
                    (time.time(), actor, action, task_type, detail,
                     duration_ms, provider, error),
                )

    def log_inference(self, agent: str, task_type: str, prompt: str,
                      response: str, provider: str | None,
                      duration_ms: float | None = None) -> None:
        self.log(actor=f"agent:{agent}", action="inference", task_type=task_type,
                 detail={"prompt": prompt[:500], "response": response[:500]},
                 duration_ms=duration_ms, provider=provider)

    def query(self, actor: str | None = None, action: str | None = None,
              limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audit_log WHERE 1=1"
        params: list[Any] = []
        if actor:
            sql += " AND actor LIKE ?"
            params.append(f"%{actor}%")
        if action:
            sql += " AND action = ?"
            params.append(action)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.pop("ts")))
            out.append(d)
        return out

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            by_action = dict(conn.execute(
                "SELECT action, COUNT(*) FROM audit_log GROUP BY action").fetchall())
        return {"total": total, "by_action": by_action}
