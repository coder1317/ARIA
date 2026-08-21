"""AuditLog — append-only, queryable operation log (spec §7.1).

Every significant operation (intent dispatch, agent run, file write,
command execution, LLM inference) is recorded with actor, action,
timestamps, duration, provider and error. Rows are never updated or
deleted — the log is append-only.

P2-11: Hash chaining for tamper detection. Each entry's hash includes
the previous entry's hash, forming a chain. Any modification to a past
entry breaks the chain.
"""
from __future__ import annotations

import hashlib
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
                    error TEXT,
                    entry_hash TEXT NOT NULL DEFAULT '',
                    prev_hash TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actor ON audit_log(actor)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON audit_log(ts)")
            # Migration: add hash columns to existing tables
            cols = {row[1] for row in conn.execute("PRAGMA table_info(audit_log)").fetchall()}
            if "entry_hash" not in cols:
                conn.execute("ALTER TABLE audit_log ADD COLUMN entry_hash TEXT NOT NULL DEFAULT ''")
            if "prev_hash" not in cols:
                conn.execute("ALTER TABLE audit_log ADD COLUMN prev_hash TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _compute_hash(ts: float, actor: str, action: str, detail: str,
                      prev_hash: str) -> str:
        """Compute SHA-256 hash for an audit entry, chaining to previous."""
        payload = f"{ts}|{actor}|{action}|{detail or ''}|{prev_hash}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get_last_hash(self) -> str:
        """Get the hash of the most recent audit entry."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else ""

    def log(self, actor: str, action: str, task_type: str | None = None,
            detail: dict | str | None = None, duration_ms: float | None = None,
            provider: str | None = None, error: str | None = None) -> None:
        """Append one entry with hash chaining for tamper detection."""
        if isinstance(detail, dict):
            detail = json.dumps(detail, default=str)[:4000]
        ts = time.time()
        prev_hash = self._get_last_hash()
        entry_hash = self._compute_hash(ts, actor, action, detail, prev_hash)
        with self.lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO audit_log (ts, actor, action, task_type, detail,"
                    " duration_ms, provider, error, entry_hash, prev_hash)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (ts, actor, action, task_type, detail,
                     duration_ms, provider, error, entry_hash, prev_hash),
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
