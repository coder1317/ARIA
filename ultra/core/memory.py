"""Persistent memory — SQLite with FTS5 full-text search.

Stores conversations, facts, projects, interactions, lessons, and skills.
Vector (semantic) search lives in `vectors.py` on top of this DB.

Thread-safe: uses threading.local() for per-thread connections.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class Memory:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._local = threading.local()
        # Initialize schema on main thread
        conn = self._get_conn()
        # Don't close - keep thread-local connection open

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
            self._init_schema(conn)
        return self._local.conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        c = conn
        c.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(content);
        CREATE TABLE IF NOT EXISTS facts (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT,
            problem TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            intent TEXT,
            success INTEGER DEFAULT 1,
            failure_reason TEXT,
            duration_ms REAL
        );
        CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT,
            what_went_wrong TEXT,
            better_approach TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            applied_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            trigger_phrases TEXT,
            steps TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """)
        conn.commit()

    # ── conversations ───────────────────────────────────────────────

    def add_message(self, role: str, content: str, embed: bool = True) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO conversations (role, content) VALUES (?, ?)", (role, content)
        )
        rowid = cur.lastrowid
        conn.execute(
            "INSERT INTO conversations_fts (rowid, content) VALUES (?, ?)",
            (rowid, content),
        )
        conn.commit()
        return rowid

    def recent(self, limit: int = 6) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def thread(self, limit: int = 6) -> list[dict]:
        """Messages in OpenAI-style format for the /api/chat endpoint."""
        return [{"role": r["role"], "content": r["content"]} for r in self.recent(limit)]

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search over past conversation content."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT c.content, c.created_at "
                "FROM conversations_fts i JOIN conversations c ON i.rowid = c.id "
                "WHERE conversations_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS5 syntax error (e.g. punctuation in query) — fall back to LIKE
            rows = conn.execute(
                "SELECT content, created_at FROM conversations "
                "WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── facts ───────────────────────────────────────────────────────

    def set_fact(self, key: str, value: str, category: str = "general") -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO facts (key, value, category) VALUES (?, ?, ?)",
            (key.lower().strip(), value, category),
        )
        conn.commit()

    def get_facts(self, category: str | None = None) -> dict[str, str]:
        conn = self._get_conn()
        if category:
            rows = conn.execute(
                "SELECT key, value FROM facts WHERE category = ?", (category,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT key, value FROM facts").fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ── projects ────────────────────────────────────────────────────

    def save_project(self, name: str, path: str, problem: str = "") -> str:
        pid = f"{int(time.time())}-{abs(hash(name)) % 10000}"
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO projects (id, name, path, problem) VALUES (?, ?, ?, ?)",
            (pid, name, path, problem),
        )
        conn.commit()
        return pid

    def get_projects(self, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT name, path, problem, created_at FROM projects "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── interactions (telemetry) ────────────────────────────────────

    def log_interaction(self, user_input: str, intent: str, success: bool,
                        duration_ms: float, failure: str = "") -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO interactions (timestamp, user_input, intent, success, "
            "failure_reason, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), user_input, intent,
             int(success), failure, duration_ms),
        )
        conn.commit()

    def stats(self) -> dict:
        conn = self._get_conn()
        counts = {}
        for table in ("conversations", "facts", "projects", "interactions", "lessons", "skills"):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return counts

    # ── lessons (self-improvement) ──────────────────────────────────

    def add_lesson(self, agent: str, what: str, better: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO lessons (agent, what_went_wrong, better_approach) VALUES (?, ?, ?)",
            (agent, what, better),
        )
        conn.commit()

    def get_lessons(self, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT agent, what_went_wrong, better_approach, created_at "
            "FROM lessons ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── skills ──────────────────────────────────────────────────────

    def save_skill(self, name: str, triggers: list[str], steps: list[str]) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO skills (name, trigger_phrases, steps) VALUES (?, ?, ?)",
            (name, json.dumps(triggers), json.dumps(steps)),
        )
        conn.commit()

    def get_skills(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute("SELECT name, trigger_phrases, steps FROM skills").fetchall()
        return [dict(r) for r in rows]
