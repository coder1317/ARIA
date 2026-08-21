"""Semantic (vector) memory — Ollama embeddings stored in SQLite.

No ChromaDB needed: embeddings are cheap JSON blobs, and cosine search over
a few thousand vectors is fast enough on any laptop. This keeps ARIA
local-only and dependency-free.

Thread-safe: each thread gets its own SQLite connection via threading.local().
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ultra.llm import OllamaClient, cosine_similarity


class VectorStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._local = threading.local()
        # Initialize schema on the main thread
        conn = self._get_conn()
        conn.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )""")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return self._local.conn

    def add(self, collection: str, content: str, client: OllamaClient) -> bool:
        try:
            vec = client.embed(content)
        except Exception:
            return False
        if not vec:
            return False
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO vectors (collection, content, embedding) VALUES (?, ?, ?)",
            (collection, content, json.dumps(vec)),
        )
        conn.commit()
        return True

    def search(self, collection: str, query: str, client: OllamaClient,
               limit: int = 5) -> list[dict]:
        try:
            qvec = client.embed(query)
        except Exception:
            return []
        if not qvec:
            return []
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT content, embedding FROM vectors WHERE collection = ?",
            (collection,),
        ).fetchall()
        scored = []
        for r in rows:
            vec = json.loads(r["embedding"])
            sim = cosine_similarity(qvec, vec)
            scored.append((sim, r["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"content": content, "score": round(sim, 4)}
            for sim, content in scored[:limit]
        ]

    def count(self, collection: str | None = None) -> int:
        conn = self._get_conn()
        if collection:
            return conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection = ?", (collection,)
            ).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
