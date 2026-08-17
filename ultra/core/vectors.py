"""Semantic (vector) memory — Ollama embeddings stored in SQLite.

No ChromaDB needed: embeddings are cheap JSON blobs, and cosine search over
a few thousand vectors is fast enough on any laptop. This keeps ARIA
local-only and dependency-free.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ultra.llm import OllamaClient, cosine_similarity


class VectorStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )""")
        self.conn.commit()

    def add(self, collection: str, content: str, client: OllamaClient) -> bool:
        try:
            vec = client.embed(content)
        except Exception:
            return False
        if not vec:
            return False
        self.conn.execute(
            "INSERT INTO vectors (collection, content, embedding) VALUES (?, ?, ?)",
            (collection, content, json.dumps(vec)),
        )
        self.conn.commit()
        return True

    def search(self, collection: str, query: str, client: OllamaClient,
               limit: int = 5) -> list[dict]:
        try:
            qvec = client.embed(query)
        except Exception:
            return []
        if not qvec:
            return []
        rows = self.conn.execute(
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
        if collection:
            return self.conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection = ?", (collection,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
