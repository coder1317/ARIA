"""Phase 2 Memory Intelligence — episodic, procedural, user model.

Extends the base Memory with three new memory types:

1. Episodic:  "What happened?" — task history, events, conversations with context
2. Procedural: "How do I do this?" — skill execution patterns, workflows
3. User Model: "Who is Hari?" — preferences, goals, profile, working style

Also provides `retrieve_context()` which ranks and injects relevant memories
into the chat prompt — the key missing piece from Phase 1.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ultra.core.memory import Memory
from ultra.core.vectors import VectorStore
from ultra.llm import OllamaClient


# ── Data models ──────────────────────────────────────────────────────

@dataclass
class Episode:
    """A single episodic memory — what happened and when."""
    event_type: str           # "task_completed", "conversation", "decision", "error", "milestone"
    summary: str              # human-readable description
    detail: str = ""          # full context
    project: str = ""         # which project this relates to
    outcome: str = "success"  # success | failure | partial
    duration_ms: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    timestamp: str = ""
    importance: float = 0.5   # 0.0 = trivial, 1.0 = critical
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec="seconds")


@dataclass
class Procedure:
    """A procedural memory — how to accomplish a specific task."""
    name: str                 # e.g. "research_embedded_vision"
    description: str          # what this procedure does
    steps: list[str]          # ordered list of steps
    tools_used: list[str] = field(default_factory=list)
    success_count: int = 0
    fail_count: int = 0
    last_used: str = ""
    avg_duration_ms: float = 0.0
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5   # 0.0 = unreliable, 1.0 = battle-tested

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.5

    def record_result(self, success: bool, duration_ms: float = 0.0) -> None:
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
        self.last_used = datetime.now().isoformat(timespec="seconds")
        # Update running average
        total = self.success_count + self.fail_count
        if total == 1:
            self.avg_duration_ms = duration_ms
        else:
            self.avg_duration_ms = (
                self.avg_duration_ms * (total - 1) + duration_ms
            ) / total
        # Update confidence based on success rate
        self.confidence = min(0.95, self.success_rate * 0.9 + 0.1)


@dataclass
class UserProfile:
    """Structured user model — preferences, goals, profile."""
    # Identity
    name: str = ""
    timezone: str = "Asia/Kolkata"
    os: str = "Ubuntu"

    # Preferences
    preferred_language: str = "Python"
    preferred_editor: str = "VS Code"
    preferred_model: str = "lfm2.5"
    response_style: str = "concise"  # concise | detailed | adaptive

    # Technical profile
    hardware: list[str] = field(default_factory=list)  # ["ESP32", "Arduino Uno", "RPi 4"]
    skills: list[str] = field(default_factory=list)     # ["embedded", "python", "pcb-design"]

    # Goals
    active_goals: list[str] = field(default_factory=list)
    long_term_goals: list[str] = field(default_factory=list)

    # Working style
    work_hours: str = "flexible"
    auto_approve_level: str = "moderate"  # low | moderate | high
    verbosity: str = "normal"  # quiet | normal | verbose

    # Metadata
    last_updated: str = ""
    facts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        # Filter out unknown keys
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ── Enhanced Memory ─────────────────────────────────────────────────

class MemoryV2:
    """Extended memory with episodic, procedural, and user model.

    Wraps the base Memory and VectorStore, adding:
    - Episodic event log with importance scoring
    - Procedural skill patterns with success tracking
    - User profile with preferences and goals
    - Context retrieval that ranks memories for chat injection
    """

    def __init__(self, db_path: Path, vectors: VectorStore | None = None):
        self.base = Memory(db_path)
        self.vectors = vectors
        self._local = sqlite3.connect(str(db_path))
        self._local.row_factory = sqlite3.Row
        self._local.execute("PRAGMA journal_mode=WAL")
        self._local.execute("PRAGMA busy_timeout=5000")
        self._init_phase2_schema()

    def _init_phase2_schema(self) -> None:
        self._local.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            detail TEXT DEFAULT '',
            project TEXT DEFAULT '',
            outcome TEXT DEFAULT 'success',
            duration_ms REAL DEFAULT 0,
            tools_used TEXT DEFAULT '[]',
            importance REAL DEFAULT 0.5,
            tags TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS procedures (
            name TEXT PRIMARY KEY,
            description TEXT DEFAULT '',
            steps TEXT DEFAULT '[]',
            tools_used TEXT DEFAULT '[]',
            success_count INTEGER DEFAULT 0,
            fail_count INTEGER DEFAULT 0,
            last_used TEXT DEFAULT '',
            avg_duration_ms REAL DEFAULT 0,
            tags TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.5
        );

        CREATE TABLE IF NOT EXISTS user_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(event_type);
        CREATE INDEX IF NOT EXISTS idx_episodes_project ON episodes(project);
        CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance DESC);
        """)
        self._local.commit()

    # ── Episodic Memory ─────────────────────────────────────────────

    def record_episode(self, episode: Episode) -> int:
        """Record what happened — an event in ARIA's life."""
        cur = self._local.execute(
            "INSERT INTO episodes (event_type, summary, detail, project, "
            "outcome, duration_ms, tools_used, importance, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (episode.event_type, episode.summary, episode.detail,
             episode.project, episode.outcome, episode.duration_ms,
             json.dumps(episode.tools_used), episode.importance,
             json.dumps(episode.tags)),
        )
        self._local.commit()
        eid = cur.lastrowid

        # Also store in vectors for semantic search
        if self.vectors:
            text = f"{episode.event_type}: {episode.summary}"
            if episode.detail:
                text += f" — {episode.detail[:500]}"
            self.vectors.add("episodes", text, self._get_embed_client())

        return eid

    def get_episodes(self, event_type: str | None = None,
                     project: str | None = None,
                     limit: int = 20,
                     min_importance: float = 0.0) -> list[dict]:
        """Retrieve episodic memories with optional filters."""
        query = "SELECT * FROM episodes WHERE importance >= ?"
        params: list[Any] = [min_importance]
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if project:
            query += " AND project = ?"
            params.append(project)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._local.execute(query, params).fetchall()
        return [self._episode_to_dict(r) for r in rows]

    def search_episodes(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic search over episodic memories."""
        if not self.vectors:
            return self.get_episodes(limit=limit)
        results = self.vectors.search("episodes", query, self._get_embed_client(), limit=limit)
        # Enrich with full episode data
        enriched = []
        for r in results:
            rows = self._local.execute(
                "SELECT * FROM episodes WHERE summary LIKE ? LIMIT 1",
                (f"%{r['content'][:50]}%",),
            ).fetchall()
            if rows:
                enriched.append(self._episode_to_dict(rows[0]))
            else:
                enriched.append({"summary": r["content"], "score": r["score"]})
        return enriched

    def recent_episodes(self, days: int = 7, limit: int = 20) -> list[dict]:
        """Episodes from the last N days."""
        rows = self._local.execute(
            "SELECT * FROM episodes WHERE created_at >= datetime('now', ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (f"-{days} days", limit),
        ).fetchall()
        return [self._episode_to_dict(r) for r in rows]

    def _episode_to_dict(self, row) -> dict:
        d = dict(row)
        d["tools_used"] = json.loads(d.get("tools_used", "[]"))
        d["tags"] = json.loads(d.get("tags", "[]"))
        return d

    # ── Procedural Memory ───────────────────────────────────────────

    def save_procedure(self, proc: Procedure) -> None:
        """Save or update a procedural pattern."""
        self._local.execute(
            "INSERT OR REPLACE INTO procedures "
            "(name, description, steps, tools_used, success_count, fail_count, "
            "last_used, avg_duration_ms, tags, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (proc.name, proc.description, json.dumps(proc.steps),
             json.dumps(proc.tools_used), proc.success_count, proc.fail_count,
             proc.last_used, proc.avg_duration_ms, json.dumps(proc.tags),
             proc.confidence),
        )
        self._local.commit()

    def get_procedure(self, name: str) -> Procedure | None:
        row = self._local.execute(
            "SELECT * FROM procedures WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            return None
        return self._procedure_from_row(row)

    def search_procedures(self, query: str, tags: list[str] | None = None,
                          limit: int = 5) -> list[Procedure]:
        """Find procedures matching a query or tags."""
        # Try semantic search first
        if self.vectors:
            results = self.vectors.search("procedures", query, self._get_embed_client(), limit=limit)
            procs = []
            for r in results:
                proc = self.get_procedure(r["content"].split(":")[0].strip())
                if proc:
                    procs.append(proc)
            if procs:
                return procs

        # Fallback: tag-based search
        if tags:
            placeholders = ",".join("?" for _ in tags)
            rows = self._local.execute(
                f"SELECT * FROM procedures WHERE tags LIKE ? "
                f"ORDER BY confidence DESC LIMIT ?",
                (f"%{tags[0]}%", limit),
            ).fetchall()
            return [self._procedure_from_row(r) for r in rows]

        # Last resort: all procedures sorted by confidence
        rows = self._local.execute(
            "SELECT * FROM procedures ORDER BY confidence DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._procedure_from_row(r) for r in rows]

    def record_procedure_result(self, name: str, success: bool,
                                duration_ms: float = 0.0) -> None:
        """Update a procedure's success stats after execution."""
        proc = self.get_procedure(name)
        if proc:
            proc.record_result(success, duration_ms)
            self.save_procedure(proc)

    def list_procedures(self, limit: int = 20) -> list[dict]:
        rows = self._local.execute(
            "SELECT name, description, success_count, fail_count, confidence, "
            "last_used FROM procedures ORDER BY confidence DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _procedure_from_row(self, row) -> Procedure:
        return Procedure(
            name=row["name"],
            description=row["description"],
            steps=json.loads(row["steps"]),
            tools_used=json.loads(row["tools_used"]),
            success_count=row["success_count"],
            fail_count=row["fail_count"],
            last_used=row["last_used"],
            avg_duration_ms=row["avg_duration_ms"],
            tags=json.loads(row["tags"]),
            confidence=row["confidence"],
        )

    # ── User Model ──────────────────────────────────────────────────

    def get_user_profile(self) -> UserProfile:
        """Load the user profile from the database."""
        rows = self._local.execute("SELECT key, value FROM user_profile").fetchall()
        if not rows:
            return UserProfile()
        data = {}
        for r in rows:
            key, value = r["key"], r["value"]
            # Try to parse JSON values (lists, dicts)
            try:
                data[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                data[key] = value
        return UserProfile.from_dict(data)

    def set_user_profile(self, profile: UserProfile) -> None:
        """Save the entire user profile."""
        data = profile.to_dict()
        data["last_updated"] = datetime.now().isoformat(timespec="seconds")
        for key, value in data.items():
            self._local.execute(
                "INSERT OR REPLACE INTO user_profile (key, value, updated_at) "
                "VALUES (?, ?, datetime('now'))",
                (key, json.dumps(value) if isinstance(value, (list, dict)) else str(value)),
            )
        self._local.commit()

    def update_user_fact(self, key: str, value: str) -> None:
        """Update a single user profile field."""
        self._local.execute(
            "INSERT OR REPLACE INTO user_profile (key, value, updated_at) "
            "VALUES (?, ?, datetime('now'))",
            (key, value),
        )
        self._local.commit()

    # ── Context Retrieval (the key Phase 2 feature) ─────────────────

    def retrieve_context(self, query: str, client: OllamaClient,
                         max_tokens: int = 2000) -> str:
        """Build a context block from relevant memories for chat injection.

        This is the core Phase 2 feature: before sending a user message to
        the LLM, we retrieve relevant episodic memories, procedures, user
        facts, and project context to enrich the prompt.

        Returns a formatted string suitable for injection into the system prompt.
        """
        sections = []

        # 1. User profile (always include)
        profile = self.get_user_profile()
        profile_lines = []
        if profile.name:
            profile_lines.append(f"User: {profile.name}")
        if profile.preferred_language:
            profile_lines.append(f"Preferred language: {profile.preferred_language}")
        if profile.active_goals:
            profile_lines.append(f"Active goals: {', '.join(profile.active_goals[:3])}")
        if profile.hardware:
            profile_lines.append(f"Hardware: {', '.join(profile.hardware[:5])}")
        if profile.skills:
            profile_lines.append(f"Skills: {', '.join(profile.skills[:5])}")
        if profile.response_style:
            profile_lines.append(f"Response style: {profile.response_style}")
        if profile.facts:
            for k, v in list(profile.facts.items())[:5]:
                profile_lines.append(f"{k}: {v}")
        if profile_lines:
            sections.append("## User Profile\n" + "\n".join(profile_lines))

        # 2. Relevant episodic memories
        episodes = self.search_episodes(query, limit=3)
        if episodes:
            ep_lines = []
            for ep in episodes:
                outcome = ep.get("outcome", "")
                icon = "✓" if outcome == "success" else "✗" if outcome == "failure" else "•"
                ep_lines.append(f"{icon} [{ep.get('event_type', '?')}] {ep.get('summary', '')[:150]}")
            sections.append("## Relevant Past Events\n" + "\n".join(ep_lines))

        # 3. Relevant procedures
        procs = self.search_procedures(query, limit=2)
        if procs:
            proc_lines = []
            for p in procs:
                conf = f"{p.confidence:.0%}" if p.confidence else "?"
                proc_lines.append(f"- {p.name}: {p.description[:100]} (confidence: {conf})")
            sections.append("## Known Procedures\n" + "\n".join(proc_lines))

        # 4. Recent projects
        projects = self.base.get_projects(limit=3)
        if projects:
            proj_lines = [f"- {p['name']}: {p.get('problem', '')[:80]}" for p in projects]
            sections.append("## Recent Projects\n" + "\n".join(proj_lines))

        # 5. Relevant facts
        facts = self.base.get_facts()
        if facts:
            fact_lines = [f"- {k}: {v}" for k, v in list(facts.items())[:5]]
            sections.append("## Known Facts\n" + "\n".join(fact_lines))

        # 6. Lessons (what went wrong before)
        lessons = self.base.get_lessons(limit=3)
        if lessons:
            lesson_lines = [f"- {l['what_went_wrong'][:80]} → {l['better_approach'][:80]}"
                           for l in lessons]
            sections.append("## Lessons Learned\n" + "\n".join(lesson_lines))

        if not sections:
            return ""

        result = "\n\n".join(sections)
        # Truncate if too long
        if len(result) > max_tokens * 4:  # rough char estimate
            result = result[:max_tokens * 4] + "\n[...truncated]"
        return result

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_embed_client(self) -> OllamaClient:
        """Get an OllamaClient for embedding (reuse if possible)."""
        if not hasattr(self, "_embed_client"):
            self._embed_client = OllamaClient()
        return self._embed_client

    def stats(self) -> dict:
        """Extended stats including Phase 2 tables."""
        base_stats = self.base.stats()
        base_stats["episodes"] = self._local.execute(
            "SELECT COUNT(*) FROM episodes"
        ).fetchone()[0]
        base_stats["procedures"] = self._local.execute(
            "SELECT COUNT(*) FROM procedures"
        ).fetchone()[0]
        base_stats["user_profile_fields"] = self._local.execute(
            "SELECT COUNT(*) FROM user_profile"
        ).fetchone()[0]
        return base_stats
