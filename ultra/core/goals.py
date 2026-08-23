"""Goal Manager — persistent goals for autonomous ARIA operation.

Goals give ARIA long-term direction beyond single prompts. They persist
across sessions and influence ARIA's behavior:
- Active goals shape research and build priorities
- Completed goals build a history of achievements
- Goal progress is tracked via episodes and memory

Goal lifecycle:
  active → in_progress → completed | abandoned
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Goal:
    """A single goal ARIA is working toward."""
    id: int | None = None
    title: str = ""
    description: str = ""
    status: str = "active"  # "active" | "in_progress" | "completed" | "abandoned"
    priority: int = 2  # 0=critical, 1=high, 2=normal, 3=low
    progress: float = 0.0  # 0.0 to 1.0
    tags: list[str] = field(default_factory=list)
    parent_id: int | None = None  # sub-goals
    created_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "description": self.description,
            "status": self.status, "priority": self.priority, "progress": self.progress,
            "tags": self.tags, "parent_id": self.parent_id,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "completed_at": self.completed_at, "notes": self.notes,
        }


class GoalManager:
    """Persistent goal storage with progress tracking.

    Phase 3: Added goal decomposition → task generation pipeline.
    Goals now drive autonomous task creation via the agent runtime.

    Usage:
        goals = GoalManager(db_path)
        goal = goals.add("Build ARIA production-ready", priority=1)
        goals.update_progress(goal.id, 0.5, notes="Phase 1+2 complete")
        goals.complete(goal.id)
        active = goals.list_active()
        tasks = goals.decompose_to_tasks(goal.id, llm_fn)
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                priority INTEGER DEFAULT 2,
                progress REAL DEFAULT 0.0,
                tags TEXT DEFAULT '[]',
                parent_id INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                completed_at TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            )
        """)
        conn.commit()

    def add(self, title: str, description: str = "", priority: int = 2,
            tags: list[str] | None = None, parent_id: int | None = None) -> Goal:
        """Add a new goal."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO goals (title, description, priority, tags, parent_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (title, description, priority, json.dumps(tags or []), parent_id, now, now),
        )
        conn.commit()
        return Goal(
            id=cur.lastrowid, title=title, description=description,
            priority=priority, tags=tags or [], parent_id=parent_id,
            created_at=now, updated_at=now,
        )

    def update_progress(self, goal_id: int, progress: float, notes: str = "") -> bool:
        """Update goal progress (0.0 to 1.0)."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE goals SET progress = ?, notes = ?, updated_at = ?, "
            "status = 'in_progress' WHERE id = ?",
            (min(1.0, max(0.0, progress)), notes, now, goal_id),
        )
        conn.commit()
        return cur.rowcount > 0

    def complete(self, goal_id: int, notes: str = "") -> bool:
        """Mark a goal as completed."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE goals SET status = 'completed', progress = 1.0, "
            "completed_at = ?, notes = ?, updated_at = ? WHERE id = ?",
            (now, notes, now, goal_id),
        )
        conn.commit()
        return cur.rowcount > 0

    def abandon(self, goal_id: int, reason: str = "") -> bool:
        """Abandon a goal."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE goals SET status = 'abandoned', notes = ?, updated_at = ? WHERE id = ?",
            (reason, now, goal_id),
        )
        conn.commit()
        return cur.rowcount > 0

    def list_active(self) -> list[Goal]:
        """Get all active/in_progress goals sorted by priority."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM goals WHERE status IN ('active', 'in_progress') "
            "ORDER BY priority, updated_at DESC"
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def list_completed(self, limit: int = 20) -> list[Goal]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM goals WHERE status = 'completed' "
            "ORDER BY completed_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def get(self, goal_id: int) -> Goal | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
        return self._row_to_goal(row) if row else None

    def stats(self) -> dict[str, int]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM goals GROUP BY status"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def _row_to_goal(self, row: sqlite3.Row) -> Goal:
        return Goal(
            id=row["id"], title=row["title"], description=row["description"],
            status=row["status"], priority=row["priority"], progress=row["progress"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            parent_id=row["parent_id"], created_at=row["created_at"],
            updated_at=row["updated_at"], completed_at=row["completed_at"],
            notes=row["notes"],
        )

    # Phase 3: Goal → Task decomposition

    def decompose_to_tasks(self, goal_id: int, llm_json_fn) -> list[dict]:
        """Use LLM to decompose a goal into actionable tasks.

        Returns list of task dicts: [{"type": str, "objective": str, "priority": int}]
        """
        goal = self.get(goal_id)
        if not goal:
            return []

        system = (
            "You are ARIA's goal decomposer. Break a high-level goal into "
            "concrete, actionable tasks. Each task should be something ARIA "
            "can accomplish using its tools (research, build, review, debug).\n\n"
            "Rules:\n"
            "- 2-5 tasks per goal\n"
            "- Tasks should be ordered by dependency\n"
            "- Use 'research' for information gathering\n"
            "- Use 'build' for creating/implementing\n"
            "- Use 'review' for verification\n"
            "- Each task must have a clear, measurable objective"
        )
        prompt = (
            f"Goal: {goal.title}\n"
            f"Description: {goal.description}\n"
            f"Priority: {goal.priority}\n"
            f"Tags: {', '.join(goal.tags)}\n\n"
            "Decompose into tasks. Return ONLY a JSON array:\n"
            '[{"type": "research"|"build"|"review", '
            '"objective": "specific task description", '
            '"priority": 0-3}]'
        )
        try:
            result = llm_json_fn(prompt, system)
            tasks = result if isinstance(result, list) else []
            valid = []
            for t in tasks:
                if isinstance(t, dict) and t.get("type") in ("research", "build", "review"):
                    if t.get("objective"):
                        valid.append(t)
            return valid[:5]  # cap at 5 tasks
        except Exception:
            return []

    def get_next_task(self, goal_id: int) -> dict | None:
        """Get the next actionable task for a goal.

        Checks what's been done and suggests the next step.
        """
        goal = self.get(goal_id)
        if not goal or goal.status == "completed":
            return None

        # Simple heuristic based on progress
        if goal.progress < 0.2:
            return {"type": "research", "objective": f"Research and analyze: {goal.title}"}
        elif goal.progress < 0.5:
            return {"type": "build", "objective": f"Implement initial prototype for: {goal.title}"}
        elif goal.progress < 0.8:
            return {"type": "review", "objective": f"Review and improve: {goal.title}"}
        else:
            return {"type": "build", "objective": f"Finalize and polish: {goal.title}"}
