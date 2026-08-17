"""Trainer agent — skill extraction from completed projects (spec §5.9).

After a successful build, the trainer reads the project, asks the model
for the reusable patterns/techniques it demonstrated, and writes them as
a SKILL.md into the skills library — so future planning prompts include
what ARIA learned. This closes the self-improvement loop.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from ultra.config import Config
from ultra.display import info, ok, warn
from ultra.llm import OllamaClient
from ultra.persona import system_prompt

SYSTEM = system_prompt("knowledge engineer") + """

You extract reusable skills from a project. A skill captures patterns,
techniques and pitfalls that would help future similar projects.

Output ONLY this exact format:
SKILL NAME: <short snake_case name>
DESCRIPTION: <one sentence: when to use this skill>
TRIGGERS: <comma-separated keywords that should activate this skill>
BODY:
- <pattern 1: what to do, with a short code/pseudo example if useful>
- <pitfall 1: what to avoid>
..."""


class TrainerAgent:
    def __init__(self, client: OllamaClient, config: Config):
        self.client = client
        self.config = config
        self.skills_dir = config.data_dir / "skills"  # learned skills live here

    def extract_skill(self, project_dir: Path, description: str = "") -> str | None:
        """Extract a skill from a project. Returns the skill name or None."""
        py_files = sorted(project_dir.rglob("*.py"))
        if not py_files:
            return None
        bundle = "\n\n".join(
            f"=== {f.relative_to(project_dir)} ===\n"
            f"{f.read_text(encoding='utf-8', errors='ignore')[:4000]}"
            for f in py_files[:6]
        )
        prompt = f"""Project: {project_dir.name}
{('Request: ' + description) if description else ''}

Project files:
{bundle}

Extract the reusable skill using the exact output format."""
        raw = self.client.generate(prompt, system=SYSTEM, max_tokens=2048,
                                   temperature=0.3, task_type="research")
        skill = self._parse(raw)
        if not skill:
            warn("trainer: could not parse a skill from the response")
            return None
        return self._write_skill(skill)

    # ── parsing ─────────────────────────────────────────────────────

    def _parse(self, raw: str) -> dict | None:
        lines = raw.splitlines()
        out: dict[str, str] = {}
        current: str | None = None
        body: list[str] = []
        for line in lines:
            s = line.strip()
            low = s.lower()
            if low.startswith("skill name:"):
                current = "name"
                out["name"] = s.split(":", 1)[1].strip()
            elif low.startswith("description:"):
                current = "description"
                out["description"] = s.split(":", 1)[1].strip()
            elif low.startswith("triggers:"):
                current = "triggers"
                out["triggers"] = s.split(":", 1)[1].strip()
            elif low.startswith("body:"):
                current = "body"
            elif current == "body" and s:
                body.append(s)
            elif current in ("name", "description", "triggers") and s:
                out[current] += " " + s
        if not out.get("name") or not out.get("description"):
            return None
        out["body"] = "\n".join(body[:30])
        return out

    def _write_skill(self, skill: dict) -> str:
        name = re.sub(r"[^a-z0-9_]+", "-", skill["name"].lower()).strip("-")
        if not name:
            name = f"skill-{int(time.time())}"
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        triggers = skill.get("triggers", "")
        body = skill.get("body", "")
        content = f"""---
name: {name}
description: {skill['description']}
triggers: {triggers}
---

{body}
"""
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        ok(f"learned skill '{name}' → {skill_dir / 'SKILL.md'}")
        return name
