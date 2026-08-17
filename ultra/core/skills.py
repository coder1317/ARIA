"""Skills manager — SKILL.md files with YAML-ish frontmatter.

Skills are folders containing SKILL.md; they get injected into planning
prompts so ARIA "knows how" to do specialized work. Drop a folder to
extend ARIA — same pattern as Prototype 1 / OpenClaw.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
KEY_VALUE = re.compile(r"^(\w[\w\s-]*?):\s*(.+)$", re.MULTILINE)


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    body: str = ""

    @property
    def prompt_fragment(self) -> str:
        return f"[SKILL: {self.name}] {self.description}\n{self.body[:800]}"


class SkillManager:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self._cache: dict[str, Skill] | None = None

    def _load(self) -> dict[str, Skill]:
        if self._cache is not None:
            return self._cache
        skills: dict[str, Skill] = {}
        if self.skills_dir.is_dir():
            for folder in self.skills_dir.iterdir():
                skill_file = folder / "SKILL.md"
                if not skill_file.is_file():
                    continue
                text = skill_file.read_text(encoding="utf-8")
                skills[folder.name] = self._parse(folder.name, text)
        self._cache = skills
        return skills

    @staticmethod
    def _parse(name: str, text: str) -> Skill:
        fm = FRONTMATTER.match(text)
        description, triggers = "", []
        body = text
        if fm:
            meta = dict(KEY_VALUE.findall(fm.group(1)))
            description = meta.get("description", "")
            triggers = [t.strip() for t in meta.get("triggers", "").split(",") if t.strip()]
            body = text[fm.end():].strip()
        return Skill(name=name, description=description,
                     triggers=triggers, body=body)

    def list(self) -> list[Skill]:
        return sorted(self._load().values(), key=lambda s: s.name)

    def get(self, name: str) -> Skill | None:
        return self._load().get(name)

    def for_request(self, text: str) -> list[Skill]:
        """Skills whose triggers match the request text."""
        low = text.lower()
        return [s for s in self._load().values()
                if any(t in low for t in s.triggers)]

    def context(self, text: str = "") -> str:
        """All skills (or matching ones) formatted for prompt injection."""
        matched = self.for_request(text) if text else self.list()
        if not matched:
            return ""
        return "\n\n".join(s.prompt_fragment for s in matched)
