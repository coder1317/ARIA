"""Skills manager — SKILL.md + skill.json with GitHub install/search.

Skills are folders containing SKILL.md and optionally skill.json.
They get injected into planning prompts so ARIA "knows how" to do
specialized work. Install from GitHub: skill install owner/repo.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("aria.skills")

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
KEY_VALUE = re.compile(r"^(\w[\w\s-]*?):\s*(.+)$", re.MULTILINE)
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    body: str = ""
    # skill.json metadata
    version: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    source_repo: str = ""  # "owner/repo"
    installed_at: str = ""

    @property
    def prompt_fragment(self) -> str:
        header = f"[SKILL: {self.name}] {self.description}"
        if self.version:
            header += f" (v{self.version})"
        return f"{header}\n{self.body[:800]}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "source_repo": self.source_repo,
        }


class SkillManager:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Skill] | None = None

    def _load(self) -> dict[str, Skill]:
        if self._cache is not None:
            return self._cache
        skills: dict[str, Skill] = {}
        if self.skills_dir.is_dir():
            # 1. Load traditional SKILL.md folders (backward compatible)
            for folder in self.skills_dir.iterdir():
                if not folder.is_dir():
                    continue
                skill = self._load_skill(folder)
                if skill:
                    skills[folder.name] = skill
            # 2. Load .md files in subdirectories (new curated skills)
            for md_file in self.skills_dir.rglob("*.md"):
                if md_file.name == "SKILL.md":
                    continue  # already loaded above
                name = md_file.stem  # filename without extension
                if name not in skills:
                    skill = self._load_md_skill(md_file)
                    if skill:
                        skills[name] = skill
        self._cache = skills
        return skills

    def _load_md_skill(self, md_file: Path) -> Skill | None:
        """Load a skill from a standalone .md file."""
        try:
            text = md_file.read_text(encoding="utf-8")
        except OSError:
            return None
        skill = self._parse(md_file.stem, text)
        # Derive category from parent directory
        parent = md_file.parent.name
        if parent != "skills":
            skill.tags = [parent]
            skill.source_repo = f"skills/{parent}"
        return skill

    def _load_skill(self, folder: Path) -> Skill | None:
        """Load a skill from a folder (SKILL.md + optional skill.json)."""
        meta = {}
        skill_json = folder / "skill.json"
        if skill_json.is_file():
            try:
                meta = json.loads(skill_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Bad skill.json in {folder.name}: {e}")

        skill_file = folder / "SKILL.md"
        if not skill_file.is_file():
            # no SKILL.md — use skill.json description as body
            if meta:
                return Skill(
                    name=meta.get("name", folder.name),
                    description=meta.get("description", ""),
                    triggers=meta.get("triggers", []),
                    body=meta.get("description", ""),
                    version=meta.get("version", ""),
                    author=meta.get("author", ""),
                    tags=meta.get("tags", []),
                    source_repo=meta.get("source_repo", ""),
                    installed_at=meta.get("installed_at", ""),
                )
            return None

        text = skill_file.read_text(encoding="utf-8")
        skill = self._parse(folder.name, text)
        # merge skill.json metadata
        if meta:
            skill.version = meta.get("version", skill.version)
            skill.author = meta.get("author", skill.author)
            skill.tags = meta.get("tags", skill.tags)
            skill.source_repo = meta.get("source_repo", skill.source_repo)
            skill.installed_at = meta.get("installed_at", skill.installed_at)
            if not skill.description:
                skill.description = meta.get("description", "")
            if not skill.triggers:
                skill.triggers = meta.get("triggers", [])
        return skill

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

    # ── GitHub install ───────────────────────────────────────────

    def install(self, repo: str, branch: str = "main") -> Skill | None:
        """Install a skill from GitHub. repo = 'owner/repo'."""
        url = f"https://github.com/{repo}.git"
        target = self.skills_dir / repo.split("/")[-1]

        if target.exists():
            logger.info(f"Skill '{repo.split('/')[-1]}' already installed, updating...")
            try:
                subprocess.run(
                    ["git", "-C", str(target), "pull", "--quiet"],
                    capture_output=True, timeout=30,
                )
            except Exception as e:
                logger.warning(f"Git pull failed: {e}")
        else:
            logger.info(f"Installing skill from {repo}...")
            try:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", branch,
                     url, str(target)],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    logger.error(f"Git clone failed: {result.stderr}")
                    return None
            except Exception as e:
                logger.error(f"Git clone failed: {e}")
                return None

        # write/update skill.json with source info
        self._mark_installed(target, repo)
        self._cache = None  # invalidate cache
        skill = self._load_skill(target)
        if skill:
            logger.info(f"Installed: {skill.name} v{skill.version or '?'}")
        return skill

    def uninstall(self, name: str) -> bool:
        """Remove an installed skill."""
        target = self.skills_dir / name
        if not target.exists():
            return False
        import shutil
        shutil.rmtree(target)
        self._cache = None
        return True

    def _mark_installed(self, folder: Path, repo: str) -> None:
        """Write/update skill.json with installation metadata."""
        skill_json = folder / "skill.json"
        meta: dict[str, Any] = {}
        if skill_json.is_file():
            try:
                meta = json.loads(skill_json.read_text(encoding="utf-8"))
            except Exception:
                pass
        from datetime import datetime
        meta["source_repo"] = repo
        meta["installed_at"] = datetime.now().isoformat()
        if "name" not in meta:
            meta["name"] = folder.name
        skill_json.write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

    # ── GitHub search ────────────────────────────────────────────

    @staticmethod
    def search_github(query: str, limit: int = 10) -> list[dict]:
        """Search GitHub for repos tagged 'aria-skill'."""
        try:
            resp = requests.get(
                GITHUB_SEARCH_URL,
                params={
                    "q": f"aria-skill {query}",
                    "sort": "stars",
                    "per_page": limit,
                },
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for repo in data.get("items", []):
                results.append({
                    "repo": repo["full_name"],
                    "description": repo.get("description", ""),
                    "stars": repo.get("stargazers_count", 0),
                    "url": repo.get("html_url", ""),
                    "updated": repo.get("updated_at", ""),
                })
            return results
        except Exception as e:
            logger.warning(f"GitHub search failed: {e}")
            return []
