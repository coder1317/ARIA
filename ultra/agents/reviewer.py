"""Reviewer agent — reviews generated code and returns structured issues."""
from __future__ import annotations

import re
from pathlib import Path

from ultra.llm import OllamaClient
from ultra.persona import system_prompt

SYSTEM = system_prompt("reviewer") + """

You review code for bugs before it runs. Report only REAL problems.

CATEGORIES: bugs, syntax, missing imports, api misuse, security, style.

Output ONLY:
ISSUE: <category> | <filename> | <problem>
...
If the code is fine, output exactly: NO ISSUES"""

ISSUE_RE = re.compile(r"ISSUE:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)")


def review_project(client: OllamaClient, project_dir: Path) -> list[dict]:
    """Review all Python files in a project. Returns list of issue dicts."""
    py_files = sorted(project_dir.rglob("*.py"))
    if not py_files:
        return []
    bundle = "\n\n".join(
        f"=== {f.relative_to(project_dir)} ===\n{f.read_text(encoding='utf-8')[:6000]}"
        for f in py_files
    )
    prompt = f"Review this project:\n\n{bundle}"
    raw = client.generate(prompt, system=SYSTEM, max_tokens=2048, temperature=0.2,
                          model=client.config.coding_model)
    issues = []
    for line in raw.splitlines():
        m = ISSUE_RE.match(line.strip())
        if m:
            issues.append({"category": m.group(1).strip(),
                           "file": m.group(2).strip(),
                           "problem": m.group(3).strip()})
    return issues


def has_issues(issues: list[dict]) -> bool:
    return len(issues) > 0
