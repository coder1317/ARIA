"""Deployer agent — deployment configuration generation (spec §5.8).

Given a built project, generates Dockerfile, docker-compose.yml,
GitHub Actions CI, and a deploy README by reusing the coder's tested
---FILE: parser. Configs are written into the project (deploy/ folder
for CI files, root for Dockerfile/compose).
"""
from __future__ import annotations

import re
from pathlib import Path

from ultra.agents.coder import _parse_files
from ultra.config import Config
from ultra.display import info, ok, step, warn
from ultra.llm import OllamaClient
from ultra.persona import system_prompt
from ultra.tools.editor import Editor

SYSTEM = system_prompt("devops") + """

You generate deployment configuration files. Rules:
- Complete, working configs. No placeholders like <your-key>.
- Match the project's language (Python → python:3.12-slim base image).
- Never embed real secrets — reference env vars only.
- Output format: each file starts on its own line with  ---FILE: <path> ---
  and ends at the next marker (or end of output).
- Paths: Dockerfile and docker-compose.yml at project root; CI under
  .github/workflows/ci.yml; deployment notes in DEPLOY.md."""

PLATFORMS = ("docker", "github_actions", "vercel")


def generate_deployment(client: OllamaClient, project_dir: Path,
                        platform: str = "docker") -> list[Path]:
    """Generate deployment files for a project. Returns written paths."""
    platform = platform if platform in PLATFORMS else "docker"
    py_files = sorted(project_dir.rglob("*.py"))
    lang_hint = "python" if py_files else "unknown"
    sample = ""
    if py_files:
        sample = py_files[0].read_text(encoding="utf-8", errors="ignore")[:1500]

    prompt = f"""Project: {project_dir.name}
Language: {lang_hint}
Platform: {platform}

Main entry sample:
{sample}

Generate the deployment files for {platform} using the ---FILE: <path> --- format.
Begin now."""
    files: dict[str, str] = {}
    for attempt in range(3):
        temp = 0.4 if attempt == 0 else 0.2
        raw = client.generate(prompt, system=SYSTEM, max_tokens=4096,
                              temperature=temp,
                              model=client.config.coding_model,
                              task_type="code")
        files = _parse_files(raw)
        if files:
            break

    written: list[Path] = []
    root = project_dir.resolve()
    for name, content in files.items():
        # sanitize paths: strip traversal, keep files inside the project
        clean = name.strip().lstrip("/\\").replace("..", "")
        path = (project_dir / clean).resolve()
        if not str(path).startswith(str(root)):
            warn(f"deployer: skipping path outside project: {name}")
            continue
        if ".github" in clean:
            path.parent.mkdir(parents=True, exist_ok=True)
        result = Editor.write_file(str(path), content, validate=False)
        if result.ok:
            written.append(path)
    return written
