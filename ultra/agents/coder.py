"""Coder agent — writes complete, working file contents.

Uses a ---FILE: marker format for multi-file generation in one call,
which is dramatically cheaper/faster than one call per file on a small
local model.
"""
from __future__ import annotations

import re
from pathlib import Path

from ultra.llm import OllamaClient
from ultra.persona import system_prompt
from ultra.tools.editor import Editor

SYSTEM = system_prompt("coder") + """

You write complete, working code files. The user will run and test them.

RULES:
- Complete implementations only. NO TODO comments, NO stubs, NO placeholders.
- Follow the file plan exactly — every listed filename must appear.
- Correct syntax above all (the code will be parsed).
- Include README.md with install + run instructions.
- Output format: each file starts on its own line with  ---FILE: <path> ---
  and ends at the next marker (or end of output)."""

FILE_MARKER = re.compile(r"---FILE:\s*([^\s]+?)\s*---")
FENCE = re.compile(r"```([a-zA-Z0-9_./\- ]*)\s*\n(.*?)```", re.DOTALL)

# files always worth keeping even if the model didn't list them in the plan
ALWAYS_KEEP = {"README.md", "requirements.txt", "pyproject.toml", "setup.py"}


def _parse_files(raw: str) -> dict[str, str]:
    """Parse multi-file output into {filename: content}.

    Handles two formats small local models actually produce:
      1. ---FILE: <path> ---  ...  (next marker / end)
      2. ```python <path>  ...  ```  (fenced blocks with a filename)
    """
    files: dict[str, str] = {}

    matches = list(FILE_MARKER.finditer(raw))
    if matches:
        for i, m in enumerate(matches):
            name = m.group(1)
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            content = raw[m.end():end]
            # model sometimes puts a fence right after the marker
            content = content.strip("\n")
            if name and content:
                files[name] = content
        return files

    for info, body in FENCE.findall(raw):
        parts = info.split()
        if not parts:
            continue
        name = parts[-1]
        if "." not in name:  # e.g. ```python with no filename — can't place it
            continue
        files[name] = body.strip("\n")
    return files


def generate_files(client: OllamaClient, project_dir: Path,
                   description: str, plan: list[dict]) -> list[Path]:
    plan_text = "\n".join(
        f"- {f['filename']}: {f['description']}" for f in plan
    )
    prompt = f"""Project request: {description}

FILE PLAN:
{plan_text}

Generate every file. Use the ---FILE: <path> --- marker format.
Begin now."""
    # small local models drift — retry with lower temperature until we
    # actually get parseable files (Prototype 1 lesson: assume failure)
    files: dict[str, str] = {}
    for attempt in range(3):
        temp = 0.5 if attempt == 0 else 0.2
        raw = client.generate(prompt, system=SYSTEM, max_tokens=8192,
                              temperature=temp,
                              model=client.config.coding_model)
        files = _parse_files(raw)
        if files:
            break

    planned_names = {f["filename"] for f in plan}
    written: list[Path] = []
    kept: list[str] = []
    for name, content in files.items():
        if name in planned_names or name in ALWAYS_KEEP:
            kept.append(name)
        elif name.endswith((".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".html", ".css", ".js")):
            kept.append(name)  # plausible real file — keep it
        else:
            continue
        path = project_dir / name
        result = Editor.write_file(str(path), content, validate=False)
        if result.ok:
            written.append(path)
    return written
