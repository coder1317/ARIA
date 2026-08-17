"""Architect agent — turns a description into a concrete file plan.

Returns a JSON list of { filename, description, depends_on } so the coder
can generate each file with full context.
"""
from __future__ import annotations

from ultra.llm import OllamaClient
from ultra.persona import system_prompt

SYSTEM = system_prompt("architect") + """

You design small software projects. Given a request, produce a plan.

RULES:
- Small, complete projects. Prefer Python unless the user asks otherwise.
- Every file gets a one-line purpose.
- Include README.md and requirements.txt / pyproject.toml.
- Keep dependencies minimal and installable (pip).
- Output ONLY JSON: {"files": [{"filename": "...", "description": "...", "depends_on": []}]}
- 3 to 12 files max."""


def plan_project(client: OllamaClient, description: str) -> list[dict]:
    prompt = f"Project request: {description}\n\nReturn the file plan as JSON."
    files: list = []
    for attempt in range(3):
        temp = 0.3 if attempt == 0 else 0.1
        result = client.json(prompt, system=SYSTEM, max_tokens=2048,
                             temperature=temp,
                             model=client.config.coding_model)
        files = (result or {}).get("files", [])
        if len(files) >= 3:
            break
    # sanitize: only dicts with a filename string
    clean = []
    for f in files:
        if isinstance(f, dict) and isinstance(f.get("filename"), str):
            clean.append({
                "filename": f["filename"].lstrip("/"),
                "description": str(f.get("description", "")),
                "depends_on": f.get("depends_on", []),
            })
    if not clean:
        # fallback plan so the pipeline never dies
        clean = [{
            "filename": "app.py",
            "description": "Main application entry point",
            "depends_on": [],
        }, {
            "filename": "README.md",
            "description": "Project overview and usage",
            "depends_on": [],
        }, {
            "filename": "requirements.txt",
            "description": "Python dependencies",
            "depends_on": [],
        }]
    return clean
