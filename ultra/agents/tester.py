"""Test running for generated projects.

Detects unittest vs pytest from the test files themselves (a file that
imports unittest must NOT be run with pytest), runs with a timeout, and
reports real exit codes. Returns None when the project has no tests.
"""
from __future__ import annotations

from pathlib import Path

from ultra.tools.terminal import Terminal


def detect_framework(project_dir: Path) -> str | None:
    test_files = list(project_dir.rglob("test_*.py")) + \
                 list(project_dir.rglob("*_test.py")) + \
                 list(project_dir.rglob("tests/*.py"))
    if not test_files:
        return None
    for f in test_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "import unittest" in text or "from unittest" in text:
            return "unittest"
    return "pytest"


def run_tests(project_dir: Path, terminal: Terminal):
    framework = detect_framework(project_dir)
    if framework is None:
        return None
    if framework == "pytest":
        return terminal.run(f"cd {project_dir} && python3 -m pytest -q")
    return terminal.run(
        f"cd {project_dir} && python3 -m unittest discover -v"
    )
