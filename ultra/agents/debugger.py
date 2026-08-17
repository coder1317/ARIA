"""Debugger agent — fixes reported issues with complete file rewrites.

Modeled on ARIA v3's debugger: max 3 fix passes, full-file generation,
"NO BUGS FOUND" detection. Reuses the coder's tested output parser so a
fix that comes back in either ---FILE: or fenced-block format lands.
"""
from __future__ import annotations

from pathlib import Path

from ultra.agents.coder import _parse_files
from ultra.llm import OllamaClient
from ultra.persona import system_prompt
from ultra.security import Security
from ultra.tools.editor import Editor

SYSTEM = system_prompt("debugger") + """

Fix the reported issues by rewriting the ENTIRE affected file with
correct code. Keep the same filename and overall structure.

Output exactly one file using:
---FILE: <path> ---
<complete corrected file>

If the issues are already fixed, output: NO BUGS FOUND"""

MAX_FIX_PASSES = 3


def fix_issues(client: OllamaClient, project_dir: Path,
               issues: list[dict]) -> tuple[bool, int]:
    """Attempt to fix issues. Returns (all_clear, passes_used)."""
    if not issues:
        return True, 0

    # group issues by file
    by_file: dict[str, list[str]] = {}
    security_files: set[str] = set()
    for issue in issues:
        fname = issue.get("file", "")
        if fname:
            by_file.setdefault(fname, []).append(
                f"[{issue.get('category', '?')}] {issue.get('problem', '')}"
            )
            if issue.get("category") == "security":
                security_files.add(fname)

    for pass_no in range(1, MAX_FIX_PASSES + 1):
        remaining = []
        for fname, problems in by_file.items():
            if fname == "(project)":
                fixed = _repair_project(client, project_dir, problems)
                if not fixed:
                    remaining.append(fname)
                continue
            path = project_dir / fname
            if not path.exists():
                remaining.append(fname)
                continue
            content = path.read_text(encoding="utf-8")
            prompt = (
                f"Fix these issues in {fname}:\n"
                + "\n".join(f"- {p}" for p in problems)
                + f"\n\nCurrent file:\n{content}"
            )
            raw = client.generate(prompt, system=SYSTEM,
                                  max_tokens=8192, temperature=0.2,
                                  model=client.config.coding_model)
            if "NO BUGS FOUND" in raw.upper():
                # never trust a claim of "fixed" — security issues are
                # deterministically checkable, so re-scan before believing
                if _still_has_critical(path):
                    remaining.append(fname)
                continue
            new_files = _parse_files(raw)
            if not new_files:
                remaining.append(fname)
                continue
            fixed_this_file = False
            for name, new_content in new_files.items():
                if name != fname:
                    continue
                if not _better_than(new_content, content):
                    continue  # don't make it worse
                result = Editor.write_file(str(project_dir / name),
                                           new_content, validate=False)
                if result.ok:
                    fixed_this_file = True
            if fixed_this_file and fname in security_files and _still_has_critical(path):
                # written, but the deterministic scanner says it's still
                # critical — the "fix" didn't actually fix the issue
                fixed_this_file = False
            if not fixed_this_file:
                remaining.append(fname)
        by_file = {f: p for f, p in by_file.items() if f in remaining}
        if not by_file:
            return True, pass_no
    return False, MAX_FIX_PASSES


def _repair_project(client: OllamaClient, project_dir: Path,
                    problems: list[str]) -> bool:
    """Whole-project repair: show every file + the failure, fix what's broken.

    Used when the traceback doesn't point at a project file. The model
    returns corrected files in ---FILE: format; all of them are written.
    """
    py_files = sorted(project_dir.rglob("*.py"))
    if not py_files:
        return False
    bundle = "\n\n".join(
        f"=== {f.relative_to(project_dir)} ===\n{f.read_text(encoding='utf-8')[:6000]}"
        for f in py_files
    )
    prompt = (
        "The project fails to import/run. Find the bug(s) and rewrite ONLY "
        "the broken files. Problems:\n"
        + "\n".join(f"- {p}" for p in problems)
        + f"\n\nPROJECT FILES:\n{bundle}"
    )
    raw = client.generate(prompt, system=SYSTEM, max_tokens=8192,
                          temperature=0.2, model=client.config.coding_model)
    if "NO BUGS FOUND" in raw.upper():
        return True
    new_files = _parse_files(raw)
    if not new_files:
        return False
    written = False
    for name, new_content in new_files.items():
        if name in ("app.py",) or any(p.name == name for p in py_files) or "." in name:
            target = project_dir / name
            previous = target.read_text(encoding="utf-8") if target.exists() else ""
            if not _better_than(new_content, previous):
                continue
            result = Editor.write_file(str(target), new_content, validate=False)
            written = written or result.ok
    return written


def _still_has_critical(path: Path) -> bool:
    """Re-scan a file for critical security findings.

    The 3B model will happily declare "NO BUGS FOUND" for patterns it
    thinks are safe (e.g. eval with restricted globals). The scanner is
    deterministic — trust it over the model's self-assessment.
    """
    try:
        code = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True
    return Security().scan_code(code).has_critical()


def _better_than(new_content: str, previous: str) -> bool:
    """A fix must not be worse than what's there — at minimum, it must
    parse if the file is Python. Prevents the debugger from corrupting
    files with truncated/prose-laden output from a weak model."""
    import ast
    if not previous.strip():
        return bool(new_content.strip())
    try:
        ast.parse(new_content)
        return True
    except SyntaxError:
        return False
