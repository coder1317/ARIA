"""File editing with FIND/REPLACE diffs, timestamped backups, and Python
syntax validation — inspired by the diff engine from aria_v3_js and
Claude Code-style editing.
"""
from __future__ import annotations

import ast
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

REPLACE_BLOCK = re.compile(
    r"<<<<<<< SEARCH\n(?P<old>.*?)\n=======\n(?P<new>.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)


@dataclass
class EditResult:
    path: str
    applied: bool
    message: str
    backup: str = ""

    @property
    def ok(self) -> bool:
        return self.applied


def _backup(path: Path) -> str:
    backup_dir = path.parent / ".ultra_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = backup_dir / f"{path.name}.{stamp}.bak"
    shutil.copy2(path, backup)
    return str(backup)


def syntax_ok(path: Path) -> bool:
    """Validate Python syntax without executing."""
    if path.suffix != ".py":
        return True
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        return True
    except SyntaxError:
        return False


class Editor:
    @staticmethod
    def apply_replace(path_str: str, old: str, new: str) -> EditResult:
        """Replace one exact block of text in a file."""
        path = Path(path_str)
        if not path.exists():
            return EditResult(path_str, False, "file does not exist")
        text = path.read_text(encoding="utf-8")
        if old not in text:
            return EditResult(path_str, False, "SEARCH block not found in file")
        if text.count(old) > 1:
            return EditResult(
                path_str, False, "SEARCH block is ambiguous (found more than once)"
            )
        backup = _backup(path)
        new_text = text.replace(old, new, 1)
        path.write_text(new_text, encoding="utf-8")
        if not syntax_ok(path):
            path.write_text(text, encoding="utf-8")  # roll back
            return EditResult(path_str, False, "edit produced invalid Python — rolled back")
        return EditResult(path_str, True, "edit applied", backup)

    @staticmethod
    def apply_diff(path_str: str, diff_text: str) -> EditResult:
        """Apply a SEARCH/REPLACE diff block to a file."""
        match = REPLACE_BLOCK.search(diff_text)
        if not match:
            return EditResult(path_str, False, "no valid SEARCH/REPLACE block found")
        return Editor.apply_replace(path_str, match.group("old"), match.group("new"))

    @staticmethod
    def write_file(path_str: str, content: str, validate: bool = True) -> EditResult:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = _backup(path)
        else:
            backup = ""
        path.write_text(content, encoding="utf-8")
        if validate and not syntax_ok(path):
            # keep the file anyway so the debugger can fix it, but report
            return EditResult(path_str, True, "file written (syntax error — needs fix)", backup)
        return EditResult(path_str, True, "file written", backup)

    @staticmethod
    def read(path_str: str) -> str | None:
        path = Path(path_str)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
