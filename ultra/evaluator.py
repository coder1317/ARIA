"""Evaluator — output quality validation + iteration caps (spec §6.4).

Checks: file presence/completeness, full Python compile, a runtime
smoke test of the entry point, and a secret scan via Security. The
engineering pipeline and orchestrator use these scores to decide retry
vs give-up, and a CircuitBreaker stops repeated failure loops instead of
grinding forever.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ultra.security import Security

SMOKE_TIMEOUT = 15  # seconds


def _compile_ok(source: str) -> bool:
    """Full Python compile — stricter than ast.parse (catches e.g.
    duplicate arguments, invalid awaits at compile time)."""
    try:
        compile(source, "<generated>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def _entry_point(project_dir: Path) -> Path | None:
    """Find the most likely CLI entry point."""
    for name in ("main.py", "app.py", "cli.py", "__main__.py"):
        p = project_dir / name
        if p.is_file():
            return p
    for p in sorted(project_dir.rglob("*.py")):
        try:
            if 'if __name__ == "__main__"' in p.read_text(
                    encoding="utf-8", errors="ignore"):
                return p
        except OSError:
            continue
    return None


@dataclass
class EvalResult:
    score: float            # 0..1
    passed: bool
    checks: dict[str, dict] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"score {self.score:.2f}"]
        for name, c in self.checks.items():
            parts.append(f"{name}={'ok' if c['ok'] else '✗'}")
        return " · ".join(parts)


class Evaluator:
    def __init__(self, security: Security | None = None, threshold: float = 0.6):
        self.security = security or Security()
        self.threshold = threshold

    def evaluate_project(self, project_dir: Path) -> EvalResult:
        """Score a generated project on completeness/correctness/runtime/safety.

        "Files parse" is not enough — the entry point is actually executed
        (--help) so a project that crashes on startup is scored down.
        """
        py_files = [f for f in project_dir.rglob("*.py") if f.is_file()]
        checks: dict[str, dict] = {}

        # completeness — real .py files with substance
        substantial = [f for f in py_files
                       if len(f.read_text(encoding="utf-8", errors="ignore")) > 50]
        completeness = len(substantial) / max(len(py_files), 1) if py_files else 0.0
        checks["completeness"] = {"ok": completeness >= 0.5,
                                  "score": completeness,
                                  "detail": f"{len(substantial)}/{len(py_files)} substantial files"}

        # correctness — full compile of every file
        compiled = 0
        for f in py_files:
            try:
                src = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if _compile_ok(src):
                compiled += 1
        correctness = compiled / len(py_files) if py_files else 0.0
        checks["correctness"] = {"ok": correctness >= 0.8, "score": correctness,
                                 "detail": f"{compiled}/{len(py_files)} files compile"}

        # safety — secret scan (before smoke: never execute risky code)
        bundle = "\n".join(
            f.read_text(encoding="utf-8", errors="ignore") for f in py_files)
        scan = self.security.scan_code(bundle, "python")
        safety = 0.0 if scan.has_critical() else 1.0
        checks["safety"] = {"ok": not scan.has_critical(), "score": safety,
                            "detail": scan.summary()}

        # runtime smoke — actually run the entry point
        if scan.has_critical():
            smoke = 0.0
            checks["smoke"] = {"ok": False, "score": smoke,
                                "detail": "skipped — project has critical security findings"}
        else:
            entry = _entry_point(project_dir)
            if entry is None:
                smoke = 0.0
                checks["smoke"] = {"ok": False, "score": smoke,
                                    "detail": "no entry point detected"}
            else:
                smoke, detail = _run_smoke(entry, project_dir)
                checks["smoke"] = {"ok": smoke >= 0.8, "score": smoke,
                                    "detail": detail}

        # P1-10: Multi-dimensional evaluation
        # requirements — does it have README, requirements.txt/pyproject.toml?
        has_readme = any(f.name.lower() == "readme.md" for f in project_dir.iterdir())
        has_deps = any(f.name in ("requirements.txt", "pyproject.toml", "setup.py")
                       for f in project_dir.iterdir())
        has_tests = any("test" in f.name.lower() and f.suffix == ".py"
                        for f in project_dir.rglob("*.py"))
        requirements = (
            (0.4 if has_readme else 0.0) +
            (0.3 if has_deps else 0.0) +
            (0.3 if has_tests else 0.0)
        )
        checks["requirements"] = {
            "ok": requirements >= 0.4,
            "score": requirements,
            "detail": f"readme={'yes' if has_readme else 'no'} deps={'yes' if has_deps else 'no'} tests={'yes' if has_tests else 'no'}"
        }

        # code quality — average file size (substantial != trivial)
        sizes = []
        for f in py_files:
            try:
                sizes.append(len(f.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                pass
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        quality = min(1.0, avg_size / 200)  # 200+ chars avg = full score
        checks["quality"] = {
            "ok": quality >= 0.3,
            "score": quality,
            "detail": f"avg file size: {avg_size:.0f} chars"
        }

        # weighted scoring: runtime is strongest, then correctness, safety, completeness
        score = (0.10 * completeness + 0.20 * correctness
                 + 0.30 * smoke + 0.20 * safety
                 + 0.10 * requirements + 0.10 * quality)
        return EvalResult(score=score, passed=score >= self.threshold, checks=checks)


class CircuitBreaker:
    """Stops repeated failure loops: N failures → open for cooldown."""

    def __init__(self, name: str, max_failures: int = 3,
                 cooldown_sec: float = 120.0):
        self.name = name
        self.max_failures = max_failures
        self.cooldown_sec = cooldown_sec
        self.failures = 0
        self.open_until = 0.0

    @property
    def is_open(self) -> bool:
        if time.time() < self.open_until:
            return True
        if self.open_until and time.time() >= self.open_until:
            self.failures = 0
            self.open_until = 0.0
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.open_until = 0.0

    def record_failure(self) -> bool:
        """Record a failure; returns True if the breaker just tripped."""
        self.failures += 1
        if self.failures >= self.max_failures and not self.open_until:
            self.open_until = time.time() + self.cooldown_sec
            return True
        return False

    def remaining(self) -> int:
        if self.open_until:
            return max(1, int(self.open_until - time.time()))
        return self.max_failures - self.failures


def _run_smoke(entry: Path, project_dir: Path) -> tuple[float, str]:
    """Run `<entry> --help` in the project dir; score 1.0 on clean exit.

    Timeout is reported honestly (could be a server that stays up) and a
    non-zero exit reports the first stderr lines. Never passes silently.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(entry), "--help"],
            capture_output=True, text=True, timeout=SMOKE_TIMEOUT,
            cwd=str(project_dir),
        )
    except subprocess.TimeoutExpired:
        return 0.0, (f"entry {entry.name} did not exit within "
                     f"{SMOKE_TIMEOUT}s (may be a server — verify manually)")
    except OSError as e:
        return 0.0, f"could not run {entry.name}: {e}"
    if proc.returncode == 0:
        return 1.0, f"{entry.name} --help ran cleanly"
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return 0.0, f"{entry.name} --help exited {proc.returncode}: " + \
        " ".join(err[:2])[:160]
