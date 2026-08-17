"""Evaluator — output quality validation + iteration caps (spec §6.4).

Checks are deterministic and offline (file presence, Python parse,
secret scan via Security). The engineering pipeline and orchestrator use
these scores to decide retry vs give-up, and a CircuitBreaker stops
repeated failure loops instead of grinding forever.
"""
from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from pathlib import Path

from ultra.security import Security


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
        """Score a generated project on completeness/correctness/safety."""
        py_files = [f for f in project_dir.rglob("*.py") if f.is_file()]
        checks: dict[str, dict] = {}

        # completeness — real .py files with substance
        substantial = [f for f in py_files
                       if len(f.read_text(encoding="utf-8", errors="ignore")) > 50]
        completeness = len(substantial) / max(len(py_files), 1) if py_files else 0.0
        checks["completeness"] = {"ok": completeness >= 0.5,
                                  "score": completeness,
                                  "detail": f"{len(substantial)}/{len(py_files)} substantial files"}

        # correctness — files parse
        parsed = 0
        for f in py_files:
            try:
                ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
                parsed += 1
            except SyntaxError:
                pass
        correctness = parsed / len(py_files) if py_files else 0.0
        checks["correctness"] = {"ok": correctness >= 0.8, "score": correctness,
                                 "detail": f"{parsed}/{len(py_files)} files parse"}

        # safety — secret scan
        bundle = "\n".join(
            f.read_text(encoding="utf-8", errors="ignore") for f in py_files)
        scan = self.security.scan_code(bundle, "python")
        safety = 0.0 if scan.has_critical() else 1.0
        checks["safety"] = {"ok": not scan.has_critical(), "score": safety,
                            "detail": scan.summary()}

        score = (completeness + correctness + safety) / 3.0
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
