"""Security — defense-in-depth validation for ARIA Ultra.

Two independent gates, per the spec:
  1. validate_input()  — runs BEFORE any LLM call (prompt-injection defense)
  2. scan_code()       — runs BEFORE generated files are trusted
     (hardcoded secrets, dangerous patterns)

A failure in one gate does not affect the other.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

# Prompt-injection patterns (from the Ultra spec §4.6)
INJECTION_PATTERNS = [
    r"(?i)(ignore\s+(?:all\s+|any\s+|your\s+)?previous|disregard\s+all|forget\s+everything)",
    r"(?i)(you\s+are\s+now\s+|system\s*:\s*you\s+are)",
    r"(?i)(</instruction>|</prompt>|\[\[\[)",
    r"(?i)(DAN|jailbreak|mode\s*:\s*developer)",
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]",  # control characters
]

# Dangerous code patterns → severity
DANGEROUS_PATTERNS = {
    "os.system": "critical",
    "subprocess.call": "critical",
    "eval(": "critical",
    "exec(": "critical",
    "__import__": "critical",
    "pickle.loads": "high",
    "yaml.load": "high",
    "input(": "medium",
}

# Hardcoded secrets → critical
SECRET_PATTERNS = [
    (r"sk-[a-zA-Z0-9-]{20,}", "OpenAI API key"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub token"),
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"AIza[0-9A-Za-z_-]{30,}", "Google API key"),
    (r"xox[baprs]-[0-9A-Za-z-]{10,}", "Slack token"),
]


@dataclass
class SecurityReport:
    passed: bool
    findings: list[dict] = field(default_factory=list)
    severity: str = "clean"  # clean | low | medium | high | critical

    def has_critical(self) -> bool:
        return any(f["severity"] == "critical" for f in self.findings)

    def summary(self) -> str:
        if not self.findings:
            return "clean"
        return f"{self.severity}: {len(self.findings)} finding(s) — " + \
            "; ".join(f["detail"][:80] for f in self.findings[:3])


def _calculate_severity(findings: list[dict]) -> str:
    if not findings:
        return "clean"
    sevs = {f["severity"] for f in findings}
    for sev in ("critical", "high", "medium"):
        if sev in sevs:
            return sev
    return "low"


class Security:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.max_prompt_length = 100_000

    # ── gate 1: input validation ────────────────────────────────────

    def validate_input(self, text: str) -> SecurityReport:
        """Reject prompt-injection attempts before any LLM call."""
        if not self.enabled:
            return SecurityReport(passed=True)
        findings: list[dict] = []
        if len(text) > self.max_prompt_length:
            findings.append({"type": "length_violation", "severity": "medium",
                             "detail": f"input {len(text)} chars > {self.max_prompt_length}"})
        for pattern in INJECTION_PATTERNS:
            for m in re.finditer(pattern, text):
                findings.append({"type": "injection_attempt", "severity": "critical",
                                 "detail": f"pattern: {m.group(0)[:50]}"})
        severity = _calculate_severity(findings)
        return SecurityReport(passed=severity not in ("critical", "high"),
                              findings=findings, severity=severity)

    # ── gate 2: generated-code scanning ─────────────────────────────

    def scan_code(self, code: str, language: str = "python") -> SecurityReport:
        """Scan generated code for secrets and dangerous patterns."""
        if not self.enabled:
            return SecurityReport(passed=True)
        findings: list[dict] = []
        for pattern, name in SECRET_PATTERNS:
            for m in re.finditer(pattern, code):
                findings.append({"type": "hardcoded_secret", "severity": "critical",
                                 "detail": f"potential {name}: {m.group(0)[:20]}..."})
        for pattern, sev in DANGEROUS_PATTERNS.items():
            if pattern in code:
                findings.append({"type": "dangerous_pattern", "severity": sev,
                                 "detail": f"pattern: {pattern}"})
        if language == "python":
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                            and node.func.id in ("eval", "exec"):
                        findings.append({"type": "dangerous_call", "severity": "critical",
                                         "detail": f"direct call to {node.func.id}()"})
            except SyntaxError:
                findings.append({"type": "syntax_error", "severity": "low",
                                 "detail": "code has syntax errors — cannot fully analyze"})
        severity = _calculate_severity(findings)
        return SecurityReport(passed=severity not in ("high", "critical"),
                              findings=findings, severity=severity)

    def scan_project(self, files: dict[str, str]) -> SecurityReport:
        """Scan a whole generated project {filename: content}."""
        findings: list[dict] = []
        for fname, content in files.items():
            report = self.scan_code(content, "python" if fname.endswith(".py") else "text")
            for f in report.findings:
                f["file"] = fname
            findings.extend(report.findings)
        severity = _calculate_severity(findings)
        return SecurityReport(passed=severity not in ("high", "critical"),
                              findings=findings, severity=severity)
