"""Engineering pipeline — the core "Design & Build" mode.

architect → coder → reviewer → debugger → test loop (up to 2 test-fix
passes), then README + git init. This is the v3/v4 pipeline distilled
into ~150 lines with a local model in mind.
"""
from __future__ import annotations

from pathlib import Path

from ultra.agents import architect, coder, debugger, reviewer
from ultra.agents.tester import run_tests
from ultra.config import Config
from ultra.display import info, ok, step, warn
from ultra.llm import OllamaClient
from ultra.security import Security
from ultra.tools.terminal import Terminal

MAX_TEST_FIX_PASSES = 2


class EngineeringPipeline:
    def __init__(self, client: OllamaClient, config: Config, terminal: Terminal):
        self.client = client
        self.config = config
        self.terminal = terminal
        self.security = Security(config.security_enabled)

    def build(self, description: str, project_dir: Path | None = None,
              report=None) -> Path:
        if project_dir is None:
            project_dir = self.config.projects_dir / _slug(description)
        project_dir.mkdir(parents=True, exist_ok=True)

        # quality gate: a weak model occasionally generates something that
        # parses but isn't a real project — restart the whole generation
        generated = False
        for attempt in range(1, 4):
            for f in project_dir.iterdir():
                if f.is_file():
                    f.unlink()
            step(1, f"Architecture planning (attempt {attempt})")
            plan = architect.plan_project(self.client, description)
            info(f"{len(plan)} files planned")

            step(2, "Code generation")
            written = coder.generate_files(self.client, project_dir, description, plan)
            ok(f"{len(written)} files written")

            good, why = _quality_gate(project_dir)
            if good:
                generated = True
                break
            warn(f"generation failed quality gate: {why} — retrying")
        if not generated:
            warn("could not generate a valid project after 3 attempts — continuing anyway")

        step(3, "Review")
        issues = reviewer.review_project(self.client, project_dir)
        # security scan findings join the review queue so the debugger
        # can actually fix them (eval/exec/os.system → rewritten without)
        security_issues = self._security_findings(project_dir)
        if security_issues:
            warn(f"{len(security_issues)} security finding(s) from code scan")
            issues.extend(security_issues)
        if issues:
            warn(f"{len(issues)} issues found")
        else:
            ok("no issues found")

        step(4, "Fix loop")
        all_clear, passes = debugger.fix_issues(self.client, project_dir, issues)
        if all_clear:
            ok(f"clean after {passes} fix pass(es)")
        else:
            warn("some issues remain after max fix passes")

        step(5, "Test loop")
        for pass_no in range(1, MAX_TEST_FIX_PASSES + 1):
            result = run_tests(project_dir, self.terminal)
            if result is None:
                info("no test framework detected — skipping")
                break
            if result.ok:
                ok("tests pass")
                break
            warn(f"tests failing (pass {pass_no}): {result.summary()}")
            if pass_no == MAX_TEST_FIX_PASSES:
                break
            # feed test output back to the debugger, targeting the files
            # that actually appear in the traceback
            from ultra.agents.debugger import fix_issues
            synthetic = _issues_from_traceback(result.output, project_dir)
            fix_issues(self.client, project_dir, synthetic)

        step(6, "README + git")
        self._ensure_readme(project_dir, description)
        self._git_init(project_dir)
        ok(f"project ready at {project_dir}")
        if report:
            report.project(project_dir, description)
        return project_dir

    # ── helpers ─────────────────────────────────────────────────────

    def _security_findings(self, project_dir: Path) -> list[dict]:
        """Scan generated files and convert critical findings to issues."""
        findings = []
        for f in project_dir.rglob("*.py"):
            if not f.is_file():
                continue
            code = f.read_text(encoding="utf-8", errors="ignore")
            report = self.security.scan_code(code, "python")
            if not report.has_critical():
                continue
            rel = f.relative_to(project_dir)
            for finding in report.findings:
                if finding["severity"] == "critical":
                    findings.append({
                        "category": "security",
                        "file": str(rel),
                        "problem": f"[{finding['type']}] {finding['detail']} — "
                                   f"rewrite without this pattern",
                    })
        return findings

    def _ensure_readme(self, project_dir: Path, description: str) -> None:
        readme = project_dir / "README.md"
        if readme.exists():
            return
        text = f"# {description}\n\nLocal-first project generated by ARIA v4.\n"
        readme.write_text(text, encoding="utf-8")

    def _git_init(self, project_dir: Path) -> None:
        if not (project_dir / ".git").exists():
            self.terminal.run(f"cd {project_dir} && git init -q && git add -A && "
                              "git -c user.name=aria -c user.email=aria@local "
                              "commit -qm 'Initial ARIA v4 generation'")

    def improve(self, description: str, path: str | None = None) -> None:
        """improve/fix mode — review an existing project and fix issues."""
        target = Path(path) if path and Path(path).exists() else None
        if target is None:
            # find the most recently touched project as a fallback
            projects = sorted(self.config.projects_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            target = projects[-1] if projects else None
        if target is None:
            warn("no project found to improve")
            return
        step(1, f"Reviewing {target}")
        issues = reviewer.review_project(self.client, target)
        if not issues:
            ok("no issues found")
            return
        warn(f"{len(issues)} issues found")
        step(2, "Fixing")
        all_clear, passes = debugger.fix_issues(self.client, target, issues)
        ok(f"clean after {passes} pass(es)" if all_clear else "best effort applied")


def _slug(text: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "project"


def _quality_gate(project_dir: Path) -> tuple[bool, str]:
    """Cheap structural check that a generated project is plausible.

    - at least one real .py file with substantial content
    - most .py files parse (so the test loop gets a chance)
    - a README or requirements exists
    """
    import ast
    py_files = [f for f in project_dir.rglob("*.py") if f.is_file()]
    if not py_files:
        return False, "no .py files"
    substantial = [f for f in py_files if len(f.read_text(encoding="utf-8", errors="ignore")) > 50]
    if not substantial:
        return False, "python files empty or trivial"
    bad = 0
    for f in py_files:
        try:
            ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            bad += 1
    if bad > len(py_files) // 2:
        return False, f"{bad}/{len(py_files)} files have syntax errors"
    extras = [f for f in project_dir.iterdir()
              if f.is_file() and f.suffix not in (".py", ".md", ".txt", ".json",
                                                   ".toml", ".yaml", ".yml", ".cfg", ".ini")]
    if extras:
        return False, f"weird files: {', '.join(e.name for e in extras[:3])}"
    return True, "ok"


def _issues_from_traceback(output: str, project_dir: Path) -> list[dict]:
    """Turn failing-test output into per-file issues for the debugger.

    Parses `File "...py"` lines from the traceback and creates an issue for
    each file in the project. If nothing can be matched, uses the special
    '(project)' pseudo-file which triggers whole-project repair.
    """
    import re
    files = set(re.findall(r'File "([^"]+\.py)"', output))
    rel_files = []
    for f in files:
        p = Path(f)
        try:
            rel = p.relative_to(project_dir)
        except ValueError:
            continue
        rel_files.append(str(rel))
    if rel_files:
        return [{"file": f, "category": "test",
                 "problem": output[:1000]} for f in sorted(rel_files)]
    return [{"file": "(project)", "category": "test",
             "problem": output[:1000]}]
