#!/usr/bin/env python3
"""ARIA — autonomous multi-agent engineering assistant.

Usage:
    aria                       interactive REPL
    aria "build a cli tool"   one-shot (takes input from argv)
    aria --api [port]          run the FastAPI server instead
"""
from __future__ import annotations

import sys
from pathlib import Path

from rich.markdown import Markdown

from ultra.audit import AuditLog
from ultra.config import Config
from ultra.core.memory import Memory
from ultra.core.skills import SkillManager
from ultra.core.vectors import VectorStore
from ultra.display import banner, console, error, info, label, ok, warn
from ultra.orchestrator import Orchestrator
from ultra.provider_pool import ProviderPool
from ultra.security import Security
from ultra.task_manager import TaskManager
from ultra.tools.terminal import Terminal

BANNER_TITLE = "ARIA — Autonomous Multi-Agent Assistant"
BANNER_SUBTITLE = "ProviderPool · Research · Build · Market · Deploy · Orchestrate · Memory · Audit"

HELP = """[bold cyan]Commands[/bold cyan]
  [white]help[/white]                 this menu
  [white]status[/white]               provider health + memory stats
  [white]mode 1|2|3|auto[/white]      research / build / orchestrate / auto
  [white]bash [cmd][/white]           run a shell command (safety-checked)
  [white]read [file][/white]          read a file
  [white]save [name][/white]          save last report to markdown
  [white]memory[/white]               show memory stats + facts
  [white]memory fact key: val[/white] save a fact about you
  [white]memory recall [q][/white]    search past conversations
  [white]skill list[/white]           list skills
  [white]skill extract [project][/white]  learn a skill from a project
  [white]tasks[/white]               list background tasks
  [white]audit [n][/white]           show last n audit-log entries
  [white]api [port][/white]          start the FastAPI server (8000)
  [white]model list[/white]           show pulled Ollama models
  [white]exit[/white]                 quit

[bold cyan]Just type naturally:[/bold cyan]
  "compare Flask and FastAPI"          → research
  "build a todo CLI app"               → engineering pipeline
  "market analysis for ai dev tools"   → market intelligence
  "deploy the last project to docker"  → deployment configs
  "research and build a chat app"      → full pipeline
  "fix the code in ~/myapp"            → improve mode
"""

WORK_MODES = {"1": "research", "research": "research",
              "2": "build", "plan": "build", "build": "build",
              "3": "orchestrate", "orchestrate": "orchestrate"}


class AriaCLI:
    def __init__(self, config: Config):
        self.config = config
        config.ensure_dirs()
        self.client = ProviderPool(config)
        self.security = Security(config.security_enabled)
        self.terminal = Terminal(config)
        self.memory = Memory(config.data_dir / "memory" / "ultra.db")
        self.vectors = VectorStore(config.data_dir / "memory" / "vectors.db")
        self.skills = SkillManager(Path(__file__).resolve().parent.parent / "skills")
        self.audit = AuditLog(config.audit_db)
        self.tasks = TaskManager(config.data_dir / "memory" / "tasks.db",
                                 max_concurrent=config.max_concurrent_tasks,
                                 default_timeout=config.task_timeout_sec)
        self.orch = Orchestrator(self.client, config, self.memory,
                                 self.vectors, self.skills, self.terminal,
                                 security=self.security, audit=self.audit,
                                 tasks=self.tasks)
        self.work_mode = None  # None=auto, 'research', 'build', 'orchestrate'
        self.last_report = ""

    # ── startup ─────────────────────────────────────────────────────

    def start(self) -> None:
        banner(BANNER_TITLE, BANNER_SUBTITLE)
        if not self.client.ping(retries=3):
            warn("Ollama not reachable.")
            info("Start it:  ollama serve   (or:  systemctl --user start ollama)")
            info(f"Then pull:  ollama pull {self.config.chat_model} && "
                 f"ollama pull {self.config.fallback_model}")
            return
        ok("Ollama connected")
        models = self.client.available_models()
        missing = [m for m in (self.config.chat_model, self.config.fallback_model)
                   if m not in models]
        if missing:
            warn("Models not pulled yet:")
            for m in missing:
                info(f"  ollama pull {m}")
        else:
            label("Models:", ", ".join(models[:6]))
        label("Providers:", ", ".join(
            f"{n}({s.model})" for n, s in self.config.providers.items()) if hasattr(
                self.config.providers, "items") else ", ".join(
                f"{p.name}({p.model})" for p in self.config.providers))
        label("Memory:", f"{self.memory.stats()['conversations']} messages · "
                         f"{self.vectors.count()} vectors · "
                         f"{self.audit.stats()['total']} audit entries")
        console.print()

    # ── main loop ───────────────────────────────────────────────────

    def run(self, one_shot: str | None = None) -> None:
        self.start()
        if one_shot:
            self.handle(one_shot)
            self._shutdown()
            return
        try:
            while True:
                try:
                    mode = f"[{self.work_mode}]" if self.work_mode else ""
                    prompt = input(f"\nARIA{mode} > ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\nGoodbye.")
                    self._print_report()
                    break
                if not prompt:
                    continue
                try:
                    self.handle(prompt)
                except Exception as e:
                    error(str(e))
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        try:
            self.tasks.shutdown()
        except Exception:
            pass

    def handle(self, prompt: str) -> None:
        low = prompt.lower().strip()

        if low in ("exit", "quit", "bye"):
            self._print_report()
            self._shutdown()
            sys.exit(0)
        if low in ("help", "?"):
            console.print(Markdown(HELP))
            return
        if low == "status":
            self._status()
            return
        if low.startswith("mode "):
            self._mode(prompt[5:].strip())
            return
        if low.startswith("bash "):
            self._bash(prompt[5:].strip())
            return
        if low.startswith("read "):
            self._read(prompt[5:].strip())
            return
        if low == "save" or low.startswith("save "):
            self._save(prompt[5:].strip())
            return
        if low.startswith("memory"):
            self._memory(prompt[6:].strip())
            return
        if low.startswith("skill "):
            self._skill(prompt[6:].strip())
            return
        if low.startswith("audit"):
            self._audit(prompt[5:].strip())
            return
        if low == "tasks":
            self._tasks()
            return
        if low.startswith("status "):
            self._task_status(prompt[7:].strip())
            return
        if low.startswith("api"):
            self._api(prompt[3:].strip())
            return
        if low == "model list":
            self._models()
            return

        # work-mode override
        if self.work_mode == "research":
            prompt = "research " + prompt
        elif self.work_mode == "build":
            prompt = "build " + prompt
        elif self.work_mode == "orchestrate":
            self._orchestrate(prompt)
            return

        self._dispatch(prompt)

    # ── handlers ────────────────────────────────────────────────────

    def _dispatch(self, text: str) -> None:
        self.last_report = self.orch.dispatch(text)
        if self.last_report and len(self.last_report) > 5:
            console.print()
            console.print(Markdown(self.last_report[:6000]))

    def _status(self) -> None:
        ok(f"Ollama: {self.config.ollama_url}")
        label("Mode:", self.work_mode or "auto")
        console.print("\n[bold cyan]Provider health[/bold cyan]")
        for name, h in self.client.health_report().items():
            color = {"healthy": "green", "degraded": "yellow",
                     "unhealthy": "red", "circuit_open": "red"}.get(h["health"], "white")
            console.print(
                f"  {name:<10} [{color}]{h['health']}[/{color}]  "
                f"{h['model']}  success {h['success_rate']:.0%}  "
                f"latency {h['avg_latency_ms']}ms  calls {h['calls']}")
            if h["last_error"]:
                info(f"    last error: {h['last_error'][:120]}")
        stats = self.memory.stats()
        info(f"memory: {stats['conversations']} msgs · {stats['facts']} facts · "
             f"{stats['projects']} projects · {stats['lessons']} lessons")
        build_circuit = self.orch.status()["build_circuit"]
        info(f"build circuit: {build_circuit}")

    def _mode(self, arg: str) -> None:
        mode = WORK_MODES.get(arg.lower())
        if mode is None:
            warn(f"unknown mode '{arg}' — try 1, 2, 3, or auto")
            return
        self.work_mode = mode if mode != "auto" else None
        ok(f"mode set to {self.work_mode or 'auto'}")

    def _bash(self, command: str) -> None:
        result = self.terminal.run(command)
        self.audit.log(actor="user", action="bash", detail=command[:200],
                       error=None if result.ok else result.summary())
        if result.blocked:
            warn(result.reason)
        elif result.output:
            console.print(result.output)
        if not result.ok and not result.blocked:
            warn(result.summary())

    def _read(self, path: str) -> None:
        p = Path(path).expanduser()
        if not p.exists():
            warn(f"no such file: {path}")
            return
        from ultra.tools.editor import Editor
        content = Editor.read(str(p))
        if content:
            console.print(content[:5000])

    def _save(self, name: str) -> None:
        if not self.last_report:
            warn("nothing to save yet")
            return
        out = self.config.projects_dir / "reports"
        out.mkdir(parents=True, exist_ok=True)
        target = out / (name or "report") if name else out / "report.md"
        if name and not name.endswith(".md"):
            target = out / f"{name}.md"
        target.write_text(self.last_report, encoding="utf-8")
        ok(f"saved to {target}")

    def _memory(self, arg: str) -> None:
        low = arg.lower()
        if not low:
            stats = self.memory.stats()
            info(f"messages: {stats['conversations']} · facts: {stats['facts']} · "
                 f"projects: {stats['projects']} · lessons: {stats['lessons']}")
            facts = self.memory.get_facts()
            if facts:
                info("facts:")
                for k, v in facts.items():
                    info(f"  {k}: {v}")
            return
        if low.startswith("fact "):
            rest = arg[5:].strip()
            if ":" in rest:
                key, _, value = rest.partition(":")
                self.memory.set_fact(key.strip(), value.strip())
                ok(f"fact saved: {key.strip()}")
            else:
                warn("usage: memory fact key: value")
            return
        if low.startswith("recall "):
            query = arg[7:].strip()
            results = self.memory.search(query)
            if not results:
                warn("no matches")
                return
            for r in results[:5]:
                console.print(f"  [dim]{r['created_at']}[/dim] {r['content'][:200]}")
            return
        warn("usage: memory | memory fact key: value | memory recall query")

    def _skill(self, arg: str) -> None:
        if arg == "list":
            skills = self.skills.list()
            if not skills:
                warn("no skills loaded")
                return
            for s in skills:
                label(s.name, s.description)
            return
        if arg.startswith("show "):
            skill = self.skills.get(arg[5:].strip())
            if skill:
                console.print(skill.body[:2000])
            else:
                warn("skill not found")
            return
        if arg.startswith("extract"):
            target = arg[8:].strip()
            path = Path(target).expanduser() if target else self.orch.last_project
            if path is None or not Path(path).is_dir():
                warn("usage: skill extract <project-dir> (or build something first)")
                return
            name = self.orch.trainer.extract_skill(Path(path))
            if name:
                ok(f"extracted skill: {name}")
            return
        warn("usage: skill list | skill show <name> | skill extract <project>")

    def _audit(self, arg: str) -> None:
        try:
            n = int(arg.strip() or "10")
        except ValueError:
            n = 10
        entries = self.audit.query(limit=n)
        if not entries:
            warn("audit log is empty")
            return
        for e in entries:
            detail = ""
            if e.get("detail"):
                detail = f"  [dim]{str(e['detail'])[:80]}[/dim]"
            console.print(
                f"  [{e['when']}] {e['actor']:<14} {e['action']:<16} "
                f"{e.get('task_type') or ''}{detail}")
            if e.get("error"):
                console.print(f"    [red]error: {e['error'][:120]}[/red]")

    def _tasks(self) -> None:
        tasks = self.tasks.list_tasks()
        if not tasks:
            warn("no tasks in queue")
            return
        for t in tasks:
            color = {"completed": "green", "running": "yellow",
                     "failed": "red", "timeout": "red"}.get(t["status"], "white")
            console.print(f"  {t['task_id']}  [{color}]{t['status']}[/{color}]  "
                          f"{t['task_type']}  retries={t['retries']}")
            if t["error"]:
                console.print(f"    [red]{t['error'][:120]}[/red]")

    def _task_status(self, task_id: str) -> None:
        t = self.tasks.get(task_id.strip())
        if t is None:
            warn(f"no task {task_id}")
            return
        label("task:", t.task_id)
        label("type:", t.task_type)
        label("status:", t.status)
        if t.error:
            info(f"error: {t.error}")
        if t.result:
            console.print(str(t.result)[:800])

    def _api(self, port: str) -> None:
        try:
            port = int(port.strip() or "8000")
        except ValueError:
            port = 8000
        try:
            import uvicorn  # noqa: F401
        except ImportError:
            warn("FastAPI not installed — run: pip install fastapi uvicorn")
            return
        from ultra.api import make_app
        import os
        host = os.getenv("ARIA4_API_HOST", "127.0.0.1").strip()
        token = os.getenv("ARIA4_API_TOKEN", "").strip()
        if token:
            ok(f"starting API on http://{host}:{port} (bearer auth enabled)")
        else:
            ok(f"starting API on http://{host}:{port} (no auth — set ARIA4_API_TOKEN for protection)")
        uvicorn.run(make_app(self.orch, self.config), host=host, port=port)

    def _models(self) -> None:
        models = self.client.available_models()
        if not models:
            warn("no models pulled — run: ollama pull granite4.1:3b")
            return
        for m in models:
            console.print(f"  {m}")

    def _orchestrate(self, text: str) -> None:
        """Decompose into background tasks via the TaskManager."""
        result = self.orch.orchestrate(text)
        console.print(result)

    def _print_report(self) -> None:
        self.orch.report.print()


def main() -> None:
    config = Config.load()
    if "--api" in sys.argv:
        idx = sys.argv.index("--api")
        port = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "8000"
        AriaCLI(config)._api(port)
        return
    cli = AriaCLI(config)
    one_shot = " ".join(sys.argv[1:]).strip() or None
    cli.run(one_shot)


if __name__ == "__main__":
    main()
