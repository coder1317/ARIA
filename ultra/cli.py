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
BANNER_SUBTITLE = "ProviderPool · Research · Build · Market · Deploy · Orchestrate · Memory · Audit · Runtime"

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
  [white]skill list[/white]           list installed skills
  [white]skill install [repo][/white] install from GitHub (owner/repo)
  [white]skill search [q][/white]     search GitHub for aria-skill repos
  [white]skill show [name][/white]    show skill details
  [white]skill extract [project][/white]  learn a skill from a project
  [white]skill uninstall [name][/white]   remove a skill
  [white]tasks[/white]               list background tasks
  [white]audit [n][/white]           show last n audit-log entries
  [white]api [port][/white]          start the FastAPI server (8000)
  [white]model list[/white]           show pulled Ollama models
  [white]model set <name>[/white]     switch active model (e.g. lfm2.5:latest)
  [white]model show[/white]           show current model config
  [white]telegram[/white]             start Telegram bot (needs TELEGRAM_BOT_TOKEN)
  [white]schedule list[/white]        show scheduled tasks
  [white]schedule add ...[/white]     add a scheduled task
  [white]schedule run [id][/white]    run a task now
  [white]mcp status[/white]           show MCP server connections
  [white]browse <url>[/white]         open page in headless browser
  [white]screenshot <url>[/white]     capture page screenshot
  [white]runtime[/white]              toggle Agent Runtime (PLAN→ACT→OBSERVE→EVALUATE)
  [white]mission <objective>[/white]  run a mission through the Agent Runtime
  [white]mission status[/white]       show last mission execution trace
  [white]tools[/white]               list all registered tools
  [white]profile[/white]             show user profile (preferences, goals)
  [white]profile set k v[/white]     update profile field
  [white]profile goal [desc][/white] add an active goal
  [white]episodes[/white]            view episodic memories (what happened)
  [white]episodes search [q][/white] search past events
  [white]procedures[/white]          view procedural memories (how to do things)
  [white]procedures search [q][/white] search known procedures
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
        # Check models considering Ollama's :latest suffix
        def _model_available(name: str) -> bool:
            return name in models or f"{name}:latest" in models
        missing = [m for m in (self.config.chat_model, self.config.fallback_model)
                   if not _model_available(m)]
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
        if low == "profile" or low.startswith("profile "):
            self._profile(prompt[7:].strip())
            return
        if low == "episodes" or low.startswith("episodes "):
            self._episodes(prompt[8:].strip())
            return
        if low == "procedures" or low.startswith("procedures "):
            self._procedures(prompt[11:].strip())
            return
        if low == "model list" or low == "model":
            self._models()
            return
        if low.startswith("model set "):
            self._model_set(prompt[10:].strip())
            return
        if low == "model show":
            self._model_show()
            return
        if low == "telegram":
            self._telegram()
            return
        if low.startswith("schedule"):
            self._schedule(prompt[8:].strip())
            return
        if low.startswith("mcp"):
            self._mcp(prompt[3:].strip())
            return
        if low.startswith("browse "):
            self._browse(prompt[7:].strip())
            return
        if low.startswith("screenshot "):
            self._screenshot(prompt[11:].strip())
            return
        if low == "mission" or low.startswith("mission "):
            self._mission(prompt[7:].strip())
            return
        if low == "runtime":
            self._runtime_toggle()
            return
        if low == "tools":
            self._tools_list()
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
        low = arg.lower().strip()

        if not low or low == "list":
            skills = self.skills.list()
            if not skills:
                warn("no skills installed")
                info("Install from GitHub: skill install owner/repo")
                return
            console.print("\n[bold cyan]Installed skills[/bold cyan]")
            for s in skills:
                ver = f" v{s.version}" if s.version else ""
                src = f" ({s.source_repo})" if s.source_repo else ""
                console.print(f"  {s.name}{ver}{src}")
                if s.description:
                    console.print(f"    {s.description[:80]}")
            console.print()
            return

        if low.startswith("show "):
            skill = self.skills.get(arg[5:].strip())
            if skill:
                label("Name:", skill.name)
                if skill.version:
                    label("Version:", skill.version)
                if skill.author:
                    label("Author:", skill.author)
                if skill.source_repo:
                    label("Source:", skill.source_repo)
                if skill.tags:
                    label("Tags:", ", ".join(skill.tags))
                console.print(f"\n{skill.body[:2000]}")
            else:
                warn("skill not found")
            return

        if low.startswith("install "):
            repo = arg[8:].strip()
            if not repo or "/" not in repo:
                warn("usage: skill install owner/repo")
                info("Example: skill install coder1317/pcb-design")
                return
            info(f"Installing {repo}...")
            skill = self.skills.install(repo)
            if skill:
                ok(f"Installed: {skill.name} v{skill.version or '?'}")
                if skill.description:
                    info(skill.description[:120])
            else:
                warn(f"Failed to install {repo}")
                info("Check the repo exists and contains SKILL.md or skill.json")
            return

        if low.startswith("uninstall ") or low.startswith("remove "):
            name = arg.split(None, 1)[1].strip() if " " in arg else ""
            if not name:
                warn("usage: skill uninstall <name>")
                return
            if self.skills.uninstall(name):
                ok(f"Removed skill: {name}")
            else:
                warn(f"Skill '{name}' not found")
            return

        if low.startswith("search "):
            query = arg[7:].strip()
            if not query:
                warn("usage: skill search <query>")
                return
            info(f"Searching GitHub for '{query}'...")
            results = self.skills.search_github(query)
            if not results:
                warn("no results found")
                info("Tip: tag your skill repo with 'aria-skill' on GitHub")
                return
            console.print(f"\n[bold cyan]GitHub skills: {query}[/bold cyan]")
            for r in results:
                stars = f" ⭐{r['stars']}" if r["stars"] else ""
                console.print(f"  {r['repo']}{stars}")
                if r["description"]:
                    console.print(f"    {r['description'][:80]}")
                console.print(f"    {r['url']}")
            console.print()
            info("Install: skill install <owner/repo>")
            return

        if low.startswith("extract"):
            target = arg[8:].strip()
            path = Path(target).expanduser() if target else self.orch.last_project
            if path is None or not Path(path).is_dir():
                warn("usage: skill extract <project-dir> (or build something first)")
                return
            name = self.orch.trainer.extract_skill(Path(path))
            if name:
                ok(f"extracted skill: {name}")
            return

        warn("unknown skill command")
        info("skill list | skill install <repo> | skill search <query> | skill show <name> | skill extract <project> | skill uninstall <name>")

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

    # ── Phase 2: memory intelligence ──────────────────────────────

    def _profile(self, arg: str) -> None:
        """View or update user profile."""
        from ultra.core.memory2 import MemoryV2, UserProfile
        from ultra.core.vectors import VectorStore
        db = self.config.data_dir / "memory" / "memory.db"
        vec_db = self.config.data_dir / "memory" / "vectors.db"
        vectors = VectorStore(vec_db)
        mem2 = MemoryV2(db, vectors)

        if not arg or arg == "show":
            profile = mem2.get_user_profile()
            console.print("\n[bold cyan]User Profile[/bold cyan]")
            label("Name:", profile.name or "(not set)")
            label("OS:", profile.os)
            label("Language:", profile.preferred_language)
            label("Editor:", profile.preferred_editor)
            label("Model:", profile.preferred_model)
            label("Style:", profile.response_style)
            if profile.hardware:
                label("Hardware:", ", ".join(profile.hardware))
            if profile.skills:
                label("Skills:", ", ".join(profile.skills))
            if profile.active_goals:
                label("Active goals:", ", ".join(profile.active_goals))
            if profile.long_term_goals:
                label("Long-term:", ", ".join(profile.long_term_goals))
            if profile.facts:
                console.print("\n[bold]Facts[/bold]")
                for k, v in profile.facts.items():
                    info(f"  {k}: {v}")
            console.print()
            return

        # profile set key value
        if arg.startswith("set "):
            parts = arg[4:].strip().split(" ", 1)
            if len(parts) == 2:
                key, value = parts
                profile = mem2.get_user_profile()
                if hasattr(profile, key):
                    setattr(profile, key, value)
                    mem2.set_user_profile(profile)
                    ok(f"profile.{key} = {value}")
                else:
                    # Store as custom fact
                    profile.facts[key] = value
                    mem2.set_user_profile(profile)
                    ok(f"profile.facts[{key}] = {value}")
            else:
                warn("usage: profile set <key> <value>")
            return

        if arg.startswith("goal "):
            goal = arg[5:].strip()
            if goal:
                profile = mem2.get_user_profile()
                profile.active_goals.append(goal)
                mem2.set_user_profile(profile)
                ok(f"goal added: {goal}")
            else:
                warn("usage: profile goal <description>")
            return

        warn("usage: profile | profile show | profile set key value | profile goal desc")

    def _episodes(self, arg: str) -> None:
        """View episodic memories."""
        from ultra.core.memory2 import MemoryV2
        from ultra.core.vectors import VectorStore
        db = self.config.data_dir / "memory" / "memory.db"
        vec_db = self.config.data_dir / "memory" / "vectors.db"
        vectors = VectorStore(vec_db)
        mem2 = MemoryV2(db, vectors)

        if not arg or arg == "list":
            episodes = mem2.get_episodes(limit=10)
            if not episodes:
                warn("no episodes recorded yet")
                return
            console.print("\n[bold cyan]Recent Episodes[/bold cyan]")
            for ep in episodes:
                icon = {"success": "✓", "failure": "✗", "partial": "~"}.get(
                    ep.get("outcome", ""), "•")
                imp = ep.get("importance", 0)
                color = "green" if imp > 0.7 else "yellow" if imp > 0.4 else "dim"
                console.print(
                    f"  [{color}]{icon} {ep['event_type']}[/{color}] "
                    f"{ep['summary'][:100]}")
                if ep.get("project"):
                    info(f"    project: {ep['project']}")
            console.print()
            return

        if arg.startswith("search "):
            query = arg[7:].strip()
            episodes = mem2.search_episodes(query, limit=5)
            if not episodes:
                warn("no matching episodes")
                return
            console.print(f"\n[bold cyan]Episodes matching '{query}'[/bold cyan]")
            for ep in episodes:
                info(f"  [{ep.get('event_type', '?')}] {ep.get('summary', '')[:120]}")
            console.print()
            return

        warn("usage: episodes | episodes list | episodes search <query>")

    def _procedures(self, arg: str) -> None:
        """View procedural memories."""
        from ultra.core.memory2 import MemoryV2
        from ultra.core.vectors import VectorStore
        db = self.config.data_dir / "memory" / "memory.db"
        vec_db = self.config.data_dir / "memory" / "vectors.db"
        vectors = VectorStore(vec_db)
        mem2 = MemoryV2(db, vectors)

        if not arg or arg == "list":
            procs = mem2.list_procedures(limit=10)
            if not procs:
                warn("no procedures recorded yet")
                info("Procedures are learned automatically from successful tasks")
                return
            console.print("\n[bold cyan]Known Procedures[/bold cyan]")
            for p in procs:
                conf = p.get("confidence", 0)
                color = "green" if conf > 0.7 else "yellow" if conf > 0.4 else "dim"
                total = p["success_count"] + p["fail_count"]
                console.print(
                    f"  [{color}]{p['name']}[/{color}] "
                    f"confidence: {conf:.0%}  "
                    f"success: {p['success_count']}/{total}  "
                    f"{p['description'][:60]}")
            console.print()
            return

        if arg.startswith("search "):
            query = arg[7:].strip()
            procs = mem2.search_procedures(query, limit=5)
            if not procs:
                warn("no matching procedures")
                return
            console.print(f"\n[bold cyan]Procedures matching '{query}'[/bold cyan]")
            for p in procs:
                info(f"  {p.name}: {p.description[:80]} (confidence: {p.confidence:.0%})")
            console.print()
            return

        warn("usage: procedures | procedures list | procedures search <query>")

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
        """Show available models and which one is active."""
        models = self.client.available_models()
        if not models:
            warn("no models pulled — run: ollama pull granite4.1:3b")
            return
        console.print("\n[bold cyan]Available models[/bold cyan]")
        for m in models:
            active = ""
            if m == self.config.chat_model:
                active = " [green]← chat + code (primary)[/green]"
            elif m == self.config.fallback_model:
                active = " [yellow]← fallback[/yellow]"
            elif m == self.config.embed_model:
                active = " [dim]← embeddings[/dim]"
            console.print(f"  {m}{active}")
        console.print()
        label("Active:", f"chat={self.config.chat_model}  code={self.config.coding_model}  fallback={self.config.fallback_model}")
        info("Use: model set <name> to switch the primary model")

    def _model_set(self, name: str) -> None:
        """Switch the active chat + coding model at runtime."""
        if not name:
            warn("usage: model set <model-name>")
            info("Example: model set lfm2.5:latest")
            return
        models = self.client.available_models()
        # Accept both "lfm2.5" and "lfm2.5:latest"
        if name not in models and f"{name}:latest" not in models:
            warn(f"'{name}' not pulled. Available: {', '.join(models)}")
            info(f"Pull it first: ollama pull {name}")
            return
        old = self.config.chat_model
        self.config.chat_model = name
        self.config.coding_model = name
        # rebuild providers so the pool routes to the new model
        from ultra.config import ProviderSpec
        self.config.providers = Config._build_providers(self.config)
        self.client = ProviderPool(self.config)
        ok(f"Switched: {old} → {name}")
        info(f"chat={self.config.chat_model}  code={self.config.coding_model}  fallback={self.config.fallback_model}")
        self.audit.log(actor="user", action="model_switch",
                       detail={"from": old, "to": name})

    def _model_show(self) -> None:
        """Show current model configuration."""
        console.print("\n[bold cyan]Model configuration[/bold cyan]")
        label("Chat model:", self.config.chat_model)
        label("Code model:", self.config.coding_model)
        label("Fallback:", self.config.fallback_model)
        label("Embeddings:", self.config.embed_model)
        label("LLM timeout:", f"{self.config.llm_timeout}s")
        console.print()
        label("Providers:", ", ".join(
            f"{p.name}({p.model})" for p in self.config.providers))
        console.print()
        info("Switch: model set <name>  |  List: model list")

    # ── telegram ────────────────────────────────────────────────

    def _telegram(self) -> None:
        """Start the Telegram bot channel."""
        import asyncio
        from ultra.channels.telegram import TelegramChannel, _HAS_TELEGRAM
        if not _HAS_TELEGRAM:
            warn("python-telegram-bot not installed.")
            info("Install: pip install python-telegram-bot")
            return
        if not self.config.telegram_token:
            warn("No Telegram token configured.")
            info("Set TELEGRAM_BOT_TOKEN in .env")
            return

        def dispatch(msg):
            return self.orch.dispatch(msg.text)

        channel = TelegramChannel(
            token=self.config.telegram_token,
            dispatch_fn=dispatch,
            allowed_users=self.config.telegram_allowed_users or None,
        )
        ok(f"Starting Telegram bot (allowed: {self.config.telegram_allowed_users or 'everyone'})")
        info("Press Ctrl+C to stop")
        try:
            asyncio.run(channel.start())
            # keep running until interrupted
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\nStopping...")
            asyncio.run(channel.stop())

    # ── scheduler ────────────────────────────────────────────────

    def _schedule(self, arg: str) -> None:
        """Handle schedule commands."""
        from ultra.scheduler import Scheduler
        db_path = self.config.data_dir / "memory" / "scheduler.db"
        scheduler = Scheduler(db_path, dispatch_fn=self.orch.dispatch)
        low = arg.lower()

        if not low or low == "list":
            tasks = scheduler.list_tasks()
            if not tasks:
                info("No scheduled tasks.")
                info("Add: schedule add name='daily research' command='research AI news' daily=08:00")
                return
            console.print("\n[bold cyan]Scheduled tasks[/bold cyan]")
            for t in tasks:
                status = "enabled" if t.enabled else "disabled"
                console.print(
                    f"  [{t.id}] {t.name} — {t.command[:50]}\n"
                    f"       {t.schedule_type} | {status} | next: {t.next_run[:16]}")
            console.print()
            return

        if low.startswith("add "):
            # parse: schedule add name='x' command='y' daily=08:00
            #     or: schedule add name='x' command='y' interval=3600
            import re as _re
            params = {}
            for m in _re.finditer(r"(\w+)=('[^']*'|\"[^\"]*\"|\S+)", arg[4:]):
                k, v = m.group(1), m.group(2).strip("'\"")
                params[k] = v
            name = params.get("name", "unnamed")
            command = params.get("command", "")
            if not command:
                warn("usage: schedule add name='x' command='research AI news' daily=08:00")
                return
            if "daily" in params:
                task = scheduler.add(name, command, "daily_time",
                                     daily_time=params["daily"])
            elif "interval" in params:
                task = scheduler.add(name, command, "interval",
                                     interval_seconds=int(params["interval"]))
            else:
                task = scheduler.add(name, command, "once", interval_seconds=3600)
            ok(f"Scheduled: {task.name} (id={task.id}, next: {task.next_run[:16]})")
            return

        if low.startswith("run "):
            try:
                task_id = int(arg[4:].strip())
            except ValueError:
                warn("usage: schedule run <id>")
                return
            task = scheduler.get(task_id)
            if not task:
                warn(f"Task {task_id} not found")
                return
            info(f"Running: {task.name}...")
            result = scheduler._execute_task(task)
            console.print(result[:3000])
            return

        if low.startswith("remove ") or low.startswith("rm "):
            try:
                task_id = int(arg.split()[1])
            except (ValueError, IndexError):
                warn("usage: schedule remove <id>")
                return
            if scheduler.remove(task_id):
                ok(f"Removed task {task_id}")
            else:
                warn(f"Task {task_id} not found")
            return

        warn(f"Unknown schedule command: {arg}")
        info("Usage: schedule list | schedule add ... | schedule run <id> | schedule remove <id>")

    # ── MCP ──────────────────────────────────────────────────────

    def _mcp(self, arg: str) -> None:
        """Handle MCP commands."""
        import asyncio
        from ultra.tools.mcp_client import MCPManager, _HAS_MCP
        if not _HAS_MCP:
            warn("MCP not installed.")
            info("Install: pip install mcp")
            return
        low = arg.lower()
        manager = MCPManager()
        if not manager.servers:
            info("No MCP servers configured.")
            info("Set ARIA_MCP_SERVERS in .env (e.g. filesystem:npx -y @modelcontextprotocol/server-filesystem /tmp)")
            return
        if not low or low == "status":
            asyncio.run(manager.start())
            status = manager.get_status()
            console.print("\n[bold cyan]MCP servers[/bold cyan]")
            for name, info_dict in status.items():
                color = "green" if info_dict["connected"] else "red"
                console.print(
                    f"  {name} [{color}]{info_dict['connected']}[/{color}] "
                    f"tools: {', '.join(info_dict['tool_names']) or 'none'}")
            console.print()
            asyncio.run(manager.stop())

    # ── browser ──────────────────────────────────────────────────

    def _browse(self, url: str) -> None:
        """Browse a URL with the headless browser."""
        import asyncio
        from ultra.tools.browser import Browser, _HAS_PLAYWRIGHT
        if not _HAS_PLAYWRIGHT:
            warn("Playwright not installed.")
            info("Install: pip install playwright && playwright install chromium")
            return
        if not url:
            warn("usage: browse <url>")
            return
        if not url.startswith("http"):
            url = "https://" + url
        info(f"Browsing {url}...")
        browser = Browser()
        try:
            result = asyncio.run(browser.goto(url))
            if result.error:
                warn(f"Error: {result.error}")
                return
            label("Title:", result.title)
            label("URL:", result.url)
            console.print(result.text[:4000])
            if result.links:
                info(f"\n{len(result.links)} links found")
        finally:
            asyncio.run(browser.close())

    def _screenshot(self, url: str) -> None:
        """Take a screenshot of a URL."""
        import asyncio
        from ultra.tools.browser import Browser, _HAS_PLAYWRIGHT
        if not _HAS_PLAYWRIGHT:
            warn("Playwright not installed.")
            info("Install: pip install playwright && playwright install chromium")
            return
        if not url:
            warn("usage: screenshot <url>")
            return
        if not url.startswith("http"):
            url = "https://" + url
        out = str(self.config.data_dir / "screenshots" / f"{url.split('//')[-1][:50]}.png")
        info(f"Screenshotting {url}...")
        browser = Browser()
        try:
            result = asyncio.run(browser.screenshot(url, out))
            if result.error:
                warn(f"Error: {result.error}")
                return
            ok(f"Saved: {result.screenshot_path}")
            label("Title:", result.title)
        finally:
            asyncio.run(browser.close())

    def _orchestrate(self, text: str) -> None:
        """Decompose into background tasks via the TaskManager."""
        result = self.orch.orchestrate(text)
        console.print(result)

    # ── Agent Runtime commands ──────────────────────────────────────

    def _mission(self, arg: str) -> None:
        """Run a mission through the Agent Runtime."""
        if not arg:
            # Show mission status
            status = self.orch.mission_status()
            console.print(status)
            return
        if arg == "status":
            status = self.orch.mission_status()
            console.print(status)
            return
        # Run a mission
        if self.orch.runtime is None:
            warn("Agent Runtime not enabled. Type 'runtime' to enable it.")
            return
        step(1, "Planning")
        result = self.orch.dispatch_runtime(arg)
        self.last_report = result
        console.print()
        console.print(Markdown(result[:6000]))

    def _runtime_toggle(self) -> None:
        """Toggle the Agent Runtime on/off."""
        if self.orch.runtime is not None:
            self.config.runtime_enabled = False
            self.orch.runtime = None
            warn("Agent Runtime disabled — using pipeline mode")
        else:
            self.config.runtime_enabled = True
            self.orch._init_runtime()
            ok("Agent Runtime enabled — using PLAN→ACT→OBSERVE→EVALUATE→REPLAN")
            tools = self.orch._tool_registry.tool_count
            info(f"{tools} tools registered")

    def _tools_list(self) -> None:
        """List all registered tools."""
        if self.orch.runtime is None:
            warn("Agent Runtime not enabled. Type 'runtime' to enable it.")
            return
        registry = self.orch._tool_registry
        console.print("\n[bold cyan]Registered Tools[/bold cyan]")
        for cat in registry.categories:
            tools = registry.list_tools(category=cat)
            console.print(f"\n  [bold]{cat}[/bold]")
            for t in tools:
                risk = t.risk_level.value
                console.print(f"    {t.name} [{risk}] — {t.description[:60]}")

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
