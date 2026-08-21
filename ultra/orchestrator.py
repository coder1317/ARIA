"""Brain orchestrator — routes intents to the right pipeline (spec §6).

Intent → pipeline:
  research_only   → ResearchAgent
  build_only      → EngineeringPipeline.build  (+ evaluator + skill extraction)
  full_pipeline   → research first, then build
  improve         → EngineeringPipeline.improve
  market          → MarketAgent
  deploy          → DeployerAgent
  memory          → memory commands (handled by the CLI)
  orchestrate     → background tasks via TaskManager
  chat            → persona-aware chat

Every dispatch passes through the Security gate (prompt-injection
defense) and is written to the AuditLog.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ultra.agents import architect, coder, debugger, reviewer, tester
from ultra.agents.base import Agent
from ultra.agents.deployer import generate_deployment
from ultra.agents.engineering import EngineeringPipeline
from ultra.agents.market import MarketAgent
from ultra.agents.research import ResearchAgent
from ultra.agents.trainer import TrainerAgent
from ultra.audit import AuditLog
from ultra.config import Config
from ultra.core.events import EventBus, EventType
from ultra.core.goals import GoalManager
from ultra.core.memory import Memory
from ultra.core.memory2 import MemoryV2, Episode, Procedure
from ultra.core.notifications import NotificationManager
from ultra.core.skills import SkillManager
from ultra.core.vectors import VectorStore
from ultra.core.watchers import FileWatcher
from ultra.display import info, ok, step, warn
from ultra.evaluator import CircuitBreaker, Evaluator
from ultra.intent import detect, extract_problem
from ultra.llm import OllamaClient
from ultra.persona import IDENTITY
from ultra.provider_pool import ProviderPool
from ultra.security import Security
from ultra.task_manager import TaskManager
from ultra.tools.terminal import Terminal

# Optional: Agent Runtime + Tool Registry
try:
    from ultra.core.tool_registry import ToolRegistry, ToolCall
    from ultra.core.runtime import AgentRuntime, Plan, PlanStatus
    from ultra.core.register_tools import register_all_tools
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False


@dataclass
class SessionReport:
    tasks: int = 0
    successes: int = 0
    durations: list[float] = field(default_factory=list)

    def record(self, success: bool, duration: float) -> None:
        self.tasks += 1
        self.successes += int(success)
        self.durations.append(duration)

    def print(self) -> None:
        from ultra.display import label
        avg = sum(self.durations) / len(self.durations) if self.durations else 0
        label("Tasks:", f"{self.successes}/{self.tasks} succeeded")
        label("Avg time:", f"{avg:.1f}s")


class Orchestrator:
    def __init__(self, client: ProviderPool, config: Config,
                 memory: Memory, vectors: VectorStore,
                 skills: SkillManager, terminal: Terminal,
                 security: Security | None = None,
                 audit: AuditLog | None = None,
                 tasks: TaskManager | None = None,
                 memory2: MemoryV2 | None = None):
        self.client = client
        self.config = config
        self.memory = memory
        self.vectors = vectors
        self.skills = skills
        # Phase 2: accept pre-built MemoryV2 or create one
        if memory2 is not None:
            self.memory2 = memory2
        else:
            try:
                self.memory2 = MemoryV2(memory.db_path, vectors)
            except Exception:
                self.memory2 = None  # graceful fallback
        self.terminal = terminal
        self.security = security or Security(config.security_enabled)
        self.audit = audit
        self.tasks = tasks
        self.report = SessionReport()
        self.last_project: Path | None = None

        # pipelines + agents
        self.engineering = EngineeringPipeline(client, config, terminal)
        self.research = ResearchAgent(client, config)
        self.market = MarketAgent(client, config)
        self.trainer = TrainerAgent(client, config)
        self.evaluator = Evaluator(self.security)
        self.build_circuit = CircuitBreaker("build", max_failures=3)

        # Phase 3: EventBus + Watchers + Notifications + Goals
        self.event_bus = EventBus()
        self.watcher = FileWatcher(self.event_bus)
        self.notify = NotificationManager(
            log_path=config.data_dir / "memory" / "notifications.log")
        self.notify.subscribe_event_bus(self.event_bus)
        self.goals = GoalManager(config.data_dir / "memory" / "goals.db")
        # Auto-watch the projects directory
        if config.projects_dir.exists():
            self.watcher.watch(config.projects_dir)
        # Emit startup event
        self.event_bus.system_startup()

        # agent registry (spec §6: Brain.agents)
        self.agents: dict[str, Agent] = {
            "research": Agent("research", "research", self.research.run, audit),
            "market": Agent("market", "market", self.market.run, audit),
            "build": Agent("build", "build", self._build, audit),
            "improve": Agent("improve", "improve", self.engineering.improve, audit),
            "architect": Agent("architect", "architecture",
                               architect.plan_project, audit),
            "coder": Agent("coder", "code_generation",
                           coder.generate_files, audit),
            "reviewer": Agent("reviewer", "review", reviewer.review_project, audit),
            "debugger": Agent("debugger", "code_debug",
                              debugger.fix_issues, audit),
            "deployer": Agent("deployer", "deployment",
                              generate_deployment, audit),
        }

        # background task handlers
        if self.tasks is not None:
            self.tasks.register("research", self._research_task)
            self.tasks.register("build", self._build_task)
            self.tasks.register("market", self._market_task)
            self.tasks.register("chat", self._chat_task)

        # Agent Runtime (optional — enabled via ARIA4_RUNTIME=1)
        self.runtime = None
        if _HAS_RUNTIME and config.runtime_enabled:
            self._init_runtime()

    def project(self, path: Path, description: str) -> None:
        self.last_project = path
        self.memory.save_project(description, str(path), description)

    # ── dispatch ────────────────────────────────────────────────────

    def dispatch(self, text: str, context: dict | None = None) -> str:
        start = time.time()
        # gate 1: security — reject prompt injection before any LLM call
        report = self.security.validate_input(text)
        if not report.passed:
            if self.audit:
                self.audit.log(actor="security", action="input_blocked",
                               detail={"findings": report.findings[:5]})
            warn(f"input blocked by security: {report.summary()}")
            return "Blocked: this input looks like a prompt-injection attempt."

        intent = detect(self.client, text)
        problem = extract_problem(text)
        # Emit user message event
        self.event_bus.emit(EventType.USER_MESSAGE, {"text": text[:300], "intent": intent}, source="cli")
        if self.audit:
            detail = {"input": text[:300]}
            if context:
                detail["context"] = context
            self.audit.log(actor="brain", action="dispatch", task_type=intent,
                           detail=detail)
        try:
            if intent == "research_only":
                outcome = self._research(problem)
            elif intent == "build_only":
                outcome = self._build(problem)
            elif intent == "full_pipeline":
                outcome = self._pipeline(problem)
            elif intent == "improve":
                outcome = self._improve(problem)
            elif intent == "market":
                outcome = self._market(problem)
            elif intent == "deploy":
                outcome = self._deploy(problem)
            else:
                outcome = self._chat(text)
            success = True
        except Exception as e:
            warn(f"error: {e}")
            outcome = f"Something went wrong: {e}"
            success = False
            try:
                self.memory.add_lesson(
                    intent, str(e)[:300],
                    "Check Ollama connectivity, model availability, and input format",
                )
            except Exception:
                pass
            # Phase 2: record error episode
            if self.memory2:
                try:
                    self.memory2.record_episode(Episode(
                        event_type="error",
                        summary=f"{intent} failed: {str(e)[:200]}",
                        detail=f"Input: {text[:200]}",
                        outcome="failure",
                        importance=0.8,
                        tags=[intent, "error"],
                    ))
                except Exception:
                    pass
        duration = time.time() - start
        self.report.record(success, duration)
        self.memory.log_interaction(text, intent, success, duration * 1000)
        # Emit completion/failure event
        if success:
            self.event_bus.emit(EventType.TASK_COMPLETED,
                              {"intent": intent, "duration": duration}, source="brain")
        else:
            self.event_bus.emit(EventType.TASK_FAILED,
                              {"intent": intent, "error": outcome[:200]}, source="brain")
        return outcome

    # ── pipelines ───────────────────────────────────────────────────

    def _research(self, topic: str) -> str:
        mode = self._detect_research_mode(topic)
        report = self.research.run(topic, mode)
        path = self.research.save(report)
        ok(f"report saved: {path}")
        self.vectors.add("research", topic, self.client)
        # Emit research event
        self.event_bus.research_completed(
            topic=topic,
            sources=len(report.sources) if hasattr(report, 'sources') else 0,
            confidence=report.confidence if hasattr(report, 'confidence') else 0,
        )
        # Phase 2: record research episode
        if self.memory2:
            try:
                self.memory2.record_episode(Episode(
                    event_type="research",
                    summary=f"Researched: {topic[:150]}",
                    detail=f"Mode: {mode}, sources: {len(report.sources) if hasattr(report, 'sources') else '?'}",
                    outcome="success",
                    importance=0.5,
                    tags=["research", mode],
                ))
            except Exception:
                pass
        # Add confidence and freshness metadata to output
        meta = []
        if report.confidence > 0:
            conf_label = "high" if report.confidence > 0.7 else "medium" if report.confidence > 0.4 else "low"
            meta.append(f"Confidence: {conf_label} ({report.confidence:.0%})")
        if report.source_freshness:
            meta.append(report.source_freshness)
        if meta:
            return report.markdown + "\n\n---\n*" + " · ".join(meta) + "*"
        return report.markdown

    def _market(self, topic: str) -> str:
        mode = self._detect_market_mode(topic)
        report = self.market.run(topic, mode)
        path = self.market.save(report)
        ok(f"market report saved: {path}")
        return report.markdown

    def _build(self, description: str) -> str:
        if self.build_circuit.is_open:
            return (f"Build circuit is cooling down after repeated failures "
                    f"({self.build_circuit.remaining()}s) — try again shortly.")
        self.event_bus.build_started(description)
        path = self.engineering.build(description, report=self)
        # evaluator: score the result for the audit trail
        eval_result = self.evaluator.evaluate_project(path)
        info(f"evaluator: {eval_result.summary()}")
        self.event_bus.build_completed(str(path), eval_result.score)
        if self.audit:
            self.audit.log(actor="brain", action="build_evaluated",
                           detail=eval_result.summary(), error=None if eval_result.passed
                           else "below threshold")
        if eval_result.passed:
            self.build_circuit.record_success()
            # Phase 2: record successful build as episode
            if self.memory2:
                try:
                    self.memory2.record_episode(Episode(
                        event_type="task_completed",
                        summary=f"Built: {description[:100]}",
                        detail=f"Score: {eval_result.summary()}",
                        project=str(path),
                        outcome="success",
                        importance=0.6,
                        tags=["build", "engineering"],
                    ))
                except Exception:
                    pass
            if self.config.auto_extract_skills:
                try:
                    self.trainer.extract_skill(path, description)
                except Exception as e:
                    warn(f"skill extraction failed: {e}")
        else:
            # Phase 2: record failed build
            if self.memory2:
                try:
                    self.memory2.record_episode(Episode(
                        event_type="task_completed",
                        summary=f"Build failed: {description[:100]}",
                        detail=eval_result.summary(),
                        outcome="failure",
                        importance=0.7,
                        tags=["build", "failure"],
                    ))
                except Exception:
                    pass
            tripped = self.build_circuit.record_failure()
            if tripped:
                warn(f"build circuit opened after {self.build_circuit.max_failures} "
                     "failed builds — will cool down before retrying")
        return f"Project built at {path}"

    def _pipeline(self, problem: str) -> str:
        step(1, "Research phase")
        report = self.research.run(problem, "deep")
        path = self.research.save(report)
        ok(f"research saved: {path}")
        step(2, "Build phase")
        project = self._build(problem)
        return f"Researched + built. Report: {path}\n{project}"

    def _improve(self, problem: str) -> str:
        path = extract_path(problem)
        self.engineering.improve(problem, path)
        return "Improvement pass complete."

    def _deploy(self, problem: str) -> str:
        path = extract_path(problem)
        target = Path(path).expanduser() if path and Path(path).exists() else self.last_project
        if target is None:
            projects = sorted(self.config.projects_dir.iterdir(),
                              key=lambda p: p.stat().st_mtime)
            target = projects[-1] if projects else None
        if target is None or not target.is_dir():
            return "No project found to deploy. Build one first, or name a path."
        platform = "docker"
        low = problem.lower()
        for p in ("github", "actions", "ci"):
            if p in low:
                platform = "github_actions"
        written = generate_deployment(self.client, target, platform)
        ok(f"{len(written)} deployment files written to {target}")
        return f"Deployment config generated for {platform} in {target}"

    def _chat(self, text: str) -> str:
        system = IDENTITY + "\n\n" + self._memory_context(text)
        if self.skills.context(text):
            system += "\n\n" + self.skills.context(text)
        messages = self.memory.thread(self.config.context_window) + [
            {"role": "user", "content": text}
        ]
        response = self.client.chat(messages, system=system, task_type="chat")
        self.memory.add_message("user", text)
        self.memory.add_message("assistant", response)
        self.vectors.add("chat", text, self.client)
        # Phase 2: record episode for significant conversations
        if self.memory2 and len(text) > 10:
            try:
                self.memory2.record_episode(Episode(
                    event_type="conversation",
                    summary=text[:200],
                    detail=response[:500],
                    importance=0.3,
                    tags=["chat"],
                ))
            except Exception:
                pass
        if self.audit:
            self.audit.log(actor="brain", action="chat", task_type="chat",
                           provider=None, detail={"len": len(response)})
        return response

    # ── background tasks (orchestrate mode) ─────────────────────────

    def _research_task(self, topic: str, mode: str = "deep") -> str:
        report = self.research.run(topic, mode)
        return self.research.save(report)

    def _build_task(self, description: str) -> str:
        path = self.engineering.build(description, report=self)
        return f"built: {path}"

    def _market_task(self, topic: str, mode: str = "overview") -> str:
        report = self.market.run(topic, mode)
        return self.market.save(report)

    def _chat_task(self, text: str) -> str:
        return self._chat(text)

    def orchestrate(self, problem: str) -> str:
        """Decompose into background tasks and submit them."""
        if self.tasks is None:
            return "Task manager not available — run `aria` with it enabled."
        step(1, "Decomposing objective")
        plan = _decompose(self.client, problem)
        if not plan:
            warn("could not decompose — submitting as a single build task")
            task_id = self.tasks.submit("build", {"description": problem})
            return f"submitted task {task_id} (build)"
        step(2, "Submitting tasks")
        ids = []
        for item in plan:
            ttype = item.get("type", "build")
            payload = item.get("payload", {})
            tid = self.tasks.submit(ttype, payload)
            ids.append(tid)
            info(f"  {tid}  {ttype}: {list(payload.values())[0][:50]}")
        return ("Submitted " + ", ".join(ids) +
                " — check with `tasks` or `status <id>`")

    # ── Agent Runtime ─────────────────────────────────────────────

    def _init_runtime(self) -> None:
        """Initialize the Agent Runtime with Tool Registry."""
        registry = ToolRegistry()
        register_all_tools(
            registry, self.config,
            terminal=self.terminal,
            memory=self.memory,
            vectors=self.vectors,
            client=self.client,
            skills=self.skills,
        )

        def llm_fn(prompt: str, system: str = "") -> str:
            return self.client.chat(
                [{"role": "user", "content": prompt}],
                system=system, task_type="runtime",
            )

        def json_fn(prompt: str, system: str = "") -> dict:
            return self.client.json(prompt, system=system, task_type="runtime")

        self.runtime = AgentRuntime(
            registry=registry,
            llm_fn=llm_fn,
            json_fn=json_fn,
            max_iterations=self.config.runtime_max_iterations,
            max_replans=self.config.runtime_max_replans,
        )
        self._tool_registry = registry

    def dispatch_runtime(self, text: str) -> str:
        """Route through the Agent Runtime instead of hardcoded pipelines.

        The runtime creates a plan, executes steps via the tool registry,
        observes results, and replans if needed.
        """
        if self.runtime is None:
            return "Agent Runtime not enabled. Set ARIA4_RUNTIME=1 in .env"

        start = time.time()
        if self.audit:
            self.audit.log(actor="brain", action="runtime_dispatch",
                           detail={"input": text[:300]})

        try:
            plan = self.runtime.create_plan(text)
            plan = self.runtime.run(plan)
        except Exception as e:
            warn(f"runtime error: {e}")
            return f"Runtime failed: {e}"

        duration = time.time() - start
        self.report.record(plan.status == PlanStatus.COMPLETED, duration)

        # Format result
        lines = [plan.summary(), ""]
        for obs in plan.observations:
            if obs.tool_results:
                for r in obs.tool_results:
                    lines.append(f"  {r.summary()}")
        if plan.status == PlanStatus.COMPLETED:
            ok(f"mission complete ({duration:.1f}s, {plan.replan_count} replans)")
        else:
            warn(f"mission {plan.status.value} ({duration:.1f}s)")

        self.memory.log_interaction(text, "runtime",
                                    plan.status == PlanStatus.COMPLETED,
                                    duration * 1000)
        return "\n".join(lines)

    def mission_status(self) -> str:
        """Show status of the last runtime execution."""
        if self.runtime is None:
            return "Agent Runtime not enabled."
        return self.runtime.summary()

    # ── helpers ─────────────────────────────────────────────────────

    def _memory_context(self, query: str = "") -> str:
        """Build memory context for chat — episodic, procedural, user model, facts."""
        # Phase 2: full memory retrieval with actual query
        if self.memory2:
            try:
                return self.memory2.retrieve_context(query, self.client, max_tokens=2000)
            except Exception:
                pass
        # Fallback: basic facts only
        facts = self.memory.get_facts()
        if not facts:
            return ""
        lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        return f"KNOWN FACTS ABOUT THE USER:\n{lines}"

    def _detect_research_mode(self, topic: str) -> str:
        low = topic.lower()
        if any(w in low for w in ("compare", "vs ", "versus", "difference")):
            return "compare"
        if "feasib" in low:
            return "feasibility"
        if any(w in low for w in ("competitive", "swot", "market")):
            return "competitive"
        return "deep"

    def _detect_market_mode(self, topic: str) -> str:
        low = topic.lower()
        if "swot" in low:
            return "swot"
        if "trend" in low:
            return "trends"
        if "competit" in low:
            return "competitors"
        return "overview"

    def status(self) -> dict:
        return {
            "health": self.client.health_report(),
            "memory": self.memory.stats(),
            "build_circuit": "open" if self.build_circuit.is_open else "closed",
        }


def extract_path(text: str) -> str | None:
    """Pull a filesystem path out of free text like 'fix the login bug in ~/app'.

    Handles Unix (~/app, /home/user/app) and Windows (C:\\Users\\app, D:\\projects)."""
    import re
    # Unix: ~/foo, /foo  |  Windows: C:\foo, D:\foo, or \\server\share
    m = re.search(
        r"(?:in|at|from)\s+"
        r"((?:[a-zA-Z]:[\\/]|[\\/]|[~/])[\w\\/\-\.]+)",
        text,
    )
    return m.group(1) if m else None


def _decompose(client, problem: str) -> list[dict]:
    """LLM decomposition of an objective into background task specs."""
    prompt = (
        f"Objective: {problem}\n\n"
        "Break this into background tasks. Each task: "
        '{"type": "research"|"build"|"market"|"chat", '
        '"payload": {"topic"|"description"|"text": "..."}}. '
        "Research first when the topic is unfamiliar, then build. "
        "Return ONLY a JSON array of 1-4 tasks."
    )
    result = client.json(prompt, max_tokens=1024, temperature=0.2,
                         task_type="routing")
    tasks = result if isinstance(result, list) else []
    clean = []
    for t in tasks:
        if not isinstance(t, dict) or t.get("type") not in ("research", "build", "market", "chat"):
            continue
        payload = t.get("payload", {})
        key = {"research": "topic", "build": "description",
               "market": "topic", "chat": "text"}[t["type"]]
        if not payload.get(key):
            continue
        clean.append({"type": t["type"], "payload": {key: payload[key]}})
    return clean
