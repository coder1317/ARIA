"""Register all existing ARIA tools into the unified ToolRegistry.

This bridges the current tool implementations (terminal, editor,
researcher, browser, MCP, memory, skills) with the new ToolRegistry
so the Agent Runtime can use them as structured tools.

The LLM receives these as tool definitions and responds with
structured tool_calls instead of relying on intent classification.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ultra.config import Config
from ultra.core.tool_registry import RiskLevel, ToolDefinition, ToolParam, ToolRegistry

logger = logging.getLogger("aria.tools.register")


def register_all_tools(
    registry: ToolRegistry,
    config: Config,
    *,
    terminal=None,
    editor=None,
    memory=None,
    vectors=None,
    client=None,
    skills=None,
    orchestrator=None,
) -> None:
    """Register all ARIA tools into the given registry.

    Args:
        registry: The ToolRegistry to populate.
        config: ARIA config.
        terminal: Terminal tool instance (optional).
        editor: Editor class (static methods).
        memory: Memory instance for facts/conversations.
        vectors: VectorStore instance.
        client: ProviderPool for LLM calls.
        skills: SkillManager instance.
        orchestrator: Orchestrator instance for pipeline tools.
    """
    _register_terminal(registry, terminal, config)
    _register_filesystem(registry, config)
    _register_memory(registry, memory, vectors, client)
    _register_research(registry, client, config)
    _register_skills(registry, skills)
    _register_model(registry, client, config)
    _register_chat(registry, client)
    if orchestrator is not None:
        _register_pipelines(registry, orchestrator)
    logger.info("registered %d tools", registry.tool_count)


# ── Terminal ───────────────────────────────────────────────────

def _register_terminal(registry: ToolRegistry, terminal, config: Config) -> None:
    def _run(command: str, cwd: str = "") -> str:
        if terminal is None:
            raise RuntimeError("terminal not available")
        result = terminal.run(command, auto_approve=True,
                              cwd=cwd or None)
        if result.blocked:
            return f"[BLOCKED] {result.reason}"
        if result.timed_out:
            return f"[TIMED OUT] Command took too long"
        output = result.output[:config.output_cap]
        status = "ok" if result.ok else f"exit {result.exit_code}"
        return f"{status}\n{output}"

    registry.register(ToolDefinition(
        name="terminal.execute",
        description="Execute a shell command and return its output. "
                    "Use for running tests, installing packages, "
                    "checking system state, git operations, etc.",
        handler=_run,
        params=[
            ToolParam("command", "string", "Shell command to execute", required=True),
            ToolParam("cwd", "string", "Working directory (optional)"),
        ],
        risk_level=RiskLevel.MODERATE,
        category="terminal",
    ))


# ── Filesystem ─────────────────────────────────────────────────

def _register_filesystem(registry: ToolRegistry, config: Config) -> None:
    def _read(path: str) -> str:
        p = Path(path).expanduser()
        if not p.is_file():
            return f"file not found: {path}"
        try:
            return p.read_text(encoding="utf-8", errors="replace")[:50_000]
        except Exception as e:
            return f"error reading {path}: {e}"

    def _write(path: str, content: str) -> str:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(content, encoding="utf-8")
            return f"wrote {len(content)} chars to {path}"
        except Exception as e:
            return f"error writing {path}: {e}"

    def _list_dir(path: str = ".") -> str:
        p = Path(path).expanduser()
        if not p.is_dir():
            return f"not a directory: {path}"
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        lines = []
        for e in entries[:200]:
            prefix = "📁" if e.is_dir() else "📄"
            size = f" ({e.stat().st_size:,} bytes)" if e.is_file() else ""
            lines.append(f"  {prefix} {e.name}{size}")
        return f"{len(entries)} entries in {path}:\n" + "\n".join(lines)

    def _search(pattern: str, path: str = ".") -> str:
        """Search for files matching a glob pattern."""
        p = Path(path).expanduser()
        if not p.is_dir():
            return f"not a directory: {path}"
        matches = list(p.rglob(pattern))[:100]
        if not matches:
            return f"no files matching '{pattern}' in {path}"
        return "\n".join(str(m) for m in matches)

    registry.register(ToolDefinition(
        name="filesystem.read",
        description="Read the contents of a file.",
        handler=_read,
        params=[ToolParam("path", "string", "File path to read", required=True)],
        risk_level=RiskLevel.READ_ONLY,
        category="filesystem",
    ))
    registry.register(ToolDefinition(
        name="filesystem.write",
        description="Write content to a file. Creates parent directories if needed.",
        handler=_write,
        params=[
            ToolParam("path", "string", "File path to write", required=True),
            ToolParam("content", "string", "Content to write", required=True),
        ],
        risk_level=RiskLevel.MODERATE,
        category="filesystem",
    ))
    registry.register(ToolDefinition(
        name="filesystem.list",
        description="List files and directories in a path.",
        handler=_list_dir,
        params=[ToolParam("path", "string", "Directory path (default: current)")],
        risk_level=RiskLevel.READ_ONLY,
        category="filesystem",
    ))
    registry.register(ToolDefinition(
        name="filesystem.search",
        description="Search for files matching a glob pattern.",
        handler=_search,
        params=[
            ToolParam("pattern", "string", "Glob pattern (e.g. '*.py')", required=True),
            ToolParam("path", "string", "Directory to search in"),
        ],
        risk_level=RiskLevel.READ_ONLY,
        category="filesystem",
    ))


# ── Memory ─────────────────────────────────────────────────────

def _register_memory(
    registry: ToolRegistry, memory, vectors, client
) -> None:
    def _search_memory(query: str, n: int = 5) -> str:
        """Search memory using both keyword and vector search."""
        results = []
        # Keyword search
        if memory:
            facts = memory.get_facts()
            for k, v in facts.items():
                if any(w.lower() in k.lower() or w.lower() in v.lower()
                       for w in query.split()):
                    results.append(f"fact: {k} = {v}")

        # Vector search
        if vectors and client:
            try:
                vec_results = vectors.search(query, client, n=n)
                for r in vec_results:
                    results.append(f"semantic: {r.get('content', str(r))[:200]}")
            except Exception:
                pass

        # Recent conversation
        if memory:
            recent = memory.thread(3)
            for msg in recent:
                results.append(f"recent [{msg['role']}]: {msg['content'][:200]}")

        if not results:
            return f"no memories found for: {query}"
        return "\n".join(results[:n * 2])

    def _remember(content: str, category: str = "facts") -> str:
        """Store a fact or memory."""
        if not memory:
            return "memory not available"
        try:
            memory.save_fact(content[:500], content[:500])
            return f"remembered: {content[:100]}"
        except Exception as e:
            return f"failed to remember: {e}"

    registry.register(ToolDefinition(
        name="memory.search",
        description="Search ARIA's memory for relevant facts, "
                    "conversations, and knowledge about the user.",
        handler=_search_memory,
        params=[
            ToolParam("query", "string", "Search query", required=True),
            ToolParam("n", "integer", "Max results (default 5)"),
        ],
        risk_level=RiskLevel.READ_ONLY,
        category="memory",
    ))
    registry.register(ToolDefinition(
        name="memory.remember",
        description="Store a fact or piece of knowledge for future reference.",
        handler=_remember,
        params=[
            ToolParam("content", "string", "What to remember", required=True),
            ToolParam("category", "string", "Category (facts, project, lesson)"),
        ],
        risk_level=RiskLevel.LOW,
        category="memory",
    ))


# ── Research ───────────────────────────────────────────────────

def _register_research(registry: ToolRegistry, client, config: Config) -> None:
    def _web_search(query: str, max_results: int = 5) -> str:
        """Search the web using Bing/DuckDuckGo."""
        try:
            from ultra.tools.researcher import search_bing, search_ddg
            results = search_bing(query, config.search_timeout)
            if not results:
                results = search_ddg(query, config.search_timeout)
            if not results:
                return "no web results found"
            lines = []
            for r in results[:max_results]:
                lines.append(f"- [{r.title}]({r.url})\n  {r.snippet[:200]}")
            return "\n".join(lines)
        except Exception as e:
            return f"search failed: {e}"

    def _web_fetch(url: str, max_chars: int = 10_000) -> str:
        """Fetch and extract text from a URL."""
        try:
            from ultra.tools.researcher import fetch_page
            source = fetch_page(url, config.search_timeout)
            if source is None:
                return f"failed to fetch: {url}"
            return source.text(max_chars)
        except Exception as e:
            return f"fetch failed: {e}"

    registry.register(ToolDefinition(
        name="web.search",
        description="Search the web for information on any topic.",
        handler=_web_search,
        params=[
            ToolParam("query", "string", "Search query", required=True),
            ToolParam("max_results", "integer", "Max results (default 5)"),
        ],
        risk_level=RiskLevel.READ_ONLY,
        category="web",
    ))
    registry.register(ToolDefinition(
        name="web.fetch",
        description="Fetch and extract readable text from a URL.",
        handler=_web_fetch,
        params=[
            ToolParam("url", "string", "URL to fetch", required=True),
            ToolParam("max_chars", "integer", "Max chars to return"),
        ],
        risk_level=RiskLevel.READ_ONLY,
        category="web",
    ))


# ── Skills ─────────────────────────────────────────────────────

def _register_skills(registry: ToolRegistry, skills) -> None:
    def _skill_list() -> str:
        if not skills:
            return "skills not available"
        loaded = skills.list()
        if not loaded:
            return "no skills loaded"
        return "\n".join(f"- {s.name}: {s.description}" for s in loaded)

    def _skill_get(name: str) -> str:
        if not skills:
            return "skills not available"
        skill = skills.get(name)
        if skill is None:
            return f"skill not found: {name}"
        return skill.content[:5000]

    def _skill_search(query: str) -> str:
        if not skills:
            return "skills not available"
        loaded = skills.list()
        matching = [s for s in loaded
                    if query.lower() in s.name.lower()
                    or query.lower() in s.description.lower()
                    or query.lower() in s.content.lower()]
        if not matching:
            return f"no skills match: {query}"
        return "\n".join(f"- {s.name}: {s.description}" for s in matching)

    registry.register(ToolDefinition(
        name="skills.list",
        description="List all installed skills.",
        handler=_skill_list,
        risk_level=RiskLevel.READ_ONLY,
        category="skills",
    ))
    registry.register(ToolDefinition(
        name="skills.get",
        description="Get the content of a specific skill.",
        handler=_skill_get,
        params=[ToolParam("name", "string", "Skill name", required=True)],
        risk_level=RiskLevel.READ_ONLY,
        category="skills",
    ))
    registry.register(ToolDefinition(
        name="skills.search",
        description="Search installed skills by query.",
        handler=_skill_search,
        params=[ToolParam("query", "string", "Search query", required=True)],
        risk_level=RiskLevel.READ_ONLY,
        category="skills",
    ))


# ── Model ──────────────────────────────────────────────────────

def _register_model(registry: ToolRegistry, client, config: Config) -> None:
    def _list_models() -> str:
        if not client:
            return "provider pool not available"
        try:
            models = client.available_models()
            return "available models:\n" + "\n".join(f"  {m}" for m in models)
        except Exception as e:
            return f"error listing models: {e}"

    def _provider_health() -> str:
        if not client:
            return "provider pool not available"
        try:
            report = client.health_report()
            return json.dumps(report, indent=2, default=str)
        except Exception as e:
            return f"error: {e}"

    registry.register(ToolDefinition(
        name="model.list",
        description="List all available Ollama models.",
        handler=_list_models,
        risk_level=RiskLevel.READ_ONLY,
        category="system",
    ))
    registry.register(ToolDefinition(
        name="model.health",
        description="Check provider pool health and statistics.",
        handler=_provider_health,
        risk_level=RiskLevel.READ_ONLY,
        category="system",
    ))


# ── Chat (safe fallback) ──────────────────────────────────────
def _register_chat(registry: ToolRegistry, client) -> None:
    def _chat_respond(message: str) -> str:
        """Safe fallback: respond to a message via the LLM."""
        if not client:
            return "provider pool not available"
        try:
            from ultra.persona import IDENTITY
            response = client.chat(
                [{"role": "user", "content": message}],
                system=IDENTITY,
                task_type="chat",
            )
            return response[:5000]
        except Exception as e:
            return f"chat failed: {e}"

    registry.register(ToolDefinition(
        name="chat.respond",
        description="Respond to a user message using the LLM. "
                    "Safe fallback when other planning fails.",
        handler=_chat_respond,
        params=[
            ToolParam("message", "string", "The message to respond to", required=True),
        ],
        risk_level=RiskLevel.READ_ONLY,
        category="chat",
    ))


# ── Pipeline tools (research, build, market) ─────────────────
def _register_pipelines(registry: ToolRegistry, orchestrator) -> None:
    """Register high-level pipeline functions as tools.

    This allows the Agent Runtime to use research, build, and market
    capabilities as first-class tools, enabling the runtime to plan
    multi-pipeline workflows like "research X, then build Y".
    """
    def _research_topic(topic: str, mode: str = "deep") -> str:
        """Run deep research on a topic and return the report."""
        try:
            report = orchestrator.research.run(topic, mode)
            path = orchestrator.research.save(report)
            # Build summary with confidence
            summary = report.markdown[:3000]
            if report.confidence > 0:
                conf = "high" if report.confidence > 0.7 else "medium" if report.confidence > 0.4 else "low"
                summary += f"\n\nConfidence: {conf} ({report.confidence:.0%}) | Report saved: {path}"
            else:
                summary += f"\n\nReport saved: {path}"
            return summary
        except Exception as e:
            return f"research failed: {e}"

    def _build_project(description: str) -> str:
        """Build a complete project from a description."""
        try:
            path = orchestrator.engineering.build(description, report=orchestrator)
            eval_result = orchestrator.evaluator.evaluate_project(path)
            return f"Project built at {path}\nEvaluation: {eval_result.summary()}"
        except Exception as e:
            return f"build failed: {e}"

    def _market_analysis(topic: str, mode: str = "overview") -> str:
        """Run market analysis on a topic."""
        try:
            report = orchestrator.market.run(topic, mode)
            path = orchestrator.market.save(report)
            return f"Market report saved: {path}\n\n{report.markdown[:3000]}"
        except Exception as e:
            return f"market analysis failed: {e}"

    def _code_review(project_path: str) -> str:
        """Review code in a project directory."""
        try:
            from ultra.agents import reviewer
            result = reviewer.review_project(
                orchestrator.client, Path(project_path), orchestrator.config
            )
            return result[:3000]
        except Exception as e:
            return f"review failed: {e}"

    def _debug_code(project_path: str, issue: str = "") -> str:
        """Debug issues in a project."""
        try:
            from ultra.agents import debugger
            result = debugger.fix_issues(
                orchestrator.client, Path(project_path),
                issue or "find and fix issues", orchestrator.config
            )
            return result[:3000]
        except Exception as e:
            return f"debug failed: {e}"

    registry.register(ToolDefinition(
        name="pipeline.research",
        description="Run deep research on a topic. Returns a comprehensive "
                    "report with sources, citations, and analysis. "
                    "Use mode='compare' for comparisons, 'deep' for detailed research.",
        handler=_research_topic,
        params=[
            ToolParam("topic", "string", "Research topic or question", required=True),
            ToolParam("mode", "string", "Research mode: deep, compare, feasibility",
                     enum=["deep", "compare", "feasibility", "competitive"]),
        ],
        risk_level=RiskLevel.READ_ONLY,
        category="pipeline",
    ))
    registry.register(ToolDefinition(
        name="pipeline.build",
        description="Build a complete project from a description. Creates "
                    "architecture, code, tests, and documentation.",
        handler=_build_project,
        params=[
            ToolParam("description", "string",
                     "Detailed description of what to build", required=True),
        ],
        risk_level=RiskLevel.MODERATE,
        category="pipeline",
    ))
    registry.register(ToolDefinition(
        name="pipeline.market",
        description="Run market intelligence analysis. SWOT, competitive "
                    "analysis, trends, and market reports.",
        handler=_market_analysis,
        params=[
            ToolParam("topic", "string", "Market analysis topic", required=True),
            ToolParam("mode", "string", "Analysis mode",
                     enum=["overview", "swot", "trends", "competitors"]),
        ],
        risk_level=RiskLevel.READ_ONLY,
        category="pipeline",
    ))
    registry.register(ToolDefinition(
        name="pipeline.review",
        description="Review code in a project directory. Finds bugs, "
                    "style issues, and suggests improvements.",
        handler=_code_review,
        params=[
            ToolParam("project_path", "string",
                     "Path to the project directory", required=True),
        ],
        risk_level=RiskLevel.READ_ONLY,
        category="pipeline",
    ))
    registry.register(ToolDefinition(
        name="pipeline.debug",
        description="Debug issues in a project. Analyzes errors and "
                    "applies fixes.",
        handler=_debug_code,
        params=[
            ToolParam("project_path", "string",
                     "Path to the project directory", required=True),
            ToolParam("issue", "string", "Description of the issue to fix"),
        ],
        risk_level=RiskLevel.MODERATE,
        category="pipeline",
    ))
    logger.info("registered 5 pipeline tools")
