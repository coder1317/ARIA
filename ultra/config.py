"""Configuration for ARIA.

Layered config: defaults → optional config.yaml → environment/.env.
Environment variables always win. No API keys required — the default
setup is local-only Ollama, with optional OpenAI-compatible cloud
providers activated only when their env vars are present.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def _load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no external dependency)."""
    candidates = [
        path,
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            # strip inline comments ("KEY=6  # note")
            if " #" in value:
                value = value.split(" #", 1)[0].strip()
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass
class ProviderSpec:
    """A single LLM provider known to the pool."""

    name: str
    kind: str  # "ollama" | "openai"
    model: str
    base_url: str
    api_key: str | None = None
    capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset({"chat", "research", "json"})
    )
    priority: int = 1  # lower = preferred
    rpm: int = 60

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None or self.kind == "ollama"


@dataclass
class Config:
    ollama_url: str = "http://localhost:11434"
    chat_model: str = "lfm2.5"
    coding_model: str = "lfm2.5"
    fallback_model: str = "hermes3:3b"
    embed_model: str = "nomic-embed-text"
    search_timeout: int = 15
    research_max_sources: int = 6
    data_dir: Path = field(default_factory=lambda: Path.home() / ".aria4")
    projects_dir: Path = field(default_factory=lambda: Path.home() / "aria4_projects")
    context_window: int = 6  # conversation exchanges kept in memory
    # Safety defaults for the bash tool
    command_timeout: int = 60
    output_cap: int = 10_000
    llm_timeout: int = 900  # seconds — deep research/build generations are slow on 3B models
    # Ultra additions
    providers: list[ProviderSpec] = field(default_factory=list)
    audit_db: Path = field(default_factory=lambda: Path.home() / ".aria4" / "memory" / "audit.db")
    max_concurrent_tasks: int = 2
    task_timeout_sec: float = 600.0
    max_build_attempts: int = 3
    auto_extract_skills: bool = False
    security_enabled: bool = True
    # Telegram channel
    telegram_token: str = ""
    telegram_allowed_users: list[int] = field(default_factory=list)
    # Scheduler
    scheduler_enabled: bool = False
    scheduler_interval: int = 30  # seconds between checks
    # MCP
    mcp_servers: list[str] = field(default_factory=list)  # "name:command args"
    # Browser
    browser_enabled: bool = False
    # Agent Runtime
    runtime_enabled: bool = False
    runtime_max_iterations: int = 30
    runtime_max_replans: int = 5

    @classmethod
    def load(cls, env_path: Path | None = None,
             config_path: Path | None = None) -> "Config":
        _load_dotenv(env_path)
        home = Path.home()

        cfg = cls(
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "lfm2.5"),
            coding_model=os.getenv("OLLAMA_CODING_MODEL", "lfm2.5"),
            fallback_model=os.getenv("OLLAMA_FALLBACK_MODEL", "hermes3:3b"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            search_timeout=int(os.getenv("SEARCH_TIMEOUT_SECONDS", "15")),
            research_max_sources=int(os.getenv("RESEARCH_MAX_SOURCES", "6")),
            data_dir=Path(os.getenv("ARIA4_HOME", str(home / ".aria4"))),
            projects_dir=Path(os.getenv("ARIA4_PROJECTS", str(home / "aria4_projects"))),
            context_window=int(os.getenv("ARIA4_CONTEXT", "6")),
            command_timeout=int(os.getenv("ARIA4_CMD_TIMEOUT", "60")),
            output_cap=int(os.getenv("ARIA4_CMD_OUTPUT_CAP", "10000")),
            llm_timeout=int(os.getenv("ARIA4_LLM_TIMEOUT", "900")),
            audit_db=Path(os.getenv(
                "ARIA4_AUDIT_DB", str(home / ".aria4" / "memory" / "audit.db"))),
            max_concurrent_tasks=int(os.getenv("ARIA4_MAX_TASKS", "2")),
            task_timeout_sec=float(os.getenv("ARIA4_TASK_TIMEOUT", "600")),
            max_build_attempts=int(os.getenv("ARIA4_MAX_BUILD_ATTEMPTS", "3")),
            auto_extract_skills=os.getenv("ARIA4_EXTRACT_SKILLS", "").lower() in ("1", "true", "yes"),
            security_enabled=os.getenv("ARIA4_SECURITY", "1").lower() not in ("0", "false", "no"),
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_allowed_users=[int(x) for x in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",") if x.strip().isdigit()],
            scheduler_enabled=os.getenv("ARIA4_SCHEDULER", "").lower() in ("1", "true", "yes"),
            scheduler_interval=int(os.getenv("ARIA4_SCHEDULER_INTERVAL", "30")),
            mcp_servers=[s.strip() for s in os.getenv("ARIA_MCP_SERVERS", "").split(",") if s.strip()],
            browser_enabled=os.getenv("ARIA4_BROWSER", "").lower() in ("1", "true", "yes"),
            runtime_enabled=os.getenv("ARIA4_RUNTIME", "").lower() in ("1", "true", "yes"),
            runtime_max_iterations=int(os.getenv("ARIA4_RUNTIME_MAX_ITER", "30")),
            runtime_max_replans=int(os.getenv("ARIA4_RUNTIME_MAX_REPLAN", "5")),
        )
        cfg._apply_yaml(config_path)
        cfg.providers = cls._build_providers(cfg)
        return cfg

    # ── layering ────────────────────────────────────────────────────

    def _apply_yaml(self, config_path: Path | None) -> None:
        """Optional config.yaml overrides (env still wins after this)."""
        if not _HAS_YAML:
            return  # PyYAML not installed — skip YAML layer entirely
        candidates = [config_path, Path.cwd() / "config.yaml",
                      Path(__file__).resolve().parent.parent / "config.yaml"]
        raw: dict = {}
        for c in candidates:
            if c and c.is_file():
                raw = yaml.safe_load(c.read_text(encoding="utf-8")) or {}
                break
        if not raw:
            return
        for key in ("ollama_url", "chat_model", "coding_model", "fallback_model",
                    "embed_model", "search_timeout", "research_max_sources",
                    "context_window", "command_timeout", "output_cap",
                    "llm_timeout", "max_concurrent_tasks", "task_timeout_sec",
                    "max_build_attempts", "auto_extract_skills", "security_enabled"):
            if key in raw and os.getenv(_ENV_MAP.get(key, "")) is None:
                setattr(self, key, raw[key])
        if "data_dir" in raw and os.getenv("ARIA4_HOME") is None:
            self.data_dir = Path(raw["data_dir"]).expanduser()
        if "projects_dir" in raw and os.getenv("ARIA4_PROJECTS") is None:
            self.projects_dir = Path(raw["projects_dir"]).expanduser()
        if "audit_db" in raw and os.getenv("ARIA4_AUDIT_DB") is None:
            self.audit_db = Path(raw["audit_db"]).expanduser()

    @staticmethod
    def _build_providers(cfg: "Config") -> list[ProviderSpec]:
        """Default provider set: two local models + optional cloud.

        The chat and coding models are separate providers so code paths
        route to the coding model first; the fallback model (hermes) is a
        last-resort provider that catches any task type.
        """
        specs = [
            ProviderSpec(name="chat", kind="ollama", model=cfg.chat_model,
                         base_url=cfg.ollama_url,
                         capabilities=frozenset({"chat", "research", "json"}),
                         priority=1),
            ProviderSpec(name="coding", kind="ollama", model=cfg.coding_model,
                         base_url=cfg.ollama_url,
                         capabilities=frozenset({"code", "json"}),
                         priority=1),
            ProviderSpec(name="fallback", kind="ollama", model=cfg.fallback_model,
                         base_url=cfg.ollama_url,
                         capabilities=frozenset({"chat", "code", "research", "json"}),
                         priority=5),
        ]
        # Optional OpenAI-compatible cloud provider (Groq, OpenRouter,
        # Cerebras, ...). Activates only when both vars are set.
        if os.getenv("ULTRA_CLOUD_BASE_URL") and os.getenv("ULTRA_CLOUD_API_KEY"):
            specs.append(ProviderSpec(
                name="cloud",
                kind="openai",
                model=os.getenv("ULTRA_CLOUD_MODEL", ""),
                base_url=os.getenv("ULTRA_CLOUD_BASE_URL", "").rstrip("/"),
                api_key=os.getenv("ULTRA_CLOUD_API_KEY"),
                capabilities=frozenset({"chat", "research", "json", "code"}),
                priority=3,
                rpm=int(os.getenv("ULTRA_CLOUD_RPM", "30")),
            ))
        return specs

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.projects_dir,
                  self.data_dir / "memory", self.audit_db.parent):
            d.mkdir(parents=True, exist_ok=True)


_ENV_MAP = {
    "ollama_url": "OLLAMA_BASE_URL",
    "chat_model": "OLLAMA_CHAT_MODEL",
    "coding_model": "OLLAMA_CODING_MODEL",
    "fallback_model": "OLLAMA_FALLBACK_MODEL",
    "embed_model": "OLLAMA_EMBED_MODEL",
    "search_timeout": "SEARCH_TIMEOUT_SECONDS",
    "research_max_sources": "RESEARCH_MAX_SOURCES",
    "context_window": "ARIA4_CONTEXT",
    "command_timeout": "ARIA4_CMD_TIMEOUT",
    "output_cap": "ARIA4_CMD_OUTPUT_CAP",
    "llm_timeout": "ARIA4_LLM_TIMEOUT",
    "max_concurrent_tasks": "ARIA4_MAX_TASKS",
    "task_timeout_sec": "ARIA4_TASK_TIMEOUT",
    "max_build_attempts": "ARIA4_MAX_BUILD_ATTEMPTS",
    "auto_extract_skills": "ARIA4_EXTRACT_SKILLS",
    "security_enabled": "ARIA4_SECURITY",
}
