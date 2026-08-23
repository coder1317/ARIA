"""ProviderPool — multi-provider LLM routing with health + failover.

Implements the same interface as OllamaClient (generate / json / chat /
embed / available_models / ping) so every agent keeps working unchanged,
while calls are routed across providers with:

- capability routing (code tasks → coding model, chat → chat model, ...)
- health tracking (latency EMA, success rate, consecutive failures)
- circuit breaker (3 consecutive failures → open for 60s)
- automatic failover (next provider in priority order on any error)
- optional OpenAI-compatible cloud providers, activated via env vars

Local-first: with no API keys configured the pool is just the two Ollama
models (granite primary, hermes fallback) — exactly the proven v4 setup.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ultra.config import Config, ProviderSpec
from ultra.llm import OllamaClient, OpenAICompatClient, OllamaError

# task_type → required capability
TASK_CAPABILITY = {
    "chat": "chat",
    "research": "research",
    "code": "code",
    "code_generation": "code",
    "code_debug": "code",
    "json": "json",
    "routing": "chat",
    "embedding": "embed",
    "generic": "chat",
    # Phase 4: complexity-based routing
    "planning": "chat",      # complex planning needs strong model
    "evaluation": "chat",    # evaluation needs strong model
    "simple": "chat",        # simple tasks can use any model
}

CIRCUIT_OPEN_SECONDS = 60.0
CONSECUTIVE_FAILURES_TO_OPEN = 3
MAX_RETRIES = 2  # retry transient failures before failover
RETRY_BACKOFF = [0.5, 1.5]  # seconds between retries


class _RateLimiter:
    """Simple sliding-window rate limiter per provider."""

    def __init__(self, rpm: int = 60):
        self.rpm = rpm
        self._window: list[float] = []

    def acquire(self) -> float:
        """Wait if necessary; returns wait time in seconds."""
        now = time.time()
        # Remove timestamps older than 60s
        self._window = [t for t in self._window if now - t < 60.0]
        if len(self._window) >= self.rpm:
            # Must wait until the oldest call falls out of the window
            wait = 60.0 - (now - self._window[0]) + 0.1
            time.sleep(wait)
            now = time.time()
            self._window = [t for t in self._window if now - t < 60.0]
        self._window.append(now)
        return 0.0


class ProviderHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class ProviderStats:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    last_error: str | None = None
    last_called: float = 0.0
    health: ProviderHealth = ProviderHealth.HEALTHY

    @property
    def success_rate(self) -> float:
        return self.successful_calls / max(self.total_calls, 1)

    def record_success(self, latency_ms: float) -> None:
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_failures = 0
        self.last_called = time.time()
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency_ms
        if self.health in (ProviderHealth.DEGRADED, ProviderHealth.UNHEALTHY):
            self.health = ProviderHealth.HEALTHY

    def record_failure(self, error: str) -> None:
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.last_error = error
        self.last_called = time.time()
        if self.consecutive_failures >= CONSECUTIVE_FAILURES_TO_OPEN:
            self.health = ProviderHealth.UNHEALTHY


class ProviderExhaustedError(RuntimeError):
    """All providers failed for a task."""


class ProviderPool:
    """Routes LLM calls across providers with health + failover."""

    def __init__(self, config: Config):
        self.config = config
        self.providers: dict[str, Any] = {}   # name → client
        self.specs: dict[str, ProviderSpec] = {}
        self.stats: dict[str, ProviderStats] = {}
        self.circuit_open_until: dict[str, float] = {}
        self._rate_limiters: dict[str, _RateLimiter] = {}  # P1-7
        self._ollama: OllamaClient | None = None
        self._build_providers()
        self._available_cache: list[str] | None = None

    # ── setup ───────────────────────────────────────────────────────

    def _build_providers(self) -> None:
        """Register providers from config, de-duplicating by (url, model)."""
        merged: dict[tuple, ProviderSpec] = {}
        for spec in self.config.providers:
            if not spec.is_configured or not spec.model:
                continue
            key = (spec.kind, spec.base_url, spec.model, spec.api_key)
            if key in merged:
                old = merged[key]
                merged[key] = ProviderSpec(
                    name=old.name, kind=old.kind, model=old.model,
                    base_url=old.base_url, api_key=old.api_key,
                    capabilities=old.capabilities | spec.capabilities,
                    priority=min(old.priority, spec.priority),
                    rpm=min(old.rpm, spec.rpm),
                )
            else:
                merged[key] = spec
        for spec in merged.values():
            if spec.kind == "ollama":
                if self._ollama is None:
                    self._ollama = OllamaClient(self.config)
                client = self._ollama
            else:
                client = OpenAICompatClient(spec.base_url, spec.api_key or "", spec.model)
            self.providers[spec.name] = client
            self.specs[spec.name] = spec
            self.stats[spec.name] = ProviderStats()
            self._rate_limiters[spec.name] = _RateLimiter(rpm=spec.rpm)

    # ── routing ─────────────────────────────────────────────────────

    def _required_capability(self, task_type: str) -> str:
        return TASK_CAPABILITY.get(task_type, "chat")

    def _candidates(self, task_type: str, model: str | None) -> list[str]:
        """Providers eligible for this call, best-first."""
        cap = self._required_capability(task_type)
        now = time.time()
        # reset circuit breakers that have cooled down
        for name in list(self.circuit_open_until):
            if now >= self.circuit_open_until[name]:
                del self.circuit_open_until[name]
                self.stats[name].health = ProviderHealth.HEALTHY
                self.stats[name].consecutive_failures = 0

        ranked: list[tuple[int, float, str]] = []
        for name, spec in self.specs.items():
            if name in self.circuit_open_until:
                continue
            if self.stats[name].health == ProviderHealth.UNHEALTHY:
                continue
            if model is not None and spec.model != model:
                continue  # caller asked for a specific model
            if model is None and cap not in spec.capabilities:
                continue
            ranked.append((spec.priority, self.stats[name].avg_latency_ms, name))
        ranked.sort()
        return [name for _, _, name in ranked]

    # ── public interface (OllamaClient-compatible) ──────────────────

    def generate(self, prompt: str, system: str | None = None,
                 model: str | None = None, max_tokens: int = 4096,
                 temperature: float = 0.7, format: str | None = None,
                 task_type: str = "chat", **_) -> str:
        text, _ = self.execute(task_type, prompt, system=system, model=model,
                               max_tokens=max_tokens, temperature=temperature,
                               format=format)
        return text

    def json(self, prompt: str, system: str | None = None,
             model: str | None = None, max_tokens: int = 2048,
             temperature: float = 0.0, task_type: str = "json", **_) -> dict | None:
        import json as _json
        try:
            raw = self.generate(prompt, system=system, model=model,
                                max_tokens=max_tokens, temperature=temperature,
                                format="json", task_type=task_type)
            return _json.loads(raw)
        except (_json.JSONDecodeError, ProviderExhaustedError, OllamaError):
            return None

    def chat(self, messages: list[dict], model: str | None = None,
             system: str | None = None, task_type: str = "chat", **opts) -> str:
        """Conversational completion — full history + system prompt.

        The system prompt is injected as a leading system message (the
        v4 client silently dropped it in chat; this fixes that).
        """
        msgs = list(messages)
        if system and not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": system}] + msgs
        attempted: list[str] = []
        last_error: str | None = None
        for name in self._candidates(task_type, model):
            spec = self.specs[name]
            client = self.providers[name]
            attempted.append(name)
            # P1-7: Rate limit
            limiter = self._rate_limiters.get(name)
            if limiter:
                limiter.acquire()
            # P1-9: Retry with backoff
            for retry in range(MAX_RETRIES + 1):
                try:
                    start = time.time()
                    text = client.chat(msgs, model=spec.model, **opts)
                    self.stats[name].record_success((time.time() - start) * 1000)
                    return text
                except Exception as e:
                    last_error = f"{name}: {e}"
                    if retry < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF[min(retry, len(RETRY_BACKOFF) - 1)])
            self.stats[name].record_failure(str(last_error))
            if self.stats[name].health == ProviderHealth.UNHEALTHY:
                self.circuit_open_until[name] = time.time() + CIRCUIT_OPEN_SECONDS
                self.stats[name].health = ProviderHealth.CIRCUIT_OPEN
        raise ProviderExhaustedError(
            f"All providers failed for chat. Attempted: {attempted}. "
            f"Last error: {last_error}"
        )

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        model: str | None = None,
                        task_type: str = "chat", **opts) -> dict:
        """Chat with native tool calling support across providers.

        Returns dict with: text, tool_calls, finish_reason.
        Routes through providers with retry + failover.
        """
        msgs = list(messages)
        attempted: list[str] = []
        last_error: str | None = None
        for name in self._candidates(task_type, model):
            spec = self.specs[name]
            client = self.providers[name]
            attempted.append(name)
            limiter = self._rate_limiters.get(name)
            if limiter:
                limiter.acquire()
            for retry in range(MAX_RETRIES + 1):
                try:
                    start = time.time()
                    result = client.chat_with_tools(
                        msgs, tools, model=spec.model, **opts)
                    self.stats[name].record_success((time.time() - start) * 1000)
                    return result
                except Exception as e:
                    last_error = f"{name}: {e}"
                    if retry < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF[min(retry, len(RETRY_BACKOFF) - 1)])
            self.stats[name].record_failure(str(last_error))
            if self.stats[name].health == ProviderHealth.UNHEALTHY:
                self.circuit_open_until[name] = time.time() + CIRCUIT_OPEN_SECONDS
                self.stats[name].health = ProviderHealth.CIRCUIT_OPEN
        raise ProviderExhaustedError(
            f"All providers failed for chat_with_tools. Attempted: {attempted}. "
            f"Last error: {last_error}"
        )

    def embed(self, text: str, model: str | None = None) -> list[float]:
        if self._ollama is None:
            raise ProviderExhaustedError("No local Ollama provider for embeddings")
        return self._ollama.embed(text, model=model)

    def available_models(self) -> list[str]:
        if self._ollama is None:
            return []
        return self._ollama.available_models()

    def ping(self, retries: int = 1) -> bool:
        if self._ollama is None:
            return False
        return self._ollama.ping(retries=retries)

    def ensure_model(self, model: str) -> bool:
        return model in self.available_models()

    # ── core execute with failover ──────────────────────────────────

    def execute(self, task_type: str, prompt: str, system: str | None = None,
                model: str | None = None, max_tokens: int = 4096,
                temperature: float = 0.7, format: str | None = None,
                timeout: int = 300) -> tuple[str, str]:
        """Run a call with retry + rate limiting, failing over across providers.

        Returns (text, provider). Each provider gets up to MAX_RETRIES attempts
        before failing over to the next candidate.
        """
        attempted: list[str] = []
        last_error: str | None = None
        for name in self._candidates(task_type, model):
            spec = self.specs[name]
            client = self.providers[name]
            attempted.append(name)
            # P1-7: Rate limit before calling
            limiter = self._rate_limiters.get(name)
            if limiter:
                limiter.acquire()
            # P1-9: Retry with backoff per provider
            for retry in range(MAX_RETRIES + 1):
                try:
                    start = time.time()
                    text = client.generate(
                        prompt, system=system, model=spec.model,
                        max_tokens=max_tokens, temperature=temperature,
                        format=format,
                    )
                    latency_ms = (time.time() - start) * 1000
                    self.stats[name].record_success(latency_ms)
                    return text, name
                except Exception as e:
                    last_error = f"{name}: {e}"
                    if retry < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF[min(retry, len(RETRY_BACKOFF) - 1)])
            # All retries exhausted for this provider
            self.stats[name].record_failure(str(last_error))
            if self.stats[name].health == ProviderHealth.UNHEALTHY:
                self.circuit_open_until[name] = time.time() + CIRCUIT_OPEN_SECONDS
                self.stats[name].health = ProviderHealth.CIRCUIT_OPEN
        raise ProviderExhaustedError(
            f"All providers failed for task_type={task_type} (model={model}). "
            f"Attempted: {attempted}. Last error: {last_error}"
        )

    # ── introspection ───────────────────────────────────────────────

    def health_report(self) -> dict[str, dict]:
        return {
            name: {
                "model": self.specs[name].model,
                "kind": self.specs[name].kind,
                "health": self.stats[name].health.value,
                "success_rate": round(self.stats[name].success_rate, 3),
                "avg_latency_ms": round(self.stats[name].avg_latency_ms, 1),
                "calls": self.stats[name].total_calls,
                "last_error": self.stats[name].last_error,
            }
            for name in self.specs
        }

    def reset_health(self) -> None:
        for name in self.stats:
            self.stats[name] = ProviderStats()
        self.circuit_open_until.clear()

    # Phase 4: Model routing based on task complexity

    def route_for_complexity(self, task_description: str) -> str:
        """Select the best provider based on task complexity.

        Simple tasks (short, factual) → fastest model
        Complex tasks (long, analytical) → strongest model
        """
        desc = task_description.lower()
        word_count = len(task_description.split())

        # Complex tasks: long, multi-step, analytical
        complex_markers = [
            "analyze", "research", "compare", "design", "architect",
            "plan", "evaluate", "review", "debug", "optimize",
            "comprehensive", "detailed", "thorough",
        ]
        is_complex = (word_count > 30 or
                      any(m in desc for m in complex_markers))

        if is_complex:
            # Use the strongest available model
            candidates = self._candidates("chat", None)
            if candidates:
                # Prefer cloud models for complex tasks
                for name in candidates:
                    if "cloud" in self.specs[name].model.lower():
                        return name
                return candidates[0]
        else:
            # Use the fastest model
            candidates = self._candidates("chat", None)
            if candidates:
                # Prefer lowest latency
                return min(candidates,
                          key=lambda n: self.stats[n].avg_latency_ms or 999)
        return ""
