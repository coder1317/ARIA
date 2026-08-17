"""Tests for ultra.provider_pool — routing, failover, circuit breaker."""
import pytest

from ultra.config import Config, ProviderSpec
from ultra.provider_pool import (
    CIRCUIT_OPEN_SECONDS,
    CONSECUTIVE_FAILURES_TO_OPEN,
    ProviderExhaustedError,
    ProviderPool,
)


class FakeClient:
    """Deterministic stand-in for a provider client."""

    def __init__(self, name, fail=False, fail_times=0):
        self.name = name
        self.fail = fail
        self.fail_times = fail_times
        self.calls = 0
        self.last_messages = None

    def generate(self, prompt, **kwargs):
        self.calls += 1
        if self.fail and (self.fail_times == 0 or self.calls <= self.fail_times):
            raise RuntimeError(f"{self.name} down")
        return f"answer-from-{self.name}"

    def chat(self, messages, **kwargs):
        self.calls += 1
        self.last_messages = messages
        if self.fail and (self.fail_times == 0 or self.calls <= self.fail_times):
            raise RuntimeError(f"{self.name} down")
        return f"chat-from-{self.name}"


def _pool_with(clients: dict[str, FakeClient], specs: list[ProviderSpec]) -> ProviderPool:
    pool = ProviderPool.__new__(ProviderPool)
    pool.config = Config()
    pool.providers = clients
    pool.specs = {s.name: s for s in specs}
    from ultra.provider_pool import ProviderStats
    pool.stats = {s.name: ProviderStats() for s in specs}
    pool.circuit_open_until = {}
    pool._ollama = None
    pool._available_cache = None
    return pool


def _specs():
    return [
        ProviderSpec(name="primary", kind="ollama", model="m1",
                     base_url="http://x", capabilities=frozenset({"chat", "code"}),
                     priority=1),
        ProviderSpec(name="backup", kind="ollama", model="m2",
                     base_url="http://x", capabilities=frozenset({"chat", "code"}),
                     priority=2),
    ]


def test_routes_to_primary():
    clients = {"primary": FakeClient("primary"), "backup": FakeClient("backup")}
    pool = _pool_with(clients, _specs())
    text, provider = pool.execute("chat", "hello")
    assert text == "answer-from-primary"
    assert provider == "primary"
    assert clients["backup"].calls == 0


def test_failover_to_backup():
    clients = {"primary": FakeClient("primary", fail=True),
               "backup": FakeClient("backup")}
    pool = _pool_with(clients, _specs())
    text, provider = pool.execute("chat", "hello")
    assert text == "answer-from-backup"
    assert provider == "backup"


def test_all_providers_fail_raises():
    clients = {"primary": FakeClient("primary", fail=True),
               "backup": FakeClient("backup", fail=True)}
    pool = _pool_with(clients, _specs())
    with pytest.raises(ProviderExhaustedError):
        pool.execute("chat", "hello")


def test_circuit_breaker_opens_after_failures():
    clients = {"primary": FakeClient("primary", fail=True),
               "backup": FakeClient("backup")}
    pool = _pool_with(clients, _specs())
    for _ in range(CONSECUTIVE_FAILURES_TO_OPEN):
        pool.execute("chat", "hi")  # fails over to backup each time
    # primary should now be circuit-open and skipped entirely
    assert "primary" in pool.circuit_open_until
    assert "primary" not in pool._candidates("chat", None)
    assert pool.stats["primary"].health.value == "circuit_open"


def test_circuit_recovers_after_cooldown():
    clients = {"primary": FakeClient("primary", fail=True),
               "backup": FakeClient("backup")}
    pool = _pool_with(clients, _specs())
    for _ in range(CONSECUTIVE_FAILURES_TO_OPEN):
        pool.execute("chat", "hi")
    # simulate cooldown expiry
    pool.circuit_open_until["primary"] = 0
    candidates = pool._candidates("chat", None)
    assert "primary" in candidates


def test_model_param_filters_candidates():
    clients = {"primary": FakeClient("primary"), "backup": FakeClient("backup")}
    pool = _pool_with(clients, _specs())
    text, provider = pool.execute("code", "write code", model="m2")
    assert provider == "backup"
    assert text == "answer-from-backup"


def test_capability_routing_skips_primary():
    specs = [
        ProviderSpec(name="chat-only", kind="ollama", model="m1",
                     base_url="http://x", capabilities=frozenset({"chat"}),
                     priority=1),
        ProviderSpec(name="coder", kind="ollama", model="m2",
                     base_url="http://x", capabilities=frozenset({"code"}),
                     priority=2),
    ]
    clients = {"chat-only": FakeClient("chat-only"), "coder": FakeClient("coder")}
    pool = _pool_with(clients, specs)
    text, provider = pool.execute("code", "write code")
    assert provider == "coder"


def test_chat_injects_system_prompt_and_passes_history():
    clients = {"primary": FakeClient("primary"), "backup": FakeClient("backup")}
    pool = _pool_with(clients, _specs())
    messages = [{"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"}]
    out = pool.chat(messages, system="YOU ARE ARIA")
    assert out == "chat-from-primary"
    sent = clients["primary"].last_messages
    assert sent[0] == {"role": "system", "content": "YOU ARE ARIA"}
    assert len(sent) == 4  # system + 3 history messages preserved


def test_health_report_shape():
    clients = {"primary": FakeClient("primary"), "backup": FakeClient("backup")}
    pool = _pool_with(clients, _specs())
    pool.execute("chat", "hello")
    report = pool.health_report()
    assert "primary" in report
    assert report["primary"]["success_rate"] == 1.0
    assert report["primary"]["calls"] == 1
