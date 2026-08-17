"""Tests for the Ultra agents: market, deployer, trainer."""
import pytest

from ultra.agents.deployer import generate_deployment, PLATFORMS
from ultra.agents.trainer import TrainerAgent
from ultra.config import Config


# ── deployer ─────────────────────────────────────────────────────────

class FakePool:
    def __init__(self, response):
        self.response = response
        self.config = Config()

    def generate(self, prompt, **kwargs):
        return self.response


def test_deployer_parses_files(tmp_path):
    (tmp_path / "app.py").write_text("print('hi')")
    pool = FakePool(
        "---FILE: Dockerfile ---\nFROM python:3.12-slim\n"
        "---FILE: docker-compose.yml ---\nservices:\n  app:\n    build: .\n"
        "---FILE: .github/workflows/ci.yml ---\nname: CI\n"
    )
    written = generate_deployment(pool, tmp_path, "docker")
    paths = {str(p.relative_to(tmp_path)) for p in written}
    assert "Dockerfile" in paths
    assert "docker-compose.yml" in paths
    assert ".github/workflows/ci.yml" in paths
    assert (tmp_path / "Dockerfile").read_text().startswith("FROM python")


def test_deployer_sanitizes_paths(tmp_path):
    (tmp_path / "app.py").write_text("print('hi')")
    pool = FakePool("---FILE: ../../etc/evil.sh ---\nrm -rf /\n")
    written = generate_deployment(pool, tmp_path, "docker")
    # path traversal must be neutralized — file lands inside the project
    assert all(str(p).startswith(str(tmp_path)) for p in written)


def test_deployer_platforms():
    assert "docker" in PLATFORMS
    assert "github_actions" in PLATFORMS


# ── trainer ──────────────────────────────────────────────────────────

@pytest.fixture
def trainer(tmp_path):
    config = Config()
    config.data_dir = tmp_path
    return TrainerAgent(FakePool(""), config)


def test_trainer_parses_skill(trainer):
    raw = (
        "SKILL NAME: rest_api_patterns\n"
        "DESCRIPTION: Use this when building REST APIs\n"
        "TRIGGERS: rest, api, endpoint\n"
        "BODY:\n"
        "- Use nouns for resources\n"
        "- Version your API\n"
    )
    skill = trainer._parse(raw)
    assert skill["name"] == "rest_api_patterns"
    assert "REST APIs" in skill["description"]
    assert "rest" in skill["triggers"]
    assert "nouns" in skill["body"]


def test_trainer_parse_requires_name_and_description(trainer):
    assert trainer._parse("just some text") is None
    assert trainer._parse("SKILL NAME: only_name\nDESCRIPTION: desc\n") is not None


def test_trainer_writes_skill_file(trainer):
    skill = {"name": "test-skill", "description": "desc here",
             "triggers": "test, demo", "body": "- do things"}
    name = trainer._write_skill(skill)
    assert name == "test-skill"
    md = (trainer.skills_dir / "test-skill" / "SKILL.md").read_text()
    assert "description: desc here" in md
    assert "- do things" in md


def test_trainer_extract_returns_none_without_python(tmp_path, trainer):
    (tmp_path / "notes.txt").write_text("hello")
    assert trainer.extract_skill(tmp_path) is None


# ── market mode detection ────────────────────────────────────────────

# ── debugger security verification ────────────────────────────────

def test_debugger_does_not_trust_no_bugs_found(tmp_path):
    from ultra.agents.debugger import _still_has_critical, fix_issues
    f = tmp_path / "app.py"
    f.write_text("def calc():\n    return eval('1+1')\n")

    class Fake:
        config = Config()

        def generate(self, prompt, **kw):
            return "NO BUGS FOUND"  # model wrongly claims it's fine

    all_clear, _ = fix_issues(Fake(), tmp_path,
                              [{"category": "security", "file": "app.py",
                                "problem": "eval() is dangerous"}])
    assert all_clear is False
    assert _still_has_critical(f) is True


def test_debugger_accepts_real_fix(tmp_path):
    from ultra.agents.debugger import fix_issues
    f = tmp_path / "app.py"
    f.write_text("def calc():\n    return eval('1+1')\n")

    class Fake:
        config = Config()

        def generate(self, prompt, **kw):
            return "---FILE: app.py ---\ndef calc():\n    return 2\n"

    all_clear, passes = fix_issues(Fake(), tmp_path,
                                   [{"category": "security", "file": "app.py",
                                     "problem": "eval() is dangerous"}])
    assert all_clear is True
    assert passes == 1
    assert "eval" not in f.read_text()


def test_debugger_rejects_fix_that_keeps_eval(tmp_path):
    from ultra.agents.debugger import fix_issues
    f = tmp_path / "app.py"
    f.write_text("def calc():\n    return eval('1+1')\n")

    class Fake:
        config = Config()

        def generate(self, prompt, **kw):
            # parseable output, but eval is still there → not a real fix
            return "---FILE: app.py ---\ndef calc():\n    return eval('2+2')\n"

    all_clear, _ = fix_issues(Fake(), tmp_path,
                              [{"category": "security", "file": "app.py",
                                "problem": "eval() is dangerous"}])
    assert all_clear is False


def test_still_has_critical_detects_clean_file(tmp_path):
    from ultra.agents.debugger import _still_has_critical
    f = tmp_path / "app.py"
    f.write_text("def add(a, b):\n    return a + b\n")
    assert _still_has_critical(f) is False


def test_market_mode_detection():
    from ultra.orchestrator import Orchestrator
    orch = object.__new__(Orchestrator)
    assert orch._detect_market_mode("swot analysis of tesla") == "swot"
    assert orch._detect_market_mode("ai trends 2026") == "trends"
    assert orch._detect_market_mode("competitive landscape for saas") == "competitors"
    assert orch._detect_market_mode("the ev market") == "overview"


def test_market_citation_stripping():
    from ultra.agents.market import CITATION_MARKER, CITATION_LINE
    text = "Source: [1] Fake\n\nThe market is big [1][2]."
    text = CITATION_LINE.sub("", text)
    text = CITATION_MARKER.sub("", text)
    assert "Fake" not in text
    assert "[1]" not in text
