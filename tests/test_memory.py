import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultra.core.memory import Memory
from ultra.core.vectors import VectorStore
from ultra.llm import cosine_similarity


def test_memory_roundtrip(tmp_path):
    mem = Memory(tmp_path / "m.db")
    mem.add_message("user", "hello world")
    mem.add_message("assistant", "hi there")
    recent = mem.recent()
    assert len(recent) == 2
    assert recent[-1]["role"] == "assistant"


def test_memory_fts_search(tmp_path):
    mem = Memory(tmp_path / "m.db")
    mem.add_message("user", "I like raspberry pi projects")
    hits = mem.search("raspberry")
    assert any("raspberry" in h["content"] for h in hits)


def test_facts(tmp_path):
    mem = Memory(tmp_path / "m.db")
    mem.set_fact("name", "Hari", "user")
    mem.set_fact("os", "Ubuntu", "user")
    facts = mem.get_facts("user")
    assert facts["name"] == "Hari"
    assert facts["os"] == "Ubuntu"


def test_projects(tmp_path):
    mem = Memory(tmp_path / "m.db")
    pid = mem.save_project("test-app", "/tmp/test-app", "a test")
    projects = mem.get_projects()
    assert projects[0]["name"] == "test-app"


def test_lessons(tmp_path):
    mem = Memory(tmp_path / "m.db")
    mem.add_lesson("coder", "forgot imports", "list imports explicitly")
    lessons = mem.get_lessons()
    assert lessons[0]["agent"] == "coder"


def test_skills_crud(tmp_path):
    mem = Memory(tmp_path / "m.db")
    mem.save_skill("deploy", ["deploy", "ship"], ["run tests", "push"])
    skills = mem.get_skills()
    assert skills[0]["name"] == "deploy"


def test_cosine():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    c = [2.0, 0.0]
    assert abs(cosine_similarity(a, c) - 1.0) < 1e-9
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_vectors_store(tmp_path):
    vs = VectorStore(tmp_path / "v.db")
    # offline: no Ollama call, add() should fail gracefully
    assert vs.count() == 0
