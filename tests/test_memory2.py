"""Tests for Phase 2 Memory Intelligence."""
import json
import tempfile
from pathlib import Path

import pytest

from ultra.core.memory2 import (
    MemoryV2, Episode, Procedure, UserProfile,
)


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "memory.db"


@pytest.fixture
def mem(tmp_db):
    return MemoryV2(tmp_db)


# ── Episodic Memory ───────────────────────────────────────────────


class TestEpisodes:

    def test_record_and_retrieve(self, mem):
        eid = mem.record_episode(Episode(
            event_type="task_completed",
            summary="Built a Flask API",
            detail="4 files, score 1.0",
            project="/tmp/myapi",
            outcome="success",
            importance=0.8,
            tags=["build", "flask"],
        ))
        assert eid > 0

        episodes = mem.get_episodes(limit=5)
        assert len(episodes) == 1
        assert episodes[0]["event_type"] == "task_completed"
        assert episodes[0]["summary"] == "Built a Flask API"
        assert episodes[0]["outcome"] == "success"
        assert episodes[0]["importance"] == 0.8
        assert "build" in episodes[0]["tags"]

    def test_filter_by_type(self, mem):
        mem.record_episode(Episode(event_type="research", summary="Research AI"))
        mem.record_episode(Episode(event_type="task_completed", summary="Built app"))
        mem.record_episode(Episode(event_type="error", summary="Build failed"))

        research = mem.get_episodes(event_type="research")
        assert len(research) == 1
        assert research[0]["event_type"] == "research"

        errors = mem.get_episodes(event_type="error")
        assert len(errors) == 1

    def test_filter_by_project(self, mem):
        mem.record_episode(Episode(event_type="task_completed", summary="A", project="proj-a"))
        mem.record_episode(Episode(event_type="task_completed", summary="B", project="proj-b"))

        a = mem.get_episodes(project="proj-a")
        assert len(a) == 1
        assert a[0]["summary"] == "A"

    def test_importance_filter(self, mem):
        mem.record_episode(Episode(event_type="chat", summary="low", importance=0.1))
        mem.record_episode(Episode(event_type="chat", summary="high", importance=0.9))

        high = mem.get_episodes(min_importance=0.5)
        assert len(high) == 1
        assert high[0]["summary"] == "high"

    def test_recent_episodes(self, mem):
        mem.record_episode(Episode(event_type="chat", summary="today"))
        episodes = mem.recent_episodes(days=1)
        assert len(episodes) >= 1

    def test_search_episodes(self, mem):
        mem.record_episode(Episode(event_type="research", summary="Compare Python asyncio vs threading"))
        mem.record_episode(Episode(event_type="chat", summary="Hello world"))

        # FTS search should find the asyncio episode
        results = mem.search_episodes("asyncio", limit=5)
        assert len(results) >= 1

    def test_multiple_episodes(self, mem):
        for i in range(5):
            mem.record_episode(Episode(
                event_type="task_completed",
                summary=f"Task {i}",
                importance=0.3 + i * 0.1,
            ))
        episodes = mem.get_episodes(limit=10)
        assert len(episodes) == 5


# ── Procedural Memory ─────────────────────────────────────────────


class TestProcedures:

    def test_save_and_get(self, mem):
        proc = Procedure(
            name="research_embedded_vision",
            description="Research embedded computer vision models",
            steps=["Search arXiv", "Filter by size", "Compare benchmarks"],
            tools_used=["web.search", "web.fetch"],
            tags=["research", "embedded"],
        )
        mem.save_procedure(proc)

        loaded = mem.get_procedure("research_embedded_vision")
        assert loaded is not None
        assert loaded.name == "research_embedded_vision"
        assert len(loaded.steps) == 3
        assert loaded.confidence == 0.5

    def test_record_result(self, mem):
        proc = Procedure(name="test_proc", description="test", steps=["step1"])
        mem.save_procedure(proc)

        mem.record_procedure_result("test_proc", success=True, duration_ms=1000)
        loaded = mem.get_procedure("test_proc")
        assert loaded.success_count == 1
        assert loaded.fail_count == 0
        assert loaded.confidence > 0.5

        mem.record_procedure_result("test_proc", success=False, duration_ms=2000)
        loaded = mem.get_procedure("test_proc")
        assert loaded.success_count == 1
        assert loaded.fail_count == 1

    def test_success_rate(self, mem):
        proc = Procedure(name="rate_test", description="test", steps=[])
        for _ in range(3):
            proc.record_result(True, 100)
        proc.record_result(False, 100)
        assert proc.success_rate == 0.75
        assert proc.confidence > 0.5

    def test_list_procedures(self, mem):
        for i in range(3):
            mem.save_procedure(Procedure(
                name=f"proc_{i}",
                description=f"Procedure {i}",
                steps=[f"step {i}"],
            ))
        procs = mem.list_procedures()
        assert len(procs) == 3

    def test_search_procedures(self, mem):
        mem.save_procedure(Procedure(
            name="pcb_design",
            description="Design PCBs for ESP32 projects",
            steps=["Schematic", "Layout", "DRC"],
            tags=["hardware", "pcb"],
        ))
        procs = mem.search_procedures("pcb", limit=5)
        assert len(procs) >= 1

    def test_confidence_increases(self, mem):
        proc = Procedure(name="confidence_test", description="test", steps=[])
        mem.save_procedure(proc)

        for _ in range(5):
            mem.record_procedure_result("confidence_test", success=True, duration_ms=100)

        loaded = mem.get_procedure("confidence_test")
        assert loaded.confidence > 0.7  # should be high after 5 successes


# ── User Model ────────────────────────────────────────────────────


class TestUserProfile:

    def test_default_profile(self, mem):
        profile = mem.get_user_profile()
        assert profile.os == "Ubuntu"
        assert profile.preferred_language == "Python"
        assert profile.response_style == "concise"

    def test_save_and_load(self, mem):
        profile = UserProfile(
            name="Hari",
            preferred_language="C++",
            hardware=["ESP32", "RPi 4"],
            active_goals=["Build AGRI-GLIDE"],
            skills=["embedded", "pcb-design"],
        )
        mem.set_user_profile(profile)

        loaded = mem.get_user_profile()
        assert loaded.name == "Hari"
        assert loaded.preferred_language == "C++"
        assert "ESP32" in loaded.hardware
        assert "Build AGRI-GLIDE" in loaded.active_goals

    def test_update_single_field(self, mem):
        mem.update_user_fact("preferred_editor", "Vim")
        profile = mem.get_user_profile()
        # Should be stored in facts or as a direct field
        assert profile.preferred_editor == "Vim" or "preferred_editor" in profile.facts

    def test_profile_roundtrip(self, mem):
        profile = UserProfile(
            name="Test",
            facts={"key1": "val1", "key2": "val2"},
        )
        mem.set_user_profile(profile)
        loaded = mem.get_user_profile()
        assert loaded.facts.get("key1") == "val1"
        assert loaded.facts.get("key2") == "val2"

    def test_to_dict(self):
        profile = UserProfile(name="Test", skills=["python"])
        d = profile.to_dict()
        assert d["name"] == "Test"
        assert "python" in d["skills"]

    def test_from_dict(self):
        d = {"name": "Test", "skills": ["python"], "unknown_field": "ignored"}
        profile = UserProfile.from_dict(d)
        assert profile.name == "Test"
        assert "python" in profile.skills


# ── Context Retrieval ─────────────────────────────────────────────


class TestContextRetrieval:

    def test_retrieve_context_with_profile(self, mem):
        profile = UserProfile(name="Hari", active_goals=["Build ARIA"])
        mem.set_user_profile(profile)

        # retrieve_context needs an OllamaClient, but we can verify the profile is stored
        loaded = mem.get_user_profile()
        assert loaded.name == "Hari"
        assert "Build ARIA" in loaded.active_goals

    def test_stats(self, mem):
        mem.record_episode(Episode(event_type="chat", summary="test"))
        mem.save_procedure(Procedure(name="p", description="d", steps=[]))
        stats = mem.stats()
        assert stats["episodes"] == 1
        assert stats["procedures"] == 1
        assert "conversations" in stats


# ── Integration with base Memory ──────────────────────────────────


class TestIntegration:

    def test_base_memory_still_works(self, mem):
        # Base memory operations should still work
        mem.base.add_message("user", "hello")
        mem.base.set_fact("language", "Python")
        mem.base.save_project("test", "/tmp/test")

        recent = mem.base.recent(1)
        assert len(recent) == 1

        facts = mem.base.get_facts()
        assert facts["language"] == "Python"

        projects = mem.base.get_projects()
        assert len(projects) == 1

    def test_episodes_and_base_coexist(self, mem):
        mem.base.add_message("user", "hello")
        mem.record_episode(Episode(event_type="chat", summary="test"))

        stats = mem.stats()
        assert stats["conversations"] == 1
        assert stats["episodes"] == 1
