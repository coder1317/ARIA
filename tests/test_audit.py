"""Tests for ultra.audit — append-only operation log."""
import pytest

from ultra.audit import AuditLog


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.db")


def test_log_and_query(audit):
    audit.log(actor="brain", action="dispatch", task_type="build",
              detail={"input": "build app"})
    rows = audit.query()
    assert len(rows) == 1
    assert rows[0]["actor"] == "brain"
    assert rows[0]["action"] == "dispatch"
    assert rows[0]["task_type"] == "build"
    assert "when" in rows[0]


def test_filter_by_actor(audit):
    audit.log(actor="agent:coder", action="execute")
    audit.log(actor="brain", action="dispatch")
    rows = audit.query(actor="coder")
    assert len(rows) == 1
    assert rows[0]["actor"] == "agent:coder"


def test_query_limited(audit):
    for i in range(5):
        audit.log(actor="a", action=f"act{i}")
    assert len(audit.query(limit=3)) == 3


def test_stats(audit):
    audit.log(actor="a", action="execute")
    audit.log(actor="a", action="execute")
    audit.log(actor="b", action="dispatch")
    stats = audit.stats()
    assert stats["total"] == 3
    assert stats["by_action"]["execute"] == 2


def test_log_inference(audit):
    audit.log_inference("coder", "code", "write a parser", "ok", "granite")
    rows = audit.query(actor="coder", action="inference")
    assert len(rows) == 1
    assert rows[0]["provider"] == "granite"


def test_error_recorded(audit):
    audit.log(actor="brain", action="dispatch", error="timeout")
    rows = audit.query()
    assert rows[0]["error"] == "timeout"


def test_persists_across_instances(tmp_path):
    db = tmp_path / "audit.db"
    AuditLog(db).log(actor="a", action="x")
    fresh = AuditLog(db)
    assert len(fresh.query()) == 1
