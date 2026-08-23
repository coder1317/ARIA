"""Tests for Phase 1-5 upgrades: runtime, tool fabric, autonomy, reliability, safety."""
from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ultra.core.tool_registry import (
    ToolRegistry, ToolDefinition, ToolParam, ToolCall, ToolResult,
    RiskLevel,
)
from ultra.core.runtime import (
    AgentRuntime, Plan, PlanStatus, StepStatus,
    MissionBudget, ArtifactRegistry, Artifact,
)
from ultra.core.policy_engine import PolicyEngine, Policy, load_default_policies
from ultra.core.security_policy import ExecutionSecurity, SecurityPolicy, ApprovalRequest
from ultra.evaluator import GoalEvaluator, VerificationLoop, Evaluator
from ultra.config import Config, ProviderSpec


# ── Phase 1: Runtime as primary ──────────────────────────────

class TestRuntimePhase1:
    def test_mission_budget_check(self):
        budget = MissionBudget(max_iterations=10, max_tool_calls=50,
                               max_runtime_seconds=60, max_replans=3,
                               max_failed_actions=5)
        assert budget.check(5, 20, 30, 1, 2) is None  # within limits
        assert budget.check(10, 20, 30, 1, 2) is not None  # iterations
        assert budget.check(5, 50, 30, 1, 2) is not None  # tool calls
        assert budget.check(5, 20, 60, 1, 2) is not None  # time
        assert budget.check(5, 20, 30, 3, 2) is not None  # replans
        assert budget.check(5, 20, 30, 1, 5) is not None  # failures

    def test_artifact_registry(self):
        reg = ArtifactRegistry()
        art = Artifact(id="a1", path="/tmp/test.py", artifact_type="code",
                       created_by="s1", mission_id="m1", description="test file")
        reg.register(art)
        assert len(reg.get_mission_artifacts("m1")) == 1
        assert reg.get_mission_artifacts("m2") == []
        summary = reg.summary("m1")
        assert "1 artifacts" in summary
        assert "test.py" in summary

    def test_tool_result_enhanced_semantics(self):
        result = ToolResult(
            tool="test", success=True, output="ok",
            artifacts=["file.py"], metadata={"exit_code": 0},
            retryable=True, source="test",
        )
        d = result.to_dict()
        assert d["artifacts"] == ["file.py"]
        assert d["metadata"] == {"exit_code": 0}
        summary = result.summary()
        assert "test.py" in summary or "artifacts" in summary

    def test_parallel_step_execution(self):
        """Test that independent steps execute in parallel."""
        reg = ToolRegistry()
        results = []

        def tool_a():
            time.sleep(0.05)
            results.append("a")
            return "done"

        def tool_b():
            time.sleep(0.05)
            results.append("b")
            return "done"

        reg.register(ToolDefinition(
            name="a", description="tool a", handler=tool_a, risk_level=RiskLevel.LOW))
        reg.register(ToolDefinition(
            name="b", description="tool b", handler=tool_b, risk_level=RiskLevel.LOW))

        runtime = AgentRuntime(
            registry=reg,
            llm_fn=lambda p, s="": "ok",
            json_fn=lambda p, s="": {"steps": [
                {"id": "s1", "description": "a", "tool": "a", "args": {}, "depends_on": []},
                {"id": "s2", "description": "b", "tool": "b", "args": {}, "depends_on": []},
            ]},
            max_iterations=5,
        )
        plan = runtime.create_plan("parallel test")
        start = time.time()
        plan = runtime.run(plan)
        elapsed = time.time() - start

        assert plan.status == PlanStatus.COMPLETED
        assert len(results) == 2
        # Should be faster than sequential (0.1s) due to parallelism
        assert elapsed < 0.15


# ── Phase 2: Tool fabric ─────────────────────────────────────

class TestToolFabric:
    def test_pipeline_tools_registered(self):
        """Pipeline tools should be registered when orchestrator is provided."""
        reg = ToolRegistry()
        # Create mock orchestrator
        mock_orch = MagicMock()
        mock_orch.research = MagicMock()
        mock_orch.research.run.return_value = MagicMock(
            markdown="Report", confidence=0.8, sources=[], source_freshness=""
        )
        mock_orch.research.save.return_value = Path("/tmp/report.md")
        mock_orch.engineering = MagicMock()
        mock_orch.evaluator = MagicMock()
        mock_orch.market = MagicMock()
        mock_orch.market.run.return_value = MagicMock(
            markdown="Market", sources=[]
        )
        mock_orch.market.save.return_value = Path("/tmp/market.md")

        from ultra.core.register_tools import register_all_tools
        config = Config()
        register_all_tools(reg, config, orchestrator=mock_orch)

        tool_names = [t.name for t in reg.list_tools()]
        assert "pipeline.research" in tool_names
        assert "pipeline.build" in tool_names
        assert "pipeline.market" in tool_names
        assert "pipeline.review" in tool_names
        assert "pipeline.debug" in tool_names

    def test_dynamic_tool_add_remove(self):
        reg = ToolRegistry()
        reg.dynamic_add("custom.tool", "A custom tool", lambda: "ok")
        assert reg.get("custom.tool") is not None
        assert reg.dynamic_remove("custom.tool")
        assert reg.get("custom.tool") is None
        assert not reg.dynamic_remove("nonexistent")

    def test_tool_json_schemas(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="test.tool", description="Test",
            handler=lambda: "ok",
            params=[ToolParam("x", "string", "input", required=True)],
        ))
        schemas = reg.to_json_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "test.tool"
        assert "x" in schemas[0]["function"]["parameters"]["properties"]

    def test_ollama_tools_format(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="test.tool", description="Test",
            handler=lambda: "ok",
            params=[ToolParam("x", "string", "input", required=True)],
        ))
        tools = reg.to_ollama_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"


# ── Phase 3: Autonomy ────────────────────────────────────────

class TestAutonomy:
    def test_policy_engine_triggers(self):
        from ultra.core.events import EventBus, EventType
        bus = EventBus()
        engine = PolicyEngine(bus)
        triggered = []

        engine.add_policy(Policy(
            name="test_policy",
            event_type=EventType.TASK_COMPLETED,
            action_type="log",
            cooldown_seconds=0,
        ))

        bus.emit(EventType.TASK_COMPLETED, {"test": True}, source="test")
        assert engine.stats()["actions_taken"] >= 1

    def test_default_policies_loaded(self):
        from ultra.core.events import EventBus
        bus = EventBus()
        engine = PolicyEngine(bus)
        load_default_policies(engine)
        assert len(engine.policies) >= 4
        assert engine.stats()["enabled"] >= 4

    def test_goal_decomposition(self):
        """Test goal → task decomposition structure."""
        from ultra.core.goals import GoalManager
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "goals.db"
            goals = GoalManager(db)
            goal = goals.add("Build a web scraper", priority=1)

            # Mock LLM response
            def mock_json(prompt, system=""):
                return [
                    {"type": "research", "objective": "Research scraping techniques", "priority": 1},
                    {"type": "build", "objective": "Implement the scraper", "priority": 2},
                    {"type": "review", "objective": "Review and test", "priority": 2},
                ]

            tasks = goals.decompose_to_tasks(goal.id, mock_json)
            assert len(tasks) == 3
            assert tasks[0]["type"] == "research"
            assert tasks[1]["type"] == "build"
            assert tasks[2]["type"] == "review"

    def test_goal_next_task(self):
        from ultra.core.goals import GoalManager
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "goals.db"
            goals = GoalManager(db)
            goal = goals.add("Build something", priority=1)

            # Low progress → research first
            task = goals.get_next_task(goal.id)
            assert task["type"] == "research"

            # Medium progress → build
            goals.update_progress(goal.id, 0.4)
            task = goals.get_next_task(goal.id)
            assert task["type"] == "build"

            # High progress → review
            goals.update_progress(goal.id, 0.7)
            task = goals.get_next_task(goal.id)
            assert task["type"] == "review"

            # Completed → None
            goals.complete(goal.id)
            assert goals.get_next_task(goal.id) is None


# ── Phase 4: Reliability ─────────────────────────────────────

class TestReliability:
    def test_goal_evaluator(self):
        evaluator = GoalEvaluator()
        result = evaluator.evaluate(
            "Build a todo app",
            "Created a Python CLI todo app with add, list, delete commands. "
            "Has README and tests.",
            artifacts=["main.py", "test_main.py", "README.md"],
        )
        assert "achieved" in result
        assert "confidence" in result

    def test_verification_loop(self):
        evaluator = Evaluator()
        verifier = VerificationLoop(evaluator)

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "test_project"
            project.mkdir()
            # Create a minimal valid Python project
            (project / "main.py").write_text(
                'def main():\n    print("hello")\n\nif __name__ == "__main__":\n    main()\n'
            )
            (project / "README.md").write_text("# Test\nA test project")
            (project / "requirements.txt").write_text("")

            result = verifier.verify_project(project, "Build a hello world app")
            assert "passed" in result
            assert "score" in result
            assert "goal_achieved" in result

    def test_model_routing(self):
        from ultra.provider_pool import ProviderPool
        config = Config()
        pool = ProviderPool(config)

        # Simple task → fast model
        simple = pool.route_for_complexity("hello")
        # Complex task → strong model
        complex_task = pool.route_for_complexity(
            "Analyze the complete architecture of a distributed system "
            "and compare microservices vs monolith for this use case"
        )
        # Both should return non-empty strings
        assert isinstance(simple, str)
        assert isinstance(complex_task, str)


# ── Phase 5: Safety ──────────────────────────────────────────

class TestSafety:
    def test_security_policy_blocks_critical(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="dangerous.tool", description="Dangerous",
            handler=lambda: "boom",
            risk_level=RiskLevel.CRITICAL,
        ))
        result = reg.execute(ToolCall("dangerous.tool", {}))
        assert not result.success
        assert "CRITICAL" in result.error

    def test_security_policy_blocks_blocked_commands(self):
        security = ExecutionSecurity()
        call = ToolCall("terminal.execute", {"command": "rm -rf /"})
        result = security.check_tool_call(call, RiskLevel.MODERATE)
        assert result is not None
        assert not result.success
        assert "Blocked" in result.error

    def test_security_policy_blocks_blocked_patterns(self):
        security = ExecutionSecurity()
        call = ToolCall("terminal.execute", {"command": "sudo rm -rf /home"})
        result = security.check_tool_call(call, RiskLevel.MODERATE)
        assert result is not None
        assert not result.success

    def test_security_policy_allows_safe_commands(self):
        security = ExecutionSecurity()
        call = ToolCall("terminal.execute", {"command": "ls -la"})
        result = security.check_tool_call(call, RiskLevel.MODERATE)
        assert result is None  # allowed

    def test_credential_redaction(self):
        security = ExecutionSecurity()
        output = 'API_KEY=sk-abc123def456ghi789jkl012mno Token=secret123'
        redacted = security.redact_output(output)
        assert "sk-abc123" not in redacted
        assert "secret123" not in redacted
        assert "REDACTED" in redacted

    def test_workspace_sandboxing(self):
        security = ExecutionSecurity()
        security.policy.allowed_read_dirs = ["/home/user/projects"]
        security.policy.allowed_write_dirs = ["/home/user/projects"]

        # Read outside workspace
        call = ToolCall("filesystem.read", {"path": "/etc/passwd"})
        result = security.check_tool_call(call, RiskLevel.READ_ONLY)
        assert result is not None
        assert "outside workspace" in result.error

        # Write outside workspace
        call = ToolCall("filesystem.write", {"path": "/tmp/evil.py", "content": "bad"})
        result = security.check_tool_call(call, RiskLevel.MODERATE)
        assert result is not None
        assert "outside workspace" in result.error

    def test_web_domain_blocking(self):
        security = ExecutionSecurity()
        security.policy.blocked_domains = ["malware.com"]

        call = ToolCall("web.fetch", {"url": "https://malware.com/payload"})
        result = security.check_tool_call(call, RiskLevel.READ_ONLY)
        assert result is not None
        assert "Blocked domain" in result.error

    def test_tool_registry_with_security(self):
        reg = ToolRegistry()
        security = ExecutionSecurity()
        reg.set_security(security)

        # Blocked command should be rejected
        reg.register(ToolDefinition(
            name="terminal.execute", description="Run command",
            handler=lambda command: f"executed: {command}",
            params=[ToolParam("command", "string", required=True)],
            risk_level=RiskLevel.MODERATE,
        ))
        result = reg.execute(ToolCall("terminal.execute", {"command": "rm -rf /"}))
        assert not result.success
        assert "Blocked" in result.error

        # Safe command should work
        result = reg.execute(ToolCall("terminal.execute", {"command": "echo hello"}))
        assert result.success
