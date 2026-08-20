"""Tests for Tool Registry and Agent Runtime."""
import json
import pytest
from ultra.core.tool_registry import (
    ToolRegistry, ToolDefinition, ToolParam, ToolCall,
    ToolResult, RiskLevel,
)
from ultra.core.runtime import (
    AgentRuntime, Plan, Step, StepStatus, PlanStatus,
    Observation, ExecutionTrace,
)


# ── Tool Registry tests ────────────────────────────────────────

def _make_registry() -> ToolRegistry:
    """Create a registry with sample tools for testing."""
    reg = ToolRegistry()

    def echo(text: str) -> str:
        return f"echo: {text}"

    def add(a: int, b: int) -> int:
        return a + b

    def search(query: str, limit: int = 5) -> str:
        return f"results for {query} (limit={limit})"

    def dangerous_action() -> str:
        return "done"

    reg.register(ToolDefinition(
        name="test.echo",
        description="Echo text back",
        handler=echo,
        params=[ToolParam("text", "string", "Text to echo", required=True)],
        risk_level=RiskLevel.READ_ONLY,
        category="test",
    ))
    reg.register(ToolDefinition(
        name="test.add",
        description="Add two numbers",
        handler=add,
        params=[
            ToolParam("a", "integer", "First number", required=True),
            ToolParam("b", "integer", "Second number", required=True),
        ],
        risk_level=RiskLevel.READ_ONLY,
        category="test",
    ))
    reg.register(ToolDefinition(
        name="test.search",
        description="Search for things",
        handler=search,
        params=[
            ToolParam("query", "string", "Search query", required=True),
            ToolParam("limit", "integer", "Max results"),
        ],
        risk_level=RiskLevel.READ_ONLY,
        category="test",
    ))
    reg.register(ToolDefinition(
        name="test.dangerous",
        description="A dangerous action",
        handler=dangerous_action,
        risk_level=RiskLevel.CRITICAL,
        category="dangerous",
    ))
    return reg


class TestToolDefinition:
    def test_to_prompt_schema(self):
        tool = ToolDefinition(
            name="test.echo",
            description="Echo text",
            handler=lambda text: text,
            params=[ToolParam("text", "string", "Input text", required=True)],
            risk_level=RiskLevel.LOW,
            category="test",
        )
        schema = tool.to_prompt_schema()
        assert "test.echo" in schema
        assert "Echo text" in schema
        assert "text" in schema
        assert "required" in schema
        assert "low" in schema

    def test_to_json_schema(self):
        tool = ToolDefinition(
            name="test.echo",
            description="Echo text",
            handler=lambda text: text,
            params=[ToolParam("text", "string", "Input text", required=True)],
        )
        schema = tool.to_json_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test.echo"
        assert "text" in schema["function"]["parameters"]["properties"]
        assert "text" in schema["function"]["parameters"]["required"]

    def test_optional_param_not_required(self):
        tool = ToolDefinition(
            name="test.search",
            description="Search",
            handler=lambda query, limit=5: query,
            params=[
                ToolParam("query", "string", "query", required=True),
                ToolParam("limit", "integer", "max", required=False),
            ],
        )
        schema = tool.to_json_schema()
        assert "query" in schema["function"]["parameters"]["required"]
        assert "limit" not in schema["function"]["parameters"]["required"]


class TestToolRegistry:
    def test_register_and_get(self):
        reg = _make_registry()
        tool = reg.get("test.echo")
        assert tool is not None
        assert tool.name == "test.echo"

    def test_unregister(self):
        reg = _make_registry()
        reg.unregister("test.echo")
        assert reg.get("test.echo") is None

    def test_list_tools(self):
        reg = _make_registry()
        assert reg.tool_count == 4
        tools = reg.list_tools(category="test")
        assert len(tools) == 3
        tools = reg.list_tools(category="dangerous")
        assert len(tools) == 1

    def test_categories(self):
        reg = _make_registry()
        assert "test" in reg.categories
        assert "dangerous" in reg.categories

    def test_to_prompt(self):
        reg = _make_registry()
        prompt = reg.to_prompt()
        assert "Available Tools" in prompt
        assert "test.echo" in prompt
        assert "tool_call" in prompt

    def test_execute_success(self):
        reg = _make_registry()
        result = reg.execute(ToolCall("test.echo", {"text": "hello"}))
        assert result.success
        assert result.output == "echo: hello"

    def test_execute_with_default_param(self):
        reg = _make_registry()
        result = reg.execute(ToolCall("test.search", {"query": "python"}))
        assert result.success
        assert "limit=5" in result.output

    def test_execute_with_explicit_param(self):
        reg = _make_registry()
        result = reg.execute(ToolCall("test.search", {"query": "python", "limit": 10}))
        assert result.success
        assert "limit=10" in result.output

    def test_execute_missing_required_param(self):
        reg = _make_registry()
        result = reg.execute(ToolCall("test.echo", {}))
        assert not result.success
        assert "missing required args" in result.error

    def test_execute_unknown_tool(self):
        reg = _make_registry()
        result = reg.execute(ToolCall("nonexistent", {}))
        assert not result.success
        assert "unknown tool" in result.error

    def test_execute_disabled_tool(self):
        reg = _make_registry()
        tool = reg.get("test.echo")
        tool.enabled = False
        result = reg.execute(ToolCall("test.echo", {"text": "hi"}))
        assert not result.success
        assert "disabled" in result.error
        tool.enabled = True  # restore

    def test_execute_exception_handling(self):
        reg = _make_registry()

        def boom():
            raise ValueError("test error")

        reg.register(ToolDefinition(
            name="test.boom",
            description="Raises error",
            handler=boom,
        ))
        result = reg.execute(ToolCall("test.boom", {}))
        assert not result.success
        assert "ValueError" in result.error

    def test_history_tracking(self):
        reg = _make_registry()
        reg.execute(ToolCall("test.echo", {"text": "a"}))
        reg.execute(ToolCall("test.echo", {"text": "b"}))
        assert len(reg.history) == 2
        assert reg.history[0].tool == "test.echo"

    def test_result_summary(self):
        r1 = ToolResult("test", True, output="hello world")
        assert "[test]" in r1.summary()
        assert "hello world" in r1.summary()

        r2 = ToolResult("test", False, error="something broke")
        assert "[test FAILED]" in r2.summary()
        assert "something broke" in r2.summary()


# ── Runtime tests ──────────────────────────────────────────────

def _make_runtime() -> AgentRuntime:
    """Create a runtime with mock LLM for testing."""
    reg = _make_registry()

    call_count = [0]
    responses = [
        # Plan response
        {"steps": [
            {"id": "step_1", "description": "Echo hello",
             "tool": "test.echo", "args": {"text": "hello"}, "depends_on": []},
            {"id": "step_2", "description": "Add numbers",
             "tool": "test.add", "args": {"a": 1, "b": 2}, "depends_on": []},
        ]},
        # Evaluation 1
        {"success": True, "analysis": "step completed", "should_continue": True, "should_replan": False},
        # Evaluation 2
        {"success": True, "analysis": "step completed", "should_continue": True, "should_replan": False},
    ]

    def mock_json(prompt: str, system: str = "") -> dict:
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    def mock_llm(prompt: str, system: str = "") -> str:
        return "ok"

    return AgentRuntime(
        registry=reg,
        llm_fn=mock_llm,
        json_fn=mock_json,
        max_iterations=10,
        max_replans=3,
    )


class TestPlan:
    def test_plan_creation(self):
        plan = Plan(id="test-1", objective="test objective")
        assert plan.status == PlanStatus.PLANNING
        assert plan.progress()["total"] == 0

    def test_plan_steps(self):
        plan = Plan(id="test-1", objective="test", steps=[
            Step(id="s1", description="step 1"),
            Step(id="s2", description="step 2", depends_on=["s1"]),
        ])
        # s1 should be ready, s2 should not
        ready = plan.ready_steps()
        assert len(ready) == 1
        assert ready[0].id == "s1"

    def test_plan_all_done(self):
        plan = Plan(id="test-1", objective="test", steps=[
            Step(id="s1", description="step 1", status=StepStatus.SUCCEEDED),
        ])
        assert plan.all_done()

    def test_plan_any_failed(self):
        plan = Plan(id="test-1", objective="test", steps=[
            Step(id="s1", description="step 1", status=StepStatus.SUCCEEDED),
            Step(id="s2", description="step 2", status=StepStatus.FAILED),
        ])
        assert plan.any_failed()

    def test_plan_progress(self):
        plan = Plan(id="test-1", objective="test", steps=[
            Step(id="s1", description="step 1", status=StepStatus.SUCCEEDED),
            Step(id="s2", description="step 2", status=StepStatus.RUNNING),
            Step(id="s3", description="step 3", status=StepStatus.PENDING),
        ])
        p = plan.progress()
        assert p["total"] == 3
        assert p["done"] == 1
        assert p["percent"] == 33

    def test_plan_summary(self):
        plan = Plan(id="test-1", objective="test obj", steps=[
            Step(id="s1", description="do stuff"),
        ])
        s = plan.summary()
        assert "test obj" in s
        assert "do stuff" in s


class TestAgentRuntime:
    def test_create_plan(self):
        runtime = _make_runtime()
        plan = runtime.create_plan("echo hello and add numbers")
        assert plan.status == PlanStatus.EXECUTING
        assert len(plan.steps) == 2
        assert plan.steps[0].tool_calls[0].tool == "test.echo"

    def test_run_plan(self):
        runtime = _make_runtime()
        plan = runtime.create_plan("echo hello and add numbers")
        plan = runtime.run(plan)
        assert plan.status == PlanStatus.COMPLETED
        assert plan.all_done()

    def test_execution_trace(self):
        runtime = _make_runtime()
        plan = runtime.create_plan("echo hello")
        plan = runtime.run(plan)
        assert len(runtime.traces) == 1
        trace = runtime.traces[0]
        assert trace.objective == "echo hello"
        assert trace.tool_calls_total >= 1

    def test_tool_execution_in_plan(self):
        runtime = _make_runtime()
        plan = runtime.create_plan("echo test")
        plan = runtime.run(plan)
        # Find the echo step
        echo_step = plan.steps[0]
        assert echo_step.status == StepStatus.SUCCEEDED
        assert echo_step.observation is not None
        assert echo_step.observation.success

    def test_fallback_plan(self):
        """When LLM fails to produce a plan, create a fallback."""
        reg = _make_registry()
        call_count = [0]

        def failing_json(prompt, system=""):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("LLM failed")
            return {"success": True, "analysis": "ok", "should_continue": True, "should_replan": False}

        def mock_llm(prompt, system=""):
            return "ok"

        runtime = AgentRuntime(
            registry=reg,
            llm_fn=mock_llm,
            json_fn=failing_json,
            max_iterations=10,
        )
        plan = runtime.create_plan("do something")
        assert len(plan.steps) == 1
        assert plan.steps[0].id == "step_1"

    def test_replan(self):
        """Test that the runtime replans after failures."""
        reg = _make_registry()

        def boom():
            raise RuntimeError("tool broken")

        reg.register(ToolDefinition(
            name="test.boom",
            description="Breaks",
            handler=boom,
            params=[],
            risk_level=RiskLevel.LOW,
        ))

        call_count = [0]

        def mock_json(prompt, system=""):
            call_count[0] += 1
            if call_count[0] == 1:
                # Initial plan
                return {"steps": [
                    {"id": "s1", "description": "break stuff",
                     "tool": "test.boom", "args": {}},
                ]}
            elif "Replan" in prompt or "replan" in prompt.lower() or "revised" in prompt.lower():
                # Replan — new step
                return {"steps": [
                    {"id": "s1_fix", "description": "echo fallback",
                     "tool": "test.echo", "args": {"text": "recovered"}},
                ]}
            else:
                # Evaluation — check if boom tool was used
                if "boom" in prompt.lower():
                    return {"success": False, "analysis": "tool broke",
                            "should_continue": True, "should_replan": True}
                return {"success": True, "analysis": "step worked",
                        "should_continue": True, "should_replan": False}

        def mock_llm(prompt, system=""):
            return "ok"

        runtime = AgentRuntime(
            registry=reg,
            llm_fn=mock_llm,
            json_fn=mock_json,
            max_iterations=10,
            max_replans=3,
        )
        plan = runtime.create_plan("break stuff")
        plan = runtime.run(plan)
        # Should have replanned and found a working step
        assert plan.replan_count >= 1
        assert plan.status == PlanStatus.COMPLETED

    def test_get_context(self):
        runtime = _make_runtime()
        plan = Plan(id="test", objective="test objective", steps=[
            Step(id="s1", description="step one"),
        ])
        ctx = runtime.get_context(plan)
        assert "test objective" in ctx
        assert "step one" in ctx

    def test_summary(self):
        runtime = _make_runtime()
        assert "No plans" in runtime.summary()
        plan = runtime.create_plan("test")
        plan = runtime.run(plan)
        s = runtime.summary()
        assert "Executed 1 plans" in s
