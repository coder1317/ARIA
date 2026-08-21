"""Agent Runtime — PLAN → ACT → OBSERVE → EVALUATE → REPLAN loop.

This replaces the current hardcoded intent → pipeline routing with a
genuine agent loop. The runtime:

1. Takes an objective from the user
2. Creates a plan (list of steps with dependencies)
3. Executes each step by calling tools via the Tool Registry
4. Observes results and evaluates whether the step succeeded
5. Replans if a step fails or produces unexpected results
6. Continues until the objective is met or max iterations reached

The runtime maintains a TaskGraph with dependency tracking, so steps
like "research → architecture → implement → test" execute in the
correct order, with parallelism where possible.

Design principles:
- The LLM plans and decides; tools execute and report
- Every step produces an Observation that feeds back into reasoning
- The runtime can abort, retry, or replan at any step
- Full execution trace for debugging and improvement
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ultra.core.tool_registry import ToolCall, ToolRegistry, ToolResult

logger = logging.getLogger("aria.runtime")


# ── Step / Plan data structures ────────────────────────────────

class StepStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    REPLANNED = "replanned"


class PlanStatus(Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class Observation:
    """What happened after a step executed."""
    step_id: str
    tool_results: list[ToolResult] = field(default_factory=list)
    llm_analysis: str = ""       # LLM's assessment of the result
    success: bool = False
    should_continue: bool = True
    should_replan: bool = False

    def summary(self) -> str:
        parts = [f"step={self.step_id}", f"success={self.success}"]
        if self.tool_results:
            ok = sum(1 for r in self.tool_results if r.success)
            parts.append(f"tools={ok}/{len(self.tool_results)}")
        if self.llm_analysis:
            parts.append(f"analysis={self.llm_analysis[:100]}")
        return " ".join(parts)


@dataclass
class Step:
    """A single step in a plan."""
    id: str
    description: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    observation: Observation | None = None
    result: str | None = None     # human-readable result
    attempt: int = 0
    max_attempts: int = 3

    def ready(self, completed: set[str]) -> bool:
        """Is this step ready to execute?"""
        if self.status == StepStatus.PENDING:
            return all(dep in completed for dep in self.depends_on)
        if self.status == StepStatus.FAILED:
            # Only retry if we haven't exhausted attempts
            return (self.attempt < self.max_attempts and
                    all(dep in completed for dep in self.depends_on))
        return False


@dataclass
class Plan:
    """A plan is a list of steps with a status."""
    id: str
    objective: str
    steps: list[Step] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PLANNING
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    observations: list[Observation] = field(default_factory=list)
    replan_count: int = 0
    max_replans: int = 5

    def ready_steps(self) -> list[Step]:
        """Get steps that are ready to execute."""
        completed = {
            s.id for s in self.steps
            if s.status in (StepStatus.SUCCEEDED, StepStatus.SKIPPED)
        }
        return [s for s in self.steps if s.ready(completed)]

    def all_done(self) -> bool:
        """Are all steps finished (succeeded, skipped, or failed)?"""
        return all(
            s.status in (StepStatus.SUCCEEDED, StepStatus.SKIPPED, StepStatus.FAILED)
            for s in self.steps
        )

    def any_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def progress(self) -> dict:
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.status in (
            StepStatus.SUCCEEDED, StepStatus.SKIPPED))
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        return {
            "total": total,
            "done": done,
            "failed": failed,
            "running": sum(1 for s in self.steps if s.status == StepStatus.RUNNING),
            "pending": sum(1 for s in self.steps if s.status == StepStatus.PENDING),
            "percent": round(done / max(total, 1) * 100),
        }

    def summary(self) -> str:
        p = self.progress()
        lines = [
            f"Plan: {self.objective}",
            f"Status: {self.status.value}",
            f"Progress: {p['done']}/{p['total']} ({p['percent']}%) "
            f"failed={p['failed']} replans={self.replan_count}",
            "",
        ]
        for s in self.steps:
            icon = {
                StepStatus.SUCCEEDED: "✓",
                StepStatus.FAILED: "✗",
                StepStatus.RUNNING: "●",
                StepStatus.PENDING: "○",
                StepStatus.SKIPPED: "–",
                StepStatus.REPLANNED: "↻",
            }.get(s.status, "?")
            dep = f" (after: {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"  {icon} {s.id}: {s.description}{dep}")
        return "\n".join(lines)


@dataclass
class ExecutionTrace:
    """Full execution trace for debugging and improvement."""
    plan_id: str
    objective: str
    steps: list[dict] = field(default_factory=list)
    total_duration_ms: float = 0.0
    tool_calls_total: int = 0
    replan_count: int = 0
    final_status: str = ""

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "objective": self.objective,
            "steps": self.steps,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "tool_calls_total": self.tool_calls_total,
            "replan_count": self.replan_count,
            "final_status": self.final_status,
        }


# ── Agent Runtime ──────────────────────────────────────────────

class AgentRuntime:
    """The core agent loop: PLAN → ACT → OBSERVE → EVALUATE → REPLAN.

    This is what transforms ARIA from a pipeline engine into a
    genuine autonomous agent.

    Usage:
        runtime = AgentRuntime(registry, llm_fn)
        plan = await runtime.create_plan("Build a Flask API for todo items")
        result = await runtime.execute(plan)
        print(result.summary())
    """

    def __init__(
        self,
        registry: ToolRegistry,
        llm_fn: Callable[[str, str], str],
        json_fn: Callable[[str, str], Any],
        max_iterations: int = 30,
        max_replans: int = 5,
    ):
        """
        Args:
            registry: The tool registry for executing tools.
            llm_fn: Function(prompt, system) -> text response.
            json_fn: Function(prompt, system) -> parsed JSON.
            max_iterations: Max tool-call steps before forced stop.
            max_replans: Max replanning attempts before giving up.
        """
        self.registry = registry
        self.llm = llm_fn
        self.json_fn = json_fn
        self.max_iterations = max_iterations
        self.max_replans = max_replans
        self._traces: list[ExecutionTrace] = []

    # ── Planning ───────────────────────────────────────────────

    def create_plan(self, objective: str) -> Plan:
        """Use the LLM to decompose an objective into a step plan.

        The LLM receives the available tools and produces a structured
        plan with dependencies.
        """
        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        tools_prompt = self.registry.to_prompt()

        system = (
            "You are ARIA's planner. Given an objective and available tools, "
            "create a step-by-step plan. Each step should be a concrete action "
            "that uses one or more tools.\n\n"
            "Rules:\n"
            "- Keep plans to 1-8 steps\n"
            "- Steps with dependencies must list them in depends_on\n"
            "- Make steps concrete and measurable\n"
            "- Research before building when the topic is unfamiliar\n"
            "- Test after building\n"
            f"{tools_prompt}"
        )

        prompt = (
            f"Objective: {objective}\n\n"
            "Create a plan. Return ONLY a JSON object:\n"
            "```json\n"
            '{\n'
            '  "steps": [\n'
            '    {\n'
            '      "id": "step_1",\n'
            '      "description": "what to do",\n'
            '      "tool": "tool.name",\n'
            '      "args": {"param": "value"},\n'
            '      "depends_on": []\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "```\n"
            "Use tool names exactly as listed above. "
            "For steps that need multiple tools, list the most important one "
            "as 'tool' and describe the rest in the description."
        )

        try:
            result = self.json_fn(prompt, system)
        except Exception as e:
            logger.warning("planning failed: %s", e)
            # Fallback: single step using the objective as a terminal command
            return self._fallback_plan(objective)

        steps_data = result.get("steps", []) if isinstance(result, dict) else []
        if not steps_data:
            return self._fallback_plan(objective)

        steps = []
        for i, sd in enumerate(steps_data):
            step_id = sd.get("id", f"step_{i+1}")
            tool_name = sd.get("tool", "")
            args = sd.get("args", {})
            depends = sd.get("depends_on", [])

            # Build tool calls
            tool_calls = []
            if tool_name and self.registry.get(tool_name):
                tool_calls.append(ToolCall(tool=tool_name, args=args))

            steps.append(Step(
                id=step_id,
                description=sd.get("description", f"Step {i+1}"),
                tool_calls=tool_calls,
                depends_on=depends,
            ))

        plan = Plan(id=plan_id, objective=objective, steps=steps)
        plan.status = PlanStatus.EXECUTING
        return plan

    def _fallback_plan(self, objective: str) -> Plan:
        """Create a minimal plan when LLM planning fails."""
        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        # Try terminal.execute as a fallback
        steps = [Step(
            id="step_1",
            description=objective,
            tool_calls=[ToolCall("terminal.execute", {"command": objective})],
        )]
        plan = Plan(id=plan_id, objective=objective, steps=steps)
        plan.status = PlanStatus.EXECUTING
        return plan

    # ── Execution Loop ─────────────────────────────────────────

    def run(self, plan: Plan) -> Plan:
        """Execute a plan through the agent loop.

        This is synchronous for compatibility with the current codebase.
        For async use, wrap in asyncio.

        P3-17: Includes loop detection — if the same step fails 3 times
        with the same error, the runtime force-replans or aborts.
        """
        start = time.time()
        iterations = 0
        # P3-17: Loop detection — track consecutive failures per step
        _failure_history: dict[str, list[str]] = {}  # step_id → [error messages]

        while iterations < self.max_iterations:
            iterations += 1
            ready = plan.ready_steps()

            if not ready:
                if plan.any_failed():
                    # Try replanning
                    if plan.replan_count < plan.max_replans:
                        plan = self._replan(plan)
                        continue
                    else:
                        plan.status = PlanStatus.FAILED
                        break
                elif plan.all_done():
                    # All steps finished successfully (or skipped)
                    break
                else:
                    # All remaining steps are blocked — shouldn't happen
                    plan.status = PlanStatus.FAILED
                    break

            # Execute the first ready step (could parallelize later)
            step = ready[0]
            self._execute_step(plan, step)

            # P3-17: Loop detection — check for repeated failures
            if step.status == StepStatus.FAILED:
                error_key = step.observation.llm_analysis[:100] if step.observation else "unknown"
                hist = _failure_history.setdefault(step.id, [])
                hist.append(error_key)
                if len(hist) >= 5 and len(set(hist[-5:])) == 1:
                    # Same error 3 times in a row — force skip this step
                    logger.warning("Loop detected for step %s: same error 3x — skipping", step.id)
                    step.status = StepStatus.SKIPPED
                    _failure_history[step.id] = []  # reset to avoid cascade

        if plan.status not in (PlanStatus.FAILED, PlanStatus.ABORTED, PlanStatus.REPLANNING):
            if plan.any_failed():
                plan.status = PlanStatus.FAILED
            elif plan.all_done():
                plan.status = PlanStatus.COMPLETED

        plan.completed_at = time.time()

        # Record trace
        trace = ExecutionTrace(
            plan_id=plan.id,
            objective=plan.objective,
            steps=[
                {
                    "id": s.id,
                    "description": s.description,
                    "status": s.status.value,
                    "result": s.result,
                    "tool_calls": len(s.tool_calls),
                    "attempt": s.attempt,
                }
                for s in plan.steps
            ],
            total_duration_ms=(time.time() - start) * 1000,
            tool_calls_total=sum(len(s.tool_calls) for s in plan.steps),
            replan_count=plan.replan_count,
            final_status=plan.status.value,
        )
        self._traces.append(trace)
        return plan

    def _execute_step(self, plan: Plan, step: Step) -> None:
        """Execute a single step: run tools, observe, evaluate."""
        step.status = StepStatus.RUNNING
        step.attempt += 1
        tool_results: list[ToolResult] = []

        for call in step.tool_calls:
            result = self.registry.execute(call)
            tool_results.append(result)
            logger.info("tool %s: success=%s", call.tool, result.success)

        # Evaluate: did the step succeed?
        all_ok = all(r.success for r in tool_results) if tool_results else False
        combined_output = "\n".join(
            r.summary() for r in tool_results
        )

        # Ask the LLM to evaluate the observation
        analysis = self._evaluate_observation(
            step.description, combined_output, all_ok
        )

        observation = Observation(
            step_id=step.id,
            tool_results=tool_results,
            llm_analysis=analysis.get("analysis", ""),
            success=analysis.get("success", all_ok),
            should_continue=analysis.get("should_continue", True),
            should_replan=analysis.get("should_replan", False),
        )

        step.observation = observation
        step.result = combined_output
        plan.observations.append(observation)

        if observation.success:
            step.status = StepStatus.SUCCEEDED
        elif observation.should_replan and step.attempt < step.max_attempts:
            step.status = StepStatus.FAILED  # will be caught by replan
        elif step.attempt >= step.max_attempts:
            step.status = StepStatus.FAILED
        else:
            step.status = StepStatus.FAILED

    def _evaluate_observation(
        self, step_desc: str, output: str, tool_success: bool
    ) -> dict:
        """Ask the LLM to evaluate whether a step succeeded.

        Returns dict with: success, analysis, should_continue, should_replan.
        """
        system = (
            "You are ARIA's evaluator. Assess whether a step succeeded "
            "based on its description and the tool output.\n"
            "Return ONLY a JSON object:\n"
            '{"success": bool, "analysis": "brief assessment", '
            '"should_continue": bool, "should_replan": bool}\n'
            "Rules:\n"
            "- success=true if the step achieved its goal\n"
            "- should_replan=true if a different approach might work\n"
            "- should_continue=false only for unrecoverable errors"
        )
        prompt = (
            f"Step: {step_desc}\n"
            f"Tool execution success: {tool_success}\n"
            f"Output:\n{output[:3000]}"
        )
        try:
            return self.json_fn(prompt, system)
        except Exception:
            return {
                "success": tool_success,
                "analysis": "evaluation failed — using tool success",
                "should_continue": True,
                "should_replan": False,
            }

    def _replan(self, plan: Plan) -> Plan:
        """Replan after failures.

        The LLM sees the original objective, the plan so far, and
        the failures, and produces a revised plan for remaining steps.
        """
        plan.replan_count += 1
        plan.status = PlanStatus.REPLANNING
        logger.info("replan %d/%d for plan %s",
                     plan.replan_count, plan.max_replans, plan.id)

        # Summarize what succeeded and what failed
        completed = []
        failed = []
        for s in plan.steps:
            if s.status == StepStatus.SUCCEEDED:
                completed.append(f"✓ {s.id}: {s.description}")
            elif s.status == StepStatus.FAILED:
                err = s.observation.llm_analysis if s.observation else "unknown"
                failed.append(f"✗ {s.id}: {s.description} — {err}")

        tools_prompt = self.registry.to_prompt()

        system = (
            "You are ARIA's replanner. The original plan partially failed. "
            "Create a revised plan for the REMAINING work only.\n"
            "Don't repeat steps that already succeeded.\n"
            f"{tools_prompt}"
        )
        prompt = (
            f"Objective: {plan.objective}\n\n"
            f"Completed:\n" + "\n".join(completed) + "\n\n"
            f"Failed:\n" + "\n".join(failed) + "\n\n"
            "Create a revised plan for remaining work. Return ONLY:\n"
            "```json\n"
            '{"steps": [{"id": "...", "description": "...", '
            '"tool": "...", "args": {...}, "depends_on": [...]}]}\n'
            "```\n"
            "If no further steps are needed, return {\"steps\": []}."
        )

        try:
            result = self.json_fn(prompt, system)
            steps_data = result.get("steps", []) if isinstance(result, dict) else []
        except Exception:
            steps_data = []

        if not steps_data:
            # No more steps — mark remaining as skipped
            for s in plan.steps:
                if s.status == StepStatus.PENDING:
                    s.status = StepStatus.SKIPPED
            plan.status = PlanStatus.COMPLETED if not plan.any_failed() else PlanStatus.FAILED
            return plan

        # Mark exhausted failed steps as SKIPPED so they don't
        # keep triggering replans after new steps succeed
        for s in plan.steps:
            if s.status == StepStatus.FAILED and s.attempt >= s.max_attempts:
                s.status = StepStatus.SKIPPED

        # Add new steps to the plan
        existing_ids = {s.id for s in plan.steps}
        for i, sd in enumerate(steps_data):
            step_id = sd.get("id", f"replan_{plan.replan_count}_{i+1}")
            if step_id in existing_ids:
                step_id = f"{step_id}_r{plan.replan_count}"

            tool_name = sd.get("tool", "")
            args = sd.get("args", {})
            depends = sd.get("depends_on", [])

            tool_calls = []
            if tool_name and self.registry.get(tool_name):
                tool_calls.append(ToolCall(tool=tool_name, args=args))

            plan.steps.append(Step(
                id=step_id,
                description=sd.get("description", f"Replan step {i+1}"),
                tool_calls=tool_calls,
                depends_on=depends,
            ))

        plan.status = PlanStatus.EXECUTING
        return plan

    # ── Context / State ────────────────────────────────────────

    def get_context(self, plan: Plan) -> str:
        """Build a context string from the plan state for the LLM.

        This replaces the current simple memory retrieval with a
        rich context that includes the active plan, observations,
        and tool history.
        """
        lines = [
            f"## Current Objective",
            plan.objective,
            "",
            f"## Plan Progress ({plan.progress()['percent']}%)",
        ]
        for s in plan.steps:
            icon = "✓" if s.status == StepStatus.SUCCEEDED else \
                   "✗" if s.status == StepStatus.FAILED else \
                   "●" if s.status == StepStatus.RUNNING else "○"
            lines.append(f"  {icon} {s.description}")
        lines.append("")
        return "\n".join(lines)

    def summary(self) -> str:
        """Summary of all execution traces."""
        if not self._traces:
            return "No plans executed yet."
        lines = [f"Executed {len(self._traces)} plans:"]
        for t in self._traces:
            lines.append(f"  {t.final_status}: {t.objective[:60]} "
                         f"({t.tool_calls_total} tool calls, "
                         f"{t.total_duration_ms/1000:.1f}s, "
                         f"{t.replan_count} replans)")
        return "\n".join(lines)

    @property
    def traces(self) -> list[ExecutionTrace]:
        return list(self._traces)
