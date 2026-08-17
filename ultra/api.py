"""API mode — FastAPI server over the Brain (spec §11).

Endpoints:
  POST /process          {objective, context} → dispatch through the Brain
  GET  /health           provider health + memory stats
  GET  /memory/search?q= semantic memory search
  GET  /tasks            background task list
  GET  /status/{id}      single task status
  GET  /agents           registered agent registry

Optional dependency: pip install fastapi uvicorn
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ultra.config import Config


class ProcessRequest(BaseModel):
    objective: str
    context: dict = {}


def make_app(orch, config: Config) -> FastAPI:
    app = FastAPI(title="ARIA Ultra", version="1.0.0")

    @app.post("/process")
    def process(req: ProcessRequest):
        try:
            output = orch.dispatch(req.objective)
            return {"success": True, "output": output[:20_000],
                    "intent_hint": req.objective[:100]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/health")
    def health():
        return {"status": "healthy", **orch.status()}

    @app.get("/memory/search")
    def memory_search(q: str, limit: int = 5):
        results = orch.memory.search(q)
        return {"results": [dict(r) for r in results[:limit]]}

    @app.get("/tasks")
    def tasks():
        if orch.tasks is None:
            return {"tasks": []}
        return {"tasks": orch.tasks.list_tasks()}

    @app.get("/status/{task_id}")
    def status(task_id: str):
        if orch.tasks is None:
            return {"error": "task manager unavailable"}
        t = orch.tasks.get(task_id)
        if t is None:
            return {"error": "not found"}
        return {"task_id": t.task_id, "task_type": t.task_type,
                "status": t.status, "error": t.error,
                "result": str(t.result)[:2000] if t.result else None}

    @app.get("/agents")
    def agents():
        return {"agents": sorted(orch.agents.keys())}

    return app
