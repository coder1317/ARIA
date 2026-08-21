"""API mode — FastAPI server over the Brain (spec §11).

Endpoints:
  POST /process          {objective, context} → dispatch through the Brain
  GET  /health           provider health + memory stats
  GET  /memory/search?q= semantic memory search
  GET  /tasks            background task list
  GET  /status/{id}      single task status
  GET  /agents           registered agent registry

Security:
  - Bearer token required when ARIA4_API_TOKEN is set (recommended).
  - Binds to 127.0.0.1 by default; binding 0.0.0.0 requires explicit
    configuration (ARIA4_API_HOST) and a token.
  - Errors are redacted: clients get a generic message + request_id; the
    full detail goes to the audit log, never the wire.

Optional dependency: pip install fastapi uvicorn
"""
from __future__ import annotations

import os
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ultra.config import Config


class ProcessRequest(BaseModel):
    objective: str
    context: dict = Field(default_factory=dict)


def make_app(orch, config: Config) -> FastAPI:
    app = FastAPI(title="ARIA", version="1.0.0")

    token = os.getenv("ARIA4_API_TOKEN", "").strip()
    require_auth = bool(token)
    host = os.getenv("ARIA4_API_HOST", "127.0.0.1").strip()

    if require_auth:
        def check_auth(authorization: str | None = Header(default=None)) -> None:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="missing bearer token")
            if authorization[7:].strip() != token:
                raise HTTPException(status_code=401, detail="invalid token")
        auth = Depends(check_auth)
    else:
        auth = None

    def _fail(request_id: str, exc: Exception) -> dict:
        """Redact internal details — full trace goes to the audit log."""
        # Log full error internally
        if orch.audit:
            orch.audit.log(actor="api", action="error",
                           detail={"request_id": request_id,
                                   "error": f"{type(exc).__name__}: {exc}"})
        # P2-12: Never expose file paths, provider details, or stack traces
        import re as _re
        safe_msg = str(exc)
        # Remove file paths
        safe_msg = _re.sub(r"(/[\w./-]+)", "[path]", safe_msg)
        # Remove provider URLs
        safe_msg = _re.sub(r"https?://[^\s]+", "[url]", safe_msg)
        # Remove stack traces
        safe_msg = _re.sub(r"Traceback.*", "", safe_msg, flags=_re.DOTALL)
        # Cap length
        safe_msg = safe_msg[:200] if safe_msg else "unknown error"
        return {"success": False, "error": "internal error",
                "request_id": request_id, "hint": safe_msg}

    @app.post("/process", dependencies=[auth] if auth else [])
    def process(req: ProcessRequest):
        request_id = uuid.uuid4().hex[:12]
        if orch.audit:
            orch.audit.log(actor="api", action="request", task_type="process",
                           detail={"request_id": request_id,
                                   "objective": req.objective[:300],
                                   "context": req.context})
        try:
            output = orch.dispatch(req.objective, context=req.context)
            return {"success": True, "output": output[:20_000],
                    "intent_hint": req.objective[:100],
                    "request_id": request_id}
        except Exception as e:
            return _fail(request_id, e)

    @app.get("/health", dependencies=[auth] if auth else [])
    def health():
        return {"status": "healthy", **orch.status()}

    @app.get("/memory/search", dependencies=[auth] if auth else [])
    def memory_search(q: str, limit: int = 5):
        results = orch.memory.search(q)
        return {"results": [dict(r) for r in results[:limit]]}

    @app.get("/tasks", dependencies=[auth] if auth else [])
    def tasks():
        if orch.tasks is None:
            return {"tasks": []}
        return {"tasks": orch.tasks.list_tasks()}

    @app.get("/status/{task_id}", dependencies=[auth] if auth else [])
    def status(task_id: str):
        if orch.tasks is None:
            return {"error": "task manager unavailable"}
        t = orch.tasks.get(task_id)
        if t is None:
            return {"error": "not found"}
        return {"task_id": t.task_id, "task_type": t.task_type,
                "status": t.status, "error": t.error,
                "result": str(t.result)[:2000] if t.result else None}

    @app.get("/agents", dependencies=[auth] if auth else [])
    def agents():
        return {"agents": sorted(orch.agents.keys())}

    return app
