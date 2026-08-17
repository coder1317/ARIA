# ARIA Ultra — Autonomous Multi-Agent Engineering Assistant

A new prototype built from the **ARIA Ultra build specification**, layered on top of everything proven in ARIA v4. Local-first (Ollama, zero API keys), with optional OpenAI-compatible cloud providers that activate automatically when configured.

```
ProviderPool · Research · Build · Market · Deploy · Orchestrate · Memory · Audit
```

## What's new vs ARIA v4

| Capability | ARIA v4 | ARIA Ultra |
|---|---|---|
| **ProviderPool** | static granite→hermes fallback | multi-provider routing: health stats, circuit breaker (3 fails → 60s open), automatic failover, optional cloud (Groq/OpenRouter/...) |
| **Security** | terminal blocklist only | prompt-injection gate on every input + secret/dangerous-pattern scan on generated code |
| **Audit log** | — | append-only SQLite log of every dispatch, agent run, command, and error |
| **Evaluator** | quality gate (structural) | scored evaluation (completeness/correctness/safety) + build circuit breaker |
| **Market agent** | — | market intelligence, SWOT, competitive analysis reports |
| **Deployer agent** | — | Dockerfile, docker-compose, GitHub Actions generation for built projects |
| **Trainer agent** | lessons on failure | extracts reusable skills from successful projects into `skills/` |
| **TaskManager** | synchronous pipelines | persistent background task queue: priorities, retries, timeouts, survives restarts |
| **API mode** | — | FastAPI server: `/process`, `/health`, `/memory/search`, `/tasks`, `/agents` |
| **Config** | `.env` only | `.env` + optional `config.yaml` (env wins) |

Plus a real bug fixed during the port: **the v4 chat client silently dropped the system prompt** (persona/facts never reached plain chat) — the pool now injects it properly.

## Setup (Ubuntu)

```bash
cd ARIA_ULTRA
./setup.sh                # venv + deps + installs package + pulls models
source .venv/bin/activate
aria-ultra                # or: python -m ultra
```

Requires: Python 3.10+, Ollama with `granite4.1:3b`, `hermes3:3b`, `nomic-embed-text`.

## Quick start

```
ARIA > status                          # provider health, memory, circuit state
ARIA > hello                           # chat (persona + your facts now injected)
ARIA > compare Flask and FastAPI       # research with cited report
ARIA > build a todo CLI app            # architect → coder → review → test → git
ARIA > market analysis for ai dev tools   # market intelligence report
ARIA > deploy the last project         # Dockerfile + compose + CI generation
ARIA > research and build a weather cli tool   # full pipeline
ARIA > fix the code in ~/myapp         # improve mode
ARIA > tasks                           # background task queue
ARIA > audit                           # last operations
ARIA > mode 3                          # orchestrate — decomposes into background tasks
ARIA > memory fact goal: GATE EC 2028  # permanent facts
ARIA > skill extract ~/aria4_projects/myapp   # learn a skill from a project
```

One-shot (non-interactive):

```bash
aria-ultra "build a python cli calculator"
```

API server:

```bash
aria-ultra --api 8000
curl -X POST localhost:8000/process -d '{"objective": "research ollama"}' -H 'Content-Type: application/json'
```

The API binds to `127.0.0.1` by default. Set `ARIA4_API_TOKEN` in `.env`
to require `Authorization: Bearer <token>` on every request; only bind
`0.0.0.0` (via `ARIA4_API_HOST`) when a token is set.

## Architecture

```
ultra/
├── cli.py            REPL + commands
├── provider_pool.py  routing, health, circuit breakers, failover
├── llm.py            OllamaClient + OpenAICompatClient
├── security.py       input validation + code scanning
├── audit.py          append-only operation log
├── evaluator.py      quality scoring + build circuit breaker
├── task_manager.py   persistent background queue
├── orchestrator.py   the Brain: dispatch, agent registry, pipelines
├── api.py            FastAPI mode
├── agents/           research, market, architect, coder, reviewer,
│                     debugger, tester, deployer, trainer, engineering pipeline
├── tools/            safe bash, diff editor, web researcher (Bing + citations)
└── core/             memory (SQLite+FTS5), vectors (embeddings), skills (SKILL.md)
```

**Provider pool routing** — tasks are routed by capability: code tasks → coding model, chat/research → chat model, anything failing falls over to the next provider in priority order. A provider that fails 3× consecutively is circuit-open for 60s. Configure a cloud provider by setting `ULTRA_CLOUD_BASE_URL`, `ULTRA_CLOUD_API_KEY`, `ULTRA_CLOUD_MODEL` in `.env`.

**Security gates** — every input passes the injection check before any LLM call; every generated project is scanned for hardcoded secrets (`sk-`, `ghp_`, `AKIA`, ...) and dangerous patterns (`eval`, `os.system`, ...). Blocked inputs are recorded in the audit log. Additional hardening:
- **Path sandboxing** — every model-supplied filename is resolved against the project workspace (`resolve_inside`); `../../` traversal, absolute paths, and symlink escapes are silently dropped.
- **Untrusted web content** — fetched pages are wrapped in `<untrusted_web_source>` blocks and the model is told they are data, never instructions (prompt-injection defense); leaked tags are stripped from reports.
- **SSRF defense** — research only fetches http(s) URLs that resolve to public addresses; localhost, private ranges, and metadata endpoints (169.254.169.254) are refused, including mid-redirect.
- **API auth** — bearer token (`ARIA4_API_TOKEN`), localhost bind by default, and redacted error messages (full detail goes to the audit log).

**Self-improvement loop** — build → evaluate → on success (optionally) extract a skill → on failure write a lesson and open the build circuit. `ARIA4_EXTRACT_SKILLS=true` in `.env` turns on automatic skill extraction.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 117 tests, offline
```

## Roadmap

- WebSocket streaming in API mode
- Plugin system for custom tools
- Multi-modal input
- Distributed Brain (multi-node)
