# ARIA — Autonomous Multi-Agent Engineering Assistant

```
ProviderPool · Research · Build · Market · Deploy · Orchestrate · Memory · Audit
Telegram · Browser · Scheduler · MCP
```

## What is ARIA?

**ARIA** (Autonomous Research & Intelligent Agent) is a local-first, multi-agent AI system that can **research, plan, code, debug, test, deploy, and self-improve** software projects — all from your terminal, with zero API keys and zero cloud dependency.

Think of it as an **AI engineering team on your laptop**. Instead of a single chatbot, ARIA is a collection of specialized agents — a researcher, architect, coder, debugger, tester, deployer, and more — coordinated by a central "Brain" orchestrator that decides which agent handles what, in what order, and how to recover when things go wrong.

### Core ideas

| Principle | What it means in practice |
|---|---|
| **Local-first** | Runs entirely on your machine via Ollama. Your code, research, and data never leave your laptop unless you explicitly add a cloud provider. Works on **Linux, macOS, and Windows**. |
| **Multi-agent** | Specialized agents (research, code, debug, test, deploy, market intelligence, skill extraction) collaborate under a central Brain orchestrator. |
| **Self-correcting** | Generated code is reviewed, tested, and security-scanned before delivery. Failures feed lessons back into memory. The system gets smarter over time. |
| **Provider-agnostic** | A ProviderPool routes tasks to the best available LLM (local or cloud), with health tracking, circuit breakers, and automatic failover. |
| **Auditable** | Every dispatch, agent call, command, and error is logged in an append-only audit trail. Nothing happens invisibly. |
| **Secure** | Path sandboxing, SSRF defense, untrusted-content isolation, API auth, and a hardened terminal blocklist (Unix + Windows patterns). |

### How a request flows

```
User: "build a todo CLI app in Python"
        │
        ▼
    Security Gate  ──── injection check, input validation
        │
        ▼
       Brain  ──────── intent classified → build pipeline
        │
        ▼
    Research  ──────── best practices, tech choices (with citations)
        │
        ▼
    Architect ──────── file structure, module dependencies, API design
        │
        ▼
     Coder    ──────── full implementation (routed to coding-capable model)
        │
        ▼
    Reviewer  ──────── code review + security scan
        │
        ▼
    Debugger  ──────── fix issues (re-scans to verify fixes are real)
        │
        ▼
     Tester   ──────── unit tests, syntax check, runtime smoke test
        │
        ▼
   Evaluator  ──────── scored build (completeness / correctness / safety)
        │
        ▼
      Git     ──────── init, add, commit
        │
        ▼
    Result    ──────── working project in ~/aria4_projects/<name>
```

### What ARIA can do today

- **Research** — deep research with Bing, real citations, and a written report (saved to disk)
- **Build** — architect → code → review → debug → test → git commit, all autonomous
- **Market intelligence** — SWOT, competitive analysis, trend reports with sources
- **Deploy** — generates Dockerfile, docker-compose, and GitHub Actions for a built project
- **Learn** — extracts reusable skills from successful projects; writes lessons from failures
- **Background tasks** — submit long-running work, check status, survive process restarts
- **API server** — FastAPI mode with auth, for integration into other tools
- **Telegram** — talk to ARIA from your phone (same memory, skills, tools)
- **Browser** — headless Playwright for JS-rendered pages, screenshots, Google search
- **Scheduler** — autonomous tasks: "every morning research AI news and send summary"
- **MCP** — plug in external tool servers via Model Context Protocol

---

## What's new vs ARIA v4

| Capability | ARIA v4 | ARIA |
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
| **Platform** | Linux only | Linux, macOS, **Windows** (setup.ps1 + terminal hardening) |

Plus a real bug fixed during the port: **the v4 chat client silently dropped the system prompt** (persona/facts never reached plain chat) — the pool now injects it properly.

## Setup

### Ubuntu / macOS

```bash
cd ARIA_ULTRA
./setup.sh                  # venv + deps + installs package + pulls models
source .venv/bin/activate
aria                       # or: python -m ultra
```

### Windows

```powershell
# Option 1: double-click setup.bat
# Option 2: PowerShell
cd ARIA_ULTRA
.\setup.ps1
.\.venv\Scripts\Activate.ps1
aria
```

Requires: Python 3.10+, Ollama with `granite4.1:3b`, `hermes3:3b`, `nomic-embed-text`.
The Windows script installs Ollama automatically if missing.

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
ARIA > skill search esp32                    # search GitHub for aria-skill repos
ARIA > skill install coder1317/pcb-design    # install a skill from GitHub
ARIA > skill list                            # list installed skills
ARIA > browse https://example.com            # headless browser
ARIA > schedule add name='news' command='research AI news' daily=08:00
ARIA > telegram                              # start Telegram bot
```

## Skills

Skills are reusable instructions that make ARIA smarter at specific tasks.
Install from GitHub (tag your repo with `aria-skill`):

```bash
ARIA > skill search esp32          # find skills on GitHub
ARIA > skill install owner/repo   # install from GitHub
ARIA > skill list                  # see what's installed
ARIA > skill show pcb-design       # view skill details
ARIA > skill uninstall pcb-design  # remove a skill
```

Skills are auto-matched to your requests — if you install a `pcb-design`
skill and say "design a PCB", ARIA automatically uses it.

You can also create skills from your own projects:

```bash
ARIA > skill extract ~/aria4_projects/myapp   # extract patterns into a skill
```

### Creating skill repos

A skill repo needs either a `SKILL.md` or `skill.json` (or both):

```
my-skill/
├── SKILL.md        # instructions with frontmatter
├── skill.json      # metadata (version, tags, triggers)
└── examples/       # optional code templates
```

Tag your repo with `aria-skill` on GitHub so others can find it.

One-shot (non-interactive):

```bash
aria "build a python cli calculator"
```

API server:

```bash
aria --api 8000
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
├── scheduler.py      cron-style autonomous task execution
├── orchestrator.py   the Brain: dispatch, agent registry, pipelines
├── api.py            FastAPI mode
├── channels/         multi-interface adapters
│   ├── __init__.py   NormalizedMessage abstraction
│   └── telegram.py   Telegram bot channel
├── agents/           research, market, architect, coder, reviewer,
│                     debugger, tester, deployer, trainer, engineering pipeline
├── tools/            safe bash, diff editor, web researcher, browser, MCP
│   ├── browser.py    Playwright headless browser
│   └── mcp_client.py Model Context Protocol client
└── core/             memory (SQLite+FTS5), vectors (embeddings), skills (SKILL.md)
```

**Provider pool routing** — tasks are routed by capability: code tasks → coding model, chat/research → chat model, anything failing falls over to the next provider in priority order. A provider that fails 3× consecutively is circuit-open for 60s. Configure a cloud provider by setting `ULTRA_CLOUD_BASE_URL`, `ULTRA_CLOUD_API_KEY`, `ULTRA_CLOUD_MODEL` in `.env`.

**Security gates** — every input passes the injection check before any LLM call; every generated project is scanned for hardcoded secrets (`sk-`, `ghp_`, `AKIA`, ...) and dangerous patterns (`eval`, `os.system`, ...). Blocked inputs are recorded in the audit log. Additional hardening:
- **Path sandboxing** — every model-supplied filename is resolved against the project workspace (`resolve_inside`); `../../` traversal, absolute paths, and symlink escapes are silently dropped.
- **Untrusted web content** — fetched pages are wrapped in `<untrusted_web_source>` blocks and the model is told they are data, never instructions (prompt-injection defense); leaked tags are stripped from reports.
- **SSRF defense** — research only fetches http(s) URLs that resolve to public addresses; localhost, private ranges, and metadata endpoints (169.254.169.254) are refused, including mid-redirect.
- **API auth** — bearer token (`ARIA4_API_TOKEN`), localhost bind by default, and redacted error messages (full detail goes to the audit log).

**Self-improvement loop** — build → evaluate → on success (optionally) extract a skill → on failure write a lesson and open the build circuit. `ARIA4_EXTRACT_SKILLS=true` in `.env` turns on automatic skill extraction.

## Platform support

| Platform | Setup | Status |
|---|---|---|
| **Ubuntu / Debian** | `./setup.sh` | ✅ Primary target |
| **macOS** | `./setup.sh` | ✅ Tested |
| **Windows** | `setup.bat` or `setup.ps1` | ✅ PowerShell auto-installs Ollama |

All core Python code is cross-platform (`pathlib`, `threading.local` SQLite, platform-aware terminal). The terminal blocklist covers both Unix (`rm -rf /`) and Windows (`format c:`, `rmdir /s`) destructive patterns.

## Tests

```bash
# Linux / macOS
.venv/bin/python -m pytest tests/ -q     # 117 tests, offline

# Windows
.\.venv\Scripts\python -m pytest tests\ -q
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ARIA4_HOME` | `~/.aria4` | Data directory (memory, vectors, audit) |
| `ARIA4_PROJECTS` | `~/aria4_projects` | Where built projects are saved |
| `ARIA4_EXTRACT_SKILLS` | `false` | Auto-extract skills on successful build |
| `ARIA4_LLM_TIMEOUT` | `900` | LLM read timeout in seconds (for slow 3B generation) |
| `ARIA4_API_TOKEN` | — | Bearer token for API auth (required if binding 0.0.0.0) |
| `ARIA4_API_HOST` | `127.0.0.1` | API bind address |
| `ULTRA_CLOUD_BASE_URL` | — | OpenAI-compatible cloud endpoint (activates cloud provider) |
| `ULTRA_CLOUD_API_KEY` | — | API key for cloud provider |
| `ULTRA_CLOUD_MODEL` | — | Model name for cloud provider |

See `.env.example` for all options.

## Roadmap

- WebSocket streaming in API mode
- Plugin system for custom tools
- Multi-modal input
- Distributed Brain (multi-node)
