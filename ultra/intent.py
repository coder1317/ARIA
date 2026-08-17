"""Intent detection — keyword rules first (zero LLM cost), LLM fallback.

Order matters: personal/small-talk patterns are matched BEFORE research
patterns so "introduce yourself" never becomes a research task.
"""
from __future__ import annotations

from ultra.llm import OllamaClient

INTENTS = {
    "full_pipeline": "research and build",
    "research_only": "research",
    "build_only": "build",
    "improve": "improve",
    "market": "market",
    "deploy": "deploy",
    "memory": "memory",
    "chat": "chat",
}

# matched FIRST — things that must never be routed to research/build
_CHAT_PATTERNS = [
    "introduce yourself", "tell me about yourself", "who are you",
    "what are you", "what can you do", "how are you", "hello", "hi ",
    "hey ", "thanks", "thank you", "good morning", "good evening",
    "good afternoon", "nice to meet", "help me get started",
    "what should i do first", "just chatting", "tell me a joke",
    "how does this work", "what is this", "what is aria",
]

_KEYWORD_ROUTES: list[tuple[str, list[str]]] = [
    ("full_pipeline", ["research and build", "research & build", "research then build",
                       "build a solution for", "solve the problem of"]),
    ("research_only", ["research ", "investigate ", "survey ", "look into ",
                       "find papers", "what is ", "how does ", "how to ",
                       "compare ", "difference between", "feasibility"]),
    ("build_only", ["build ", "create ", "make ", "scaffold ", "prototype ",
                    "develop ", "generate ", "write a ", "implement "]),
    ("improve", ["improve ", "fix ", "refactor ", "debug ", "add to ", "update ",
                 "optimize ", "review "]),
    ("market", ["market ", "market analysis", "swot", "competitive analysis",
                "industry analysis", "trend analysis for "]),
    ("deploy", ["deploy ", "dockerize ", "containerize ", "generate ci ",
                "github actions for "]),
    ("memory", ["show projects", "show skills", "show lessons", "memory",
                "what did you build", "recall "]),
]


def keyword_route(text: str) -> str:
    low = text.lower().strip()
    # small talk first — a 3B model classifier is unreliable, so the
    # cheap deterministic rules carry the cases that matter
    for pattern in _CHAT_PATTERNS:
        if pattern in low:
            return "chat"
    for intent, keywords in _KEYWORD_ROUTES:
        for kw in keywords:
            if kw in low:
                return intent
    return "chat"


_CLASSIFY_PROMPT = """Classify this user request into exactly one category.

Request: "{request}"

Categories:
- full_pipeline : research a problem AND build a solution for it
- research_only : research, investigate, or answer a factual question
- build_only    : build/create a new software project
- improve       : fix, debug, or improve existing code
- market        : market analysis, SWOT, competitive landscape
- deploy        : generate deployment configs (docker, CI)
- memory        : ask about past projects, skills, or saved knowledge
- chat          : personal questions about the assistant, greetings,
                  casual conversation, or anything not covered above

Examples:
  "introduce yourself"          -> chat
  "who are you"                 -> chat
  "hello"                       -> chat
  "what is fastapi"             -> research_only
  "build a todo app"            -> build_only
  "research and build a chat app" -> full_pipeline

Return JSON: {{"intent": "category"}}"""


def llm_route(client: OllamaClient, text: str) -> str:
    """LLM-based intent classification (fallback when keywords are ambiguous)."""
    result = client.json(_CLASSIFY_PROMPT.format(request=text[:500]))
    intent = (result or {}).get("intent", "chat")
    return intent if intent in INTENTS else "chat"


def detect(client: OllamaClient, text: str) -> str:
    intent = keyword_route(text)
    if intent == "chat":
        # ambiguous — ask the model once
        intent = llm_route(client, text)
    return intent


def extract_problem(text: str) -> str:
    """Strip the routing verb from a request to get the actual problem."""
    low = text.lower().strip()
    prefixes = [
        "research and build ", "research & build ",
        "build a solution for ", "solve the problem of ",
        "research ", "investigate ", "survey ", "build ",
        "create ", "make ", "scaffold ", "prototype ",
        "develop ", "improve ", "fix ", "debug ", "refactor ",
        "review ", "implement ", "generate ", "write a ",
        "market analysis ", "market ", "deploy ", "dockerize ", "containerize ",
    ]
    for p in prefixes:
        if low.startswith(p):
            return text[len(p):].strip()
    return text.strip()
