import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultra.intent import extract_problem, keyword_route


def test_build_route():
    assert keyword_route("build a todo CLI app") == "build_only"


def test_research_route():
    assert keyword_route("what is FastAPI") == "research_only"
    assert keyword_route("compare Flask and Django") == "research_only"


def test_improve_route():
    assert keyword_route("fix the bug in my app") == "improve"


def test_full_pipeline():
    assert keyword_route("research and build a chat app") == "full_pipeline"


def test_chat_fallback():
    assert keyword_route("hello there") == "chat"
    assert keyword_route("hmm not sure what to say") == "chat"


def test_extract_problem():
    assert extract_problem("build a todo app") == "a todo app"
    assert extract_problem("research drones") == "drones"
    assert extract_problem("just chat normally") == "just chat normally"


def test_market_route():
    assert keyword_route("market analysis for ai dev tools") == "market"
    assert keyword_route("swot analysis of tesla") == "market"
    assert keyword_route("competitive analysis for saas") == "market"


def test_deploy_route():
    assert keyword_route("deploy the project to docker") == "deploy"
    assert keyword_route("dockerize my app") == "deploy"


def test_market_extract():
    assert extract_problem("market analysis for ai dev tools") == "for ai dev tools"


# ── chat-first rules: small talk must never become research ─────────

def test_introduce_yourself_is_chat():
    assert keyword_route("i want you to introduce yourself") == "chat"
    assert keyword_route("tell me about yourself") == "chat"
    assert keyword_route("who are you") == "chat"
    assert keyword_route("what can you do") == "chat"


def test_greetings_are_chat():
    assert keyword_route("hello") == "chat"
    assert keyword_route("good morning, aria") == "chat"
    assert keyword_route("thanks for the help") == "chat"


def test_how_does_this_work_is_chat_but_specific_questions_research():
    # generic meta-question → chat
    assert keyword_route("how does this work") == "chat"
    # specific factual question → research (must not be swallowed)
    assert keyword_route("how does django work") == "research_only"
