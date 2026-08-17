import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultra.agents.research import sanitize_citations
from ultra.tools.researcher import Researcher, _decode_bing_url, _safe_url
from ultra.config import Config


# ── SSRF defense ────────────────────────────────────────────────────

def test_safe_url_allows_public_https():
    assert _safe_url("https://example.com/page")
    assert _safe_url("http://fastapi.tiangolo.com/")


def test_safe_url_blocks_non_http_schemes():
    assert not _safe_url("file:///etc/passwd")
    assert not _safe_url("ftp://example.com/x")
    assert not _safe_url("data:text/plain,hello")


def test_safe_url_blocks_local_and_private():
    assert not _safe_url("http://localhost:11434/")
    assert not _safe_url("http://127.0.0.1/")
    assert not _safe_url("http://169.254.169.254/latest/meta-data/")
    assert not _safe_url("http://10.0.0.1/")
    assert not _safe_url("http://192.168.1.1/")
    assert not _safe_url("http://172.16.0.5/")


# ── citation honesty ────────────────────────────────────────────────

def test_no_sources_strips_all_citations():
    text = ("Intro [1] and details [2].\n"
            "Source: [1] – Internal design documentation.\n"
            "Some real content here.")
    cleaned = sanitize_citations(text, n_sources=0)
    assert "[1]" not in cleaned
    assert "[2]" not in cleaned
    assert "Source:" not in cleaned
    assert "Some real content here." in cleaned


def test_with_sources_keeps_valid_citations():
    text = "FastAPI [1] is built on Starlette [2]."
    cleaned = sanitize_citations(text, n_sources=3)
    assert "[1]" in cleaned and "[2]" in cleaned


def test_with_sources_drops_out_of_range():
    text = "Real claim [1]. Fabricated claim [7]."
    cleaned = sanitize_citations(text, n_sources=2)
    assert "[1]" in cleaned
    assert "[7]" not in cleaned


def test_no_sources_marker_removed_cleanly():
    text = "para [1]\n\npara2\n\nSource: [1] – made up"
    cleaned = sanitize_citations(text, n_sources=0)
    assert "[" not in cleaned


def test_leaked_untrusted_delimiters_removed():
    text = (
        "SQLite is a library [1].\n\n"
        "*Source: <untrusted_web_source id=1>*\n\n"
        "More content.\n"
        "</untrusted_web_source> trailing\n"
    )
    cleaned = sanitize_citations(text, n_sources=4)
    assert "untrusted_web_source" not in cleaned
    assert "<" not in cleaned or "<untrusted" not in cleaned
    assert "SQLite is a library [1]." in cleaned
    assert "More content." in cleaned


# ── Bing URL decoding ───────────────────────────────────────────────

def test_decode_bing_redirect():
    # u= is junk prefix + base64 of https://fastapi.tiangolo.com/
    redirect = ("https://www.bing.com/ck/a?!&&p=abc&u=a1aHR0cHM6Ly9mYXN0YXBp"
                "LnRpYW5nb2xvLmNvbS8&ntb=1")
    assert _decode_bing_url(redirect) == "https://fastapi.tiangolo.com/"


def test_decode_bing_redirect_plain_base64():
    # some bing responses have clean base64 without the junk prefix
    import base64
    target = "https://example.com/article"
    clean = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    redirect = f"https://www.bing.com/ck/a?u={clean}"
    assert _decode_bing_url(redirect) == target


def test_decode_plain_url_unchanged():
    assert _decode_bing_url("https://example.com/x") == "https://example.com/x"


# ── Bing parsing (offline fixture) ──────────────────────────────────

BING_HTML = """<html><body>
<li class="b_algo">
  <h2><a href="https://example.com/fastapi">FastAPI - Modern Web Framework</a></h2>
  <div class="b_caption"><p>FastAPI is a modern framework for building APIs.</p></div>
</li>
<li class="b_algo">
  <h2><a href="https://example.org/docs">FastAPI Docs</a></h2>
  <div class="b_caption"><p>Official documentation and tutorial.</p></div>
</li>
</body></html>"""


def test_bing_parse_fixture(monkeypatch):
    researcher = Researcher(Config.load())
    class _Resp:
        text = BING_HTML
        def raise_for_status(self): pass
    monkeypatch.setattr(researcher.session, "get", lambda *a, **k: _Resp())
    results = researcher._search_bing("fastapi", 5)
    assert len(results) == 2
    assert results[0].title == "FastAPI - Modern Web Framework"
    assert results[0].url == "https://example.com/fastapi"
    assert "modern framework" in results[0].snippet


def test_bing_parse_anomaly_page(monkeypatch):
    researcher = Researcher(Config.load())
    class _Resp:
        text = "<html>no b_algo here</html>"
        def raise_for_status(self): pass
    monkeypatch.setattr(researcher.session, "get", lambda *a, **k: _Resp())
    assert researcher._search_bing("x", 5) == []


def test_fetch_refuses_unsafe_url(monkeypatch):
    """fetch() must not even attempt a request to a private/local URL."""
    researcher = Researcher(Config.load())
    called = []
    def _get(url, *a, **k):
        called.append(url)
        raise AssertionError("should not be called")
    monkeypatch.setattr(researcher.session, "get", _get)
    assert researcher.fetch("http://127.0.0.1:11434/x") == ""
    assert called == []


def test_fetch_follows_redirect_but_validates_each_hop(monkeypatch):
    """A redirect to a private address must be refused mid-chain."""
    researcher = Researcher(Config.load())
    calls = []
    class _Resp:
        def __init__(self, status, headers=None, text=""):
            self.status_code = status
            self.headers = headers or {}
            self.text = text
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size=1):
            return [self.text.encode()]
        def close(self):
            pass
    def _get(url, *a, **k):
        calls.append(url)
        if url.startswith("https://example.com/a"):
            return _Resp(302, {"Location": "http://192.168.0.5/secret"})
        return _Resp(200, text="<html>should never reach here</html>")
    monkeypatch.setattr(researcher.session, "get", _get)
    assert researcher.fetch("https://example.com/a") == ""
    # first hop attempted, unsafe redirect refused without a second request
    assert len(calls) == 1
