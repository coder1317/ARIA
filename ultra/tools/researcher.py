"""Web research without API keys — multi-engine fallback.

DuckDuckGo's HTML endpoint frequently serves an anti-bot "anomaly" page,
so the primary engine is Bing HTML (works without keys), with DuckDuckGo
as a backup. All LLM work still happens locally via Ollama.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import urllib.parse
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup  # type: ignore

from ultra.config import Config

UA = ("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) "
      "Gecko/20100101 Firefox/126.0")

MAX_REDIRECTS = 5
MAX_RAW_BYTES = 1_000_000  # cap page download size


def _safe_url(url: str) -> bool:
    """SSRF defense — only http(s), and only destinations that resolve to
    public addresses. Blocks file://, ftp://, data://, localhost,
    loopback, link-local (169.254.x), private ranges (10/8, 172.16/12,
    192.168/16) and metadata endpoints (169.254.169.254).
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    low = host.lower()
    if low in ("localhost", "::1") or low.endswith(".local") or low.endswith(".localhost"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
        except OSError:
            return False
    return bool(getattr(ip, "is_global", False))


@dataclass
class Source:
    url: str
    title: str
    snippet: str = ""
    content: str = ""

    def text(self, max_chars: int = 3000) -> str:
        body = self.content or self.snippet
        return body[:max_chars]


def _decode_bing_url(url: str) -> str:
    """Bing wraps real URLs in bing.com/ck/a redirects; the target URL is
    base64-encoded in the `u=` query param. Decode it for clean citations.
    """
    if "bing.com/ck/a" not in url:
        return url
    import base64
    from urllib.parse import parse_qs, urlparse
    qs = parse_qs(urlparse(url).query)
    encoded = qs.get("u", [""])[0]
    if not encoded:
        return url
    # bing prefixes junk bytes ("a1") before the base64-encoded URL and
    # may drop trailing chars — try the value as-is, then with the prefix
    # stripped, and locate the real URL inside whatever decodes
    candidates = [encoded]
    if encoded.startswith("a1"):
        candidates.append(encoded[2:])
    for cand in candidates:
        try:
            padded = cand + "=" * (-len(cand) % 4)
            decoded = base64.urlsafe_b64decode(padded).decode("utf-8",
                                                              errors="ignore")
        except Exception:
            continue
        idx = decoded.find("http")
        if idx >= 0:
            decoded = decoded[idx:]
        if decoded.startswith("http"):
            return decoded
    return url


def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


class Researcher:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": UA,
            "Accept-Language": "en-US,en;q=0.8",
        })

    # ── search ──────────────────────────────────────────────────────

    def search(self, query: str, limit: int | None = None) -> list[Source]:
        """Try Bing first, then DuckDuckGo as backup. Unsafe URLs are
        filtered out (SSRF defense)."""
        limit = limit or self.config.research_max_sources
        results = self._search_bing(query, limit)
        if not results:
            results = self._search_ddg(query, limit)
        return [r for r in results if _safe_url(r.url)]

    def _search_bing(self, query: str, limit: int) -> list[Source]:
        """Bing HTML — li.b_algo result blocks. No API key needed."""
        try:
            resp = self.session.get(
                "https://www.bing.com/search",
                params={"q": query, "setlang": "en"},
                timeout=self.config.search_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException:
            return []
        if "b_algo" not in resp.text:
            return []  # bot page or no results
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select("li.b_algo")[:limit]:
            link = item.select_one("h2 a")
            if not link:
                continue
            url = link.get("href", "")
            if not url.startswith("http"):
                continue
            title = link.get_text(strip=True)
            snippet_el = item.select_one(".b_caption p") or item.select_one("p")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append(Source(url=_decode_bing_url(url), title=title,
                                  snippet=snippet))
        return results

    def _search_ddg(self, query: str, limit: int) -> list[Source]:
        """DuckDuckGo HTML — often blocked by an anomaly page; kept as backup."""
        params = {"q": query, "kl": "us-en"}
        try:
            resp = self.session.get(
                "https://html.duckduckgo.com/html/",
                params=params,
                timeout=self.config.search_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException:
            return []
        if "anomaly" in resp.text.lower() or "result" not in resp.text:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result")[:limit]:
            link = result.select_one("a.result__a")
            snippet_el = result.select_one(".result__snippet")
            if not link:
                continue
            href = link.get("href", "")
            url = urllib.parse.unquote(href)
            if url.startswith("//"):
                url = "https:" + url
            if not url.startswith("http"):
                continue
            title = link.get_text(strip=True)
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append(Source(url=url, title=title, snippet=snippet))
        return results

    def fetch(self, url: str, max_chars: int = 5000) -> str:
        """Fetch and extract readable text from a page.

        Every redirect hop is re-validated against _safe_url (a page could
        redirect to localhost/private addresses), and download size is
        capped to avoid memory abuse.
        """
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            if not _safe_url(current):
                return ""
            try:
                resp = self.session.get(
                    current, timeout=self.config.search_timeout,
                    allow_redirects=False, stream=True,
                )
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        return ""
                    current = urllib.parse.urljoin(current, location)
                    resp.close()
                    continue
                resp.raise_for_status()
                chunks = []
                total = 0
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= MAX_RAW_BYTES:
                        break
                resp.close()
                html = b"".join(chunks).decode("utf-8", errors="ignore")
                return _clean_text(html)[:max_chars]
            except requests.RequestException:
                return ""
        return ""

    def deep_search(self, query: str, n_queries: int = 3) -> list[Source]:
        """Search the same topic from multiple angles."""
        variants = [
            query,
            f"{query} overview",
            f"{query} tutorial guide",
            f"{query} comparison best practices",
        ][:n_queries]
        seen: dict[str, Source] = {}
        for variant in variants:
            for src in self.search(variant):
                if src.url not in seen:
                    seen[src.url] = src
        return list(seen.values())[:self.config.research_max_sources]
