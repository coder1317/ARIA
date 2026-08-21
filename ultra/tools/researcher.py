"""Web research without API keys — DuckDuckGo primary, Bing fallback.

Uses the `ddgs` package for DuckDuckGo search (fast, reliable, no keys).
Falls back to Bing HTML scraping if DuckDuckGo is unavailable.
All LLM work still happens locally via Ollama.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import urllib.parse
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup  # type: ignore

try:
    from ddgs import DDGS
    _HAS_DDGS = True
except ImportError:
    _HAS_DDGS = False

from ultra.config import Config

UA = ("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) "
      "Gecko/20100101 Firefox/126.0")

MAX_REDIRECTS = 5
MAX_RAW_BYTES = 1_000_000  # cap page download size


def _relevant_source(src: Source, query: str) -> bool:
    """Filter out obviously irrelevant search results.

    Drops login pages, shopping sites, social media, non-technical content,
    and results with no topical overlap with the query.
    """
    url = src.url.lower()
    title = (src.title or "").lower()
    snippet = (src.snippet or "").lower()
    combined = f"{title} {snippet}"

    # Block known irrelevant domains
    blocked_domains = (
        'login', 'signin', 'sign-in', 'account', 'checkout',
        'cart', 'shopping', 'buy', 'price', 'shop.',
        'facebook.com', 'instagram.com', 'twitter.com', 'tiktok.com',
        'pinterest.com', 'reddit.com',
        'baidu.com', 'zhihu.com', 'weibo.com', 'douyin.com',
        'taobao.com', 'jd.com', '163.com', 'sina.com',
        'amazon.com', 'ebay.com', 'walmart.com',
        'dictionary.com', 'merriam-webster.com',
        'time.is', 'onlineclock', 'vclock', 'dayspedia',
    )
    if any(d in url for d in blocked_domains):
        return False

    # Block non-English content (CJK characters in title)
    import unicodedata
    cjk_chars = sum(1 for c in title
                    if ('\u4e00' <= c <= '\u9fff') or
                    ('\u3040' <= c <= '\u30ff') or
                    ('\uac00' <= c <= '\ud7af'))
    if cjk_chars > 2:
        return False

    # Technical content boost — if title contains tech terms, be more lenient
    tech_terms = ('verilog', 'vhdl', 'fpga', 'asic', 'rtl', 'python',
                  'javascript', 'api', 'docker', 'linux', 'git',
                  'machine learning', 'neural', 'database', 'sql',
                  'algorithm', 'compiler', 'kernel', 'driver',
                  'github', 'arxiv', 'stackoverflow', 'ieee', 'acm')
    has_tech = any(t in combined for t in tech_terms)

    # Check topical relevance
    query_words = set(w.lower().strip('?,.:;!?') for w in query.split()
                      if len(w) > 2)
    if query_words:
        matches = sum(1 for w in query_words if w in combined)
        # Require fewer matches if it's a tech source
        min_matches = 1 if has_tech else 2
        if matches < min_matches:
            return False

    return True


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
        """Search using DuckDuckGo (primary) or Bing (fallback).

        Unsafe URLs are filtered out (SSRF defense). Irrelevant results
        (login pages, shopping sites, non-English content) are also dropped.
        """
        limit = limit or self.config.research_max_sources
        # Primary: DuckDuckGo via ddgs package (fast, reliable)
        results = self._search_ddgs(query, limit + 3)
        # Fallback: Bing HTML scraping
        if not results:
            results = self._search_bing(query, limit + 3)
        # Last resort: DuckDuckGo HTML (often blocked)
        if not results:
            results = self._search_ddg_html(query, limit + 3)
        filtered = [r for r in results if _safe_url(r.url) and _relevant_source(r, query)]
        return filtered[:limit]

    def _search_ddgs(self, query: str, limit: int) -> list[Source]:
        """DuckDuckGo search via the ddgs package — fast, reliable, no keys."""
        if not _HAS_DDGS:
            return []
        try:
            results = DDGS().text(query, region="wt-wt", max_results=limit)
            sources = []
            for r in results:
                url = r.get("href", "")
                title = r.get("title", "")
                snippet = r.get("body", "")
                if url.startswith("http"):
                    sources.append(Source(url=url, title=title, snippet=snippet))
            return sources
        except Exception:
            return []

    def _search_bing(self, query: str, limit: int) -> list[Source]:
        """Bing HTML — li.b_algo result blocks. No API key needed."""
        try:
            resp = self.session.get(
                "https://www.bing.com/search",
                params={"q": query, "setlang": "en", "cc": "us", "count": str(limit + 2)},
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

    def _search_ddg_html(self, query: str, limit: int) -> list[Source]:
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

    def deep_search(self, query: str, n_queries: int = 3,
                    llm: OllamaClient | None = None) -> list[Source]:
        """Search the same topic from multiple angles.

        Uses the LLM to generate targeted search queries when available,
        falling back to keyword extraction + angle suffixes.
        """
        variants = self._generate_search_queries(query, n_queries, llm)
        seen: dict[str, Source] = {}
        for variant in variants:
            for src in self.search(variant):
                if src.url not in seen:
                    seen[src.url] = src
        return list(seen.values())[:self.config.research_max_sources]

    def _generate_search_queries(self, topic: str, n: int,
                                 llm: OllamaClient | None = None) -> list[str]:
        """Generate n targeted search queries for a research topic.

        Uses the LLM when available for better query quality.
        """
        if llm:
            try:
                prompt = (
                    f"Generate {n} short web search queries for: {topic}. "
                    f"One query per line, 5-10 words each, no numbering."
                )
                raw = llm.generate(prompt, max_tokens=200, temperature=0.3)
                queries = [q.strip().strip('"').strip("'")
                          for q in raw.strip().splitlines() if q.strip()]
                if queries and len(queries) >= 2:
                    return queries[:n]
            except Exception:
                pass
        # Fallback: extract key terms and generate targeted queries
        stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                 'with', 'by', 'from', 'about', 'between', 'through', 'during',
                 'before', 'after', 'above', 'below', 'each', 'every', 'all',
                 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                 'just', 'because', 'as', 'until', 'while', 'find', 'what',
                 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                 'compare', 'best', 'worst', 'list', 'explain', 'describe',
                 'their', 'them', 'it', 'its', 'how'}
        words = topic.lower().split()
        keywords = [w.strip('?,.:;!?') for w in words
                    if w.strip('?,.:;!?') not in stop and len(w) > 2]
        if not keywords:
            keywords = [w for w in words if len(w) > 2][:5]
        if not keywords:
            keywords = words[:5]
        # Expand common abbreviations for better search results
        topic_lower = topic.lower()
        expansions = {
            'cdc': 'clock domain crossing',
            'clk': 'clock signal',
            'fsa': 'finite state automaton',
            ' fsm': ' finite state machine',
            'dut': 'design under test',
            'tb': 'testbench',
            'dff': 'D flip-flop',
            'mux': 'multiplexer',
            'alu': 'arithmetic logic unit',
            'risc': 'reduced instruction set computing',
            'cpu': 'central processing unit',
            'gpu': 'graphics processing unit',
            'dsp': 'digital signal processing',
            'soc': 'system on chip',
        }
        expanded_keywords = []
        for kw in keywords:
            if kw in expansions:
                expanded_keywords.append(expansions[kw])
            else:
                expanded_keywords.append(kw)
        core = ' '.join(expanded_keywords[:6])

        # Technical domain hints
        domain = ''
        if any(t in topic_lower for t in ('verilog', 'vhdl', 'rtl', 'fpga', 'asic', 'synthesis', 'cdc')):
            domain = 'digital design'
        elif any(t in topic_lower for t in ('python', 'javascript', 'api', 'web', 'app')):
            domain = 'software engineering'
        elif any(t in topic_lower for t in ('machine learning', 'neural', 'model', 'ai')):
            domain = 'machine learning'
        elif any(t in topic_lower for t in ('circuit', 'pcb', 'hardware', 'embedded')):
            domain = 'electronics'

        angles = [
            core,
            f"{core} {domain} overview" if domain else f"{core} overview",
            f"{core} comparison analysis" if 'compare' in topic_lower or 'vs' in topic_lower else f"{core} best practices",
        ]
        return angles[:n]
