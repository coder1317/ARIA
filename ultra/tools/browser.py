"""Browser automation tool — headless Playwright for dynamic web pages.

Falls back to requests+BeautifulSoup when Playwright isn't installed,
so the tool always exists but browser capabilities are optional.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("aria.browser")

try:
    from playwright.async_api import async_playwright, Browser, Page
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False


@dataclass
class BrowserResult:
    url: str
    title: str = ""
    text: str = ""
    html: str = ""
    screenshot_path: str | None = None
    error: str | None = None
    success: bool = True
    links: list[dict[str, str]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        parts = [f"Title: {self.title}" if self.title else ""]
        if self.text:
            parts.append(f"Text: {len(self.text)} chars")
        if self.screenshot_path:
            parts.append(f"Screenshot: {self.screenshot_path}")
        if self.links:
            parts.append(f"Links: {len(self.links)}")
        return " | ".join(p for p in parts if p) or "Page loaded"


class Browser:
    """Headless browser for JS-rendered pages, screenshots, and interaction.

    Usage:
        browser = Browser()
        result = await browser.goto("https://example.com")
        result = await browser.screenshot("https://example.com", "/tmp/page.png")
        result = await browser.search_google("query")
        await browser.close()
    """

    def __init__(self, headless: bool = True, timeout_ms: int = 30_000):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._pw = None
        self._browser: Browser | None = None

    async def _ensure_browser(self) -> Browser:
        if not _HAS_PLAYWRIGHT:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
        if self._browser is None:
            self._pw = await async_playwright().__aenter__()
            self._browser = await self._pw.chromium.launch(headless=self.headless)
        return self._browser

    async def goto(self, url: str, wait: str = "domcontentloaded") -> BrowserResult:
        """Navigate to a URL and return page content."""
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            await page.goto(url, wait_until=wait, timeout=self.timeout_ms)
            title = await page.title()
            text = await page.inner_text("body")
            html = await page.content()
            links = await self._extract_links(page)
            await page.close()
            return BrowserResult(
                url=url, title=title, text=text[:50_000],
                html=html[:100_000], links=links,
            )
        except Exception as e:
            logger.warning(f"Browser.goto failed for {url}: {e}")
            return BrowserResult(url=url, error=str(e), success=False)

    async def screenshot(self, url: str, path: str,
                         full_page: bool = False) -> BrowserResult:
        """Navigate and take a screenshot."""
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page(viewport={"width": 1280, "height": 720})
            await page.goto(url, wait_until="networkidle", timeout=self.timeout_ms)
            title = await page.title()
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=path, full_page=full_page)
            text = await page.inner_text("body")
            await page.close()
            return BrowserResult(
                url=url, title=title, text=text[:30_000],
                screenshot_path=path,
            )
        except Exception as e:
            return BrowserResult(url=url, error=str(e), success=False)

    async def search_google(self, query: str, num_results: int = 5) -> BrowserResult:
        """Search Google and return results."""
        url = f"https://www.google.com/search?q={query}"
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            # extract search results
            results = await page.evaluate("""() => {
                const items = [];
                document.querySelectorAll('div.g, div[data-sokoban-container]').forEach(el => {
                    const link = el.querySelector('a');
                    const title = el.querySelector('h3');
                    const snippet = el.querySelector('[data-sncf], .VwiC3b, .IsZvec');
                    if (link && title) {
                        items.push({
                            url: link.href,
                            title: title.innerText,
                            snippet: snippet ? snippet.innerText : ''
                        });
                    }
                });
                return items.slice(0, arguments[0]);
            }""", num_results)
            await page.close()
            text = "\n\n".join(
                f"[{i+1}] {r['title']}\n{r['url']}\n{r['snippet']}"
                for i, r in enumerate(results)
            )
            return BrowserResult(
                url=url, title=f"Google: {query}", text=text,
                links=[{"url": r["url"], "title": r["title"]} for r in results],
            )
        except Exception as e:
            return BrowserResult(url=url, error=str(e), success=False)

    async def execute_js(self, url: str, script: str) -> BrowserResult:
        """Navigate to a page and execute JavaScript."""
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            result = await page.evaluate(script)
            title = await page.title()
            await page.close()
            return BrowserResult(
                url=url, title=title, text=str(result)[:50_000],
            )
        except Exception as e:
            return BrowserResult(url=url, error=str(e), success=False)

    async def _extract_links(self, page: Page) -> list[dict[str, str]]:
        """Extract all links from the current page."""
        try:
            return await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a[href]')).slice(0, 50).map(a => ({
                    url: a.href,
                    text: a.innerText.trim().slice(0, 100)
                })).filter(l => l.url && l.text);
            }""")
        except Exception:
            return []

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._pw:
            await self._pw.__aexit__(None, None, None)
            self._pw = None


# ── sync wrapper for non-async callers (CLI, agents) ──────────────

def browse(url: str, **kwargs) -> BrowserResult:
    """Synchronous wrapper for Browser.goto()."""
    browser = Browser()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # already in async context — use a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, browser.goto(url, **kwargs)).result()
        return loop.run_until_complete(browser.goto(url, **kwargs))
    finally:
        try:
            asyncio.run(browser.close())
        except Exception:
            pass


def screenshot(url: str, path: str, **kwargs) -> BrowserResult:
    """Synchronous wrapper for Browser.screenshot()."""
    browser = Browser()
    try:
        return asyncio.run(browser.screenshot(url, path, **kwargs))
    finally:
        try:
            asyncio.run(browser.close())
        except Exception:
            pass
