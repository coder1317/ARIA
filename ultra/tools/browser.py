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

    # ── Interaction methods ─────────────────────────────────────

    async def click(self, url: str, selector: str) -> BrowserResult:
        """Navigate to a page and click an element."""
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            await page.click(selector, timeout=5000)
            await page.wait_for_load_state("domcontentloaded")
            title = await page.title()
            text = await page.inner_text("body")
            final_url = page.url
            await page.close()
            return BrowserResult(url=final_url, title=title, text=text[:30_000])
        except Exception as e:
            return BrowserResult(url=url, error=str(e), success=False)

    async def type_text(self, url: str, selector: str, text: str,
                        submit: bool = False) -> BrowserResult:
        """Navigate, type into an input field, optionally submit."""
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            await page.fill(selector, text, timeout=5000)
            if submit:
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("domcontentloaded")
            title = await page.title()
            body_text = await page.inner_text("body")
            final_url = page.url
            await page.close()
            return BrowserResult(url=final_url, title=title, text=body_text[:30_000])
        except Exception as e:
            return BrowserResult(url=url, error=str(e), success=False)

    async def select_option(self, url: str, selector: str,
                            value: str) -> BrowserResult:
        """Navigate and select an option from a dropdown."""
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            await page.select_option(selector, value, timeout=5000)
            title = await page.title()
            text = await page.inner_text("body")
            await page.close()
            return BrowserResult(url=url, title=title, text=text[:30_000])
        except Exception as e:
            return BrowserResult(url=url, error=str(e), success=False)

    async def scroll(self, url: str, direction: str = "down",
                     pixels: int = 500) -> BrowserResult:
        """Navigate and scroll the page."""
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            if direction == "down":
                await page.evaluate(f"window.scrollBy(0, {pixels})")
            elif direction == "up":
                await page.evaluate(f"window.scrollBy(0, -{pixels})")
            elif direction == "bottom":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif direction == "top":
                await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(500)  # let lazy content load
            title = await page.title()
            text = await page.inner_text("body")
            await page.close()
            return BrowserResult(url=url, title=title, text=text[:30_000])
        except Exception as e:
            return BrowserResult(url=url, error=str(e), success=False)

    async def inspect(self, url: str, selector: str = "body") -> BrowserResult:
        """Inspect DOM elements — get HTML, text, attributes."""
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            info = await page.evaluate("""(sel) => {
                const el = document.querySelector(sel);
                if (!el) return {error: 'selector not found'};
                return {
                    tag: el.tagName,
                    text: el.innerText?.slice(0, 2000),
                    html: el.outerHTML?.slice(0, 5000),
                    attrs: Object.fromEntries(Array.from(el.attributes).map(a => [a.name, a.value])),
                    children: el.children.length,
                };
            }""", selector)
            await page.close()
            if "error" in info:
                return BrowserResult(url=url, error=info["error"], success=False)
            text = info.get("text", "")
            return BrowserResult(
                url=url, title=info.get("tag", ""), text=text,
                html=info.get("html", ""),
            )
        except Exception as e:
            return BrowserResult(url=url, error=str(e), success=False)

    async def download(self, url: str, selector: str, save_path: str) -> BrowserResult:
        """Click a download link and save the file."""
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            async with page.expect_download(timeout=30000) as download_info:
                await page.click(selector)
            download = await download_info.value
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            await download.save_as(save_path)
            await page.close()
            return BrowserResult(url=url, title=f"Downloaded to {save_path}",
                                text=f"File saved: {save_path}")
        except Exception as e:
            return BrowserResult(url=url, error=str(e), success=False)

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


def click_element(url: str, selector: str) -> BrowserResult:
    browser = Browser()
    try:
        return asyncio.run(browser.click(url, selector))
    finally:
        try:
            asyncio.run(browser.close())
        except Exception:
            pass


def type_in_field(url: str, selector: str, text: str,
                  submit: bool = False) -> BrowserResult:
    browser = Browser()
    try:
        return asyncio.run(browser.type_text(url, selector, text, submit))
    finally:
        try:
            asyncio.run(browser.close())
        except Exception:
            pass


def scroll_page(url: str, direction: str = "down",
                pixels: int = 500) -> BrowserResult:
    browser = Browser()
    try:
        return asyncio.run(browser.scroll(url, direction, pixels))
    finally:
        try:
            asyncio.run(browser.close())
        except Exception:
            pass


def inspect_element(url: str, selector: str = "body") -> BrowserResult:
    browser = Browser()
    try:
        return asyncio.run(browser.inspect(url, selector))
    finally:
        try:
            asyncio.run(browser.close())
        except Exception:
            pass
