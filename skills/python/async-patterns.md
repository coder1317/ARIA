---
description: Python async/await patterns for concurrent I/O
triggers: async, await, concurrent, asyncio, parallel
---

# Python Async Patterns Skill

## When to Use Async
- Network I/O (HTTP requests, database queries, file downloads)
- Web scraping (many pages concurrently)
- API servers (handle many concurrent connections)
- NOT for CPU-bound work (use multiprocessing instead)

## Template: Concurrent HTTP Requests
```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        return await resp.text()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*errors="ignore")

urls = ["https://example.com", "https://httpbin.org/get"]
results = asyncio.run(fetch_all(urls))
```

## Template: Rate-Limited Concurrency
```python
import asyncio

async def rate_limited(coro, semaphore, *args):
    async with semaphore:
        return await coro(*args)

async def main():
    sem = asyncio.Semaphore(10)  # max 10 concurrent
    tasks = [rate_limited(fetch, sem, url) for url in urls]
    return await asyncio.gather(*tasks)
```

## Common Pitfalls
- Don't mix sync and async in the same event loop
- Use `asyncio.create_task()` for fire-and-forget
- Always set timeouts on network calls
- Use `asyncio.gather(return_exceptions=True)` for fault tolerance
