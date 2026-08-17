"""Ollama LLM client — chat, streaming, structured JSON, and embeddings.

Only talks to a local Ollama instance (default http://localhost:11434).
No API keys, no vendor lock-in.
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterator

import requests

from ultra.config import Config


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, config: Config | None = None):
        self.config = config or Config.load()
        # long generation timeout — deep research/build generations on a
        # 3B local model can exceed minutes; configurable via
        # ARIA4_LLM_TIMEOUT (default 900s)
        self.timeout = self.config.llm_timeout
        self._available_cache: list[str] | None = None

    def _resolve(self, model: str | None) -> str:
        """Return a model that actually exists on the server.

        Order: requested model → explicit fallback (hermes3:3b) → whatever
        is pulled, so a half-configured setup still works.
        """
        if model is None:
            model = self.config.chat_model
        if self._available_cache is None:
            try:
                self._available_cache = self.available_models()
            except OllamaError:
                self._available_cache = []
        available = self._available_cache
        if not available or model in available:
            return model
        for candidate in (self.config.fallback_model,
                          self.config.chat_model,
                          self.config.coding_model):
            if candidate in available:
                return candidate
        return model

    # ── low-level helpers ──────────────────────────────────────────

    def _post(self, path: str, payload: dict, timeout: int | None = None) -> dict:
        try:
            resp = requests.post(
                self.config.ollama_url + path,
                json=payload,
                timeout=timeout or self.timeout,
            )
        except requests.Timeout as e:
            # NOTE: requests.Timeout subclasses ConnectionError — catch it first
            raise OllamaError(
                f"Ollama timed out after {timeout or self.timeout}s "
                f"({path}). Long generations on small local models can "
                "exceed this — raise ARIA4_LLM_TIMEOUT if it keeps happening."
            ) from e
        except requests.ConnectionError as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.config.ollama_url}. "
                "Start it with: ollama serve  (or systemctl start ollama)"
            ) from e
        if resp.status_code != 200:
            raise OllamaError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _get(self, path: str, timeout: int | None = None) -> dict:
        """GET request — /api/tags only accepts GET/HEAD, not POST."""
        try:
            resp = requests.get(
                self.config.ollama_url + path,
                timeout=timeout or self.timeout,
            )
        except requests.Timeout as e:
            # NOTE: requests.Timeout subclasses ConnectionError — catch it first
            raise OllamaError(
                f"Ollama timed out after {timeout or self.timeout}s ({path})."
            ) from e
        except requests.ConnectionError as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.config.ollama_url}."
            ) from e
        if resp.status_code != 200:
            raise OllamaError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _stream(self, path: str, payload: dict, timeout: int | None = None) -> Iterator[dict]:
        try:
            resp = requests.post(
                self.config.ollama_url + path,
                json=payload,
                timeout=timeout or self.timeout,
                stream=True,
            )
        except requests.Timeout as e:
            raise OllamaError(
                f"Ollama timed out after {timeout or self.timeout}s ({path})."
            ) from e
        except requests.ConnectionError as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.config.ollama_url}."
            ) from e
        if resp.status_code != 200:
            raise OllamaError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        except requests.Timeout as e:
            raise OllamaError(
                f"Ollama stream timed out after {timeout or self.timeout}s ({path})."
            ) from e

    # ── public API ─────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        format: str | None = None,  # "json" to force JSON output
    ) -> str:
        """Non-streaming completion, returns the full text."""
        payload: dict[str, Any] = {
            "model": self._resolve(model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format
        data = self._post("/api/generate", payload)
        return data.get("response", "")

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream tokens as they are generated."""
        payload: dict[str, Any] = {
            "model": self._resolve(model),
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system
        for chunk in self._stream("/api/generate", payload):
            token = chunk.get("response", "")
            if token:
                yield token

    def chat(self, messages: list[dict], model: str | None = None, **opts) -> str:
        """Chat-style completion using the /api/chat endpoint."""
        payload: dict[str, Any] = {
            "model": self._resolve(model),
            "messages": messages,
            "stream": False,
            "options": {"num_predict": opts.get("max_tokens", 4096),
                        "temperature": opts.get("temperature", 0.7)},
        }
        if opts.get("format"):
            payload["format"] = opts["format"]
        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    def json(self, prompt: str, system: str | None = None, model: str | None = None,
             max_tokens: int = 2048, temperature: float = 0.0) -> dict | None:
        """Force structured JSON output. Returns None on parse failure."""
        try:
            raw = self.generate(
                prompt, system=system, model=model,
                max_tokens=max_tokens, temperature=temperature, format="json",
            )
            return json.loads(raw)
        except (json.JSONDecodeError, OllamaError):
            return None

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Embedding vector for a piece of text (Ollama embeddings API)."""
        payload = {
            "model": model or self.config.embed_model,
            "prompt": text,
        }
        data = self._post("/api/embeddings", payload, timeout=60)
        return data.get("embedding", [])

    def available_models(self) -> list[str]:
        """Names of models currently pulled on the Ollama server."""
        try:
            data = self._get("/api/tags", timeout=10)
            return [m.get("name", "") for m in data.get("models", [])]
        except OllamaError:
            return []

    def ping(self, retries: int = 1) -> bool:
        for attempt in range(retries + 1):
            try:
                self._get("/api/tags", timeout=5)
                return True
            except OllamaError:
                if attempt < retries:
                    time.sleep(1.5)
        return False

    def ensure_model(self, model: str) -> bool:
        """Return True if the model exists locally, else False (with a hint)."""
        return model in self.available_models()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class OpenAICompatClient:
    """Client for any OpenAI-compatible chat-completions endpoint.

    Same surface as OllamaClient (generate / json / chat) so the
    ProviderPool can treat both identically. No embeddings — those stay
    local (Ollama). Requires api_key.
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _post(self, payload: dict) -> dict:
        try:
            resp = requests.post(
                self.base_url + "/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        except requests.ConnectionError as e:
            raise OllamaError(f"Cannot reach {self.base_url}") from e
        if resp.status_code != 200:
            raise OllamaError(
                f"Cloud provider error {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()

    def generate(self, prompt: str, system: str | None = None,
                 model: str | None = None, max_tokens: int = 4096,
                 temperature: float = 0.7, format: str | None = None,
                 **_) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = self._post(payload)
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def json(self, prompt: str, system: str | None = None,
             model: str | None = None, max_tokens: int = 2048,
             temperature: float = 0.0, **_) -> dict | None:
        try:
            raw = self.generate(prompt, system=system, model=model,
                                max_tokens=max_tokens, temperature=temperature)
            return json.loads(raw)
        except (json.JSONDecodeError, OllamaError):
            return None

    def chat(self, messages: list[dict], model: str | None = None, **opts) -> str:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": opts.get("max_tokens", 4096),
            "temperature": opts.get("temperature", 0.7),
        }
        data = self._post(payload)
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
