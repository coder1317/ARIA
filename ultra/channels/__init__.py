"""Channel abstraction — normalize messages across interfaces.

ARIA's core processes NormalizedMessage objects regardless of whether
they came from CLI, Telegram, a web UI, or any other channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedMessage:
    """Universal message format across all channels."""
    channel: str           # "cli" | "telegram" | "web" | "api"
    user_id: str           # channel-specific user identifier
    text: str              # the user's message
    session_id: str = ""   # conversation session identifier
    reply_to: str = ""     # message ID being replied to (for threading)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_group(self) -> bool:
        return self.metadata.get("is_group", False)


@dataclass
class NormalizedResponse:
    """Standard response format back to any channel."""
    text: str
    session_id: str = ""
    reply_to: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # optional: send as markdown, split into chunks, etc.
    parse_mode: str = "markdown"


class ChannelAdapter:
    """Base class for all channel adapters.

    Subclasses implement start(), stop(), and send_message().
    The core calls send_message() to push responses back.
    """

    name: str = "base"

    def __init__(self, dispatch_fn=None):
        """
        Args:
            dispatch_fn: Function(NormalizedMessage) -> str
                         Called when a message arrives from this channel.
        """
        self.dispatch_fn = dispatch_fn
        self._running = False

    async def start(self) -> None:
        """Start listening for messages."""
        self._running = True

    async def stop(self) -> None:
        """Stop listening and clean up."""
        self._running = False

    async def send_message(self, response: NormalizedResponse) -> bool:
        """Send a response back through this channel. Returns True on success."""
        raise NotImplementedError

    def format_response(self, text: str, **kwargs) -> NormalizedResponse:
        """Create a NormalizedResponse from raw text."""
        return NormalizedResponse(text=text, **kwargs)
