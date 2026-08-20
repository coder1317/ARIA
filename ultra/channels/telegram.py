"""Telegram channel adapter — talk to ARIA from your phone.

Requires: pip install python-telegram-bot
Environment: TELEGRAM_BOT_TOKEN (from @BotFather)

Usage:
    from ultra.channels.telegram import TelegramChannel
    channel = TelegramChannel(token="YOUR_TOKEN", dispatch_fn=aria_handler)
    await channel.start()
"""
from __future__ import annotations

import logging
import os
from typing import Callable

from ultra.channels import ChannelAdapter, NormalizedMessage, NormalizedResponse

logger = logging.getLogger("aria.telegram")

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False


class TelegramChannel(ChannelAdapter):
    """Telegram bot channel for ARIA."""

    name = "telegram"

    def __init__(self, token: str | None = None,
                 dispatch_fn: Callable | None = None,
                 allowed_users: list[int] | None = None):
        """
        Args:
            token: Telegram bot token. Falls back to TELEGRAM_BOT_TOKEN env var.
            dispatch_fn: Function(NormalizedMessage) -> str
            allowed_users: List of Telegram user IDs allowed to use the bot.
                          None = anyone can use it.
        """
        super().__init__(dispatch_fn)
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.allowed_users = allowed_users
        self._app: Application | None = None
        # track sessions per user
        self._sessions: dict[int, str] = {}

    async def start(self) -> None:
        if not _HAS_TELEGRAM:
            raise RuntimeError(
                "python-telegram-bot not installed. "
                "Run: pip install python-telegram-bot"
            )
        if not self.token:
            raise RuntimeError(
                "No Telegram token. Set TELEGRAM_BOT_TOKEN env var "
                "or pass token= to TelegramChannel."
            )

        self._app = (
            Application.builder()
            .token(self.token)
            .build()
        )

        # register handlers
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))
        self._app.add_handler(CommandHandler("new", self._handle_new_session))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        self._running = True
        logger.info("Telegram bot starting...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot running.")

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        self._running = False
        logger.info("Telegram bot stopped.")

    async def send_message(self, response: NormalizedResponse) -> bool:
        """Send a response to a Telegram user/chat."""
        if not self._app:
            return False
        try:
            chat_id = int(response.metadata.get("chat_id", 0))
            if not chat_id:
                return False

            text = response.text
            # Telegram has a 4096 char limit — split if needed
            chunks = self._split_message(text, 4000)
            for chunk in chunks:
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="Markdown",
                    disable_web_page_preview=True,
                )
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            # retry without markdown if parsing fails
            try:
                await self._app.bot.send_message(
                    chat_id=int(response.metadata.get("chat_id", 0)),
                    text=response.text[:4000],
                    disable_web_page_preview=True,
                )
                return True
            except Exception:
                return False

    # ── handlers ─────────────────────────────────────────────────

    async def _handle_start(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        user = update.effective_user
        if not self._check_access(user.id):
            await update.message.reply_text("Access denied.")
            return
        welcome = (
            f"👋 Hi {user.first_name}! I'm **ARIA** — your autonomous "
            f"engineering assistant.\n\n"
            f"**What I can do:**\n"
            f"• Research with citations\n"
            f"• Build full projects (code → test → git)\n"
            f"• Market intelligence & SWOT\n"
            f"• Deploy configs (Docker, CI)\n"
            f"• Learn from projects\n\n"
            f"**Commands:**\n"
            f"/new — new conversation\n"
            f"/help — show this menu\n\n"
            f"Just type naturally to get started!"
        )
        await update.message.reply_text(welcome, parse_mode="Markdown")

    async def _handle_help(self, update: Update,
                           context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_access(update.effective_user.id):
            return
        await update.message.reply_text(
            "Just type what you need. Examples:\n\n"
            "• `research FastAPI vs Flask`\n"
            "• `build a todo CLI in Python`\n"
            "• `market analysis for AI dev tools`\n"
            "• `deploy the last project`\n"
            "• `what can you do?`\n\n"
            "Use /new to start a fresh conversation.",
            parse_mode="Markdown",
        )

    async def _handle_new_session(self, update: Update,
                                  context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._check_access(update.effective_user.id):
            return
        user_id = update.effective_user.id
        self._sessions[user_id] = f"tg_{user_id}_{int(__import__('time').time())}"
        await update.message.reply_text("✅ New conversation started.")

    async def _handle_message(self, update: Update,
                              context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle regular text messages."""
        user = update.effective_user
        if not self._check_access(user.id):
            await update.message.reply_text("Access denied.")
            return

        text = update.message.text
        if not text:
            return

        # show typing indicator
        await update.message.chat.send_action("typing")

        # build session ID
        session_id = self._sessions.get(user.id, f"tg_{user.id}")

        # normalize the message
        msg = NormalizedMessage(
            channel="telegram",
            user_id=str(user.id),
            text=text,
            session_id=session_id,
            reply_to=str(update.message.message_id),
            metadata={
                "chat_id": update.effective_chat.id,
                "username": user.username or user.first_name,
                "is_group": update.effective_chat.type in ("group", "supergroup"),
            },
        )

        # dispatch to ARIA
        response_text = ""
        try:
            if self.dispatch_fn:
                response_text = self.dispatch_fn(msg)
            else:
                response_text = "⚠️ No dispatch function configured."
        except Exception as e:
            logger.error(f"Dispatch error: {e}")
            response_text = f"❌ Error: {str(e)[:200]}"

        # send back
        response = NormalizedResponse(
            text=response_text,
            session_id=session_id,
            reply_to=msg.reply_to,
            metadata={"chat_id": update.effective_chat.id},
        )
        await self.send_message(response)

    # ── helpers ──────────────────────────────────────────────────

    def _check_access(self, user_id: int) -> bool:
        if self.allowed_users is None:
            return True
        return user_id in self.allowed_users

    @staticmethod
    def _split_message(text: str, max_len: int = 4000) -> list[str]:
        """Split a long message into chunks that fit Telegram's limit."""
        if len(text) <= max_len:
            return [text]
        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            # try to split at a newline
            split_at = text.rfind("\n", 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return chunks
