"""File Watchers — detect changes in project directories.

Watches ARIA's project directories for file changes and emits events
through the EventBus. When files are created, modified, or deleted,
ARIA can automatically react (e.g., re-run tests, update memory,
notify the user).

Uses polling (no external dependencies like watchdog) for maximum
portability across Ubuntu, Windows, and macOS.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger("aria.watchers")


class FileSnapshot:
    """Point-in-time snapshot of a directory's files."""

    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, float] = {}  # relative_path → mtime
        self._take()

    def _take(self, ignore: list[str] | None = None) -> None:
        """Capture current state of all files."""
        self.files.clear()
        if not self.root.exists():
            return
        ignore = ignore or []
        for f in self.root.rglob("*"):
            if not f.is_file():
                continue
            rel = str(f.relative_to(self.root))
            # Skip hidden files and ignored patterns
            if any(p.startswith(".") for p in f.parts):
                continue
            if any(pat in rel for pat in ignore):
                continue
            try:
                self.files[rel] = f.stat().st_mtime
            except (OSError, ValueError):
                pass

    def diff(self, other: FileSnapshot) -> dict[str, list[str]]:
        """Compare two snapshots. Returns {created: [], modified: [], deleted: []}."""
        created = []
        modified = []
        deleted = []

        for path, mtime in self.files.items():
            if path not in other.files:
                created.append(path)
            elif mtime > other.files[path]:
                modified.append(path)

        for path in other.files:
            if path not in self.files:
                deleted.append(path)

        return {"created": created, "modified": modified, "deleted": deleted}


class FileWatcher:
    """Polls project directories for file changes and emits events.

    Usage:
        from ultra.core.events import EventBus, EventType

        bus = EventBus()
        watcher = FileWatcher(bus)

        # Watch a project directory
        watcher.watch(Path("/home/user/myproject"))

        # Start background polling
        watcher.start(interval=5)  # check every 5 seconds

        # Stop when done
        watcher.stop()
    """

    def __init__(self, event_bus=None, ignore_patterns: list[str] | None = None):
        """
        Args:
            event_bus: EventBus instance to emit events to. If None, events are just logged.
            ignore_patterns: Patterns to ignore (e.g., __pycache__, .git, node_modules).
        """
        self.event_bus = event_bus
        self.ignore = ignore_patterns or [
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            ".pytest_cache", "*.pyc", ".mypy_cache", ".tox",
        ]
        self._watched: dict[str, FileSnapshot] = {}  # path_str → last snapshot
        self._running = False
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable] = []

    def watch(self, path: Path) -> None:
        """Start watching a directory."""
        path = Path(path).resolve()
        if path.exists() and path.is_dir():
            snap = FileSnapshot(path)
            snap._take(self.ignore)
            self._watched[str(path)] = snap
            logger.info("Watching: %s", path)

    def unwatch(self, path: Path) -> None:
        """Stop watching a directory."""
        path = Path(path).resolve()
        self._watched.pop(str(path), None)

    def on_change(self, callback: Callable[[str, dict[str, list[str]]], None]) -> None:
        """Register a callback for changes. Called with (path, diff_dict)."""
        self._callbacks.append(callback)

    def start(self, interval: int = 5) -> None:
        """Start background polling."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, args=(interval,), daemon=True)
        self._thread.start()
        logger.info("FileWatcher started (polling every %ds)", interval)

    def stop(self) -> None:
        """Stop background polling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def check_once(self) -> list[tuple[str, dict[str, list[str]]]]:
        """Check all watched directories once. Returns list of (path, diff)."""
        results = []
        for path_str, old_snapshot in list(self._watched.items()):
            path = Path(path_str)
            if not path.exists():
                continue
            new_snapshot = FileSnapshot(path)
            new_snapshot._take(self.ignore)
            diff = new_snapshot.diff(old_snapshot)
            if any(diff.values()):
                results.append((path_str, diff))
                self._watched[path_str] = new_snapshot
                # Emit events
                self._emit_events(path_str, diff)
                # Call callbacks
                for cb in self._callbacks:
                    try:
                        cb(path_str, diff)
                    except Exception as e:
                        logger.warning("Watcher callback error: %s", e)
        return results

    def _loop(self, interval: int) -> None:
        while self._running:
            try:
                self.check_once()
            except Exception as e:
                logger.error("Watcher loop error: %s", e)
            time.sleep(interval)

    def _emit_events(self, path: str, diff: dict[str, list[str]]) -> None:
        """Emit file change events through the EventBus."""
        if not self.event_bus:
            return
        from ultra.core.events import EventType
        for f in diff.get("created", []):
            self.event_bus.emit(EventType.FILE_CREATED,
                              {"file": f, "project": path}, source="watcher")
        for f in diff.get("modified", []):
            self.event_bus.emit(EventType.FILE_CHANGED,
                              {"file": f, "project": path}, source="watcher")
        for f in diff.get("deleted", []):
            self.event_bus.emit(EventType.FILE_DELETED,
                              {"file": f, "project": path}, source="watcher")

    @property
    def watched_count(self) -> int:
        return len(self._watched)

    def snapshot_summary(self) -> dict[str, int]:
        """Number of files in each watched directory."""
        return {path: len(snap.files) for path, snap in self._watched.items()}
