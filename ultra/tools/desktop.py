"""Desktop Awareness — screen capture and OCR.

Gives ARIA eyes on the user's screen. Captures screenshots and
extracts text via OCR, allowing ARIA to:
- See error messages and offer help
- Read forms and assist with filling them
- Monitor application state
- Detect what the user is working on

Uses platform-native tools:
- Linux: xdotool + scrot/import (ImageMagick) + tesseract
- macOS: screencapture + tesseract
- Windows: (not yet supported — use sidecar)
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("aria.desktop")


@dataclass
class ScreenCapture:
    """Result of a screen capture + OCR."""
    image_path: str = ""
    text: str = ""  # OCR extracted text
    width: int = 0
    height: int = 0
    error: str | None = None
    window_title: str = ""
    active_app: str = ""

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.image_path)


class DesktopAwareness:
    """Capture screen content and extract text via OCR.

    Usage:
        desktop = DesktopAwareness()
        capture = desktop.capture()
        if capture.success:
            print(f"Screen text: {capture.text[:500]}")
            print(f"Active app: {capture.active_app}")
    """

    def __init__(self, ocr_lang: str = "eng"):
        self.ocr_lang = ocr_lang
        self._tesseract_available: bool | None = None

    def _check_tesseract(self) -> bool:
        if self._tesseract_available is not None:
            return self._tesseract_available
        try:
            result = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True, timeout=5
            )
            self._tesseract_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._tesseract_available = False
        return self._tesseract_available

    def capture(self, region: str | None = None,
                output_path: str | None = None) -> ScreenCapture:
        """Capture the screen (or a region) and extract text.

        Args:
            region: Optional region to capture, e.g. "100,200,800,600" (x,y,w,h)
            output_path: Where to save the screenshot. Auto-generated if None.
        """
        system = platform.system()
        if system == "Linux":
            return self._capture_linux(region, output_path)
        elif system == "Darwin":
            return self._capture_macos(region, output_path)
        else:
            return ScreenCapture(error=f"Desktop awareness not supported on {system}")

    def get_active_window(self) -> str:
        """Get the title of the currently active window."""
        system = platform.system()
        try:
            if system == "Linux":
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=5
                )
                return result.stdout.strip()
            elif system == "Darwin":
                result = subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to get name of first application process whose frontmost is true'],
                    capture_output=True, text=True, timeout=5
                )
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    def get_screen_size(self) -> tuple[int, int]:
        """Get screen resolution."""
        system = platform.system()
        try:
            if system == "Linux":
                result = subprocess.run(
                    ["xdotool", "getdisplaygeometry"],
                    capture_output=True, text=True, timeout=5
                )
                parts = result.stdout.strip().split()
                return int(parts[0]), int(parts[1])
            elif system == "Darwin":
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=5
                )
                # Parse resolution from output
                for line in result.stdout.split("\n"):
                    if "Resolution" in line:
                        # e.g., "Resolution: 2560 x 1600"
                        parts = line.split(":")[1].strip().split("x")
                        return int(parts[0].strip()), int(parts[1].strip())
        except Exception:
            pass
        return 1920, 1080  # default

    def _capture_linux(self, region: str | None,
                       output_path: str | None) -> ScreenCapture:
        """Linux screen capture using scrot or import (ImageMagick)."""
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(), "aria_screenshot.png")

        try:
            # Get active window title
            window_title = self.get_active_window()

            # Capture screen
            if region:
                x, y, w, h = region.split(",")
                subprocess.run(
                    ["import", "-window", "root", "-crop", f"{w}x{h}+{x}+{y}",
                     output_path],
                    capture_output=True, timeout=10
                )
            else:
                # Try scrot first (faster), fall back to import
                try:
                    subprocess.run(
                        ["scrot", "-o", output_path],
                        capture_output=True, timeout=10
                    )
                except FileNotFoundError:
                    subprocess.run(
                        ["import", "-window", "root", output_path],
                        capture_output=True, timeout=10
                    )

            if not Path(output_path).exists():
                return ScreenCapture(error="Screenshot capture failed")

            # OCR
            text = self._ocr(output_path)

            return ScreenCapture(
                image_path=output_path,
                text=text,
                window_title=window_title,
            )

        except Exception as e:
            return ScreenCapture(error=str(e))

    def _capture_macos(self, region: str | None,
                       output_path: str | None) -> ScreenCapture:
        """macOS screen capture using screencapture."""
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(), "aria_screenshot.png")

        try:
            window_title = self.get_active_window()

            cmd = ["screencapture", "-x"]  # -x = no sound
            if region:
                cmd.extend(["-R", region])
            cmd.append(output_path)

            subprocess.run(cmd, capture_output=True, timeout=10)

            if not Path(output_path).exists():
                return ScreenCapture(error="Screenshot capture failed")

            text = self._ocr(output_path)

            return ScreenCapture(
                image_path=output_path,
                text=text,
                window_title=window_title,
            )

        except Exception as e:
            return ScreenCapture(error=str(e))

    def _ocr(self, image_path: str) -> str:
        """Extract text from an image using tesseract."""
        if not self._check_tesseract():
            return "[OCR not available — install tesseract: sudo apt install tesseract-ocr]"

        try:
            result = subprocess.run(
                ["tesseract", image_path, "stdout", "-l", self.ocr_lang],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout.strip()
        except Exception as e:
            return f"[OCR failed: {e}]"

    def watch_application(self, app_name: str, interval: int = 10,
                          callback=None) -> None:
        """Watch a specific application for changes.

        Args:
            app_name: Part of the window title to match
            interval: Seconds between captures
            callback: Called with ScreenCapture when content changes
        """
        import time
        last_text = ""
        while True:
            capture = self.capture()
            if capture.success and app_name.lower() in capture.window_title.lower():
                if capture.text != last_text:
                    last_text = capture.text
                    if callback:
                        callback(capture)
            time.sleep(interval)
