"""Safe terminal command execution.

Borrows the safety model from ARIA v1/v3: a blocklist of destructive
patterns, optional allowlist, hard timeout, and output cap.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field

from ultra.config import Config

BLOCKED_PATTERNS = [
    # Unix destructive
    "rm -rf /", "rm -rf ~", "rm -rf .", "rm -rf /usr", "rm -rf /etc",
    "rm -rf /var", "rm -rf /home", "mkfs", "mkfs.",
    ":(){:|:&};:", "dd if=/dev/zero", "shutdown", "reboot",
    "chmod -R 777 /", "chmod -R 777 /usr", "chown -R /",
    "> /dev/sda", "> /dev/sdb", "pvcreate", "fdisk /dev/sda",
    "curl -sSL | sudo bash", "curl -sL | sudo bash",
    "wget -qO- | sudo bash",
    # Windows destructive
    "format c:", "format d:", "format e:",
    "rmdir /s /q c:", "rmdir /s /q d:",
    "rd /s /q c:\\", "rd /s /q d:\\",
    "del /s /q c:\\*", "del /s /q d:\\*",
    "shutdown /s", "shutdown /r", "shutdown /f",
    "bcdedit", "diskpart", "cipher /w",
    "takeown /f c:\\", "icacls c:\\ /grant",
    "reg delete HKLM", "reg delete HKCU",
    "net user Administrator", "net localgroup Administrators",
    # Cross-platform dangerous
    "git push -f origin master",
]

DANGEROUS_PATTERNS = [  # require confirmation
    "git push", "sudo", "npm publish", "pip install -U", "pip uninstall",
    "rm -rf", "git reset --hard", "dropdb", "DROP TABLE", "kill -9",
    # Windows equivalents
    "rmdir /s", "rd /s", "del /s", "format ",
    "Remove-Item -Recurse -Force", "del /f /q",
    "taskkill /f", "Stop-Process -Force",
]


@dataclass
class CommandResult:
    command: str
    exit_code: int
    output: str
    timed_out: bool = False
    blocked: bool = False
    reason: str = ""
    output_truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.blocked

    def summary(self) -> str:
        if self.blocked:
            return f"[blocked] {self.reason}"
        if self.timed_out:
            return f"[timed out after timeout]"
        status = "ok" if self.ok else f"exit {self.exit_code}"
        return f"{status} · {len(self.output)} chars"


def check_safety(command: str, config: Config) -> str | None:
    """Return a block reason if the command is unsafe, else None."""
    low = command.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in low:
            return f"blocked pattern: {pattern}"
    if any(p in low for p in DANGEROUS_PATTERNS):
        return "requires confirmation (dangerous pattern)"
    return None


class Terminal:
    def __init__(self, config: Config, confirm: callable | None = None):
        self.config = config
        self.confirm = confirm or (lambda msg: True)

    def run(self, command: str, auto_approve: bool = False,
            cwd: str | None = None) -> CommandResult:
        reason = check_safety(command, self.config)
        if reason:
            if "requires confirmation" in reason and auto_approve:
                pass  # dangerous but user opted into auto-approve
            else:
                return CommandResult(command, -1, "", blocked=True, reason=reason)

        try:
            # pipefail so a failing command in a pipeline (e.g. | tail)
            # still reports a non-zero exit code instead of tail's 0
            full = command
            if sys.platform != "win32":
                full = "set -o pipefail; " + command
            proc = subprocess.run(
                full,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout,
                executable="/bin/bash" if sys.platform != "win32" else None,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as e:
            return CommandResult(
                command, -1, (e.stdout or "")[:self.config.output_cap],
                timed_out=True,
            )
        output = (proc.stdout or "") + (proc.stderr or "")
        truncated = len(output) > self.config.output_cap
        return CommandResult(
            command, proc.returncode, output[:self.config.output_cap],
            output_truncated=truncated,
        )
