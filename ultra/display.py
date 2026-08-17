"""Terminal display helpers (Rich)."""
from __future__ import annotations

from rich.console import Console

console = Console()


def banner(title: str, subtitle: str = "") -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    if subtitle:
        console.print(f"  [dim]{subtitle}[/dim]")
    console.print()


def step(n: int, text: str) -> None:
    console.print(f"  [bold cyan]{n}[/bold cyan] [white]{text}[/white]")


def info(text: str) -> None:
    console.print(f"  [dim]•[/dim] {text}")


def ok(text: str) -> None:
    console.print(f"  [green]✓[/green] {text}")


def warn(text: str) -> None:
    console.print(f"  [yellow]⚠[/yellow] {text}")


def error(text: str) -> None:
    console.print(f"  [red]✘[/red] {text}")


def label(text: str, value: str) -> None:
    console.print(f"  [dim]{text}[/dim] [cyan]{value}[/cyan]")


def code_block(text: str, max_len: int = 4000) -> None:
    from rich.syntax import Syntax
    if len(text) > max_len:
        text = text[:max_len] + "\n… (truncated)"
    console.print(Syntax(text, "markdown", word_wrap=True, theme="monokai"))


def markdown(text: str) -> None:
    from rich.markdown import Markdown
    console.print(Markdown(text))
