---
description: Build Python CLI tools with Click, argparse, or typer
triggers: cli, command line, terminal tool, argparse, click, typer
---

# Python CLI Tool Skill

## Recommended Stack
- **Typer** (modern, type-hint based, auto-completion)
- **Rich** (beautiful terminal output, tables, progress bars)
- **Click** (if Typer is too new for the project)

## Template: Typer CLI
```python
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Tool description")
console = Console()

@app.command()
def main(name: str, verbose: bool = False):
    """Command description."""
    if verbose:
        console.print(f"[green]Processing {name}...[/green]")
    console.print(f"Done: {name}")

@app.command()
def list_items():
    """List all items."""
    table = Table(title="Items")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_row("Item 1", "Active")
    console.print(table)

if __name__ == "__main__":
    app()
```

## Packaging
- Use `pyproject.toml` with `[project.scripts]` for entry points
- Include `--help`, `--version`, `--verbose` flags
- Add shell completion: `typer --shell-completion`
