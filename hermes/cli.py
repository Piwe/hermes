from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from . import agent as agent_mod
from . import ingest as ingest_mod
from .llm import DEFAULT_MODEL

app = typer.Typer(
    add_completion=False,
    help="Hermes — local repo-memory + Q&A on Nous Hermes 3.",
)
console = Console()

DEFAULT_STORE = Path.home() / ".hermes" / "store"


@app.command("ingest")
def ingest_cmd(
    path: Path = typer.Argument(Path("."), exists=True, file_okay=False, help="Path to a git repository."),
    max_commits: int = typer.Option(500, "--max-commits", "-n", help="Maximum commits to ingest."),
    store: Path = typer.Option(DEFAULT_STORE, "--store", help="LanceDB store path."),
):
    """Ingest a git repo's history into local memory (replaces any existing store)."""
    repo = path.resolve()
    console.print(f"[bold]Ingesting[/bold] [cyan]{repo}[/cyan] -> [dim]{store}[/dim]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as bar:
        task_id = bar.add_task("commits", total=None)

        def progress(done: int, total: int, sha: str):
            if bar.tasks[0].total is None:
                bar.update(task_id, total=total)
            bar.update(task_id, completed=done, description=f"commit {sha}")

        n = ingest_mod.ingest(repo, store, max_commits=max_commits, progress=progress)
    console.print(f"[green]done[/green] — {n} commits indexed")


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="A question about the repo."),
    store: Path = typer.Option(DEFAULT_STORE, "--store", help="LanceDB store path."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Hermes model tag, e.g. hermes3:70b."),
):
    """Ask Hermes a question. The agent will call search_memory as needed."""
    if not store.exists():
        console.print(f"[red]No store at {store}[/red] — run `hermes ingest <repo>` first.")
        raise typer.Exit(code=1)
    with console.status("[dim]thinking…[/dim]"):
        answer = agent_mod.ask(question, store, model=model)
    console.print(Markdown(answer))


if __name__ == "__main__":
    app()
