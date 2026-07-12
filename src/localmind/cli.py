from __future__ import annotations

import typer

from typing import Annotated
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from localmind.agent import LocalMindAgent
from localmind.config import (
    DEFAULT_MODEL,
    DEFAULT_SEARXNG_URL,
    DEFAULT_WORKSPACE,
    DeviceMode,
    LocalMindConfig,
)
from localmind.model import TransformersChatModel


app = typer.Typer(
    no_args_is_help=True
)
console = Console()


@app.callback()
def main() -> None:
    """LocalMind command line interface."""


@app.command()
def chat(
    model: Annotated[
        str,
        typer.Option("--model", help="Hugging Face model name to load."),
    ] = DEFAULT_MODEL,
    workspace: Annotated[
        Path,
        typer.Option("--workspace", help="Workspace root for file tools."),
    ] = DEFAULT_WORKSPACE,
    thinking: Annotated[
        bool,
        typer.Option("--thinking/--no-thinking", help="Enable or disable SmolLM3 thinking mode."),
    ] = False,
    device: Annotated[
        DeviceMode,
        typer.Option(
            "--device",
            help="Device mode for model loading: auto, cpu, or cuda.",
            case_sensitive=False,
        ),
    ] = "auto",
    search: Annotated[
        bool,
        typer.Option("--search/--no-search", help="Enable the SearXNG-backed web_search tool."),
    ] = False,
    searxng_url: Annotated[
        str,
        typer.Option(
            "--searxng-url",
            help="SearXNG base URL. Can also be set with LOCALMIND_SEARXNG_URL.",
        ),
    ] = DEFAULT_SEARXNG_URL,
) -> None:
    """Start a session-only LocalMind chat."""
    config = LocalMindConfig(
        model_name=model,
        workspace=workspace,
        enable_thinking=thinking,
        device=device,
        search_enabled=search,
        searxng_url=searxng_url,
    )
    config.workspace.mkdir(parents=True, exist_ok=True)
    chat_model = TransformersChatModel(config.model_name, device=config.device, console=console)
    agent = LocalMindAgent(config=config, model=chat_model)

    console.print(
        Panel.fit(
            f"[bold]LocalMind[/bold]\nA small local agent for everyday reasoning.\n\n"
            f"[dim]Model:[/dim] {config.model_name}\n"
            f"[dim]Workspace:[/dim] {config.workspace}\n"
            f"[dim]Requested device:[/dim] {config.device}\n"
            f"[dim]Search:[/dim] {'SearXNG at ' + config.searxng_url if config.search_enabled else 'disabled'}\n"
            f"[dim]Thinking:[/dim] {config.enable_thinking}",
            border_style="cyan",
        )
    )
    console.print("[dim]The model loads on the first message. Type /help for commands, /exit to quit.[/dim]")

    while True:
        user_text = Prompt.ask("\n[bold cyan]you[/bold cyan]").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            console.print("[dim]Goodbye.[/dim]")
            raise typer.Exit()
        if user_text.lower() == "/help":
            console.print("Commands: /help, /exit, /quit")
            continue

        try:
            answer = agent.ask(user_text)
        except RuntimeError as exc:
            console.print(
                Panel.fit(
                    str(exc),
                    title="LocalMind could not start the model",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1) from exc
        console.print(f"\n[bold green]LocalMind[/bold green]\n{answer}")


if __name__ == "__main__":
    app()
