from __future__ import annotations

import typer

from typing import Annotated
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from localmind.model import TransformersChatModel
from localmind.agent import LocalMindAgent

from localmind.config import (
    DEFAULT_MODEL,
    DEFAULT_WORKSPACE,
    DeviceMode,
    LocalMindConfig,
    PromptFormat,
)


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
    lora_model: Annotated[
        str | None,
        typer.Option(
            "--lora-model",
            help="Local path or Hugging Face repository containing a PEFT LoRA adapter.",
        ),
    ] = None,
    prompt_format: Annotated[
        PromptFormat,
        typer.Option(
            "--prompt-format",
            help="Inference prompt format: native SmolLM chat or Axolotl Alpaca.",
            case_sensitive=False,
        ),
    ] = "chat",
    workspace: Annotated[
        Path,
        typer.Option("--workspace", help="Workspace root for file tools."),
    ] = DEFAULT_WORKSPACE,
    memory: Annotated[
        bool,
        typer.Option(
            "--memory/--no-memory",
            help="Retain normal chat turns for follow-up questions.",
        ),
    ] = False,
    thinking: Annotated[
        bool,
        typer.Option("--thinking/--no-thinking", help="Enable or disable SmolLM3 thinking mode."),
    ] = False,
    max_new_tokens: Annotated[
        int,
        typer.Option(
            "--max-new-tokens",
            help="Maximum generated tokens when thinking mode is disabled.",
            min=1,
        ),
    ] = 1_024,
    thinking_max_new_tokens: Annotated[
        int,
        typer.Option(
            "--thinking-max-new-tokens",
            help="Maximum generated tokens when thinking mode is enabled.",
            min=1,
        ),
    ] = 4_096,
    coding: Annotated[
        bool,
        typer.Option("--coding/--no-coding", help="Enable programming-focused answer guidance."),
    ] = False,
    direct: Annotated[
        bool,
        typer.Option("--direct/--no-direct", help="Use a more direct, less lecture-style answer mode."),
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
        str | None,
        typer.Option(
            "--searxng-url",
            help="SearXNG base URL. Can also be set with LOCALMIND_SEARXNG_URL.",
        ),
    ] = None,
) -> None:
    """Start a session-only LocalMind chat."""
    config = LocalMindConfig(
        model_name=model,
        lora_model=lora_model,
        prompt_format=prompt_format,
        workspace=workspace,
        memory_enabled=memory,
        enable_thinking=thinking,
        max_new_tokens=max_new_tokens,
        thinking_max_new_tokens=thinking_max_new_tokens,
        coding_mode=coding,
        direct_mode=direct,
        device=device,
        search_enabled=search,
        searxng_url=searxng_url,
    )
    config.workspace.mkdir(parents=True, exist_ok=True)
    chat_model = TransformersChatModel(
        config.model_name,
        lora_model=config.lora_model,
        prompt_format=config.prompt_format,
        device=config.device,
        console=console,
        max_new_tokens=config.max_new_tokens,
        thinking_max_new_tokens=config.thinking_max_new_tokens,
    )
    agent = LocalMindAgent(config=config, model=chat_model)

    console.print(
        Panel.fit(
            f"[bold]LocalMind[/bold]\nA small local agent for everyday reasoning.\n\n"
            f"[dim]Model:[/dim] {config.model_name}\n"
            f"[dim]LoRA adapter:[/dim] {config.lora_model or 'none'}\n"
            f"[dim]Prompt format:[/dim] {config.prompt_format}\n"
            f"[dim]Workspace:[/dim] {config.workspace}\n"
            f"[dim]Memory:[/dim] {'enabled' if config.memory_enabled else 'disabled'}\n"
            f"[dim]Requested device:[/dim] {config.device}\n"
            f"[dim]Search:[/dim] {'SearXNG at ' + config.searxng_url if config.search_enabled else 'disabled'}\n"
            f"[dim]Thinking:[/dim] {config.enable_thinking}\n"
            f"[dim]Normal generation limit:[/dim] {config.max_new_tokens} tokens\n"
            f"[dim]Thinking generation limit:[/dim] {config.thinking_max_new_tokens} tokens\n"
            f"[dim]Coding mode:[/dim] {config.coding_mode}\n"
            f"[dim]Direct mode:[/dim] {config.direct_mode}",
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
            console.print("Commands: /help, /clear, /exit, /quit")
            continue
        if user_text.lower() == "/clear":
            agent.reset()
            console.print("[dim]Conversation context cleared.[/dim]")
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
