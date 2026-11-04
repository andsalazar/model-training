"""Command line entry point.

Run everything through Doppler so secrets are injected rather than stored:

    doppler run --project model-training --config dev -- wsp doctor
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from wsp.config import MissingSecret, Settings
from wsp.editors.registry import EDITORS, build_editor
from wsp.runpod import NotOwnedError, RunPodClient

app = typer.Typer(add_completion=False, help="Weight-Space Programming experiments.")
console = Console()


@app.command()
def doctor() -> None:
    """Check that every configured secret is present and usable."""
    table = Table(title="Environment")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    try:
        settings = Settings.from_env()
    except MissingSecret as exc:
        console.print(f"[red]missing secret:[/red] {exc}")
        raise typer.Exit(1) from exc

    table.add_row("OPENROUTER_API_KEY", "[green]present[/green]", "editor access")
    table.add_row(
        "HF_TOKEN",
        "[green]present[/green]" if settings.hf_token else "[yellow]absent[/yellow]",
        "gated model downloads",
    )
    table.add_row(
        "WANDB_API_KEY",
        "[green]present[/green]" if settings.wandb_api_key else "[yellow]absent[/yellow]",
        "run tracking",
    )
    table.add_row(
        "RUNPOD_MODEL_TRAINING_API_KEY",
        "[green]present[/green]" if settings.runpod_api_key else "[yellow]absent[/yellow]",
        "GPU rental",
    )
    console.print(table)


@app.command("editors")
def list_editors() -> None:
    """List candidate editors and their cost per million tokens."""
    table = Table(title="Editor candidates")
    for column in ("alias", "model id", "$/M in", "$/M out", "notes"):
        table.add_column(column)

    for spec in EDITORS.values():
        table.add_row(
            spec.alias,
            spec.model_id,
            f"{spec.usd_per_mtok_in:.2f}",
            f"{spec.usd_per_mtok_out:.2f}",
            spec.notes,
        )
    console.print(table)


@app.command()
def ping(
    editor: str = typer.Option("deepseek", help="Editor alias to test."),
    prompt: str = typer.Option("Reply with exactly: ok", help="Prompt to send."),
) -> None:
    """Send one prompt to an editor and report latency and cost."""
    settings = Settings.from_env()
    spec = EDITORS.get(editor)
    if spec is None:
        console.print(f"[red]unknown editor[/red] {editor!r}; known: {sorted(EDITORS)}")
        raise typer.Exit(1)

    async def _run() -> None:
        async with build_editor(editor, settings.openrouter_api_key) as client:
            response = await client.complete("You are a terse assistant.", prompt, max_tokens=64)
        cost = spec.estimate_cost(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        console.print(f"[bold]{spec.model_id}[/bold]")
        console.print(f"  reply:  {response.text.strip()[:200]!r}")
        console.print(f"  tokens: {response.usage.prompt_tokens} in / "
                      f"{response.usage.completion_tokens} out")
        console.print(f"  cost:   ${cost:.6f}")

    asyncio.run(_run())


@app.command("pods")
def pods() -> None:
    """List pods, marking which ones this project owns."""
    settings = Settings.from_env()
    if not settings.runpod_api_key:
        console.print("[red]RUNPOD_MODEL_TRAINING_API_KEY not set[/red]")
        raise typer.Exit(1)

    with RunPodClient(settings.runpod_api_key) as client:
        owned = client.owned_ids()
        rows = client.list_pods()

    table = Table(title="RunPod pods")
    for column in ("id", "name", "status", "$/hr", "owned"):
        table.add_column(column)

    for pod in rows:
        pod_id = pod.get("id", "")
        is_owned = pod_id in owned
        table.add_row(
            pod_id,
            pod.get("name", ""),
            pod.get("desiredStatus", ""),
            f"{pod.get('costPerHr', 0):.2f}",
            "[green]yes[/green]" if is_owned else "[yellow]no[/yellow]",
        )
    console.print(table)
    if any(p.get("id") not in owned for p in rows):
        console.print(
            "[dim]Pods marked 'no' were not created by this project and are "
            "protected from modification.[/dim]"
        )


@app.command("stop")
def stop(pod_id: str = typer.Argument(..., help="Pod id to stop.")) -> None:
    """Stop a pod this project owns. Refuses pods it did not create."""
    settings = Settings.from_env()
    if not settings.runpod_api_key:
        console.print("[red]RUNPOD_MODEL_TRAINING_API_KEY not set[/red]")
        raise typer.Exit(1)

    with RunPodClient(settings.runpod_api_key) as client:
        try:
            client.stop_pod(pod_id)
        except NotOwnedError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
    console.print(f"[green]stopped[/green] {pod_id}")


@app.command("inspect")
def inspect(
    checkpoint: Path = typer.Argument(..., help="Path to a .safetensors file."),
    out: Path = typer.Option(Path("runs/summary.json"), help="Where to write the summary."),
) -> None:
    """Summarize a checkpoint's tensors without loading the weights."""
    from wsp.weights.store import WeightStore

    with WeightStore(checkpoint) as store:
        path = store.write_summary(out)
        summary = store.summary()

    console.print(f"tensors: {summary['num_tensors']:,}")
    console.print(f"params:  {summary['num_params']:,}")
    console.print(f"written: {path}")


@app.command("slice")
def slice_tensor(
    checkpoint: Path = typer.Argument(..., help="Path to a .safetensors file."),
    tensor: str = typer.Option(..., help="Tensor name."),
    cols: str = typer.Option(None, help="Column range as start:stop."),
    rows: str = typer.Option(None, help="Row range as start:stop."),
    out: Path = typer.Option(None, help="Write the rendered text here."),
) -> None:
    """Render a tensor slice as text for an editor."""
    from wsp.weights.slices import TensorSlice, to_text
    from wsp.weights.store import WeightStore

    def parse_range(value: str | None) -> tuple[int, int] | None:
        if not value:
            return None
        start, stop = value.split(":")
        return int(start), int(stop)

    with WeightStore(checkpoint) as store:
        array = store.get(tensor)

    target = TensorSlice(name=tensor, rows=parse_range(rows), cols=parse_range(cols))
    text = to_text(target.apply(array))

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        console.print(f"written: {out} ({len(text):,} chars)")
    else:
        console.print(text)


@app.command("summary")
def summary(path: Path = typer.Argument(Path("runs/summary.json"))) -> None:
    """Print a saved checkpoint summary."""
    data = json.loads(path.read_text(encoding="utf-8"))
    console.print(f"{data['num_tensors']:,} tensors, {data['num_params']:,} params")


if __name__ == "__main__":
    app()
