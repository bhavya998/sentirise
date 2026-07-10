"""CLI - train, serve, classify from terminal."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from sentirise.classifier import SentimentClassifier
from sentirise.config import ADAPTER_DIR

app = typer.Typer(help="QLoRA sentiment classifier on Qwen3-0.6B", no_args_is_help=True)
console = Console()


@app.command()
def train(
    epochs: int = typer.Option(3, help="Number of training epochs"),
    max_train: int = typer.Option(2000, help="Max training samples"),
    max_eval: int = typer.Option(400, help="Max eval samples"),
    output: str = typer.Option(str(ADAPTER_DIR), help="Output directory for adapter"),
) -> None:
    """Fine-tune Qwen3-0.6B with QLoRA on IMDB sentiment data."""
    from sentirise.train import train as run_train

    run_train(epochs=epochs, max_train=max_train, max_eval=max_eval, output_dir=output)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev only)"),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("sentirise.api:app", host=host, port=port, reload=reload)


@app.command()
def classify(
    text: str = typer.Argument(help="Text to classify"),
) -> None:
    """Classify sentiment of the given text."""
    clf = SentimentClassifier()
    result = clf.classify(text)

    color = "green" if result["label"] == "positive" else "red"
    console.print(f"\n[{color} bold]{result['label'].upper()}[/{color} bold]")
    console.print(f"confidence: {result['confidence']:.0%}")
    console.print(f"raw: {result['raw_output']}")


@app.command()
def batch(
    file_path: str = typer.Argument(help="JSON file with list of {\"text\": ...} objects"),
) -> None:
    """Classify multiple texts from a JSON file."""
    from pathlib import Path

    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    clf = SentimentClassifier()

    for i, item in enumerate(data):
        result = clf.classify(item["text"])
        console.print(f"[{i+1}] {result['label']:10s} ({result['confidence']:.0%}) | {item['text'][:80]}")


if __name__ == "__main__":
    app()
