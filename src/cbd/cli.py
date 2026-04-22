from __future__ import annotations

import logging
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from cbd import __version__
from cbd.data.aois import normalize_aoi
from cbd.data.labels import normalize_labels
from cbd.logging_utils import configure_logging
from cbd.manifests import (
    format_validation_error,
    load_aoi_manifest,
    load_data_manifest,
    summarize_aoi_manifest,
    summarize_data_manifest,
)

app = typer.Typer(help="CBD baseline CLI")
console = Console()
logger = logging.getLogger("cbd")


@app.callback()
def main() -> None:
    configure_logging()


@app.command()
def version() -> None:
    console.print(f"cbd version: {__version__}")


@app.command()
def health() -> None:
    console.print("[green]OK[/green] CBD CLI is available.")


@app.command("validate-manifests")
def validate_manifests(
    data_manifest_path: Path = typer.Argument(..., help="Path to data_manifest.yaml"),
    aoi_manifest_path: Path = typer.Argument(..., help="Path to aoi_manifest.yaml"),
) -> None:
    try:
        data_manifest = load_data_manifest(data_manifest_path)
        aoi_manifest = load_aoi_manifest(aoi_manifest_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except ValidationError as exc:
        console.print("[red]Manifest validation failed.[/red]")
        console.print(format_validation_error(exc))
        raise typer.Exit(code=3) from exc
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]Unexpected error: {exc}[/red]")
        raise typer.Exit(code=4) from exc

    data_summary = summarize_data_manifest(data_manifest)
    aoi_summary = summarize_aoi_manifest(aoi_manifest)

    table = Table(title="Manifest validation summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Project", str(data_summary["project"]))
    table.add_row("Enabled sources", str(data_summary["enabled_sources"]))
    table.add_row("Source IDs", ", ".join(data_summary["source_ids"]))
    table.add_row("Enabled AOIs", str(aoi_summary["enabled_aois"]))
    table.add_row("Split counts", str(aoi_summary["split_counts"]))

    console.print(table)
    console.print("[green]Validation successful.[/green]")


@app.command("normalize-labels")
def normalize_labels_command(
    input_path: Path = typer.Argument(..., help="Input vector file"),
    output_path: Path = typer.Argument(..., help="Output normalized file"),
    target_crs: str = typer.Option("EPSG:4326", help="Target CRS"),
    source_id: str = typer.Option("carolina_bays_labels", help="Source identifier"),
    split: str = typer.Option("train", help="Split name"),
) -> None:
    try:
        out = normalize_labels(
            input_path=input_path,
            output_path=output_path,
            target_crs=target_crs,
            source_id=source_id,
            split=split,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=3) from exc

    console.print(f"[green]Normalized labels written to {out}[/green]")


@app.command("normalize-aoi")
def normalize_aoi_command(
    input_path: Path = typer.Argument(..., help="Input AOI vector file"),
    output_path: Path = typer.Argument(..., help="Output normalized AOI file"),
    target_crs: str = typer.Option("EPSG:4326", help="Target CRS"),
    aoi_id: str = typer.Option("aoi", help="AOI identifier prefix"),
    split: str = typer.Option("train", help="Split name"),
) -> None:
    try:
        out = normalize_aoi(
            input_path=input_path,
            output_path=output_path,
            target_crs=target_crs,
            aoi_id=aoi_id,
            split=split,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=3) from exc

    console.print(f"[green]Normalized AOI written to {out}[/green]")


if __name__ == "__main__":
    app()
