from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from cbd import __version__
from cbd.data.aois import normalize_aoi
from cbd.data.labels import normalize_labels
from cbd.data.terrain import (
    TerrainCandidatesError,
    TerrainDerivativesError,
    TerrainPreprocessingError,
    TerrainResolutionError,
    TerrainReviewError,
    derive_terrain_features,
    generate_terrain_candidates,
    load_terrain_candidates_summary,
    load_terrain_derivatives_summary,
    load_terrain_preprocessing_summary,
    load_terrain_resolution_summary,
    prepare_terrain_review_artifacts,
    preprocess_terrain_inputs,
    resolve_terrain_inputs,
    write_terrain_candidates_summary,
    write_terrain_derivatives_summary,
    write_terrain_preprocessing_summary,
    write_terrain_resolution_summary,
    write_terrain_review_summary,
)
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
    data_manifest_path: Annotated[Path, typer.Argument(help="Path to data_manifest.yaml")],
    aoi_manifest_path: Annotated[Path, typer.Argument(help="Path to aoi_manifest.yaml")],
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
    input_path: Annotated[Path, typer.Argument(help="Input vector file")],
    output_path: Annotated[Path, typer.Argument(help="Output normalized file")],
    target_crs: Annotated[str, typer.Option(help="Target CRS")] = "EPSG:4326",
    source_id: Annotated[str, typer.Option(help="Source identifier")] = "carolina_bays_labels",
    split: Annotated[str, typer.Option(help="Split name")] = "train",
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
    input_path: Annotated[Path, typer.Argument(help="Input AOI vector file")],
    output_path: Annotated[Path, typer.Argument(help="Output normalized AOI file")],
    target_crs: Annotated[str, typer.Option(help="Target CRS")] = "EPSG:4326",
    aoi_id: Annotated[str, typer.Option(help="AOI identifier prefix")] = "aoi",
    split: Annotated[str, typer.Option(help="Split name")] = "train",
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


@app.command("resolve-terrain-inputs")
def resolve_terrain_inputs_command(
    data_manifest_path: Annotated[Path, typer.Argument(help="Path to data_manifest.yaml")],
    aoi_manifest_path: Annotated[Path, typer.Argument(help="Path to aoi_manifest.yaml")],
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Project root used to resolve manifest-relative local paths.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output-path",
            help=(
                "Output JSON artifact path. Defaults to "
                "outputs/interim/terrain/terrain_input_resolution.json under the project root."
            ),
        ),
    ] = None,
) -> None:
    try:
        data_manifest = load_data_manifest(data_manifest_path)
        aoi_manifest = load_aoi_manifest(aoi_manifest_path)
        summary = resolve_terrain_inputs(
            data_manifest,
            aoi_manifest,
            data_manifest_path=data_manifest_path,
            aoi_manifest_path=aoi_manifest_path,
            project_root=project_root,
        )
        artifact_path = output_path
        if artifact_path is None:
            artifact_path = (
                Path(summary.project_root)
                / "outputs"
                / "interim"
                / "terrain"
                / "terrain_input_resolution.json"
            )
        out = write_terrain_resolution_summary(summary, artifact_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except ValidationError as exc:
        console.print("[red]Manifest validation failed.[/red]")
        console.print(format_validation_error(exc))
        raise typer.Exit(code=3) from exc
    except TerrainResolutionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=4) from exc
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=5) from exc

    table = Table(title="Terrain input resolution summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Project", summary.project_name)
    table.add_row("Enabled AOIs", str(summary.enabled_aoi_count))
    table.add_row("Terrain sources", str(summary.enabled_terrain_source_count))
    table.add_row(
        "Source IDs",
        ", ".join(source.id for source in summary.terrain_sources),
    )
    table.add_row(
        "Raster counts",
        ", ".join(f"{source.id}={source.raster_count}" for source in summary.terrain_sources),
    )
    table.add_row("Artifact", str(out))

    console.print(table)
    console.print("[green]Terrain inputs resolved successfully.[/green]")


@app.command("preprocess-terrain")
def preprocess_terrain_command(
    terrain_resolution_path: Annotated[
        Path,
        typer.Argument(help="Path to terrain_input_resolution.json"),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            help=(
                "Optional preprocessing output root. Defaults to "
                "outputs/interim/terrain under the project root."
            ),
        ),
    ] = None,
) -> None:
    try:
        terrain_summary = load_terrain_resolution_summary(terrain_resolution_path)
        preprocessing_summary = preprocess_terrain_inputs(
            terrain_summary,
            output_root=output_root,
        )
        preprocessing_summary = preprocessing_summary.model_copy(
            update={
                "terrain_resolution_artifact": str(
                    Path(terrain_resolution_path).expanduser().resolve()
                )
            }
        )
        out = write_terrain_preprocessing_summary(
            preprocessing_summary,
            preprocessing_summary.output_summary_path,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except TerrainPreprocessingError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=3) from exc
    except ValidationError as exc:
        console.print("[red]Terrain preprocessing artifact validation failed.[/red]")
        console.print(format_validation_error(exc))
        raise typer.Exit(code=4) from exc

    written = sum(1 for record in preprocessing_summary.records if record.status == "written")
    skipped = sum(1 for record in preprocessing_summary.records if record.status == "skipped")

    table = Table(title="Terrain preprocessing summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Project", preprocessing_summary.project_name)
    table.add_row("Enabled AOIs", str(preprocessing_summary.total_aois_processed))
    table.add_row(
        "Terrain sources",
        str(preprocessing_summary.total_terrain_sources_processed),
    )
    table.add_row("Outputs written", str(written))
    table.add_row("Pairs skipped", str(skipped))
    table.add_row("Artifact", str(out))

    console.print(table)
    console.print("[green]Terrain preprocessing completed successfully.[/green]")


@app.command("derive-terrain-features")
def derive_terrain_features_command(
    terrain_preprocessing_path: Annotated[
        Path,
        typer.Argument(help="Path to terrain_preprocessing_summary.json"),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            help=(
                "Optional derivatives output root. Defaults to "
                "outputs/interim/terrain under the project root."
            ),
        ),
    ] = None,
    relief_window_size: Annotated[
        int,
        typer.Option(
            "--relief-window-size",
            help="Odd square window size in pixels for local relief.",
        ),
    ] = 5,
) -> None:
    try:
        preprocessing_summary = load_terrain_preprocessing_summary(terrain_preprocessing_path)
        derivatives_summary = derive_terrain_features(
            preprocessing_summary,
            output_root=output_root,
            relief_window_size=relief_window_size,
        )
        derivatives_summary = derivatives_summary.model_copy(
            update={
                "terrain_preprocessing_artifact": str(
                    Path(terrain_preprocessing_path).expanduser().resolve()
                )
            }
        )
        out = write_terrain_derivatives_summary(
            derivatives_summary,
            derivatives_summary.output_summary_path,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except TerrainDerivativesError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=3) from exc
    except ValidationError as exc:
        console.print("[red]Terrain derivatives artifact validation failed.[/red]")
        console.print(format_validation_error(exc))
        raise typer.Exit(code=4) from exc

    table = Table(title="Terrain derivatives summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Project", derivatives_summary.project_name)
    table.add_row(
        "Input rasters",
        str(derivatives_summary.total_input_rasters_processed),
    )
    table.add_row(
        "Derivative rasters",
        str(derivatives_summary.total_derivative_rasters_written),
    )
    table.add_row(
        "Relief window",
        str(derivatives_summary.relief_window_size),
    )
    table.add_row("Artifact", str(out))

    console.print(table)
    console.print("[green]Terrain derivatives completed successfully.[/green]")


@app.command("generate-terrain-candidates")
def generate_terrain_candidates_command(
    terrain_derivatives_path: Annotated[
        Path,
        typer.Argument(help="Path to terrain_derivatives_summary.json"),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            help=(
                "Optional candidate output root. Defaults to "
                "outputs/interim/terrain under the project root."
            ),
        ),
    ] = None,
    relief_threshold: Annotated[
        float,
        typer.Option(
            "--relief-threshold",
            help="Minimum local relief value used to build candidate masks.",
        ),
    ] = 1.0,
    min_pixels: Annotated[
        int,
        typer.Option(
            "--min-pixels",
            help="Minimum connected-component size in pixels.",
        ),
    ] = 4,
) -> None:
    try:
        derivatives_summary = load_terrain_derivatives_summary(terrain_derivatives_path)
        candidates_summary = generate_terrain_candidates(
            derivatives_summary,
            output_root=output_root,
            relief_threshold=relief_threshold,
            min_pixels=min_pixels,
        )
        candidates_summary = candidates_summary.model_copy(
            update={
                "terrain_derivatives_artifact": str(
                    Path(terrain_derivatives_path).expanduser().resolve()
                )
            }
        )
        out = write_terrain_candidates_summary(
            candidates_summary,
            candidates_summary.output_summary_path,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except TerrainCandidatesError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=3) from exc
    except ValidationError as exc:
        console.print("[red]Terrain candidates artifact validation failed.[/red]")
        console.print(format_validation_error(exc))
        raise typer.Exit(code=4) from exc

    table = Table(title="Terrain candidate summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Project", candidates_summary.project_name)
    table.add_row("Input groups", str(candidates_summary.total_input_groups_processed))
    table.add_row(
        "Candidate vectors",
        str(candidates_summary.total_candidate_vectors_written),
    )
    table.add_row("Candidates", str(candidates_summary.total_candidates))
    table.add_row("Relief threshold", str(candidates_summary.relief_threshold))
    table.add_row("Min pixels", str(candidates_summary.min_pixels))
    table.add_row("Artifact", str(out))

    console.print(table)
    console.print("[green]Terrain candidate generation completed successfully.[/green]")


@app.command("prepare-terrain-review")
def prepare_terrain_review_command(
    terrain_candidates_path: Annotated[
        Path,
        typer.Argument(help="Path to terrain_candidates_summary.json"),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            help=(
                "Optional review artifact output root. Defaults to "
                "outputs/interim/terrain under the project root."
            ),
        ),
    ] = None,
) -> None:
    try:
        candidates_summary = load_terrain_candidates_summary(terrain_candidates_path)
        review_summary = prepare_terrain_review_artifacts(
            candidates_summary,
            output_root=output_root,
        )
        review_summary = review_summary.model_copy(
            update={
                "terrain_candidates_artifact": str(
                    Path(terrain_candidates_path).expanduser().resolve()
                )
            }
        )
        out = write_terrain_review_summary(
            review_summary,
            review_summary.output_summary_path,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except TerrainReviewError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=3) from exc
    except ValidationError as exc:
        console.print("[red]Terrain review artifact validation failed.[/red]")
        console.print(format_validation_error(exc))
        raise typer.Exit(code=4) from exc

    table = Table(title="Terrain review artifact summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Project", review_summary.project_name)
    table.add_row("Candidate rows", str(review_summary.total_candidate_rows))
    table.add_row("Review tables", str(review_summary.total_review_tables_written))
    table.add_row("Review overlays", str(review_summary.total_review_overlays_written))
    table.add_row("Artifact", str(out))

    console.print(table)
    console.print("[green]Terrain review artifact preparation completed successfully.[/green]")


if __name__ == "__main__":
    app()
