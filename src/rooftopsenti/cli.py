"""rooftopsenti CLI — one sub-command per pipeline stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from .config import Config, load_config
from .io_artifacts import ArtifactStore

app = typer.Typer(help="Large-scale rooftop solar detection from Sentinel-2", no_args_is_help=True)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Region config YAML")]
TilesOpt = Annotated[
    str | None, typer.Option("--tiles", help="Comma-separated MGRS tile subset")
]
RunIdOpt = Annotated[str | None, typer.Option("--run-id", help="Model run id")]
ModelCkptOpt = Annotated[
    str | None,
    typer.Option("--model-ckpt", help="External model checkpoint for transfer inference"),
]


def _setup(config_path: Path) -> tuple[Config, ArtifactStore]:
    cfg = load_config(config_path)
    logger.info("Region: {} (run id {})", cfg.region, cfg.run_id())
    return cfg, ArtifactStore(cfg)


def _tile_list(tiles: str | None) -> list[str] | None:
    return [t.strip() for t in tiles.split(",")] if tiles else None


@app.command()
def aoi(config: ConfigOpt):
    """a) Resolve region boundary + MGRS tile worklist."""
    from .stages import aoi as stage

    cfg, store = _setup(config)
    stage.run(cfg, store)


@app.command()
def labels(config: ConfigOpt):
    """b) Extract OSM rooftop solar training labels via Overpass."""
    from .stages import osm_labels as stage

    cfg, store = _setup(config)
    stage.run(cfg, store)


@app.command()
def composite(config: ConfigOpt, tiles: TilesOpt = None):
    """c) Build cloud-free Sentinel-2 composites (STAC) per MGRS tile."""
    from .stages import composite as stage

    cfg, store = _setup(config)
    stage.run(cfg, store, only_tiles=_tile_list(tiles))


@app.command()
def gba(config: ConfigOpt):
    """f-prep) Fetch large-building inference ROIs (GlobalBuildingAtlas/Overture)."""
    from .stages import gba as stage

    cfg, store = _setup(config)
    stage.run(cfg, store)


@app.command()
def chips(config: ConfigOpt):
    """d) Generate training chips + spatial train/val/test split."""
    from .stages import chips as stage

    cfg, store = _setup(config)
    stage.run(cfg, store)


@app.command()
def train(config: ConfigOpt, run_id: RunIdOpt = None):
    """e) Train the segmentation model."""
    from .stages import train as stage

    cfg, store = _setup(config)
    stage.run(cfg, store, run_id=run_id)


@app.command()
def infer(
    config: ConfigOpt,
    run_id: RunIdOpt = None,
    tiles: TilesOpt = None,
    model_ckpt: ModelCkptOpt = None,
):
    """f) Run inference on large-building ROI windows.

    Use --model-ckpt to apply a model trained in another region (transfer
    inference), e.g. the NL model on Pakistan composites.
    """
    from .stages import infer as stage

    cfg, store = _setup(config)
    stage.run(
        cfg, store, run_id=run_id, only_tiles=_tile_list(tiles), model_ckpt=model_ckpt
    )


@app.command()
def postprocess(config: ConfigOpt, run_id: RunIdOpt = None):
    """g) Polygonize, aggregate per building, build missing-in-OSM list."""
    from .stages import postprocess as stage

    cfg, store = _setup(config)
    stage.run(cfg, store, run_id=run_id)


@app.command()
def report(config: ConfigOpt, run_id: RunIdOpt = None):
    """Build the HTML map report."""
    from .report import build_map

    cfg, store = _setup(config)
    build_map(cfg, store, run_id or cfg.run_id())


@app.command()
def validate(config: ConfigOpt, run_id: RunIdOpt = None):
    """Held-out chip metrics + per-building precision/recall vs OSM."""
    from .io_artifacts import read_gdf

    cfg, store = _setup(config)
    rid = run_id or cfg.run_id()

    metrics_path = store.model_dir(rid) / "metrics.json"
    if metrics_path.exists():
        typer.echo("Test-split chip metrics:")
        typer.echo(metrics_path.read_text())

    stats_path = store.output(rid, "building_solar_stats.parquet")
    if stats_path.exists():
        stats = read_gdf(stats_path)
        tp = int((stats["has_solar"] & stats["osm_has_solar"]).sum())
        fp = int((stats["has_solar"] & ~stats["osm_has_solar"]).sum())
        fn = int((~stats["has_solar"] & stats["osm_has_solar"]).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        typer.echo(
            json.dumps(
                {
                    "buildings": len(stats),
                    "building_level_vs_osm": {
                        "tp": tp,
                        "fp_or_missing_in_osm": fp,
                        "fn": fn,
                        "precision_lower_bound": round(precision, 3),
                        "recall": round(recall, 3),
                    },
                    "note": "OSM is positive-unlabeled: 'fp' may be genuinely "
                    "missing OSM mappings — precision is a lower bound.",
                },
                indent=2,
            )
        )


@app.command(name="run-all")
def run_all(config: ConfigOpt, run_id: RunIdOpt = None):
    """Run the full pipeline a->g (each stage skips if fresh)."""
    from .report import build_map
    from .stages import aoi as s_aoi
    from .stages import chips as s_chips
    from .stages import composite as s_composite
    from .stages import gba as s_gba
    from .stages import infer as s_infer
    from .stages import osm_labels as s_labels
    from .stages import postprocess as s_post
    from .stages import train as s_train

    cfg, store = _setup(config)
    s_aoi.run(cfg, store)
    s_labels.run(cfg, store)
    s_composite.run(cfg, store)
    s_gba.run(cfg, store)
    s_chips.run(cfg, store)
    rid = s_train.run(cfg, store, run_id=run_id)
    s_infer.run(cfg, store, run_id=rid)
    s_post.run(cfg, store, run_id=rid)
    build_map(cfg, store, rid)
    typer.echo(f"Pipeline complete. Outputs: {store.output(rid, '')}")


if __name__ == "__main__":
    app()
