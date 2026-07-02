"""Artifact layout, freshness tracking, and vector IO.

Every pipeline stage writes its outputs under ``data/<region>/`` together with a
``<artifact>.meta.json`` recording the relevant config slice and input artifact
state. A stage is skipped when its artifact is *fresh*: the meta matches the
current config and none of its inputs changed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd

from .config import Config


class ArtifactStore:
    """Resolves artifact paths for one region and handles freshness checks."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.region_dir

    # ----------------------------------------------------------- paths ----
    @property
    def aoi_boundary(self) -> Path:
        return self.root / "aoi" / "boundary.parquet"

    @property
    def mgrs_tiles(self) -> Path:
        return self.root / "aoi" / "mgrs_tiles.json"

    @property
    def osm_solar(self) -> Path:
        return self.root / "osm" / "solar.parquet"

    @property
    def osm_buildings(self) -> Path:
        return self.root / "osm" / "buildings.parquet"

    @property
    def osm_labels(self) -> Path:
        return self.root / "osm" / "labels.parquet"

    @property
    def hard_negatives(self) -> Path:
        return self.root / "osm" / "hard_negatives.parquet"

    @property
    def overpass_cache(self) -> Path:
        return self.root / "osm" / "overpass_cache"

    @property
    def composites_dir(self) -> Path:
        return self.root / "composites"

    def composite_tif(self, tile: str, range_idx: int = 0) -> Path:
        return self.composites_dir / tile / f"composite_{range_idx}.tif"

    @property
    def stac_catalog(self) -> Path:
        return self.composites_dir / "catalog.json"

    @property
    def buildings(self) -> Path:
        return self.root / "buildings" / "buildings_filtered.parquet"

    @property
    def chips_dir(self) -> Path:
        return self.root / "chips"

    @property
    def chips_index(self) -> Path:
        return self.chips_dir / "index.parquet"

    @property
    def chips_h5(self) -> Path:
        return self.chips_dir / "chips.h5"

    @property
    def clean_negatives_report(self) -> Path:
        return self.chips_dir / "clean_negatives_report.json"

    @property
    def screen_head(self) -> Path:
        return self.root / "screen" / "head.pt"

    @property
    def screen_candidates(self) -> Path:
        return self.root / "screen" / "candidates.parquet"

    def model_dir(self, run_id: str) -> Path:
        return self.root / "models" / run_id

    def prediction_tif(self, run_id: str, tile: str) -> Path:
        return self.root / "predictions" / run_id / f"{tile}_solar_prob.tif"

    def output(self, run_id: str, name: str) -> Path:
        return self.root / "outputs" / run_id / name

    # ------------------------------------------------------- freshness ----
    @staticmethod
    def _meta_path(artifact: Path) -> Path:
        return artifact.with_name(artifact.name + ".meta.json")

    @staticmethod
    def _file_state(path: Path) -> str:
        st = path.stat()
        return f"{st.st_size}:{st.st_mtime_ns}"

    # static so freshness can also be checked for artifacts of *other* regions
    # (e.g. train_regions chip packs), where no Config/store is at hand
    @staticmethod
    def write_meta(artifact: Path, config_slice: dict, inputs: list[Path] | None = None) -> None:
        meta = {
            "config": _jsonify(config_slice),
            "inputs": {
                str(p): ArtifactStore._file_state(p) for p in (inputs or []) if p.exists()
            },
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        ArtifactStore._meta_path(artifact).write_text(json.dumps(meta, indent=2, sort_keys=True))

    @staticmethod
    def is_fresh(artifact: Path, config_slice: dict, inputs: list[Path] | None = None) -> bool:
        meta_path = ArtifactStore._meta_path(artifact)
        if not artifact.exists() or not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        if meta.get("config") != _jsonify(config_slice):
            return False
        recorded = meta.get("inputs", {})
        current = {str(p): ArtifactStore._file_state(p) for p in (inputs or []) if p.exists()}
        # Inputs that were recorded but have since been deleted (e.g. composites pruned
        # after chipping) are skipped — only inputs that still exist are compared.
        return {k: v for k, v in recorded.items() if k in current} == current


def _jsonify(obj: dict) -> dict:
    """Round-trip through JSON so tuples/Paths compare equal to loaded meta."""
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


# ------------------------------------------------------------- vector IO ----
def write_gdf(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    return path


def read_gdf(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_parquet(path)


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def read_json(path: Path):
    return json.loads(path.read_text())
