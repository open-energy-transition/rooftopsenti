"""Stage f-prep) Large-building inference ROIs from GlobalBuildingAtlas (or Overture).

Backends:
- ``huggingface`` — GlobalBuildingAtlas footprints. The dataset is split across
  two HF repos: ``GBA.ODbLPolygon`` (bulk, OSM/Microsoft-derived, ODbL) and the
  ``Polygon/`` folder of ``GBA.LoD1`` (additional non-ODbL detections). Tiles
  are 5°x5° GeoJSON in EPSG:3857, named ``{w}_{n}_{e}_{s}.geojson`` under a
  continent folder. Tiles for densely built regions are multi-GB — they are
  downloaded once into the HF cache, then bbox-filtered while reading.
- ``overture`` — Overture Maps buildings GeoParquet on S3 queried with DuckDB;
  no bulk download, ideal for small AOIs and smoke tests.

Output: ``gba/buildings_filtered.parquet`` — WGS84 polygons with ``area_m2``,
all >= ``gba.building_area_min_m2``.
"""

from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
from loguru import logger

from ..config import Config
from ..geo import WEB_MERCATOR, WGS84, clip_to_geom, filter_min_area
from ..io_artifacts import ArtifactStore, write_gdf
from . import aoi

CONTINENTS = [
    "europe",
    "asiaeast",
    "asiawest",
    "africa",
    "northamerica",
    "southamerica",
    "oceania",
]


# ------------------------------------------------------------------- GBA ----
def _gba_tile_names(bounds: tuple[float, float, float, float]) -> list[str]:
    """5°x5° GBA tile basenames covering a WGS84 bbox.

    Naming is ``{w}_{n}_{e}_{s}`` with zero-padded 3-digit lon / 2-digit lat,
    e.g. ``e005_n55_e010_n50`` covers lon 5..10, lat 50..55.
    """

    def lon_token(v: int) -> str:
        return f"{'e' if v >= 0 else 'w'}{abs(v):03d}"

    def lat_token(v: int) -> str:
        return f"{'n' if v >= 0 else 's'}{abs(v):02d}"

    w, s, e, n = bounds
    names = []
    for lon in range(int(math.floor(w / 5) * 5), int(math.ceil(e / 5) * 5), 5):
        for lat in range(int(math.floor(s / 5) * 5), int(math.ceil(n / 5) * 5), 5):
            names.append(
                f"{lon_token(lon)}_{lat_token(lat + 5)}_{lon_token(lon + 5)}_{lat_token(lat)}"
            )
    return names


def _hf_download_tile(repo: str, subpath_candidates: list[str]) -> str | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for sub in subpath_candidates:
        try:
            return hf_hub_download(repo_id=repo, filename=sub, repo_type="dataset")
        except EntryNotFoundError:
            continue
    return None


def _fetch_gba(boundary, cfg: Config) -> gpd.GeoDataFrame:
    import pyogrio

    bounds = boundary.bounds
    tile_names = _gba_tile_names(bounds)
    logger.info("GBA tiles covering AOI: {}", tile_names)

    # buildings straddle the bbox edge — pad the read window slightly
    pad = 0.01
    bbox_wgs84 = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
    bbox_3857 = (
        gpd.GeoSeries.from_xy([bbox_wgs84[0], bbox_wgs84[2]], [bbox_wgs84[1], bbox_wgs84[3]], crs=WGS84)
        .to_crs(WEB_MERCATOR)
        .total_bounds
    )

    frames = []
    for name in tile_names:
        found = False
        for continent in CONTINENTS:
            sources = [
                (cfg.gba.hf_repo_odbl, f"{continent}/{name}.geojson"),
                (cfg.gba.hf_repo_extra, f"Polygon/{continent}/{name}.geojson"),
            ]
            for repo, sub in sources:
                local = _hf_download_tile(repo, [sub])
                if local is None:
                    continue
                found = True
                logger.info("Reading {} (bbox-filtered)", sub)
                gdf = pyogrio.read_dataframe(local, bbox=tuple(bbox_3857))
                # GBA files are EPSG:3857 even when the header claims otherwise
                gdf = gdf.set_crs(WEB_MERCATOR, allow_override=True)
                if not gdf.empty:
                    frames.append(gdf.to_crs(WGS84))
        if not found:
            logger.warning("GBA tile {} not found in any continent folder", name)

    if not frames:
        return gpd.GeoDataFrame({"source": []}, geometry=[], crs=WGS84)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=WGS84)


# -------------------------------------------------------------- Overture ----
def _fetch_overture(boundary, cfg: Config) -> gpd.GeoDataFrame:
    from .. import overture

    logger.info(
        "Querying Overture buildings {} for bbox {}", cfg.overture.release, boundary.bounds
    )
    return overture.buildings_in_bbox(
        cfg.overture.release, boundary.bounds, min_area_m2=cfg.gba.building_area_min_m2
    )


# ------------------------------------------------------------------ stage ----
def run(cfg: Config, store: ArtifactStore) -> None:
    cfg_slice = {"gba": cfg.gba.model_dump(mode="json"), "aoi": cfg.aoi.model_dump(mode="json")}
    if store.is_fresh(store.gba_buildings, cfg_slice, inputs=[store.aoi_boundary]):
        logger.info("GBA buildings fresh — skipping")
        return

    boundary = aoi.load_boundary(store)
    if cfg.gba.source == "overture":
        buildings = _fetch_overture(boundary, cfg)
    else:
        buildings = _fetch_gba(boundary, cfg)
    logger.info("Fetched {} building footprint(s)", len(buildings))

    if not buildings.empty:
        buildings = clip_to_geom(buildings, boundary)
        buildings = filter_min_area(buildings, cfg.gba.building_area_min_m2)
        buildings = buildings.reset_index(drop=True)
        buildings["building_id"] = buildings.index
    logger.info(
        "Kept {} large building(s) >= {} m²", len(buildings), cfg.gba.building_area_min_m2
    )
    write_gdf(buildings, store.gba_buildings)
    store.write_meta(store.gba_buildings, cfg_slice, inputs=[store.aoi_boundary])
