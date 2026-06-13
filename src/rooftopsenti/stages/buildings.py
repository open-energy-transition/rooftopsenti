"""Stage f-prep) Large-building inference ROIs from Overture Maps buildings.

Overture Maps redistributes building footprints as monthly GeoParquet releases
on S3 (``theme=buildings/type=building``). They are queried by bounding box with
DuckDB — no bulk download — and area-prefiltered server-side so country-scale
AOIs transfer only the large buildings that define the inference ROIs.

Output: ``buildings/buildings_filtered.parquet`` — WGS84 polygons with
``area_m2``, all >= ``buildings.building_area_min_m2``.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from loguru import logger

from ..config import Config
from ..geo import clip_to_geom, filter_min_area
from ..io_artifacts import ArtifactStore, write_gdf
from . import aoi


def _fetch_one(source: str, boundary, cfg: Config) -> gpd.GeoDataFrame:
    fetch_min = cfg.buildings.fetch_area_min_m2
    if source == "overture":
        from .. import overture

        logger.info(
            "Querying Overture buildings {} for bbox {}", cfg.overture.release, boundary.bounds
        )
        return overture.buildings_in_bbox(
            cfg.overture.release, boundary.bounds, min_area_m2=fetch_min
        )
    if source == "vida_open_buildings":
        from .. import vida_buildings

        return vida_buildings.buildings_in_bbox(cfg, boundary.bounds)
    raise ValueError(f"Unknown building source {source!r}")


def dedupe_sources(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Union footprints from several sources, dropping cross-source duplicates.

    Sources are merged in priority order (earliest wins): a footprint from a
    later source is dropped when a representative interior point falls inside any
    already-kept footprint. Point-in-polygon is CRS-agnostic, so this runs in
    WGS84 (``representative_point`` is used over ``centroid`` — it is guaranteed
    inside the polygon and avoids the geographic-CRS centroid warning).
    """
    frames = [f for f in frames if not f.empty]
    if not frames:
        return gpd.GeoDataFrame({"id": [], "source": []}, geometry=[], crs="EPSG:4326")
    kept = frames[0]
    for extra in frames[1:]:
        points = gpd.GeoDataFrame(geometry=extra.geometry.representative_point(), crs=extra.crs)
        j = gpd.sjoin(points, kept[["geometry"]], how="left", predicate="within")
        is_dup = j.groupby(level=0)["index_right"].apply(lambda v: v.notna().any())
        fresh = extra[~is_dup.reindex(extra.index, fill_value=False).to_numpy()]
        kept = gpd.GeoDataFrame(pd.concat([kept, fresh], ignore_index=True), crs=kept.crs)
    return kept.reset_index(drop=True)


def _fetch_buildings(boundary, cfg: Config) -> gpd.GeoDataFrame:
    frames = [_fetch_one(src, boundary, cfg) for src in cfg.buildings.sources]
    merged = dedupe_sources(frames)
    if len(cfg.buildings.sources) > 1:
        logger.info(
            "Merged {} source(s) -> {} building(s) after dedupe ({})",
            len(cfg.buildings.sources),
            len(merged),
            merged["source"].value_counts().to_dict() if not merged.empty else {},
        )
    return merged


def run(cfg: Config, store: ArtifactStore) -> None:
    cfg_slice = {
        "buildings": cfg.buildings.model_dump(mode="json"),
        "overture": cfg.overture.model_dump(mode="json"),
        "aoi": cfg.aoi.model_dump(mode="json"),
    }
    if store.is_fresh(store.buildings, cfg_slice, inputs=[store.aoi_boundary]):
        logger.info("Buildings fresh — skipping")
        return

    boundary = aoi.load_boundary(store)
    buildings = _fetch_buildings(boundary, cfg)
    logger.info("Fetched {} building footprint(s)", len(buildings))

    if not buildings.empty:
        buildings = clip_to_geom(buildings, boundary)
        # keep down to the smallest area any consumer needs (ROIs may go below
        # building_area_min_m2); per-consumer thresholds are applied downstream
        buildings = filter_min_area(buildings, cfg.buildings.fetch_area_min_m2)
        buildings = buildings.reset_index(drop=True)
        buildings["building_id"] = buildings.index
    logger.info(
        "Kept {} building(s) >= {} m² (ROI min {}, negative min {})",
        len(buildings),
        cfg.buildings.fetch_area_min_m2,
        cfg.buildings.effective_roi_area_min_m2,
        cfg.buildings.building_area_min_m2,
    )
    write_gdf(buildings, store.buildings)
    store.write_meta(store.buildings, cfg_slice, inputs=[store.aoi_boundary])
