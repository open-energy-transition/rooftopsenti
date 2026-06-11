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
from loguru import logger

from ..config import Config
from ..geo import clip_to_geom, filter_min_area
from ..io_artifacts import ArtifactStore, write_gdf
from . import aoi


def _fetch_buildings(boundary, cfg: Config) -> gpd.GeoDataFrame:
    from .. import overture

    logger.info(
        "Querying Overture buildings {} for bbox {}", cfg.overture.release, boundary.bounds
    )
    return overture.buildings_in_bbox(
        cfg.overture.release, boundary.bounds, min_area_m2=cfg.buildings.building_area_min_m2
    )


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
        buildings = filter_min_area(buildings, cfg.buildings.building_area_min_m2)
        buildings = buildings.reset_index(drop=True)
        buildings["building_id"] = buildings.index
    logger.info(
        "Kept {} large building(s) >= {} m²", len(buildings), cfg.buildings.building_area_min_m2
    )
    write_gdf(buildings, store.buildings)
    store.write_meta(store.buildings, cfg_slice, inputs=[store.aoi_boundary])
