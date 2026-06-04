"""Stage b) OSM training labels: large rooftop solar polygons.

Positives are OSM solar generator polygons that (a) meet the minimum area for
Sentinel-2 detectability and (b) are rooftop installations — either tagged
``location=roof`` (or variants) or spatially intersecting a building polygon.

Backends (``osm.source``):
- ``overture`` (default) — OSM data redistributed by Overture Maps as
  bbox-queryable GeoParquet; original tags preserved in ``source_tags``.
  No rate limits; snapshot lags live OSM by up to ~a month.
- ``overpass`` — live OSM via the Overpass API (chunked, cached, mirrored).
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from loguru import logger

from .. import overture
from ..config import Config
from ..geo import WGS84, bbox_grid, clip_to_geom, filter_min_area
from ..io_artifacts import ArtifactStore, write_gdf
from ..overpass import OverpassClient, buildings_query, elements_to_polygons, solar_query
from . import aoi

ROOF_VALUES = {"roof", "rooftop"}
BUILDING_BBOX_PAD_DEG = 0.001  # ~100 m padding around solar for the building fetch
BBOXES_PER_BUILDING_QUERY = 40


def _fetch_solar(client: OverpassClient, boundary, cfg: Config) -> gpd.GeoDataFrame:
    cells = bbox_grid(boundary.bounds, cfg.osm.chunk_deg)
    cells = [c for c in cells if boundary.intersects(_cell_box(c))]
    logger.info("Querying Overpass for solar polygons in {} grid cells", len(cells))
    frames = []
    for i, cell in enumerate(cells, 1):
        payload = client.query(solar_query(cell, cfg.osm.timeout_s))
        gdf = elements_to_polygons(payload.get("elements", []))
        if not gdf.empty:
            frames.append(gdf)
        logger.debug("cell {}/{}: {} polygon(s)", i, len(cells), len(gdf))
    if not frames:
        return elements_to_polygons([])
    solar = pd.concat(frames, ignore_index=True)
    solar = gpd.GeoDataFrame(solar, crs=WGS84).drop_duplicates(subset=["osm_type", "osm_id"])
    return clip_to_geom(solar.reset_index(drop=True), boundary)


def _fetch_buildings_near(
    client: OverpassClient, solar: gpd.GeoDataFrame, cfg: Config
) -> gpd.GeoDataFrame:
    """Buildings in small padded bboxes around each solar polygon (batched)."""
    bboxes = []
    for geom in solar.geometry:
        w, s, e, n = geom.bounds
        p = BUILDING_BBOX_PAD_DEG
        bboxes.append((w - p, s - p, e + p, n + p))
    frames = []
    for i in range(0, len(bboxes), BBOXES_PER_BUILDING_QUERY):
        batch = bboxes[i : i + BBOXES_PER_BUILDING_QUERY]
        payload = client.query(buildings_query(batch, cfg.osm.timeout_s))
        gdf = elements_to_polygons(payload.get("elements", []))
        if not gdf.empty:
            frames.append(gdf)
    if not frames:
        return elements_to_polygons([])
    buildings = pd.concat(frames, ignore_index=True)
    buildings = gpd.GeoDataFrame(buildings, crs=WGS84)
    return buildings.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)


def build_labels(
    solar: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, cfg: Config
) -> gpd.GeoDataFrame:
    """Filter solar polygons to large rooftop installations."""
    large = filter_min_area(solar, cfg.osm.solar_area_min_m2)
    if large.empty:
        return large.assign(has_roof_tag=[], on_building=[])

    has_roof_tag = large["location_tag"].isin(ROOF_VALUES).fillna(False)

    if buildings.empty:
        on_building = pd.Series(False, index=large.index)
    else:
        joined = gpd.sjoin(
            large[["geometry"]], buildings[["geometry"]], how="left", predicate="intersects"
        )
        on_building = joined.groupby(level=0)["index_right"].apply(lambda s: s.notna().any())
        on_building = on_building.reindex(large.index, fill_value=False)

    large = large.assign(has_roof_tag=has_roof_tag, on_building=on_building)
    rule = cfg.osm.rooftop_rule
    if rule == "roof_tag_only":
        keep = large["has_roof_tag"]
    elif rule == "intersect_only":
        keep = large["on_building"]
    else:  # intersect_or_roof_tag
        keep = large["has_roof_tag"] | large["on_building"]
    return large[keep].reset_index(drop=True)


def _cell_box(cell):
    from shapely.geometry import box

    return box(*cell)


def _fetch_via_overture(
    boundary, cfg: Config
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    solar = overture.solar_generators(cfg.overture.release, boundary.bounds)
    solar = clip_to_geom(solar, boundary).reset_index(drop=True)
    large_solar = filter_min_area(solar, cfg.osm.solar_area_min_m2)
    buildings = overture.buildings_intersecting(cfg.overture.release, large_solar)
    return solar, buildings


def _fetch_via_overpass(
    boundary, cfg: Config, store: ArtifactStore
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    client = OverpassClient(cfg.osm.overpass_url, cfg.osm.timeout_s, store.overpass_cache)
    solar = _fetch_solar(client, boundary, cfg)
    # Only fetch buildings near *large* solar — that's all the rooftop rule needs
    large_solar = filter_min_area(solar, cfg.osm.solar_area_min_m2)
    buildings = (
        _fetch_buildings_near(client, large_solar, cfg)
        if not large_solar.empty
        else elements_to_polygons([])
    )
    return solar, buildings


def run(cfg: Config, store: ArtifactStore) -> None:
    cfg_slice = {
        "osm": cfg.osm.model_dump(mode="json"),
        "overture": cfg.overture.model_dump(mode="json"),
        "aoi": cfg.aoi.model_dump(mode="json"),
    }
    if store.is_fresh(store.osm_labels, cfg_slice, inputs=[store.aoi_boundary]):
        logger.info("OSM labels fresh — skipping")
        return

    boundary = aoi.load_boundary(store)
    if cfg.osm.source == "overture":
        solar, buildings = _fetch_via_overture(boundary, cfg)
    else:
        solar, buildings = _fetch_via_overpass(boundary, cfg, store)

    logger.info("Fetched {} solar polygon(s) in AOI", len(solar))
    write_gdf(solar, store.osm_solar)
    logger.info("Fetched {} building polygon(s) intersecting large solar", len(buildings))
    write_gdf(buildings, store.osm_buildings)

    labels = build_labels(solar, buildings, cfg)
    logger.info(
        "Built {} rooftop solar label(s) (≥{} m²)", len(labels), cfg.osm.solar_area_min_m2
    )
    write_gdf(labels, store.osm_labels)
    store.write_meta(store.osm_labels, cfg_slice, inputs=[store.aoi_boundary])
