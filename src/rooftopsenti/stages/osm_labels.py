"""Stage b) OSM training labels: large rooftop solar arrays.

Positives are rooftop solar arrays that (a) meet the minimum area for Sentinel-2
detectability and (b) are rooftop installations — either tagged ``location=roof``
(or variants) or spatially intersecting a building polygon. Solar generator/plant
polygons are first dissolved into one array per building (or per touching cluster
when on no building), so the area threshold applies to the whole array: a roof
mapped as many small panels still yields one detectable label, and multiple
generator polygons on one roof collapse to a single label.

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


TAG_COLS = ("osm_type", "osm_id", "location_tag", "tags")


def _label_groups(solar: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series]:
    """Assign each solar polygon a group key + an on-building flag.

    Polygons on the same building footprint share a ``b<idx>`` key (grouped by
    spatial intersection, not via any OSM relation). Polygons on no building are
    clustered into ``c<idx>`` keys by dissolving only those that touch/overlap.
    """
    if buildings.empty:
        bldg_of = pd.Series(pd.NA, index=solar.index)
    else:
        j = gpd.sjoin(solar[["geometry"]], buildings[["geometry"]], how="left", predicate="intersects")
        bldg_of = j.groupby(level=0)["index_right"].first().reindex(solar.index)
    on_building = bldg_of.notna()

    group = pd.Series(index=solar.index, dtype=object)
    group[on_building] = "b" + bldg_of[on_building].astype("int64").astype(str)

    off = solar.loc[~on_building]
    if not off.empty:
        clusters = gpd.GeoDataFrame(
            geometry=[off.geometry.union_all()], crs=solar.crs
        ).explode(ignore_index=True)
        cj = gpd.sjoin(off[["geometry"]], clusters[["geometry"]], how="left", predicate="intersects")
        cluster_of = cj.groupby(level=0)["index_right"].first().reindex(off.index)
        group[off.index] = "c" + cluster_of.astype("int64").astype(str)
    return group, on_building


def build_labels(
    solar: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, cfg: Config
) -> gpd.GeoDataFrame:
    """One label per rooftop array of large rooftop solar installations.

    Solar polygons are dissolved per building (or per touching/overlapping
    cluster when on no building) *before* the area threshold is applied, so a
    roof mapped as many small panels yields a single label sized by their union
    rather than several sub-threshold ones — and multiple generator polygons on
    one roof collapse to one label instead of oversampling it.
    """
    if solar.empty:
        return solar.assign(has_roof_tag=[], on_building=[])

    solar = solar.copy()
    solar["_roof"] = solar["location_tag"].isin(ROOF_VALUES).fillna(False)
    group, on_building = _label_groups(solar, buildings)
    solar["_group"] = group
    solar["_on_building"] = on_building.values

    present_tags = [c for c in TAG_COLS if c in solar.columns]
    cols = ["_group", "_roof", "_on_building", "geometry", *present_tags]
    agg = solar[cols].dissolve(
        by="_group",
        aggfunc={"_roof": "any", "_on_building": "first", **{c: "first" for c in present_tags}},
    ).reset_index(drop=True)
    agg = agg.rename(columns={"_roof": "has_roof_tag", "_on_building": "on_building"})

    agg = filter_min_area(agg, cfg.osm.solar_area_min_m2)
    rule = cfg.osm.rooftop_rule
    if rule == "roof_tag_only":
        keep = agg["has_roof_tag"]
    elif rule == "intersect_only":
        keep = agg["on_building"]
    else:  # intersect_or_roof_tag
        keep = agg["has_roof_tag"] | agg["on_building"]
    return agg[keep].reset_index(drop=True)


def _cell_box(cell):
    from shapely.geometry import box

    return box(*cell)


def _fetch_via_overture(
    boundary, cfg: Config
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    solar = overture.solar_generators(cfg.overture.release, boundary.bounds)
    solar = clip_to_geom(solar, boundary).reset_index(drop=True)
    # buildings near *all* solar (incl. small) so subdivided arrays can be
    # grouped per building and thresholded on their union (see build_labels)
    buildings = overture.buildings_intersecting(cfg.overture.release, solar)
    return solar, buildings


def _fetch_via_overpass(
    boundary, cfg: Config, store: ArtifactStore
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    client = OverpassClient(cfg.osm.overpass_url, cfg.osm.timeout_s, store.overpass_cache)
    solar = _fetch_solar(client, boundary, cfg)
    # buildings near *all* solar (incl. small) so subdivided arrays can be
    # grouped per building and thresholded on their union (see build_labels)
    buildings = (
        _fetch_buildings_near(client, solar, cfg)
        if not solar.empty
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
        "Built {} rooftop solar array label(s) (≥{} m² per array) from {} solar polygon(s)",
        len(labels),
        cfg.osm.solar_area_min_m2,
        len(solar),
    )
    write_gdf(labels, store.osm_labels)
    store.write_meta(store.osm_labels, cfg_slice, inputs=[store.aoi_boundary])
