"""Stage a) AOI resolution: region config -> boundary polygon + MGRS tile worklist."""

from __future__ import annotations

import io
import os

import geopandas as gpd
import httpx
from loguru import logger
from shapely.geometry import box

from ..config import Config
from ..geo import WGS84, dissolve_all, mgrs_tiles_for_geometry
from ..io_artifacts import ArtifactStore, read_gdf, read_json, write_gdf, write_json

GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/ADM{level}/"


def _fetch_geoboundaries(cfg: Config) -> gpd.GeoDataFrame:
    api_url = GEOBOUNDARIES_API.format(iso3=cfg.aoi.iso3, level=cfg.aoi.admin_level)
    logger.info("Fetching geoBoundaries metadata: {}", api_url)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        meta = client.get(api_url).raise_for_status().json()
        download_url = meta["gjDownloadURL"]
        logger.info("Downloading boundary GeoJSON: {}", download_url)
        geojson_bytes = client.get(download_url).raise_for_status().content
    os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")
    gdf = gpd.read_file(io.BytesIO(geojson_bytes))
    if cfg.aoi.admin_name:
        match = gdf[gdf["shapeName"].str.casefold() == cfg.aoi.admin_name.casefold()]
        if match.empty:
            available = sorted(gdf["shapeName"].unique())
            raise ValueError(
                f"admin_name {cfg.aoi.admin_name!r} not found at ADM{cfg.aoi.admin_level} "
                f"for {cfg.aoi.iso3}. Available: {available}"
            )
        gdf = match
    return gdf.to_crs(WGS84) if gdf.crs else gdf.set_crs(WGS84)


def _build_boundary(cfg: Config) -> gpd.GeoDataFrame:
    if cfg.aoi.source == "bbox":
        return gpd.GeoDataFrame(
            {"name": [cfg.region]}, geometry=[box(*cfg.aoi.bbox)], crs=WGS84
        )
    if cfg.aoi.source == "geoboundaries":
        gdf = _fetch_geoboundaries(cfg)
        return gpd.GeoDataFrame(
            {"name": [cfg.region]}, geometry=[dissolve_all(gdf)], crs=WGS84
        )
    raise NotImplementedError(
        f"aoi.source={cfg.aoi.source!r} is not implemented yet; "
        "use 'geoboundaries' or 'bbox'"
    )


def run(cfg: Config, store: ArtifactStore) -> None:
    cfg_slice = cfg.aoi.model_dump(mode="json")
    if store.is_fresh(store.aoi_boundary, cfg_slice) and store.is_fresh(
        store.mgrs_tiles, cfg_slice, inputs=[store.aoi_boundary]
    ):
        logger.info("AOI artifacts fresh — skipping")
        return

    boundary = _build_boundary(cfg)
    write_gdf(boundary, store.aoi_boundary)
    store.write_meta(store.aoi_boundary, cfg_slice)

    geom = boundary.geometry.iloc[0]
    tiles = mgrs_tiles_for_geometry(geom)
    if cfg.imagery.tiles:
        tiles = sorted(set(tiles) & set(cfg.imagery.tiles))
    write_json(tiles, store.mgrs_tiles)
    store.write_meta(store.mgrs_tiles, cfg_slice, inputs=[store.aoi_boundary])
    logger.info("AOI resolved: {} MGRS tiles: {}", len(tiles), tiles)


def load_boundary(store: ArtifactStore):
    """Boundary geometry (WGS84) for downstream stages."""
    return read_gdf(store.aoi_boundary).geometry.iloc[0]


def load_tiles(store: ArtifactStore) -> list[str]:
    return read_json(store.mgrs_tiles)
