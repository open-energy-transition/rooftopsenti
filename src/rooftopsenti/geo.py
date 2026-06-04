"""CRS and grid helpers.

Conventions used across the pipeline:
- Vectors are stored in WGS84 (EPSG:4326).
- All area thresholds are computed in World Equal-Area Cylindrical (EPSG:6933).
- Raster/model work happens in the per-MGRS-tile UTM CRS (native Sentinel-2 grid).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import geopandas as gpd
import mgrs
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

WGS84 = "EPSG:4326"
EQUAL_AREA = "EPSG:6933"
WEB_MERCATOR = "EPSG:3857"


def area_m2(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Geodesically meaningful polygon areas (m²) via equal-area projection."""
    return gdf.geometry.to_crs(EQUAL_AREA).area


def utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat)."""
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def mgrs_tiles_for_geometry(
    geom: BaseGeometry, sample_step_deg: float = 0.1, buffer_deg: float = 0.05
) -> list[str]:
    """Sentinel-2 MGRS 100 km tile ids intersecting a WGS84 geometry.

    Samples a regular lon/lat grid over the (buffered) geometry and maps each
    point to its MGRS 100 km cell. ``sample_step_deg`` of 0.1° (~11 km) is far
    below the 100 km cell size, so no intersecting tile is missed in practice.
    """
    from shapely import Point, prepare

    m = mgrs.MGRS()
    buffered = geom.buffer(buffer_deg)
    prepare(buffered)
    w, s, e, n = buffered.bounds
    lons = np.arange(w, e + sample_step_deg, sample_step_deg)
    lats = np.arange(s, n + sample_step_deg, sample_step_deg)
    tiles: set[str] = set()
    for lat in lats:
        for lon in lons:
            if buffered.intersects(Point(lon, lat)):
                tiles.add(m.toMGRS(lat, lon, MGRSPrecision=0))
    return sorted(tiles)


def bbox_grid(
    bounds: tuple[float, float, float, float], step_deg: float
) -> list[tuple[float, float, float, float]]:
    """Split a WGS84 bbox (W, S, E, N) into a grid of cell bboxes ≤ step_deg wide."""
    w, s, e, n = bounds
    cells = []
    y = s
    while y < n:
        x = w
        y2 = min(y + step_deg, n)
        while x < e:
            x2 = min(x + step_deg, e)
            cells.append((x, y, x2, y2))
            x = x2
        y = y2
    return cells


def clip_to_geom(gdf: gpd.GeoDataFrame, geom: BaseGeometry) -> gpd.GeoDataFrame:
    """Keep rows intersecting a WGS84 geometry (no geometry modification)."""
    mask = gdf.geometry.intersects(geom)
    return gdf[mask].copy()


def dissolve_all(gdf: gpd.GeoDataFrame) -> BaseGeometry:
    """Union all geometries into one (multi)polygon."""
    return gdf.geometry.union_all()


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set")
    if gdf.crs.to_epsg() != 4326:
        return gdf.to_crs(WGS84)
    return gdf


def filter_min_area(gdf: gpd.GeoDataFrame, min_m2: float) -> gpd.GeoDataFrame:
    """Keep polygons with equal-area area >= min_m2; adds an 'area_m2' column."""
    out = gdf.copy()
    out["area_m2"] = area_m2(out)
    return out[out["area_m2"] >= min_m2].copy()


def spatial_block_id(
    xs: Iterable[float], ys: Iterable[float], block_size_km: float
) -> np.ndarray:
    """Block ids for points given in EPSG:6933 metres (for spatial CV splits)."""
    size_m = block_size_km * 1000.0
    xs = np.asarray(list(xs))
    ys = np.asarray(list(ys))
    bx = np.floor(xs / size_m).astype(np.int64)
    by = np.floor(ys / size_m).astype(np.int64)
    return bx * 1_000_003 + by  # collision-free pairing for realistic extents
