"""Chunked Overpass API client with retries, on-disk caching, and polygon assembly.

Country-scale extraction works by splitting the AOI bbox into a grid of cells
(``osm.chunk_deg``) and issuing one Overpass query per cell, so individual
requests stay well under server limits and a partial run resumes from cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import geopandas as gpd
import httpx
from loguru import logger
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .geo import WGS84

RETRY_STATUS = {429, 502, 503, 504}
MAX_RETRIES = 6


class OverpassClient:
    def __init__(self, url: str, timeout_s: int, cache_dir: Path):
        self.url = url
        self.timeout_s = timeout_s
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ql: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(ql.encode()).hexdigest()[:24]}.json"

    def query(self, ql: str) -> dict:
        """POST an Overpass QL query; cached on disk, retried with backoff."""
        cache = self._cache_path(ql)
        if cache.exists():
            return json.loads(cache.read_text())

        delay = 5.0
        for attempt in range(MAX_RETRIES):
            try:
                resp = httpx.post(self.url, data={"data": ql}, timeout=self.timeout_s)
                if resp.status_code in RETRY_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable status {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                payload = resp.json()
                cache.write_text(json.dumps(payload))
                return payload
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                logger.warning(
                    "Overpass request failed ({}), retry {}/{} in {:.0f}s",
                    exc,
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")


# ------------------------------------------------------------ QL builders ----
def _bbox_ql(bbox: tuple[float, float, float, float]) -> str:
    """Overpass bbox order is (S, W, N, E); ours is (W, S, E, N)."""
    w, s, e, n = bbox
    return f"({s},{w},{n},{e})"


def solar_query(bbox: tuple[float, float, float, float], timeout_s: int) -> str:
    bb = _bbox_ql(bbox)
    return (
        f"[out:json][timeout:{timeout_s}];\n"
        f'(\n'
        f'  way["power"="generator"]["generator:source"="solar"]{bb};\n'
        f'  relation["power"="generator"]["generator:source"="solar"]{bb};\n'
        f');\n'
        f"out body geom;"
    )


def buildings_query(
    bboxes: list[tuple[float, float, float, float]], timeout_s: int
) -> str:
    parts = "".join(
        f'  way["building"]{_bbox_ql(bb)};\n  relation["building"]{_bbox_ql(bb)};\n'
        for bb in bboxes
    )
    return f"[out:json][timeout:{timeout_s}];\n(\n{parts});\nout body geom;"


# -------------------------------------------------------- response parsing ----
def _ring_from_geometry(geometry: list[dict]) -> Polygon | None:
    coords = [(pt["lon"], pt["lat"]) for pt in geometry]
    if len(coords) < 4 or coords[0] != coords[-1]:
        return None
    try:
        poly = Polygon(coords)
        return poly if poly.is_valid and poly.area > 0 else poly.buffer(0)
    except (ValueError, TypeError):
        return None


def _relation_polygon(element: dict) -> Polygon | None:
    outers, inners = [], []
    for member in element.get("members", []):
        if member.get("type") != "way" or "geometry" not in member:
            continue
        ring = _ring_from_geometry(member["geometry"])
        if ring is None or ring.is_empty:
            continue
        (outers if member.get("role") != "inner" else inners).append(ring)
    if not outers:
        return None
    shell = unary_union(outers)
    if inners:
        shell = shell.difference(unary_union(inners))
    return shell if not shell.is_empty else None


def elements_to_polygons(elements: list[dict]) -> gpd.GeoDataFrame:
    """Assemble Overpass `out geom` elements into a polygon GeoDataFrame.

    Nodes and open ways are dropped — the pipeline only uses polygon features.
    """
    records, geoms = [], []
    for el in elements:
        geom = None
        if el["type"] == "way" and "geometry" in el:
            geom = _ring_from_geometry(el["geometry"])
        elif el["type"] == "relation":
            geom = _relation_polygon(el)
        if geom is None or geom.is_empty:
            continue
        tags = el.get("tags", {})
        records.append(
            {
                "osm_type": el["type"],
                "osm_id": el["id"],
                "tags": json.dumps(tags, sort_keys=True),
                "location_tag": tags.get("location")
                or tags.get("generator:place")
                or tags.get("generator:location"),
            }
        )
        geoms.append(geom)
    if not records:
        return gpd.GeoDataFrame(
            {"osm_type": [], "osm_id": [], "tags": [], "location_tag": []},
            geometry=[],
            crs=WGS84,
        )
    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=WGS84)
    return gdf.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)
