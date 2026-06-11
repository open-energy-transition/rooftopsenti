"""Overture Maps access via DuckDB — bbox-queryable GeoParquet on S3.

Overture redistributes OSM (plus other sources) as monthly GeoParquet releases.
Used for two things:
- solar generator polygons (``theme=base/type=infrastructure``) with the original
  OSM tags preserved in ``source_tags`` — an Overpass-free label backend;
- building footprints (``theme=buildings/type=building``) for label building
  intersection and as the large-building inference ROI source.
"""

from __future__ import annotations

import duckdb
import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely import from_wkb

from .geo import WGS84

S3_ROOT = "s3://overturemaps-us-west-2/release"

ROOF_TAG_KEYS = ("location", "generator:place", "generator:location")

OSM_TYPE_BY_PREFIX = {"n": "node", "w": "way", "r": "relation"}


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")
    return con


def theme_path(release: str, theme: str, type_: str) -> str:
    return f"{S3_ROOT}/{release}/theme={theme}/type={type_}/*"


def _bbox_where(alias: str = "") -> str:
    p = f"{alias}." if alias else ""
    return (
        f"{p}bbox.xmin <= ? AND {p}bbox.xmax >= ? AND "
        f"{p}bbox.ymin <= ? AND {p}bbox.ymax >= ?"
    )


def _bbox_params(bounds: tuple[float, float, float, float]) -> list[float]:
    w, s, e, n = bounds
    return [e, w, n, s]


def parse_osm_record_id(record_id: str | None) -> tuple[str | None, int | None]:
    """Overture OSM record ids look like ``w123456789@5`` -> ('way', 123456789)."""
    if not record_id:
        return None, None
    head = record_id.split("@", 1)[0]
    osm_type = OSM_TYPE_BY_PREFIX.get(head[:1])
    try:
        return osm_type, int(head[1:])
    except ValueError:
        return None, None


def solar_generators(
    release: str, bounds: tuple[float, float, float, float], polygons_only: bool = True
) -> gpd.GeoDataFrame:
    """Solar PV features with original OSM tags, in a WGS84 bbox.

    Matches both ``power=generator`` + ``generator:source=solar`` and
    ``power=plant`` + ``plant:source=solar``. Returns the same schema as the
    Overpass backend: ``osm_type``, ``osm_id``, ``tags`` (JSON string),
    ``location_tag``, polygon geometry.
    """
    import json

    con = connect()
    geom_filter = (
        "AND ST_GeometryType(geometry) IN ('POLYGON', 'MULTIPOLYGON')" if polygons_only else ""
    )
    rows = con.execute(
        f"""
        SELECT sources[1].record_id AS record_id,
               source_tags,
               ST_AsWKB(geometry) AS wkb
        FROM read_parquet('{theme_path(release, "base", "infrastructure")}', hive_partitioning=1)
        WHERE {_bbox_where()}
          AND subtype = 'power'
          AND (source_tags['generator:source'] = 'solar'
               OR source_tags['plant:source'] = 'solar')
          {geom_filter}
        """,
        _bbox_params(bounds),
    ).fetchall()
    con.close()
    logger.info("Overture: {} solar generator/plant polygon(s) in bbox", len(rows))

    if not rows:
        return gpd.GeoDataFrame(
            {"osm_type": [], "osm_id": [], "tags": [], "location_tag": []},
            geometry=[],
            crs=WGS84,
        )

    records, geoms = [], []
    for record_id, tags, wkb in rows:
        tags = dict(tags or {})
        osm_type, osm_id = parse_osm_record_id(record_id)
        location_tag = next((tags[k] for k in ROOF_TAG_KEYS if k in tags), None)
        records.append(
            {
                "osm_type": osm_type,
                "osm_id": osm_id,
                "tags": json.dumps(tags, sort_keys=True),
                "location_tag": location_tag,
            }
        )
        geoms.append(from_wkb(wkb))
    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=WGS84)
    return gdf.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)


def buildings_in_bbox(
    release: str,
    bounds: tuple[float, float, float, float],
    min_area_m2: float | None = None,
) -> gpd.GeoDataFrame:
    """Building footprints in a WGS84 bbox, optionally pre-filtered by area.

    The area filter runs inside DuckDB (with a 20% safety margin — the exact
    equal-area filter happens later in the pipeline), so country-scale queries
    transfer only large buildings instead of the full building stock.
    ``ST_FlipCoordinates`` works around DuckDB's lat/lon axis-order assumption
    in ``ST_Area_Spheroid``.
    """
    area_filter = (
        f"AND ST_Area_Spheroid(ST_FlipCoordinates(geometry)) >= {min_area_m2 * 0.8}"
        if min_area_m2
        else ""
    )
    con = connect()
    rows = con.execute(
        f"""
        SELECT id, ST_AsWKB(geometry) AS wkb
        FROM read_parquet('{theme_path(release, "buildings", "building")}', hive_partitioning=1)
        WHERE {_bbox_where()}
          {area_filter}
        """,
        _bbox_params(bounds),
    ).fetchall()
    con.close()
    if not rows:
        return gpd.GeoDataFrame({"id": [], "source": []}, geometry=[], crs=WGS84)
    ids, wkbs = zip(*rows, strict=True)
    return gpd.GeoDataFrame(
        {"id": ids, "source": "overture"}, geometry=from_wkb(list(wkbs)), crs=WGS84
    )


def buildings_intersecting(release: str, features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Buildings that intersect any of the given WGS84 polygons.

    The spatial join runs inside DuckDB: the features are registered as a temp
    table and matched against the buildings parquet with a bbox prefilter plus
    exact ``ST_Intersects`` — no bulk download of the building stock.
    """
    if features.empty:
        return gpd.GeoDataFrame({"id": [], "source": []}, geometry=[], crs=WGS84)

    feat = pd.DataFrame(
        {
            "wkt": features.geometry.to_wkt(),
            "xmin": features.geometry.bounds["minx"],
            "ymin": features.geometry.bounds["miny"],
            "xmax": features.geometry.bounds["maxx"],
            "ymax": features.geometry.bounds["maxy"],
        }
    )
    con = connect()
    con.register("feat_raw", feat)
    con.execute(
        "CREATE TEMP TABLE feat AS "
        "SELECT ST_GeomFromText(wkt) AS geom, xmin, ymin, xmax, ymax FROM feat_raw"
    )
    rows = con.execute(
        f"""
        SELECT DISTINCT b.id, ST_AsWKB(b.geometry) AS wkb
        FROM read_parquet('{theme_path(release, "buildings", "building")}', hive_partitioning=1) b
        JOIN feat f
          ON b.bbox.xmin <= f.xmax AND b.bbox.xmax >= f.xmin
         AND b.bbox.ymin <= f.ymax AND b.bbox.ymax >= f.ymin
         AND ST_Intersects(b.geometry, f.geom)
        WHERE {_bbox_where("b")}
        """,
        _bbox_params(
            (
                float(feat["xmin"].min()),
                float(feat["ymin"].min()),
                float(feat["xmax"].max()),
                float(feat["ymax"].max()),
            )
        ),
    ).fetchall()
    con.close()
    logger.info("Overture: {} building(s) intersect the given features", len(rows))
    if not rows:
        return gpd.GeoDataFrame({"id": [], "source": []}, geometry=[], crs=WGS84)
    ids, wkbs = zip(*rows, strict=True)
    return gpd.GeoDataFrame(
        {"id": ids, "source": "overture"}, geometry=from_wkb(list(wkbs)), crs=WGS84
    )
