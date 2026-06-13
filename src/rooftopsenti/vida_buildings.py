"""VIDA Google + Microsoft combined building footprints via DuckDB.

The VIDA "Google-Microsoft Open Buildings" dataset republishes Google Open
Buildings and Microsoft Global Building Footprints as cloud-native GeoParquet on
Source Cooperative, partitioned by country ISO3. It is bbox-queryable with
DuckDB over plain HTTPS (no credentials), exactly like the Overture backend, and
gives far better coverage than Overture in the Global South (e.g. Pakistan) —
the dominant recall lever for transfer inference there.

Returns the same schema as :func:`rooftopsenti.overture.buildings_in_bbox`
(``id``, ``source``, WGS84 geometry) so the two can be unioned and deduped.
"""

from __future__ import annotations

import geopandas as gpd
from loguru import logger
from shapely import from_wkb

from .config import Config
from .geo import WGS84
from .overture import connect


def buildings_in_bbox(
    cfg: Config, bounds: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """VIDA building footprints in a WGS84 bbox, pre-filtered by area.

    The per-country parquet is located from ``aoi.iso3`` via
    ``buildings.vida_url_template``; the bbox and area filters run inside DuckDB
    so only large buildings in the AOI are transferred. As with Overture,
    ``ST_FlipCoordinates`` works around DuckDB's lat/lon axis-order assumption in
    ``ST_Area_Spheroid`` (a 20% margin is left for the exact equal-area filter
    applied later in the pipeline).
    """
    iso3 = cfg.aoi.iso3
    if not iso3:
        raise ValueError(
            "buildings.sources includes 'vida_open_buildings' but aoi.iso3 is unset — "
            "the VIDA dataset is partitioned by country, so an ISO3 code is required"
        )
    url = cfg.buildings.vida_url_template.format(iso3=iso3.upper())
    min_area = cfg.buildings.fetch_area_min_m2
    w, s, e, n = bounds
    area_filter = (
        f"AND ST_Area_Spheroid(ST_FlipCoordinates(geometry)) >= {min_area * 0.8}"
        if min_area
        else ""
    )
    logger.info("Querying VIDA buildings {} for bbox {}", url, bounds)
    con = connect()
    try:
        rows = con.execute(
            f"""
            SELECT ST_AsWKB(geometry) AS wkb
            FROM read_parquet('{url}')
            WHERE ST_Intersects(geometry, ST_MakeEnvelope(?, ?, ?, ?))
              {area_filter}
            """,
            [w, s, e, n],
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return gpd.GeoDataFrame({"id": [], "source": []}, geometry=[], crs=WGS84)
    geoms = from_wkb([r[0] for r in rows])
    ids = [f"vida_{i}" for i in range(len(geoms))]
    return gpd.GeoDataFrame({"id": ids, "source": "vida"}, geometry=geoms, crs=WGS84)
