"""HTML map report of detection results (folium)."""

from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
from loguru import logger

from .config import Config
from .io_artifacts import ArtifactStore, read_gdf

MAX_FEATURES_PER_LAYER = 5000


def _add_layer(m: folium.Map, gdf: gpd.GeoDataFrame, name: str, color: str, fields: list[str]):
    if gdf.empty:
        return
    if len(gdf) > MAX_FEATURES_PER_LAYER:
        logger.warning("{}: showing {}/{} features", name, MAX_FEATURES_PER_LAYER, len(gdf))
        gdf = gdf.head(MAX_FEATURES_PER_LAYER)
    fields = [f for f in fields if f in gdf.columns]
    folium.GeoJson(
        gdf[fields + ["geometry"]].to_json(),
        name=name,
        style_function=lambda _, c=color: {
            "color": c,
            "weight": 2,
            "fillColor": c,
            "fillOpacity": 0.25,
        },
        tooltip=folium.GeoJsonTooltip(fields=fields) if fields else None,
    ).add_to(m)


def build_map(cfg: Config, store: ArtifactStore, run_id: str) -> Path:
    boundary = read_gdf(store.aoi_boundary)
    center = boundary.geometry.iloc[0].centroid

    m = folium.Map(location=[center.y, center.x], zoom_start=10, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri satellite",
    ).add_to(m)

    _add_layer(m, boundary, "AOI", "#3388ff", [])
    labels = read_gdf(store.osm_labels)
    _add_layer(m, labels, "OSM rooftop solar (training labels)", "#2ca02c", ["area_m2"])

    predicted = read_gdf(store.output(run_id, "predicted_solar.parquet"))
    _add_layer(m, predicted, "Predicted solar", "#ff7f0e", ["area_m2", "mean_prob"])

    missing = read_gdf(store.output(run_id, "missing_in_osm.parquet"))
    _add_layer(
        m,
        missing,
        "MISSING in OSM (candidates)",
        "#d62728",
        ["area_m2", "coverage", "mean_prob", "confidence"],
    )

    folium.LayerControl(collapsed=False).add_to(m)
    out = store.output(run_id, "report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    logger.info("Report written -> {}", out)
    return out
