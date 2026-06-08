"""Stage g) Polygonize predictions, aggregate per building, compare with OSM.

Final product: ``missing_in_osm`` — large buildings with detected solar but no
OSM solar mapping, i.e. candidates for OSM enrichment / further review.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from loguru import logger
from shapely.geometry import shape

from ..config import Config
from ..geo import EQUAL_AREA, WGS84, filter_min_area
from ..io_artifacts import ArtifactStore, read_gdf, write_gdf

PROB_NODATA = -1.0


def _polygonize_predictions(store: ArtifactStore, cfg: Config, run_id: str) -> gpd.GeoDataFrame:
    from scipy import ndimage

    pred_dir = store.prediction_tif(run_id, "x").parent
    frames = []
    for tif in sorted(pred_dir.glob("*_solar_prob.tif")):
        with rasterio.open(tif) as src:
            prob = src.read(1)
            hot = (prob >= cfg.postprocess.prob_threshold) & (prob != PROB_NODATA)
            if not hot.any():
                continue
            # mean probability per connected component (O(raster), not O(polygons))
            labels, n = ndimage.label(hot)
            component_mean = ndimage.mean(prob, labels=labels, index=np.arange(1, n + 1))
            geoms, probs = [], []
            for geom, comp in rasterio.features.shapes(
                labels.astype(np.int32), mask=hot, transform=src.transform
            ):
                geoms.append(shape(geom))
                probs.append(float(component_mean[int(comp) - 1]))
            if geoms:
                gdf = gpd.GeoDataFrame({"mean_prob": probs}, geometry=geoms, crs=src.crs)
                frames.append(gdf.to_crs(WGS84))
    if not frames:
        return gpd.GeoDataFrame({"mean_prob": []}, geometry=[], crs=WGS84)

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=WGS84)
    # merge polygons split across tile seams
    dissolved = gpd.GeoDataFrame(
        geometry=[merged.geometry.union_all()], crs=WGS84
    ).explode(ignore_index=True)
    joined = gpd.sjoin(
        dissolved, merged[["mean_prob", "geometry"]], how="left", predicate="intersects"
    )
    mean_prob = joined.groupby(level=0)["mean_prob"].mean()
    dissolved["mean_prob"] = mean_prob.reindex(dissolved.index).fillna(0.0)
    return filter_min_area(dissolved, cfg.postprocess.min_solar_area_m2).reset_index(drop=True)


def _aggregate_per_building(
    predicted: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, cfg: Config
) -> gpd.GeoDataFrame:
    stats = buildings.copy()
    if predicted.empty:
        stats["pred_solar_m2"] = 0.0
        stats["coverage"] = 0.0
        stats["mean_prob"] = 0.0
        stats["has_solar"] = False
        return stats

    b = buildings.to_crs(EQUAL_AREA)
    p = predicted.to_crs(EQUAL_AREA)
    inter = gpd.overlay(
        b[["building_id", "geometry"]],
        p[["mean_prob", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    inter["inter_m2"] = inter.geometry.area
    agg = inter.groupby("building_id").agg(
        pred_solar_m2=("inter_m2", "sum"), mean_prob=("mean_prob", "mean")
    )
    stats = stats.merge(agg, on="building_id", how="left")
    stats[["pred_solar_m2", "mean_prob"]] = stats[["pred_solar_m2", "mean_prob"]].fillna(0.0)
    stats["coverage"] = stats["pred_solar_m2"] / stats["area_m2"]
    stats["has_solar"] = stats["coverage"] >= cfg.postprocess.building_coverage_min
    return stats


def _mark_osm_solar(stats: gpd.GeoDataFrame, osm_solar: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if osm_solar.empty:
        return stats.assign(osm_has_solar=False)
    joined = gpd.sjoin(
        stats[["geometry"]], osm_solar[["geometry"]], how="left", predicate="intersects"
    )
    osm_has = joined.groupby(level=0)["index_right"].apply(lambda v: v.notna().any())
    return stats.assign(osm_has_solar=osm_has.reindex(stats.index, fill_value=False))


def run(cfg: Config, store: ArtifactStore, run_id: str | None = None) -> str:
    run_id = run_id or cfg.run_id()

    predicted = _polygonize_predictions(store, cfg, run_id)
    logger.info("Polygonized {} predicted solar polygon(s)", len(predicted))
    write_gdf(predicted, store.output(run_id, "predicted_solar.parquet"))

    buildings = read_gdf(store.gba_buildings)
    stats = _aggregate_per_building(predicted, buildings, cfg)
    # inference-only transfer regions may have no OSM solar layer at all — then
    # every detection is a candidate (nothing to mark as already-mapped)
    if store.osm_solar.exists():
        osm_solar = read_gdf(store.osm_solar)
    else:
        logger.warning(
            "No OSM solar layer ({}); treating all detections as candidates "
            "(run `labels` to flag already-mapped installations)",
            store.osm_solar,
        )
        osm_solar = buildings.iloc[0:0]
    stats = _mark_osm_solar(stats, osm_solar)
    write_gdf(stats, store.output(run_id, "building_solar_stats.parquet"))

    missing = stats[stats["has_solar"] & ~stats["osm_has_solar"]].copy()
    missing["confidence"] = missing["coverage"].clip(0, 1) * missing["mean_prob"]
    missing = missing.sort_values("confidence", ascending=False).reset_index(drop=True)
    write_gdf(missing, store.output(run_id, "missing_in_osm.parquet"))
    missing.to_file(store.output(run_id, "missing_in_osm.geojson"), driver="GeoJSON")

    logger.info(
        "Buildings: {} | with detected solar: {} | already in OSM: {} | MISSING in OSM: {}",
        len(stats),
        int(stats["has_solar"].sum()),
        int((stats["has_solar"] & stats["osm_has_solar"]).sum()),
        len(missing),
    )
    return run_id
