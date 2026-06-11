"""Stage d) Training chips: image/mask pairs around labels + hard negatives.

Positives: ``chips.pos_per_label`` jittered windows per OSM rooftop solar label.
Hard negatives: windows over large buildings that have no OSM solar nearby —
this encodes the positive-unlabeled mitigation (negatives only come from
buildings in a well-mapped region that are confidently solar-free).

Chips are assigned to train/val/test by spatial block so splits are
geographically disjoint.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
from loguru import logger
from shapely.geometry import box

from ..config import Config
from ..geo import EQUAL_AREA, WGS84, spatial_block_id
from ..io_artifacts import ArtifactStore, read_gdf, write_gdf
from ..stac_catalog import composite_assets

NODATA_MAX_FRACTION = 0.5
SOLAR_FREE_BUFFER_M = 50.0  # negatives must have no OSM solar within this distance


def _jittered_window(
    src: rasterio.DatasetReader, x: float, y: float, patch: int, rng: np.random.Generator
) -> rasterio.windows.Window:
    row, col = src.index(x, y)
    jitter = patch // 4
    row += int(rng.integers(-jitter, jitter + 1))
    col += int(rng.integers(-jitter, jitter + 1))
    return rasterio.windows.Window(col - patch // 2, row - patch // 2, patch, patch)


def _read_chip(src: rasterio.DatasetReader, window) -> np.ndarray | None:
    img = src.read(window=window, boundless=True, fill_value=0)
    if (img == 0).all(axis=0).mean() > NODATA_MAX_FRACTION:
        return None
    return img


def _rasterize_labels(labels_proj: gpd.GeoDataFrame, window, src) -> np.ndarray:
    transform = rasterio.windows.transform(window, src.transform)
    shape = (int(window.height), int(window.width))
    chip_bounds = box(*rasterio.windows.bounds(window, src.transform))
    hits = labels_proj[labels_proj.intersects(chip_bounds)]
    if hits.empty:
        return np.zeros(shape, dtype=np.uint8)
    return rasterio.features.rasterize(
        ((geom, 1) for geom in hits.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def _write_pair(
    img: np.ndarray, mask: np.ndarray, window, src, images_dir: Path, masks_dir: Path, name: str
) -> tuple[Path, Path]:
    transform = rasterio.windows.transform(window, src.transform)
    profile = dict(
        driver="GTiff",
        height=img.shape[1],
        width=img.shape[2],
        crs=src.crs,
        transform=transform,
        compress="DEFLATE",
    )
    img_path = images_dir / f"{name}.tif"
    mask_path = masks_dir / f"{name}.tif"
    with rasterio.open(img_path, "w", count=img.shape[0], dtype="uint16", nodata=0, **profile) as dst:
        dst.write(img)
    with rasterio.open(mask_path, "w", count=1, dtype="uint8", **profile) as dst:
        dst.write(mask, 1)
    return img_path, mask_path


def _solar_free_buildings(
    buildings: gpd.GeoDataFrame, all_solar: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Large buildings with no OSM solar (any size) within SOLAR_FREE_BUFFER_M."""
    if all_solar.empty:
        return buildings
    b = buildings.to_crs(EQUAL_AREA)
    s = all_solar.to_crs(EQUAL_AREA)
    s = s.set_geometry(s.geometry.buffer(SOLAR_FREE_BUFFER_M))
    joined = gpd.sjoin(b[["geometry"]], s[["geometry"]], how="left", predicate="intersects")
    has_solar = joined.groupby(level=0)["index_right"].apply(lambda v: v.notna().any())
    return buildings[~has_solar.reindex(buildings.index, fill_value=False)]


def _assign_splits(index: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    """Deterministic spatial-block assignment to train/val/test."""
    cent = index.geometry.to_crs(EQUAL_AREA).centroid
    index = index.assign(block=spatial_block_id(cent.x, cent.y, cfg.split.block_size_km))
    blocks = np.sort(index["block"].unique())
    rng = np.random.default_rng(cfg.split.seed)
    rng.shuffle(blocks)
    r = cfg.split.ratios
    n = len(blocks)
    n_train = max(1, round(n * r.train))
    n_val = max(1, round(n * r.val)) if n > 2 else 0
    split_of = {}
    for i, blk in enumerate(blocks):
        if i < n_train:
            split_of[blk] = "train"
        elif i < n_train + n_val:
            split_of[blk] = "val"
        else:
            split_of[blk] = "test"
    if n <= 2:  # degenerate tiny AOIs: ensure at least train+val exist
        split_of = {blk: ("train" if i == 0 else "val") for i, blk in enumerate(blocks)}
    return index.assign(split=index["block"].map(split_of))


def run(cfg: Config, store: ArtifactStore) -> None:
    cfg_slice = {
        "chips": cfg.chips.model_dump(mode="json"),
        "split": cfg.split.model_dump(mode="json"),
        "patch_size": cfg.model.patch_size,
        # chip channel layout follows the imagery config — changing bands or
        # resolution must invalidate previously written chips
        "bands": list(cfg.imagery.bands),
        "target_resolution_m": cfg.imagery.target_resolution_m,
    }
    # track the composite COGs themselves (size+mtime), not catalog.json —
    # the catalog can be rebuilt/re-registered without the imagery changing
    composite_paths = sorted(composite_assets(store.stac_catalog).values())
    inputs = [store.osm_labels, store.buildings, *composite_paths]
    if store.is_fresh(store.chips_index, cfg_slice, inputs=inputs):
        logger.info("Chips fresh — skipping")
        return

    labels = read_gdf(store.osm_labels)
    all_solar = read_gdf(store.osm_solar)
    buildings = read_gdf(store.buildings)
    assets = composite_assets(store.stac_catalog)
    if not assets:
        raise RuntimeError("No composites in local STAC catalog — run `composite` first")
    if labels.empty:
        raise RuntimeError("No OSM labels — cannot build training chips")

    negatives = _solar_free_buildings(buildings, all_solar)
    n_neg_target = len(labels) * cfg.chips.pos_per_label * cfg.chips.neg_ratio
    rng = np.random.default_rng(cfg.split.seed)
    if len(negatives) > n_neg_target:
        negatives = negatives.sample(n=n_neg_target, random_state=cfg.split.seed)
    logger.info(
        "{} positive label(s), {} hard-negative building(s)", len(labels), len(negatives)
    )

    images_dir = store.chips_dir / "images"
    masks_dir = store.chips_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records = []
    patch = cfg.model.patch_size
    for (tile, range_idx), cog in assets.items():
        with rasterio.open(cog) as src:
            raster_bounds = box(*src.bounds)
            labels_proj = labels.to_crs(src.crs)
            negatives_proj = negatives.to_crs(src.crs)

            for kind, gdf, per_geom in (
                ("pos", labels_proj, cfg.chips.pos_per_label),
                ("neg", negatives_proj, 1),
            ):
                inside = gdf[gdf.centroid.within(raster_bounds)]
                for fid, geom in zip(inside.index, inside.geometry, strict=True):
                    c = geom.centroid
                    for k in range(per_geom):
                        window = _jittered_window(src, c.x, c.y, patch, rng)
                        img = _read_chip(src, window)
                        if img is None:
                            continue
                        mask = _rasterize_labels(labels_proj, window, src)
                        if kind == "pos" and mask.sum() == 0:
                            continue
                        name = f"{tile}_r{range_idx}_{kind}_{fid}_{k}"
                        img_path, mask_path = _write_pair(
                            img, mask, window, src, images_dir, masks_dir, name
                        )
                        chip_bounds_wgs84 = (
                            gpd.GeoSeries(
                                [box(*rasterio.windows.bounds(window, src.transform))],
                                crs=src.crs,
                            )
                            .to_crs(WGS84)
                            .iloc[0]
                        )
                        records.append(
                            {
                                "name": name,
                                "kind": kind,
                                "tile": tile,
                                "range_idx": range_idx,
                                "image": str(img_path),
                                "mask": str(mask_path),
                                "geometry": chip_bounds_wgs84,
                            }
                        )

    if not records:
        raise RuntimeError("No chips generated — check composites and labels overlap")
    index = gpd.GeoDataFrame(pd.DataFrame(records), geometry="geometry", crs=WGS84)
    index = _assign_splits(index, cfg)
    logger.info(
        "Generated {} chips ({} pos / {} neg); splits: {}",
        len(index),
        (index["kind"] == "pos").sum(),
        (index["kind"] == "neg").sum(),
        index["split"].value_counts().to_dict(),
    )
    write_gdf(index, store.chips_index)
    store.write_meta(store.chips_index, cfg_slice, inputs=inputs)
