"""Stage f) Country-scale inference restricted to large-building ROIs.

For every composite, only patch windows that contain at least one large
building are run through the model — empty countryside is skipped entirely.
Outputs one solar-probability COG per (tile, date-range).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
import torch
from loguru import logger
from shapely.geometry import box
from shapely.strtree import STRtree

from ..config import Config
from ..datamodules import REFLECTANCE_SCALE
from ..geo import mgrs_tile_polygon
from ..io_artifacts import ArtifactStore, read_gdf
from ..models import SolarSegmentationTask, resolve_accelerator
from ..stac_catalog import composite_assets

PROB_NODATA = -1.0


def _roi_windows(src: rasterio.DatasetReader, roi_geoms: list, patch: int):
    """Non-overlapping patch windows intersecting at least one ROI geometry.

    ``roi_geoms`` are in the raster CRS and already include any footprint buffer
    and embedding-screen candidate boxes.
    """
    if not roi_geoms:
        return []
    tree = STRtree(roi_geoms)
    windows = []
    for row_off in range(0, src.height, patch):
        for col_off in range(0, src.width, patch):
            window = rasterio.windows.Window(col_off, row_off, patch, patch)
            bounds = box(*rasterio.windows.bounds(window, src.transform))
            if len(tree.query(bounds, predicate="intersects")):
                windows.append(window)
    return windows


def _roi_geoms_for_tile(tile, src, buildings, building_tree, screen, screen_tree, cfg):
    """Footprint geometries (buffered) + screen candidate boxes for one tile, in src CRS."""
    tile_poly = mgrs_tile_polygon(tile)
    idx = building_tree.query(tile_poly, predicate="intersects")
    geoms = list(buildings.iloc[idx].to_crs(src.crs).geometry)
    buffer_m = cfg.buildings.roi_buffer_m
    if buffer_m:
        geoms = [g.buffer(buffer_m) for g in geoms]
    if screen is not None:
        sidx = screen_tree.query(tile_poly, predicate="intersects")
        if len(sidx):
            geoms += list(screen.iloc[sidx].to_crs(src.crs).geometry)
    return geoms


@torch.inference_mode()
def _predict_windows(model, src, windows, cfg: Config, device: str) -> np.ndarray:
    prob = np.full((src.height, src.width), PROB_NODATA, dtype=np.float32)
    batch_size = cfg.model.batch_size
    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i : i + batch_size]
        imgs, keep = [], []
        for w in batch_windows:
            img = src.read(window=w, boundless=True, fill_value=0)
            if (img == 0).all(axis=0).mean() > 0.95:
                continue
            imgs.append(img.astype(np.float32) / REFLECTANCE_SCALE)
            keep.append(w)
        if not imgs:
            continue
        x = torch.from_numpy(np.clip(np.stack(imgs), 0.0, 1.0)).to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        for w, pw in zip(keep, p, strict=True):
            r0, c0 = int(w.row_off), int(w.col_off)
            h = min(src.height - r0, int(w.height))
            wd = min(src.width - c0, int(w.width))
            prob[r0 : r0 + h, c0 : c0 + wd] = pw[:h, :wd]
        if (i // batch_size) % 20 == 0:
            logger.debug("inference {}/{} windows", min(i + batch_size, len(windows)), len(windows))
    return prob


def run(cfg: Config, store: ArtifactStore, run_id: str | None = None,
        only_tiles: list[str] | None = None, model_ckpt: str | None = None) -> str:
    run_id = run_id or cfg.run_id()
    if model_ckpt is not None:
        # transfer inference: apply a model trained in another region. Its bands
        # must match this region's (the input stem is fixed at training time).
        ckpt = Path(model_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"--model-ckpt not found: {ckpt}")
        logger.info("Transfer inference with external checkpoint {}", ckpt)
    else:
        ckpt = store.model_dir(run_id) / "best.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"No trained model at {ckpt} — run `train` first")

    accelerator = resolve_accelerator(cfg)
    device = "cuda" if accelerator == "gpu" else "cpu"
    task = SolarSegmentationTask.load_from_checkpoint(str(ckpt), map_location=device)
    model = task.model.to(device).eval()

    buildings = read_gdf(store.buildings)
    # restrict ROIs to footprints >= the (possibly lower) ROI threshold; the
    # artifact may also hold smaller buildings kept only as training negatives
    roi_min = cfg.buildings.effective_roi_area_min_m2
    if not buildings.empty and "area_m2" in buildings.columns:
        buildings = buildings[buildings["area_m2"] >= roi_min].reset_index(drop=True)

    # optional embedding pre-screen candidates (ground mounts / footprint gaps)
    screen = None
    screen_tree = None
    if cfg.buildings.use_screen_candidates and store.screen_candidates.exists():
        screen = read_gdf(store.screen_candidates)
        if not screen.empty:
            screen_tree = STRtree(list(screen.geometry))
            logger.info("Using {} embedding-screen candidate window(s) as extra ROIs", len(screen))
        else:
            screen = None

    if buildings.empty and screen is None:
        raise RuntimeError(
            "No ROIs: run `buildings` first (or enable buildings.use_screen_candidates "
            "and run `embed-screen`)"
        )
    # index the footprints once (WGS84): each tile then pulls only its local
    # buildings, instead of reprojecting + indexing the whole AOI set per tile
    building_tree = STRtree(list(buildings.geometry))

    assets = composite_assets(store.stac_catalog)
    # ROI windows depend only on the tile grid + buildings, so they are identical
    # across a tile's date ranges — compute once per tile and reuse
    window_cache: dict[str, tuple[tuple[int, int], list]] = {}
    for (tile, range_idx), cog in sorted(assets.items()):
        if only_tiles and tile not in set(only_tiles):
            continue
        out_path = store.prediction_tif(run_id, f"{tile}_r{range_idx}")
        if out_path.exists():
            logger.info("{} r{}: prediction exists — skipping", tile, range_idx)
            continue
        with rasterio.open(cog) as src:
            cached = window_cache.get(tile)
            if cached is not None and cached[0] == (src.height, src.width):
                windows = cached[1]
            else:
                roi_geoms = _roi_geoms_for_tile(
                    tile, src, buildings, building_tree, screen, screen_tree, cfg
                )
                windows = _roi_windows(src, roi_geoms, cfg.model.patch_size)
                window_cache[tile] = ((src.height, src.width), windows)
            logger.info(
                "{} r{}: {} ROI windows (of {} total)",
                tile,
                range_idx,
                len(windows),
                ((src.height // cfg.model.patch_size) + 1)
                * ((src.width // cfg.model.patch_size) + 1),
            )
            prob = _predict_windows(model, src, windows, cfg, device)
            profile = src.profile.copy()
        profile.update(
            driver="COG", count=1, dtype="float32", nodata=PROB_NODATA, compress="DEFLATE"
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(prob, 1)
        logger.info("{} r{}: prediction written -> {}", tile, range_idx, out_path)
    return run_id
