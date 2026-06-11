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


def _roi_windows(src: rasterio.DatasetReader, buildings_proj, patch: int):
    """Non-overlapping patch windows intersecting at least one building."""
    tree = STRtree(list(buildings_proj.geometry))
    windows = []
    for row_off in range(0, src.height, patch):
        for col_off in range(0, src.width, patch):
            window = rasterio.windows.Window(col_off, row_off, patch, patch)
            bounds = box(*rasterio.windows.bounds(window, src.transform))
            if len(tree.query(bounds, predicate="intersects")):
                windows.append(window)
    return windows


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
    if buildings.empty:
        raise RuntimeError("No large buildings — run `buildings` first")
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
                idx = building_tree.query(mgrs_tile_polygon(tile), predicate="intersects")
                buildings_proj = buildings.iloc[idx].to_crs(src.crs)
                windows = _roi_windows(src, buildings_proj, cfg.model.patch_size)
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
