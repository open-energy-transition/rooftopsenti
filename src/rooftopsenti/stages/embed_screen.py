"""Stage (optional) — embedding pre-screen for ROIs beyond building footprints.

Two steps, run together:

1. **Train a head** — embed every training chip with the trained encoder and fit
   a logistic-regression head (positive label vs. hard-negative) on top.
2. **Scan** — slide the encoder over every composite window, score it with the
   head, and keep windows above ``screen.prob_threshold`` as candidate ROI
   boxes (``screen/candidates.parquet``).

``infer`` unions these candidates into its ROI set when
``buildings.use_screen_candidates`` is true — catching ground-mounts and PV on
buildings missing from every footprint source, which the building-only ROI can
never surface. Cheap because it runs the encoder only (no decoder).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
import torch
from loguru import logger
from shapely.geometry import box

from ..config import Config
from ..datamodules import REFLECTANCE_SCALE, load_image
from ..embeddings import LinearHead, embed, load_encoder
from ..geo import WGS84
from ..io_artifacts import ArtifactStore, read_gdf, write_gdf
from ..models import resolve_accelerator
from ..stac_catalog import composite_assets


def _resolve_ckpt(store: ArtifactStore, run_id: str, model_ckpt: str | None) -> Path:
    if model_ckpt is not None:
        ckpt = Path(model_ckpt)
    else:
        ckpt = store.model_dir(run_id) / "best.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained model at {ckpt} — run `train` first")
    return ckpt


def _chip_embeddings(encoder, index, device: str, batch_size: int):
    """Embeddings + binary labels (1=positive chip) for the training chips."""
    paths = index["image"].tolist()
    labels = (index["kind"] == "pos").astype(int).to_numpy()
    embs = []
    for i in range(0, len(paths), batch_size):
        batch = [load_image(p) for p in paths[i : i + batch_size]]
        x = torch.stack(batch).to(device)
        embs.append(embed(encoder, x).cpu().numpy())
    return np.concatenate(embs), labels


def _train_head(emb: np.ndarray, labels: np.ndarray, cfg: Config, device: str) -> LinearHead:
    mean = emb.mean(axis=0)
    std = emb.std(axis=0) + 1e-6
    head = LinearHead(emb.shape[1]).to(device)
    head.mean.copy_(torch.tensor(mean, dtype=torch.float32, device=device))
    head.std.copy_(torch.tensor(std, dtype=torch.float32, device=device))
    x = torch.tensor(emb, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    # class-balanced BCE: positives are far rarer than hard negatives
    pos_weight = torch.tensor(
        max(1.0, (labels == 0).sum() / max(1, (labels == 1).sum())),
        dtype=torch.float32,
        device=device,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.screen.lr)
    head.train()
    for _ in range(cfg.screen.max_epochs):
        opt.zero_grad()
        loss_fn(head(x), y).backward()
        opt.step()
    head.eval()
    with torch.inference_mode():
        acc = (((head(x) > 0).float() == y).float().mean()).item()
    logger.info("Screen head trained on {} chips (train acc {:.3f})", len(y), acc)
    return head


@torch.inference_mode()
def _scan_tile(encoder, head, src, cfg: Config, device: str):
    """Candidate boxes (src CRS) for windows the head scores above threshold."""
    patch = cfg.model.patch_size
    stride = max(1, int(round(patch * cfg.screen.stride_frac)))
    windows = [
        rasterio.windows.Window(c, r, patch, patch)
        for r in range(0, src.height, stride)
        for c in range(0, src.width, stride)
    ]
    boxes, probs = [], []
    batch_size = cfg.model.batch_size
    for i in range(0, len(windows), batch_size):
        imgs, keep = [], []
        for w in windows[i : i + batch_size]:
            img = src.read(window=w, boundless=True, fill_value=0)
            if (img == 0).all(axis=0).mean() > 0.95:
                continue
            imgs.append(np.clip(img.astype(np.float32) / REFLECTANCE_SCALE, 0.0, 1.0))
            keep.append(w)
        if not imgs:
            continue
        x = torch.from_numpy(np.stack(imgs)).to(device)
        p = torch.sigmoid(head(embed(encoder, x))).cpu().numpy()
        for w, pw in zip(keep, p, strict=True):
            if pw >= cfg.screen.prob_threshold:
                boxes.append(box(*rasterio.windows.bounds(w, src.transform)))
                probs.append(float(pw))
    return boxes, probs


def run(
    cfg: Config,
    store: ArtifactStore,
    run_id: str | None = None,
    only_tiles: list[str] | None = None,
    model_ckpt: str | None = None,
) -> None:
    run_id = run_id or cfg.run_id()
    ckpt = _resolve_ckpt(store, run_id, model_ckpt)

    cfg_slice = {
        "screen": cfg.screen.model_dump(mode="json"),
        "patch_size": cfg.model.patch_size,
        "bands": list(cfg.imagery.bands),
    }
    composite_paths = sorted(composite_assets(store.stac_catalog).values())
    inputs = [store.chips_index, ckpt, *composite_paths]
    if store.is_fresh(store.screen_candidates, cfg_slice, inputs=inputs):
        logger.info("Screen candidates fresh — skipping")
        return

    accelerator = resolve_accelerator(cfg)
    device = "cuda" if accelerator == "gpu" else "cpu"
    encoder = load_encoder(str(ckpt), device)

    index = read_gdf(store.chips_index)
    if index.empty:
        raise RuntimeError("No chips — run `chips` first")
    emb, labels = _chip_embeddings(encoder, index, device, cfg.model.batch_size)
    head = _train_head(emb, labels, cfg, device)
    store.screen_head.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), store.screen_head)

    assets = composite_assets(store.stac_catalog)
    if not assets:
        raise RuntimeError("No composites — run `composite` first")
    geoms, probs = [], []
    for (tile, range_idx), cog in sorted(assets.items()):
        if only_tiles and tile not in set(only_tiles):
            continue
        with rasterio.open(cog) as src:
            tile_boxes, tile_probs = _scan_tile(encoder, head, src, cfg, device)
            if tile_boxes:
                wgs = gpd.GeoSeries(tile_boxes, crs=src.crs).to_crs(WGS84)
                geoms.extend(wgs.tolist())
                probs.extend(tile_probs)
        logger.info("{} r{}: {} screen candidate window(s)", tile, range_idx, len(tile_boxes))

    candidates = gpd.GeoDataFrame({"screen_prob": probs}, geometry=geoms, crs=WGS84)
    write_gdf(candidates, store.screen_candidates)
    store.write_meta(store.screen_candidates, cfg_slice, inputs=inputs)
    logger.info("Wrote {} screen candidate window(s) -> {}", len(candidates), store.screen_candidates)
