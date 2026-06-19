"""Pack training chips from individual GeoTIFFs into a single HDF5 file.

Eliminates ~96 file-opens per training batch (48 image TIFFs + 48 mask TIFFs)
that stall the DataLoader workers on spinning HDD. Run once after `chips`; the
freshness check re-runs only when the chip index changes.

HDF5 layout (no compression for maximum read throughput):
  images  — (N, C, H, W)  uint16, chunked (1, C, H, W)
  masks   — (N, H, W)     uint8,  chunked (1, H, W)

Row order matches index.parquet exactly, so h5_row == positional index in parquet.
"""

from __future__ import annotations

import rasterio
from loguru import logger

from ..config import Config
from ..io_artifacts import ArtifactStore, read_gdf


def run(cfg: Config, store: ArtifactStore) -> None:
    import h5py

    h5_path = store.chips_h5
    if store.is_fresh(h5_path, {}, inputs=[store.chips_index]):
        logger.info("Chips HDF5 fresh — skipping")
        return

    if not store.chips_index.exists():
        raise RuntimeError("No chip index found — run `chips` first")

    index = read_gdf(store.chips_index)
    n = len(index)
    if n == 0:
        raise RuntimeError("Chip index is empty")

    with rasterio.open(index.iloc[0]["image"]) as src:
        c, h, w = src.count, src.height, src.width

    logger.info("Packing {} chips ({} bands, {}×{}) → {}", n, c, h, w, h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        imgs_ds = f.create_dataset("images", shape=(n, c, h, w), dtype="uint16", chunks=(1, c, h, w))
        masks_ds = f.create_dataset("masks", shape=(n, h, w), dtype="uint8", chunks=(1, h, w))

        for i, (_, row) in enumerate(index.iterrows()):
            if i % 10_000 == 0 and i > 0:
                logger.info("  packed {}/{}", i, n)
            with rasterio.open(row["image"]) as src:
                imgs_ds[i] = src.read()
            with rasterio.open(row["mask"]) as src:
                masks_ds[i] = src.read(1)

    logger.info("HDF5 pack complete: {:.1f} GB", h5_path.stat().st_size / 1e9)
    store.write_meta(h5_path, {}, inputs=[store.chips_index])
