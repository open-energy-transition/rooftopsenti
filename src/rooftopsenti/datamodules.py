"""Chip dataset + Lightning datamodule for training."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
import torch
from lightning.pytorch import LightningDataModule
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .io_artifacts import ArtifactStore, read_gdf

REFLECTANCE_SCALE = 10000.0


def load_image(path: str) -> torch.Tensor:
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32) / REFLECTANCE_SCALE
    return torch.from_numpy(np.clip(img, 0.0, 1.0))


def load_mask(path: str) -> torch.Tensor:
    with rasterio.open(path) as src:
        return torch.from_numpy(src.read(1).astype(np.int64))


class ChipDataset(Dataset):
    def __init__(self, index, augment: bool):
        cols = [c for c in ("image", "mask", "h5_path", "h5_row") if c in index.columns]
        self.items = index[cols].to_dict("records")
        self.augment = augment
        # opened lazily so each DataLoader worker gets its own handle — h5py
        # handles must not cross the fork boundary
        self._h5_files: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _h5(self, path: str) -> h5py.File:
        f = self._h5_files.get(path)
        if f is None:
            f = self._h5_files[path] = h5py.File(path, "r")
        return f

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rec = self.items[i]
        if rec.get("h5_path"):
            f = self._h5(rec["h5_path"])
            row = int(rec["h5_row"])
            img = f["images"][row].astype(np.float32) / REFLECTANCE_SCALE
            image = torch.from_numpy(np.clip(img, 0.0, 1.0))
            mask = torch.from_numpy(f["masks"][row].astype(np.int64))
        else:
            image = load_image(rec["image"])
            mask = load_mask(rec["mask"])
        if self.augment:
            if torch.rand(1) < 0.5:
                image = torch.flip(image, [-1])
                mask = torch.flip(mask, [-1])
            if torch.rand(1) < 0.5:
                image = torch.flip(image, [-2])
                mask = torch.flip(mask, [-2])
            k = int(torch.randint(0, 4, (1,)))
            if k:
                image = torch.rot90(image, k, [-2, -1])
                mask = torch.rot90(mask, k, [-2, -1])
        return {"image": image, "mask": mask}


def _chip_channels(index) -> int:
    first = index.iloc[0]
    if first.get("h5_path"):  # works even after chip TIFFs were pruned
        with h5py.File(first["h5_path"], "r") as f:
            return f["images"].shape[1]
    with rasterio.open(first["image"]) as src:
        return src.count


def _attach_h5(index, chips_dir: Path):
    """Point chips at the packed HDF5 (``pack-chips`` stage) when it is fresh.

    ``h5_row`` is the positional row in that region's index.parquet, so it must
    be assigned here — before ``drop_cleaned_out`` filtering or multi-region
    concat reorder anything. A stale/absent pack falls back to per-chip TIFFs.
    """
    h5_path = chips_dir / "chips.h5"
    if ArtifactStore.is_fresh(h5_path, {}, inputs=[chips_dir / "index.parquet"]):
        logger.info("Reading chips from packed HDF5 {}", h5_path)
        return index.assign(h5_path=str(h5_path), h5_row=np.arange(len(index)))
    logger.info(
        "No fresh chip pack at {} — reading individual TIFFs "
        "(run `rooftopsenti pack-chips` to speed up training on HDD)",
        h5_path,
    )
    return index.assign(h5_path=None, h5_row=-1)


def drop_cleaned_out(index):
    """Drop hard negatives flagged by the ``clean-negatives`` stage.

    The flag is optional — indices written before cleaning (or by regions that
    were never cleaned) simply have no ``cleaned_out`` column and pass through.
    """
    if "cleaned_out" not in index.columns:
        return index
    keep = ~index["cleaned_out"].fillna(False).astype(bool)
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info("Excluding {} cleaned-out hard negative(s) from training", n_dropped)
    return index[keep].reset_index(drop=True)


def load_chip_index(cfg: Config, store: ArtifactStore):
    """Chip index of the primary region plus any ``model.train_regions``.

    Each region keeps its own spatial-block split assignment; a ``region``
    column is added so evaluation can be restricted to the primary region.
    """
    index = _attach_h5(read_gdf(store.chips_index), store.chips_dir).assign(region=cfg.region)
    frames = [index]
    n_channels = _chip_channels(index)
    for region in cfg.model.train_regions:
        if region == cfg.region:
            continue
        path = cfg.data_dir / region / "chips" / "index.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"model.train_regions includes {region!r} but {path} does not exist "
                f"— run `rooftopsenti chips -c configs/{region}.yaml` first"
            )
        extra = _attach_h5(read_gdf(path), path.parent).assign(region=region)
        extra_channels = _chip_channels(extra)
        if extra_channels != n_channels:
            raise ValueError(
                f"chips from {region!r} have {extra_channels} channels but "
                f"{cfg.region!r} has {n_channels} — regions must share imagery.bands"
            )
        frames.append(extra)
    merged = drop_cleaned_out(pd.concat(frames, ignore_index=True))
    counts = merged.groupby(["region", "split"]).size()
    logger.info("Training chip pool:\n{}", counts.to_string())
    return merged


class SolarChipDataModule(LightningDataModule):
    def __init__(self, cfg: Config, store: ArtifactStore, num_workers: int = 4):
        super().__init__()
        self.cfg = cfg
        self.store = store
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        index = load_chip_index(self.cfg, self.store)
        self.train_ds = ChipDataset(index[index["split"] == "train"], augment=True)
        self.val_ds = ChipDataset(index[index["split"] == "val"], augment=False)
        # test metrics stay on the primary region only (interpretable evaluation)
        primary = index[index["region"] == self.cfg.region]
        test = primary[primary["split"] == "test"]
        # tiny AOIs may have no test blocks — fall back to val for the test loop
        self.test_ds = ChipDataset(test if len(test) else primary[primary["split"] == "val"], False)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.cfg.model.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
