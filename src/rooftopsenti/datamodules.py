"""Chip dataset + Lightning datamodule for training."""

from __future__ import annotations

import numpy as np
import rasterio
import torch
from lightning.pytorch import LightningDataModule
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
        self.items = index[["image", "mask"]].to_dict("records")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rec = self.items[i]
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


class SolarChipDataModule(LightningDataModule):
    def __init__(self, cfg: Config, store: ArtifactStore, num_workers: int = 4):
        super().__init__()
        self.cfg = cfg
        self.store = store
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        index = read_gdf(self.store.chips_index)
        self.train_ds = ChipDataset(index[index["split"] == "train"], augment=True)
        self.val_ds = ChipDataset(index[index["split"] == "val"], augment=False)
        test = index[index["split"] == "test"]
        # tiny AOIs may have no test blocks — fall back to val for the test loop
        self.test_ds = ChipDataset(test if len(test) else index[index["split"] == "val"], False)

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
