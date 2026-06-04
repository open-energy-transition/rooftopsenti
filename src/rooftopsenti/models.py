"""Model construction: U-Net (torchgeo/smp) with SSL4EO Sentinel-2 pretrained encoder."""

from __future__ import annotations

import torch
from loguru import logger
from torchgeo.trainers import SemanticSegmentationTask

from .config import Config

# SSL4EO-S12 weights are trained on all 13 L1C bands in this order:
SSL4EO_BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]


class SolarSegmentationTask(SemanticSegmentationTask):
    """SemanticSegmentationTask with a combined focal+dice loss option."""

    def configure_losses(self) -> None:
        if self.hparams["loss"] == "focal_dice":
            import segmentation_models_pytorch as smp

            focal = smp.losses.FocalLoss(mode="multiclass", normalized=True)
            dice = smp.losses.DiceLoss(mode="multiclass")

            def combined(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return focal(y_hat, y) + dice(y_hat, y)

            self.criterion = combined
        else:
            super().configure_losses()


def _load_ssl4eo_encoder(task: SemanticSegmentationTask, cfg: Config) -> None:
    """Load SSL4EO MoCo weights, slicing the first conv to the configured bands."""
    from torchgeo.models import ResNet18_Weights, ResNet50_Weights

    weights_enum = {
        "resnet18": ResNet18_Weights.SENTINEL2_ALL_MOCO,
        "resnet50": ResNet50_Weights.SENTINEL2_ALL_MOCO,
    }.get(cfg.model.encoder)
    if weights_enum is None:
        raise ValueError(f"No SSL4EO weights mapped for encoder {cfg.model.encoder!r}")

    state_dict = weights_enum.get_state_dict(progress=True)
    band_idx = [SSL4EO_BAND_ORDER.index(b) for b in cfg.imagery.bands]
    conv1_key = "conv1.weight"
    state_dict[conv1_key] = state_dict[conv1_key][:, band_idx, :, :].clone()

    missing, unexpected = task.model.encoder.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded SSL4EO {} weights (bands {}); missing={}, unexpected={}",
        cfg.model.encoder,
        cfg.imagery.bands,
        len(missing),
        len(unexpected),
    )


def build_task(cfg: Config) -> SemanticSegmentationTask:
    task = SolarSegmentationTask(
        model=cfg.model.arch,
        backbone=cfg.model.encoder,
        weights=None,
        in_channels=cfg.in_channels,
        task="multiclass",
        num_classes=cfg.model.num_classes,
        loss=cfg.model.loss,  # type: ignore[arg-type]  ("focal_dice" handled in subclass)
        lr=cfg.model.lr,
    )
    if cfg.model.pretrained == "ssl4eo_s2_moco":
        _load_ssl4eo_encoder(task, cfg)
    elif cfg.model.pretrained:
        raise ValueError(f"Unknown pretrained option {cfg.model.pretrained!r}")
    return task


def resolve_accelerator(cfg: Config) -> str:
    if cfg.model.device == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    return "gpu" if cfg.model.device == "cuda" else "cpu"
