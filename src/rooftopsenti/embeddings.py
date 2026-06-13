"""Encoder embeddings from the trained SSL4EO U-Net, for the pre-screen stage.

The U-Net's encoder (an SSL4EO-pretrained ResNet, fine-tuned during ``train``)
is itself a Sentinel-2 feature extractor. Globally average-pooling its deepest
feature map turns any patch into a fixed-length embedding — cheap, no decoder,
no extra weights — which a light classifier head can score for "looks like PV".
This is Path A of the embedding options: reuse what the repo already trains.
"""

from __future__ import annotations

import torch

from .models import SolarSegmentationTask


def load_encoder(ckpt_path: str, device: str) -> torch.nn.Module:
    """Encoder of a trained SolarSegmentationTask, in eval mode on ``device``."""
    task = SolarSegmentationTask.load_from_checkpoint(str(ckpt_path), map_location=device)
    return task.model.encoder.to(device).eval()


@torch.inference_mode()
def embed(encoder: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Global-average-pooled deepest encoder feature map → ``(N, D)`` embeddings.

    ``x`` is ``(N, C, H, W)`` reflectance in ``[0, 1]``. ``smp`` encoders return
    the list of per-stage feature maps; the last is the deepest/most semantic.
    """
    feats = encoder(x)
    deep = feats[-1] if isinstance(feats, (list, tuple)) else feats
    return deep.mean(dim=(-2, -1))


class LinearHead(torch.nn.Module):
    """Logistic-regression head over standardized embeddings (PV vs. not)."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.linear((emb - self.mean) / self.std).squeeze(-1)
