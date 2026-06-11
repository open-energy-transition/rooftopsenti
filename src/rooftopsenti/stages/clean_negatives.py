"""Stage e-clean) Iterative hard-negative cleaning (positive-unlabeled fix).

OSM rooftop solar is positive-unlabeled: a "solar-free" hard negative may
actually be a building whose solar is simply unmapped. This stage scores each
negative chip with a trained baseline model and flags the negatives where the
model confidently predicts solar — those are likely missing-in-OSM positives,
not true negatives. Flagged chips are marked ``cleaned_out=True`` in the chip
index (non-destructive) and excluded from training by the datamodule.

Run between an initial train and a final re-train (distinct ``--run-id`` so the
cleaned model gets its own checkpoint dir):

    rooftopsenti chips           -c cfg.yaml
    rooftopsenti train           -c cfg.yaml --run-id baseline
    rooftopsenti clean-negatives -c cfg.yaml --model-ckpt data/<region>/models/baseline/best.ckpt
    rooftopsenti train           -c cfg.yaml --run-id cleaned

The cleaning is idempotent: every run re-scores *all* hard negatives from
scratch, so re-running with a different threshold (or checkpoint) recomputes the
exclusion set rather than shrinking it further.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from ..config import Config
from ..datamodules import load_image
from ..io_artifacts import ArtifactStore, read_gdf, write_gdf, write_json
from ..models import SolarSegmentationTask, resolve_accelerator


@torch.inference_mode()
def _solar_fractions(model, image_paths: list[str], prob_threshold: float,
                     device: str, batch_size: int) -> list[float]:
    """Fraction of each chip's pixels predicted solar at >= ``prob_threshold``."""
    fractions: list[float] = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        x = torch.stack([load_image(p) for p in batch]).to(device)
        prob = torch.softmax(model(x), dim=1)[:, 1]  # P(solar) per pixel
        fractions.extend((prob >= prob_threshold).float().mean(dim=(1, 2)).cpu().tolist())
    return fractions


def _resolve_ckpt(store: ArtifactStore, cfg: Config, run_id: str | None,
                  model_ckpt: str | None) -> Path:
    if model_ckpt is not None:
        ckpt = Path(model_ckpt)
    else:
        ckpt = store.model_dir(run_id or cfg.run_id()) / "best.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No baseline checkpoint at {ckpt} — train a baseline model first "
            "(or pass --model-ckpt)"
        )
    return ckpt


def run(cfg: Config, store: ArtifactStore, run_id: str | None = None,
        model_ckpt: str | None = None) -> None:
    index = read_gdf(store.chips_index)
    neg = index[index["kind"] == "neg"].copy()
    if neg.empty:
        logger.info("No hard negatives in chip index — nothing to clean")
        return

    ckpt = _resolve_ckpt(store, cfg, run_id, model_ckpt)
    accelerator = resolve_accelerator(cfg)
    device = "cuda" if accelerator == "gpu" else "cpu"
    task = SolarSegmentationTask.load_from_checkpoint(str(ckpt), map_location=device)
    model = task.model.to(device).eval()
    logger.info("Scoring {} hard negative(s) with baseline {}", len(neg), ckpt)

    cn = cfg.clean_negatives
    neg["pred_solar_fraction"] = _solar_fractions(
        model, neg["image"].tolist(), cn.prob_threshold, device, cfg.model.batch_size
    )
    suspect = neg["pred_solar_fraction"] >= cn.max_solar_fraction

    # non-destructive: re-mark from scratch every run so re-cleaning is idempotent
    index["cleaned_out"] = False
    index.loc[neg.index, "cleaned_out"] = suspect
    write_gdf(index, store.chips_index)

    n_removed = int(suspect.sum())
    logger.info(
        "Hard-negative cleaning: flagged {} of {} negative(s) "
        "(>= {:.1%} of pixels at P(solar) >= {}) as likely missing-in-OSM; "
        "{} chip(s) remain for training",
        n_removed, len(neg), cn.max_solar_fraction, cn.prob_threshold,
        int((~index["cleaned_out"]).sum()),
    )

    removed = (
        neg.loc[suspect, ["name", "pred_solar_fraction"]]
        .sort_values("pred_solar_fraction", ascending=False)
        .to_dict("records")
    )
    write_json(
        {
            "checkpoint": str(ckpt),
            "prob_threshold": cn.prob_threshold,
            "max_solar_fraction": cn.max_solar_fraction,
            "negatives_total": len(neg),
            "negatives_removed": n_removed,
            "removed": removed,
        },
        store.clean_negatives_report,
    )
    logger.info("Cleaning report -> {}", store.clean_negatives_report)
