"""Stage e) Train the segmentation model on generated chips."""

from __future__ import annotations

import json

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger

from ..config import Config
from ..datamodules import SolarChipDataModule
from ..io_artifacts import ArtifactStore
from ..models import build_task, resolve_accelerator


def run(cfg: Config, store: ArtifactStore, run_id: str | None = None) -> str:
    run_id = run_id or cfg.run_id()
    model_dir = store.model_dir(run_id)
    best_path = model_dir / "best.ckpt"
    if best_path.exists():
        logger.info("Model {} already trained — skipping (delete to retrain)", run_id)
        return run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    task = build_task(cfg)
    datamodule = SolarChipDataModule(cfg, store)

    checkpoint = ModelCheckpoint(
        dirpath=model_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    trainer = Trainer(
        accelerator=resolve_accelerator(cfg),
        devices=1,
        max_epochs=cfg.model.max_epochs,
        callbacks=[checkpoint, early_stop],
        default_root_dir=str(model_dir),
        log_every_n_steps=5,
    )
    trainer.fit(task, datamodule=datamodule)

    test_metrics = trainer.test(task, datamodule=datamodule, ckpt_path="best")
    (model_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (model_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    logger.info("Training done: {} (metrics: {})", best_path, test_metrics)
    return run_id
