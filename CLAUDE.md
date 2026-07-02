# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rooftopsenti` detects large rooftop solar PV from Sentinel-2 imagery at country/state scale and flags large buildings with visible solar but **no solar mapping in OSM** (the `missing_in_osm` deliverable). OSM solar is treated as positive-unlabeled (PU): a "false positive" against OSM may be a genuinely unmapped installation, so precision is reported as a lower bound. Read `README.md` for the domain narrative.

## Environment & commands

Dependencies are managed with **pixi** (conda-forge for the GDAL/PROJ/PyTorch stack). There is no `requirements.txt` — `pixi install` is the only setup step.

```bash
pixi install                       # build the locked env (one-time)
pixi run test                      # pytest -q
pixi run lint                      # ruff check src tests
pixi run fmt                       # ruff format src tests
pixi run smoke                     # full pipeline on tiny Venlo AOI (configs/smoke_nl_tile.yaml)
pixi run rooftopsenti <cmd> -c configs/<region>.yaml   # any CLI stage
pytest tests/test_chips.py -q                          # single test file
pytest tests/test_chips.py::test_name -q               # single test
```

GPU: the `cuda` environment is opt-in. Prefix any command with `-e cuda`, e.g. `pixi run -e cuda rooftopsenti train -c configs/germany_5country_500.yaml`. The default env is CPU-only and runs every stage (slowly for training).

Ruff is configured for line-length 100, py312, rules `E,F,I,UP,B` (E501 ignored).

## Architecture

The pipeline is a sequence of **stages**, one CLI sub-command each (`src/rooftopsenti/cli.py` → `src/rooftopsenti/stages/`), wired end-to-end by `run-all`:

```
aoi → labels → composite → buildings → chips → train → infer → postprocess → report
```

- **aoi** — region boundary (geoBoundaries API or bbox) → MGRS tile worklist.
- **labels** (`osm_labels.py`) — OSM rooftop-solar polygons → training positives; also derives hard negatives (large solar-free buildings).
- **composite** — per-MGRS-tile cloud-free Sentinel-2 composites written as COGs + a local STAC catalog. Four `imagery.stac_source` backends (`earthgenome` default, `cdse_mosaics`, `planetary_computer`, `earth_search`); see the module docstring for the trade-offs.
- **buildings** (`buildings.py`, `overture.py`, `vida_buildings.py`) — building footprints from one or more `buildings.sources` (default `overture`; add `vida_open_buildings` for the Google+Microsoft combined set, far better Global-South coverage), unioned and spatially deduped. Kept down to `buildings.fetch_area_min_m2` — the artifact serves both inference ROIs (≥ `roi_area_min_m2`, defaults to `building_area_min_m2`) and training hard negatives (≥ `building_area_min_m2`).
- **chips** — training patches around labels + sampled hard negatives, with a spatial-block train/val/test split. Hard negatives stay gated at `building_area_min_m2` even when ROIs go lower.
- **pack-chips** (`pack_chips.py`) — optional but strongly recommended on HDD: packs all chip TIFFs into one uncompressed `chips.h5` (row order = `index.parquet` position). The datamodule auto-detects a fresh pack per region (including `model.train_regions`) and falls back to per-chip TIFFs when the pack is stale or absent. Run after `chips` (and after the *last* batch when chipping with `--tiles`).
- **train** (`models.py`, `datamodules.py`) — U-Net via TorchGeo `SemanticSegmentationTask` with an SSL4EO Sentinel-2 pretrained encoder; focal+dice loss.
- **embed-screen** (`embed_screen.py`, `embeddings.py`) — *optional* recall booster. Trains a logistic-regression head on encoder embeddings of the chips, scans every composite window, and emits PV-like candidate boxes. `infer` unions them into the ROI set when `buildings.use_screen_candidates` is true — catching ground-mounts and footprint-gap PV. `--model-ckpt` for transfer.
- **infer** — runs only on ROI windows: composite windows intersecting a building footprint (buffered by `buildings.roi_buffer_m`) or, when enabled, an embed-screen candidate. `--model-ckpt` applies a model trained elsewhere (transfer inference).
- **postprocess** — polygonize predictions, aggregate per building footprint, emit the `missing_in_osm` candidate list. **Recall-first**: detections are human-validated in OSM against higher-res imagery, so `postprocess.prob_threshold` defaults low (0.35) — more candidates, more reviewer-rejectable false positives, fewer missed installations.
- **report** (`report.py`) — Folium HTML map.
- **clean-negatives** (`clean_negatives.py`) — optional PU mitigation: drop hard negatives a baseline model confidently calls solar, then re-train; non-destructive (marks a `cleaned_out` column) and idempotent.

### Cross-cutting conventions

- **Config** (`config.py`) — each region is a YAML in `configs/`, validated by a pydantic `Config`. `load_config()` is the single entry point. Validators enforce backend/band/cloud-mask compatibility (e.g. mosaic sources reject unavailable bands and non-`scl` cloud masks), so prefer fixing the YAML over bypassing validation.
- **run_id** — `Config.run_id()` is a stable 10-char hash of every field that affects the trained model (region, imagery, overture, osm, model, chips, split). Changing any of those produces a new run id and a new model/output directory. Pass `--run-id` to override with a human name (e.g. `de5_500`).
- **Artifact store & freshness** (`io_artifacts.py`) — every stage writes under `data/<region>/` with a sibling `<artifact>.meta.json` recording its config slice and input file states. A stage **skips when its artifact is fresh** (meta matches current config and no input changed). This is what makes the whole pipeline resumable — re-running after a crash continues where it left off. When adding/altering a stage, write meta via `store.write_meta(...)` and gate work behind `store.is_fresh(...)`, listing the upstream artifacts as `inputs`.
- **CRS conventions** (`geo.py`) — vectors stored in WGS84 (4326); areas computed in equal-area (6933); raster/model work in each tile's native UTM grid. Use `area_m2()` rather than `.area` on lat/lon geometries.
- **Multi-region training** — `model.train_regions` merges other regions' chips into train/val while keeping test metrics on the primary region only. Merged regions must have produced chips with the **same bands**. See `configs/nl_de_pk.yaml` and `configs/germany_5country_500.yaml`.

### Data layout (gitignored)

`data/<region>/`: `aoi/`, `osm/`, `composites/` (+ STAC `catalog.json`), `buildings/`, `chips/`, `screen/` (embed-screen head + candidates), `models/<run_id>/`, `predictions/<run_id>/`, `outputs/<run_id>/`. `data/`, `*.tif`, `*.ckpt`, `lightning_logs/` are not committed.

## Credentials

The default `earthgenome` backend needs none. The `cdse_mosaics` backend reads `s3://eodata` and needs free Copernicus keys in `CDSE_S3_ACCESS_KEY` / `CDSE_S3_SECRET_KEY`.

## Orchestration scripts

`scripts/run_full_pipeline.sh` runs the flagship workflow unattended: 5-country pooled ≥500 m² training (Germany held-out test) then Pakistan transfer inference + report. Override `RUN="pixi run -e cuda rooftopsenti"` for GPU and `TRAIN_RUN_ID` for the model name. `scripts/train_5country_500.sh` is the training-only subset. Both are resumable via stage freshness.
