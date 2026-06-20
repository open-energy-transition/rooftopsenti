#!/usr/bin/env bash
#
# Full rooftop-solar pipeline, end to end:
#   1. Train a >=500 m2 model on Germany (primary/held-out test) + Switzerland +
#      Netherlands + United Kingdom + New Zealand (pooled train/val).
#   2. Transfer-infer that model over Pakistan and build the missing-in-OSM report.
#
# Designed to run unattended on Linux / Windows-WSL. No Claude required.
#
# ---------------------------------------------------------------------------
# Quick start (inside WSL, from the repo root):
#   pixi install
#   bash scripts/run_full_pipeline.sh
#
# On a machine with an NVIDIA GPU (driver on Windows, CUDA in WSL):
#   RUN="pixi run -e cuda rooftopsenti" bash scripts/run_full_pipeline.sh
#
# It is a long job (country-scale Sentinel-2 mosaics + 60-epoch training), so run
# it detached, e.g.:   nohup bash scripts/run_full_pipeline.sh &   (then watch logs/)
#
# Every stage skips when its outputs are already fresh, so if it dies partway you
# can just re-run the same command and it resumes where it left off.
# ---------------------------------------------------------------------------
set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

# Override-able knobs (env vars):
RUN="${RUN:-pixi run rooftopsenti}"          # use "pixi run -e cuda rooftopsenti" for GPU
TRAIN_RUN_ID="${TRAIN_RUN_ID:-de5_500}"      # deterministic name for the trained model

TRAIN_CONFIG=configs/germany_5country_500.yaml
CKPT="data/germany_500/models/${TRAIN_RUN_ID}/best.ckpt"
PK=configs/pakistan_500.yaml

# Tee all output to a timestamped log as well as the terminal.
mkdir -p logs
LOG="logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
echo "RUN=$RUN  TRAIN_RUN_ID=$TRAIN_RUN_ID"

# ============================ 1. TRAIN ============================
# Per-region data prep (Germany first, then the pooled contributors).
REGIONS=(germany_500 switzerland_500 netherlands_500 united_kingdom_500 new_zealand_500)
for region in "${REGIONS[@]}"; do
  cfg="configs/${region}.yaml"
  echo "==================== data prep: ${region} ===================="
  $RUN aoi       -c "$cfg"
  $RUN labels    -c "$cfg"
  $RUN composite -c "$cfg"
  $RUN buildings -c "$cfg"
  $RUN chips     -c "$cfg"
done

echo "==================== train pooled model (${TRAIN_RUN_ID}) ===================="
$RUN train    -c "$TRAIN_CONFIG" --run-id "$TRAIN_RUN_ID"
$RUN validate -c "$TRAIN_CONFIG" --run-id "$TRAIN_RUN_ID"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: expected checkpoint not found at $CKPT" >&2
  exit 1
fi

# ===================== 2. PAKISTAN INFERENCE =====================
# Transfer inference: the trained model is applied to Pakistan composites.
echo "==================== Pakistan data prep ===================="
$RUN aoi       -c "$PK"
$RUN buildings -c "$PK"            # >=500 m2 building ROIs
$RUN composite -c "$PK"            # Sentinel-2 mosaics (the slow step)
$RUN labels    -c "$PK"            # fresh OSM solar for the missing-in-OSM audit

echo "==================== Pakistan transfer inference ===================="
$RUN infer       -c "$PK" --model-ckpt "$CKPT"
$RUN postprocess -c "$PK"
$RUN report      -c "$PK"

echo
echo "DONE."
echo "  Trained model:    $CKPT"
echo "  Germany metrics:  $RUN validate -c $TRAIN_CONFIG --run-id $TRAIN_RUN_ID"
echo "  Pakistan outputs: data/pakistan_500/outputs/<run_id>/   (missing_in_osm.geojson + HTML map)"
echo "  Full log:         $LOG"
