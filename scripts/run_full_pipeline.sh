#!/usr/bin/env bash
#
# Full rooftop-solar pipeline, end to end:
#   1. Train a >=500 m2 model on Germany (primary/held-out test) + Switzerland +
#      Netherlands + United Kingdom + New Zealand + Pakistan (pooled train/val).
#   2. Infer that model over Pakistan and build the missing-in-OSM report.
#
# Designed to run unattended on Linux / Windows-WSL. No Claude required.
#
# Disk management: composites (~95 GB each) are deleted immediately after chipping
# so only one region's composites are on disk at a time. All training chips are
# deleted after training completes (model checkpoint is kept). This keeps peak
# usage to ~175 GB, well within a 229 GB disk.
#
# ---------------------------------------------------------------------------
# Quick start (inside WSL, from the repo root):
#   pixi install
#   bash scripts/run_full_pipeline.sh
#
# CPU-only fallback (no GPU):
#   RUN="pixi run rooftopsenti" bash scripts/run_full_pipeline.sh
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
RUN="${RUN:-pixi run -e cuda rooftopsenti}"  # use "pixi run rooftopsenti" for CPU-only
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
# Pakistan is chipped first so its composites (~200 GB) are deleted before the
# European regions' chips accumulate. Composites are re-downloaded at inference.
echo "==================== Pakistan training data prep ===================="
$RUN aoi       -c "$PK"
$RUN labels    -c "$PK"            # OSM solar used as training positives
$RUN composite -c "$PK"
$RUN buildings -c "$PK"
$RUN chips     -c "$PK"
echo "--- pruning Pakistan composites (will be re-downloaded for inference) ---"
rm -rf "data/pakistan_500/composites/"
echo "--- composites deleted; disk free: $(df -h /run/media/tobi/aidisc/ | awk 'NR==2{print $4}') ---"

# Per-region data prep for the European training regions.
# Composites are deleted right after chipping to reclaim ~95 GB per region.
REGIONS=(germany_500 switzerland_500 netherlands_500 united_kingdom_500 new_zealand_500)
for region in "${REGIONS[@]}"; do
  cfg="configs/${region}.yaml"
  echo "==================== data prep: ${region} ===================="
  $RUN aoi       -c "$cfg"
  $RUN labels    -c "$cfg"
  $RUN composite -c "$cfg"
  $RUN buildings -c "$cfg"
  $RUN chips     -c "$cfg"

  echo "--- pruning composites for ${region} to reclaim disk ---"
  rm -rf "data/${region}/composites/"
  echo "--- composites deleted; disk free: $(df -h /run/media/tobi/aidisc/ | awk 'NR==2{print $4}') ---"
done

echo "==================== train pooled model (${TRAIN_RUN_ID}) ===================="
$RUN train    -c "$TRAIN_CONFIG" --run-id "$TRAIN_RUN_ID"
$RUN validate -c "$TRAIN_CONFIG" --run-id "$TRAIN_RUN_ID"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: expected checkpoint not found at $CKPT" >&2
  exit 1
fi

echo "--- pruning all training chips to reclaim disk for Pakistan inference composites ---"
for region in "${REGIONS[@]}"; do
  rm -rf "data/${region}/chips/"
done
rm -rf "data/pakistan_500/chips/"
echo "--- chips deleted; disk free: $(df -h /run/media/tobi/aidisc/ | awk 'NR==2{print $4}') ---"

# ===================== 2. PAKISTAN INFERENCE =====================
# Re-download Pakistan composites (deleted after training chips to save disk).
# OSM labels are already fresh from the training prep step above.
echo "==================== Pakistan inference ===================="
$RUN composite   -c "$PK"            # re-download Sentinel-2 mosaics
$RUN infer       -c "$PK" --model-ckpt "$CKPT"
$RUN postprocess -c "$PK"
$RUN report      -c "$PK"

echo
echo "DONE."
echo "  Trained model:    $CKPT"
echo "  Germany metrics:  $RUN validate -c $TRAIN_CONFIG --run-id $TRAIN_RUN_ID"
echo "  Pakistan outputs: data/pakistan_500/outputs/<run_id>/   (missing_in_osm.geojson + HTML map)"
echo "  Full log:         $LOG"
