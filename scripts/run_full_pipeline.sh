#!/usr/bin/env bash
#
# Full rooftop-solar pipeline, end to end:
#   1. Train a >=500 m2 model on Germany (primary/held-out test) + Switzerland +
#      Netherlands + United Kingdom + New Zealand + Pakistan (pooled train/val).
#   2. Infer that model over Pakistan and build the missing-in-OSM report.
#
# Designed to run unattended on Linux / Windows-WSL.
#
# Disk management: Pakistan composites (~200 GB) exceed the disk, so both the
# training-chips and inference stages process Pakistan in batches of 25 tiles —
# download composites, chip/infer, delete composites, repeat. European training
# regions also delete composites right after chipping. All training chips are
# deleted after training completes (the model checkpoint is kept).
#
# ---------------------------------------------------------------------------
# Quick start (from the repo root):
#   bash scripts/run_full_pipeline.sh
#
# CPU-only fallback:
#   RUN="pixi run rooftopsenti" bash scripts/run_full_pipeline.sh
#
# Resumable: every stage skips when its outputs are already fresh.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

RUN="${RUN:-pixi run -e cuda rooftopsenti}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-de5_500}"
BATCH_SIZE="${BATCH_SIZE:-25}"

TRAIN_CONFIG=configs/germany_5country_500.yaml
CKPT="data/germany_500/models/${TRAIN_RUN_ID}/best.ckpt"
PK=configs/pakistan_500.yaml

mkdir -p logs
LOG="logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
echo "RUN=$RUN  TRAIN_RUN_ID=$TRAIN_RUN_ID  BATCH_SIZE=$BATCH_SIZE"

# ---------------------------------------------------------------------------
# Helper: process a region's tiles in batches (composite → action → delete).
# Usage: tile_batches <config> <action_fn_name>
# The action receives the comma-separated tile list as $1.
# ---------------------------------------------------------------------------
_pk_tile_list() {
    # Read tile list from the AOI artifact written by `aoi`.
    python3 -c "import json; print(' '.join(json.load(open('data/pakistan_500/aoi/mgrs_tiles.json'))))"
}

_run_pk_batch() {
    local action="$1"   # "chips" or "infer"
    local extra="${2:-}"
    local tiles=($(_pk_tile_list))
    local n=${#tiles[@]}
    local total_batches=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))

    for ((i=0; i<n; i+=BATCH_SIZE)); do
        local batch=("${tiles[@]:i:BATCH_SIZE}")
        local batch_str=$(IFS=,; echo "${batch[*]}")
        local batch_num=$((i/BATCH_SIZE+1))
        echo "--- Pakistan ${action} batch ${batch_num}/${total_batches}: ${#batch[@]} tiles ---"

        $RUN composite -c "$PK" --tiles "$batch_str"
        # shellcheck disable=SC2086
        $RUN "$action"  -c "$PK" --tiles "$batch_str" $extra

        for tile in "${batch[@]}"; do
            rm -rf "data/pakistan_500/composites/${tile}/"
        done
        echo "--- batch ${batch_num} composites deleted; free: $(df -h /run/media/tobi/aidisc/ | awk 'NR==2{print $4}') ---"
    done
}

# ============================ 1. TRAIN ============================

# --- Pakistan training chips (batched — composites too large to hold at once) ---
echo "==================== Pakistan training data prep ===================="
$RUN aoi       -c "$PK"
$RUN labels    -c "$PK"
$RUN buildings -c "$PK"
_run_pk_batch chips

# --- European training regions (composites deleted after chipping) ---
REGIONS=(germany_500 switzerland_500 netherlands_500 united_kingdom_500 new_zealand_500)
for region in "${REGIONS[@]}"; do
    cfg="configs/${region}.yaml"
    echo "==================== data prep: ${region} ===================="
    $RUN aoi       -c "$cfg"
    $RUN labels    -c "$cfg"
    $RUN composite -c "$cfg"
    $RUN buildings -c "$cfg"
    $RUN chips     -c "$cfg"

    echo "--- pruning composites for ${region} ---"
    rm -rf "data/${region}/composites/"
    echo "--- composites deleted; free: $(df -h /run/media/tobi/aidisc/ | awk 'NR==2{print $4}') ---"
done

hr; log "########## TRAIN POOLED MODEL (${TRAIN_RUN_ID}) ##########"; hr
run_stage "train (${TRAIN_RUN_ID})"    "60-epoch U-Net training — LONG; watch the tqdm epoch bars below" 1 -- $RUN train    -c "$TRAIN_CONFIG" --run-id "$TRAIN_RUN_ID"
run_stage "validate (${TRAIN_RUN_ID})" "Germany held-out test metrics"                                  1 -- $RUN validate -c "$TRAIN_CONFIG" --run-id "$TRAIN_RUN_ID"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: expected checkpoint not found at $CKPT" >&2
    exit 1
fi

echo "--- pruning all training chips to free disk for inference ---"
for region in "${REGIONS[@]}"; do rm -rf "data/${region}/chips/"; done
rm -rf "data/pakistan_500/chips/"
echo "--- chips deleted; free: $(df -h /run/media/tobi/aidisc/ | awk 'NR==2{print $4}') ---"

# ===================== 2. PAKISTAN INFERENCE =====================
# OSM labels already fresh from training prep above.
echo "==================== Pakistan inference (batched) ===================="
_run_pk_batch infer "--model-ckpt $CKPT --run-id $TRAIN_RUN_ID"

$RUN postprocess -c "$PK" --run-id "$TRAIN_RUN_ID"
$RUN report      -c "$PK" --run-id "$TRAIN_RUN_ID"

echo
echo "DONE."
echo "  Trained model:    $CKPT"
echo "  Germany metrics:  data/germany_500/models/${TRAIN_RUN_ID}/"
echo "  Pakistan outputs: data/pakistan_500/outputs/${TRAIN_RUN_ID}/"
echo "  Full log:         $LOG"
