#!/usr/bin/env bash
#
# Multi-country >=500 m² rooftop-solar training pipeline.
#
# Germany is the primary (held-out test) region; Switzerland, the Netherlands,
# the United Kingdom and New Zealand contribute pooled train/val chips. Each
# region's data prep runs independently (aoi -> labels -> composite -> buildings
# -> chips), then a single model is trained on the pooled chips.
#
# Usage:
#   bash scripts/train_5country_500.sh
#   RUN="pixi run -e cuda rooftopsenti" bash scripts/train_5country_500.sh   # GPU box
#
# Re-runnable: every stage skips when its outputs are already fresh, so a crash
# mid-run can be resumed by re-invoking the script.
set -euo pipefail

RUN="${RUN:-pixi run rooftopsenti}"

# Data-prep regions, Germany first (then the pooled contributors).
REGIONS=(germany_500 switzerland_500 netherlands_500 united_kingdom_500 new_zealand_500)

# Training driver (Germany primary; the other four merged via model.train_regions).
TRAIN_CONFIG=configs/germany_5country_500.yaml

for region in "${REGIONS[@]}"; do
  cfg="configs/${region}.yaml"
  echo "==================== data prep: ${region} ===================="
  $RUN aoi       -c "$cfg"
  $RUN labels    -c "$cfg"
  $RUN composite -c "$cfg"
  $RUN buildings -c "$cfg"
  $RUN chips     -c "$cfg"
done

echo "==================== train: pooled 5-country model ===================="
$RUN train -c "$TRAIN_CONFIG"

echo "Done. Checkpoint under data/germany_500/models/<run_id>/best.ckpt"
echo "Held-out (Germany) metrics: $RUN validate -c $TRAIN_CONFIG"
