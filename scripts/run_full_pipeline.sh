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
# It is a long job (country-scale Sentinel-2 mosaics + 60-epoch training). By
# default it detaches itself into the background, writes to logs/full_pipeline_*.log,
# and prints the log path + how to follow/stop it. Pass --foreground to run inline
# (mirrored to the terminal) instead.
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

# ----- logging / self-detach -------------------------------------------------
# By default, re-launch ourselves detached in a new session so the job survives
# the terminal closing and logs straight to a file (no stray nohup.out). Pass
# --foreground to run inline with output mirrored to the terminal instead.
mkdir -p logs
LOG="${RTS_LOG:-logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log}"

if [[ "${1:-}" == "--foreground" ]]; then
  shift
  exec > >(tee -a "$LOG") 2>&1
elif [[ -z "${RTS_DETACHED:-}" ]]; then
  RTS_DETACHED=1 RTS_LOG="$LOG" setsid bash "$0" "$@" >"$LOG" 2>&1 </dev/null &
  pid=$!
  echo "Pipeline started in the background."
  echo "  PID:    $pid"
  echo "  Log:    $LOG"
  echo "  Follow: tail -f $LOG"
  echo "  Stop:   kill -- -$pid"
  exit 0
fi
# (detached child falls through here; its stdout/stderr already point at $LOG)

# ----- verbose progress helpers ----------------------------------------------
# How often (seconds) to print a "still running" heartbeat during long, quiet
# stages (composite/train/infer). Set HEARTBEAT_SECS=0 to disable.
HEARTBEAT_SECS="${HEARTBEAT_SECS:-300}"

pipeline_start=$SECONDS
STEP=0
# 5 regions x 5 prep stages + train + validate + 4 Pakistan prep + infer/postprocess/report
TOTAL=$(( 5 * 5 + 2 + 4 + 3 ))

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
hr()  { echo "----------------------------------------------------------------"; }

# Format a number of seconds as "1h 02m 03s".
fmt_elapsed() {
  local s=$1
  printf '%dh %02dm %02ds' $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
}

# run_stage <label> <hint> <slow:0|1> -- <command...>
# Banner + timestamps before, elapsed time after, optional heartbeat while it
# runs, and a clear FAILED message (then exit) if the command errors.
run_stage() {
  local label=$1 hint=$2 slow=$3; shift 3
  [[ "${1:-}" == "--" ]] && shift
  STEP=$(( STEP + 1 ))
  local start=$SECONDS
  hr
  log ">>> [${STEP}/${TOTAL}] START  ${label}"
  [[ -n "$hint" ]] && log "    note: ${hint}"
  log "    cmd:  $*"
  log "    clock: $(ts) | pipeline so far: $(fmt_elapsed $(( start - pipeline_start )))"
  hr

  # Heartbeat for long, potentially quiet stages so the log keeps showing life.
  local hb_pid=""
  if [[ "$slow" == "1" && "$HEARTBEAT_SECS" -gt 0 ]]; then
    ( n=0
      while true; do
        sleep "$HEARTBEAT_SECS"
        n=$(( n + HEARTBEAT_SECS ))
        log "    ... still running ${label} (~$(fmt_elapsed "$n") in)"
      done ) &
    hb_pid=$!
  fi

  local rc=0
  "$@" || rc=$?

  [[ -n "$hb_pid" ]] && { kill "$hb_pid" 2>/dev/null || true; wait "$hb_pid" 2>/dev/null || true; }

  local dur=$(( SECONDS - start ))
  if [[ $rc -eq 0 ]]; then
    log "<<< [${STEP}/${TOTAL}] DONE   ${label}   took $(fmt_elapsed "$dur")"
  else
    hr
    log "!!! [${STEP}/${TOTAL}] FAILED ${label}   after $(fmt_elapsed "$dur") (exit $rc)"
    log "    Re-run the same command to resume — already-fresh stages are skipped."
    hr
    exit $rc
  fi
}

log "Logging to $LOG"
log "RUN=$RUN  TRAIN_RUN_ID=$TRAIN_RUN_ID  HEARTBEAT_SECS=$HEARTBEAT_SECS"
log "Planned: ${TOTAL} stages total. Fresh artifacts are skipped (printed as a quick DONE)."

# ============================ 1. TRAIN ============================
# Per-region data prep (Germany first, then the pooled contributors).
REGIONS=(germany_500 switzerland_500 netherlands_500 united_kingdom_500 new_zealand_500)
for region in "${REGIONS[@]}"; do
  cfg="configs/${region}.yaml"
  hr; log "########## DATA PREP: ${region} ##########"; hr
  run_stage "aoi (${region})"       ""                                                          0 -- $RUN aoi       -c "$cfg"
  run_stage "labels (${region})"    "downloads OSM solar polygons via Overpass"                 0 -- $RUN labels    -c "$cfg"
  run_stage "composite (${region})" "country-scale Sentinel-2 mosaics — the SLOWEST step (can take hours)" 1 -- $RUN composite -c "$cfg"
  run_stage "buildings (${region})" "fetches Overture/VIDA footprints — can be large"           1 -- $RUN buildings -c "$cfg"
  run_stage "chips (${region})"     "cuts training patches around labels + hard negatives"      0 -- $RUN chips     -c "$cfg"
done

hr; log "########## TRAIN POOLED MODEL (${TRAIN_RUN_ID}) ##########"; hr
run_stage "train (${TRAIN_RUN_ID})"    "60-epoch U-Net training — LONG; watch the tqdm epoch bars below" 1 -- $RUN train    -c "$TRAIN_CONFIG" --run-id "$TRAIN_RUN_ID"
run_stage "validate (${TRAIN_RUN_ID})" "Germany held-out test metrics"                                  1 -- $RUN validate -c "$TRAIN_CONFIG" --run-id "$TRAIN_RUN_ID"

if [[ ! -f "$CKPT" ]]; then
  log "ERROR: expected checkpoint not found at $CKPT"
  exit 1
fi
log "Checkpoint ready: $CKPT"

# ===================== 2. PAKISTAN INFERENCE =====================
# Transfer inference: the trained model is applied to Pakistan composites.
hr; log "########## PAKISTAN DATA PREP ##########"; hr
run_stage "aoi (pakistan)"       ""                                                            0 -- $RUN aoi       -c "$PK"
run_stage "buildings (pakistan)" ">=500 m2 building ROIs"                                       1 -- $RUN buildings -c "$PK"
run_stage "composite (pakistan)" "Sentinel-2 mosaics over Pakistan — the SLOWEST step (hours)" 1 -- $RUN composite -c "$PK"
run_stage "labels (pakistan)"    "fresh OSM solar for the missing-in-OSM audit"                0 -- $RUN labels    -c "$PK"

hr; log "########## PAKISTAN TRANSFER INFERENCE ##########"; hr
run_stage "infer (pakistan)"       "model applied to every building ROI window — LONG"          1 -- $RUN infer       -c "$PK" --model-ckpt "$CKPT"
run_stage "postprocess (pakistan)" "polygonize + aggregate per building, build missing_in_osm"  0 -- $RUN postprocess -c "$PK"
run_stage "report (pakistan)"      "Folium HTML map"                                            0 -- $RUN report      -c "$PK"

hr
log "DONE — total wall time $(fmt_elapsed $(( SECONDS - pipeline_start )))."
log "  Trained model:    $CKPT"
log "  Germany metrics:  $RUN validate -c $TRAIN_CONFIG --run-id $TRAIN_RUN_ID"
log "  Pakistan outputs: data/pakistan_500/outputs/<run_id>/   (missing_in_osm.geojson + HTML map)"
log "  Full log:         $LOG"
hr
