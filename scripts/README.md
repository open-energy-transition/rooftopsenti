The script: scripts/run_full_pipeline.sh

  It does the whole thing in one run:
  1. Train the ≥500 m² model — data prep (aoi → labels → composite → buildings → chips) for
  Germany, Switzerland, Netherlands, UK, New Zealand, then trains one pooled model (Germany
  = held-out test) with a deterministic id de5_500, and prints validation metrics.
  2. Pakistan transfer inference — preps Pakistan (AOI, building ROIs, composites, fresh OSM
  solar for the audit), runs the trained checkpoint over it with --model-ckpt,
  post-processes, and builds the HTML report.

  Key properties:
  - Resumable: every stage skips when its output is already fresh, so if it crashes you just
  re-run the same command.
  - Self-locating: cds to the repo root, so it works from anywhere.
  - Logged: tees everything to logs/full_pipeline_<timestamp>.log.
  - GPU-optional: set RUN="pixi run -e cuda rooftopsenti".

  How to run it (Windows + WSL)

  # 1. Open WSL (Ubuntu). Install pixi once if you don't have it:
  curl -fsSL https://pixi.sh/install.sh | bash
  exec $SHELL                      # reload so `pixi` is on PATH

  # 2. Go to the project (keep it on the Linux filesystem, e.g. ~/, NOT /mnt/c — much faster
  I/O)
  cd ~/rooftopsenti               # wherever you cloned it

  # 3. Build the locked environment (one-time, downloads GDAL/PyTorch/etc.)
  pixi install

  # 4. Run the whole pipeline. It's long, so run it detached and watch the log:
  nohup bash scripts/run_full_pipeline.sh &
  tail -f logs/full_pipeline_*.log

  GPU box instead (NVIDIA driver on Windows, CUDA in WSL):
  RUN="pixi run -e cuda rooftopsenti" nohup bash scripts/run_full_pipeline.sh &

  When it finishes, the deliverable is
  data/pakistan_500/outputs/<run Jump to bottom (ctrl+End) ↓ lus an HTML map (report), and

