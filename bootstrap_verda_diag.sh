#!/bin/bash
# Diagnostic run driver for Verda (amendment 11). Self-driving pattern
# established for the amendment 10 run: results served on port 8000,
# status in /workspace/results/status.txt, DONE marker at the end.
# Driver gate first (stack needs host driver >= 580); if the driver
# is too old, idle WITHOUT running the pipeline. The stage scripts
# need Python 3.12 (verl pinned requires <3.13) while the Verda
# Ubuntu 26.04 image ships 3.14 as system python3, so a uv-provided
# 3.12 is shimmed onto PATH before any stage runs.
RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"
RES=/workspace/results
mkdir -p "$RES/logs" "$RES/advlogs" "$RES/tensors" /workspace/work /workspace/bin
cd /workspace/work

status() { echo "$(date -u +%H:%M:%S) $1" >> "$RES/status.txt"; echo "$1"; }
fetch() { curl -sL "$RAW/$1" -o "/workspace/work/$1"; }

nohup python3 -m http.server 8000 --directory "$RES" >/dev/null 2>&1 &
status "BOOTSTRAP_START"

nvidia-smi > "$RES/driver.txt" 2>&1 || true
DRV=$(grep -o "Driver Version: [0-9.]*" "$RES/driver.txt" | grep -o "[0-9]*" | head -1)
status "DRIVER_MAJOR ${DRV:-unknown}"

# Python 3.12 shim (idempotent).
if ! /workspace/bin/python3 -V 2>/dev/null | grep -q "3.12"; then
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv python install 3.12 >> "$RES/logs/uv_install.log" 2>&1 || true
  PY312=$(uv python find 3.12 2>/dev/null || true)
  if [ -n "$PY312" ]; then
    ln -sf "$PY312" /workspace/bin/python3
  fi
fi
export PATH="/workspace/bin:$PATH"
status "PYTHON_SHIM $(/workspace/bin/python3 -V 2>&1 || python3 -V 2>&1)"

run_stage() {
  local stage="$1"
  fetch "$stage"
  status "STAGE_START $stage"
  bash "/workspace/work/$stage" > "$RES/logs/$stage.log" 2>&1
  local rc=$?
  status "STAGE_END $stage rc=$rc"
  return $rc
}

if [ "${DRV:-0}" -ge 580 ]; then
  for s in stage_setup.sh stage_probe_gpu.sh stage_data.sh stage_train_diag.sh; do
    run_stage "$s" || status "PIPELINE_CONTINUE_AFTER_FAILURE $s"
  done
  status "PIPELINE_DONE"
  touch "$RES/DONE"
else
  status "DRIVER_TOO_OLD idle, no pipeline"
fi

while true; do sleep 300; done
