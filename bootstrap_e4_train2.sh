#!/bin/bash
# E4 stage 2 bootstrap, second form: identical pipeline to
# bootstrap_e4_train.sh, plus a driver gate before the pipeline.
# The matched stack (torch 2.13, CUDA 13.0) needs host driver >= 580;
# a Community host with driver 550 (CUDA 12.4) hard-failed vLLM init
# on 2026-11-02. If the driver is too old, idle WITHOUT running the
# pipeline so the pod can be terminated for a few cents.
RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"
RES=/workspace/results
mkdir -p "$RES/logs" /workspace/work
cd /workspace/work

status() { echo "$(date -u +%H:%M:%S) $1" >> "$RES/status.txt"; echo "$1"; }
fetch() { curl -sL "$RAW/$1" -o "/workspace/work/$1"; }

nohup python3 -m http.server 8000 --directory "$RES" >/dev/null 2>&1 &
status "BOOTSTRAP_START"

nvidia-smi > "$RES/driver.txt" 2>&1 || true
DRV=$(grep -o "Driver Version: [0-9.]*" "$RES/driver.txt" | grep -o "[0-9]*" | head -1)
status "DRIVER_MAJOR ${DRV:-unknown}"

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
  for s in stage_setup.sh stage_probe_gpu.sh stage_data.sh stage_train_vec.sh; do
    run_stage "$s" || status "PIPELINE_CONTINUE_AFTER_FAILURE $s"
  done
  status "PIPELINE_DONE"
  touch "$RES/DONE"
else
  status "DRIVER_TOO_OLD idle, no pipeline"
fi

declare -A SEEN
while true; do
  sleep 60
  CMD=$(curl -sL "$RAW/cmd.txt" | tr -d '[:space:]')
  case "$CMD" in
    *:*)
      STAGE="${CMD%%:*}"; NONCE="${CMD##*:}"
      if [ -n "$STAGE" ] && [ "${SEEN[$STAGE]}" != "$NONCE" ]; then
        SEEN[$STAGE]="$NONCE"
        run_stage "$STAGE" || true
      fi
      ;;
  esac
done
