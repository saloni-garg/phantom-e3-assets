#!/bin/bash
# Self-driving E3 bootstrap. Runs as the container start command.
# Serves /workspace/results on :8000 (status.txt, logs, artifacts),
# runs the pipeline stages in order, then idles in a command loop:
# a file cmd.txt in the assets repo containing "<stage>:<nonce>" runs
# that stage script once per nonce. Stage scripts are re-fetched from
# the repo at every execution, so fixes can be pushed mid-flight.
RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"
RES=/workspace/results
mkdir -p "$RES/logs" /workspace/work
cd /workspace/work

status() { echo "$(date -u +%H:%M:%S) $1" >> "$RES/status.txt"; echo "$1"; }
fetch() { curl -sL "$RAW/$1" -o "/workspace/work/$1"; }

nohup python3 -m http.server 8000 --directory "$RES" >/dev/null 2>&1 &
status "BOOTSTRAP_START"

run_stage() {
  local stage="$1"
  fetch "$stage"
  status "STAGE_START $stage"
  bash "/workspace/work/$stage" > "$RES/logs/$stage.log" 2>&1
  local rc=$?
  status "STAGE_END $stage rc=$rc"
  return $rc
}

for s in stage_setup.sh stage_probe.sh stage_data.sh stage_train.sh stage_posthoc.sh; do
  run_stage "$s" || status "PIPELINE_CONTINUE_AFTER_FAILURE $s"
done
status "PIPELINE_DONE"
touch "$RES/DONE"

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
