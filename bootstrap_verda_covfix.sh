#!/bin/bash
# Coverage check driver for Verda (amendment 12 step 3 rerun, wiring fix). Pipeline:
# setup, estimator probe, data, then the 30-step coverage run
# (stage_train_covfix.sh: amendment 11 shape, instrumentation v3,
# reward_fn_d). Self-driving pattern: status in
# /workspace/results/status.txt, results served on port 8000, DONE
# marker at the end. Driver gate first (needs host driver >= 580).
# The stage scripts need Python 3.12 (verl pinned requires <3.13)
# while the Verda Ubuntu 26.04 image ships 3.14 as system python3,
# so a uv-provided 3.12 is shimmed onto PATH before any stage runs.
#
# Hardened provisioning (2026-11-05, carried over from the
# amendment 15 attempt 4 bootstrap): the uv binary is fetched from
# GitHub releases, every provisioning step is logged, and the 3.12
# interpreter is verified (version check plus a real venv smoke
# test) before any stage runs. If provisioning fails the bootstrap
# writes PROVISION_FAILED to status.txt and idles without running
# the stages, so the failure stays visible over HTTP.
RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"
UV_VERSION="0.13.7"
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

UVLOG="$RES/logs/uv_install.log"
: > "$UVLOG"
if ! /workspace/bin/python3 -V 2>/dev/null | grep -q "3\.12"; then
  if [ ! -x /workspace/bin/uv ]; then
    echo "downloading uv $UV_VERSION from github releases" >> "$UVLOG"
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" -o /tmp/uv.tar.gz >> "$UVLOG" 2>&1
    tar -xzf /tmp/uv.tar.gz --strip-components=1 --no-same-owner -C /workspace/bin >> "$UVLOG" 2>&1 || echo "uv tarball extract failed" >> "$UVLOG"
  fi
  if /workspace/bin/uv --version >> "$UVLOG" 2>&1; then
    /workspace/bin/uv python install 3.12 >> "$UVLOG" 2>&1 || echo "uv python install failed" >> "$UVLOG"
    PY312=$(/workspace/bin/uv python find 3.12 2>/dev/null || true)
    echo "uv python find 3.12: ${PY312:-not found}" >> "$UVLOG"
    if [ -n "$PY312" ]; then
      ln -sf "$PY312" /workspace/bin/python3
    fi
  else
    echo "uv binary missing or not runnable" >> "$UVLOG"
  fi
fi
export PATH="/workspace/bin:$PATH"
status "PYTHON_SHIM $(/workspace/bin/python3 -V 2>&1 || python3 -V 2>&1)"

provision_ok=0
if /workspace/bin/python3 -V 2>/dev/null | grep -q "3\.12"; then
  if /workspace/bin/python3 -m venv /tmp/venv_smoke >> "$UVLOG" 2>&1; then
    rm -rf /tmp/venv_smoke
    provision_ok=1
  else
    echo "venv smoke test failed" >> "$UVLOG"
  fi
fi

run_stage() {
  local stage="$1"
  fetch "$stage"
  status "STAGE_START $stage"
  bash "/workspace/work/$stage" > "$RES/logs/$stage.log" 2>&1
  local rc=$?
  status "STAGE_END $stage rc=$rc"
  return $rc
}

if [ "$provision_ok" -ne 1 ]; then
  status "PROVISION_FAILED python 3.12 shim not verified, stages NOT run"
elif [ "${DRV:-0}" -ge 580 ]; then
  status "PROVISION_OK"
  for s in stage_setup.sh stage_probe_gpu.sh stage_data.sh stage_train_covfix.sh; do
    run_stage "$s" || status "PIPELINE_CONTINUE_AFTER_FAILURE $s"
  done
  status "PIPELINE_DONE"
  touch "$RES/DONE"
else
  status "DRIVER_TOO_OLD idle, no pipeline"
fi

while true; do sleep 300; done
