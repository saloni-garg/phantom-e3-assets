#!/bin/bash
set -x
cd /workspace/work
RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"
for f in posthoc.py reward_fn.py; do curl -sL "$RAW/$f" -o "/workspace/work/$f"; done
VLLM_WORKER_MULTIPROC_METHOD=spawn /workspace/venv-e3/bin/python posthoc.py || exit 1
echo POSTHOC_OK
