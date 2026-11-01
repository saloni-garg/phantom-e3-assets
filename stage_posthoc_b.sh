#!/bin/bash
set -x
cd /workspace/work
RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"
for f in posthoc2.py reward_fn.py; do curl -sL "$RAW/$f" -o "/workspace/work/$f"; done
export PATH="/workspace/venv-e3/bin:$PATH"
E3_ARMS="stock-b,fp64-b" E3_OUT="posthoc_b.json" VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 /workspace/venv-e3/bin/python posthoc2.py || exit 1
echo POSTHOC_B_OK
