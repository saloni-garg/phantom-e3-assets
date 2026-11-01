#!/bin/bash
set -x
cd /workspace/work
RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"
for f in make_mix.py reward_fn.py; do curl -sL "$RAW/$f" -o "/workspace/work/$f"; done
HF_HUB_ENABLE_HF_TRANSFER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /workspace/venv-e3/bin/python make_mix.py || exit 1
ls -la /workspace/data/
echo DATA_OK
