#!/bin/bash
set -x
cd /workspace/work
HF_HUB_ENABLE_HF_TRANSFER=0 /workspace/venv-e3/bin/python make_mix.py || exit 1
ls -la /workspace/data/
echo DATA_OK
