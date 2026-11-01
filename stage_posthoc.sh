#!/bin/bash
set -x
cd /workspace/work
VLLM_WORKER_MULTIPROC_METHOD=spawn /workspace/venv-e3/bin/python posthoc.py || exit 1
echo POSTHOC_OK
