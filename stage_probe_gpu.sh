#!/bin/bash
# E4 stage 1: GPU estimator probe (amendment 7). Runs the pinned tree's
# stock advantage functions on CPU and CUDA tensors via probe_gpu.py.
set -x
cd /workspace/work/verl-e3
PY=/workspace/venv-e3/bin/python
RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"
curl -sf "$RAW/probe_gpu.py" -o /workspace/work/probe_gpu.py || exit 1
curl -sf "$RAW/instrument_patch.py" -o /workspace/work/instrument_patch.py || exit 1
curl -sf "$RAW/instrument_patch2.py" -o /workspace/work/instrument_patch2.py || exit 1

# The tree must hold the stock variant (the post-setup state); enforce it.
cp /workspace/work/core_algos_stock.py verl/trainer/ppo/core_algos.py
cp /workspace/work/groupwise_stock.py verl/utils/groupwise.py

# Registry fix (idempotent): vectorized calls resolved through
# ADV_ESTIMATOR_REGISTRY must reach the logging wrappers. Refresh the
# stock snapshot afterwards so a later train stage inherits the fix.
PYTHONPATH=/workspace/work $PY /workspace/work/instrument_patch2.py verl/trainer/ppo/core_algos.py || exit 1
cp verl/trainer/ppo/core_algos.py /workspace/work/core_algos_stock.py

export E4_PROBE_OUT=/workspace/results/probe_gpu.json
$PY /workspace/work/probe_gpu.py 2>/dev/null | tee /workspace/results/probe_gpu.txt
echo PROBE_GPU_DONE
