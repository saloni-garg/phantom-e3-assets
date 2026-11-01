#!/bin/bash
set -x
cd /workspace/work/verl-e3
PY=/workspace/venv-e3/bin/python

echo "=== STOCK ==="
cp /workspace/work/core_algos_stock.py verl/trainer/ppo/core_algos.py
cp /workspace/work/groupwise_stock.py verl/utils/groupwise.py
$PY /workspace/work/probe_estimator.py 2>/dev/null | tee /workspace/results/probe_stock.txt

echo "=== FP64 ==="
cp /workspace/work/core_algos_fp64.py verl/trainer/ppo/core_algos.py
cp /workspace/work/groupwise_fp64.py verl/utils/groupwise.py
$PY /workspace/work/probe_estimator.py 2>/dev/null | tee /workspace/results/probe_fp64.txt

cp /workspace/work/core_algos_stock.py verl/trainer/ppo/core_algos.py
cp /workspace/work/groupwise_stock.py verl/utils/groupwise.py
echo PROBE_DONE
