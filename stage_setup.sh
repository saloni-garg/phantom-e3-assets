#!/bin/bash
set -x
cd /workspace/work
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -V

if [ ! -d verl-e3 ]; then
  git clone https://github.com/volcengine/verl.git verl-e3 || exit 1
fi
cd verl-e3
git checkout a4b9354968705a441fb7da7367924b15b902e771 || exit 1
git rev-parse HEAD

RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"
for f in apply_patch_fp64.py probe_estimator.py instrument_patch.py reward_fn.py make_mix.py posthoc.py; do
  curl -sL "$RAW/$f" -o "/workspace/work/$f"
done

git checkout -- verl/trainer/ppo/core_algos.py verl/utils/groupwise.py
cp verl/trainer/ppo/core_algos.py /workspace/work/core_algos_stock.py
cp verl/utils/groupwise.py /workspace/work/groupwise_stock.py
python3 /workspace/work/apply_patch_fp64.py verl/trainer/ppo/core_algos.py verl/utils/groupwise.py || exit 1
python3 /workspace/work/instrument_patch.py verl/trainer/ppo/core_algos.py || exit 1
cp verl/trainer/ppo/core_algos.py /workspace/work/core_algos_fp64.py
cp verl/utils/groupwise.py /workspace/work/groupwise_fp64.py
cp /workspace/work/core_algos_stock.py verl/trainer/ppo/core_algos.py
cp /workspace/work/groupwise_stock.py verl/utils/groupwise.py
python3 /workspace/work/instrument_patch.py verl/trainer/ppo/core_algos.py || exit 1
cp verl/trainer/ppo/core_algos.py /workspace/work/core_algos_stock.py

if [ ! -d /workspace/venv-e3 ]; then
  python3 -m venv --system-site-packages /workspace/venv-e3 || exit 1
fi
/workspace/venv-e3/bin/pip install -q -e . || exit 1
/workspace/venv-e3/bin/pip install -q "vllm==0.11.0" || exit 1
/workspace/venv-e3/bin/pip install -q "transformers==4.57.1" || exit 1
/workspace/venv-e3/bin/pip install -q "huggingface-hub==0.36.2" "datasets==3.6.0" || exit 1
/workspace/venv-e3/bin/pip install -q "transfer-queue" || exit 1
/workspace/venv-e3/bin/python -c "import verl, vllm, torch; print('verl ok, vllm', vllm.__version__, 'torch', torch.__version__)" || exit 1
echo SETUP_OK
