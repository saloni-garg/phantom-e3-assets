#!/bin/bash
# Amendment 15 probe stage: stock verl checkout (pinned), lean venv,
# run probe_cuda_map.py on CUDA with the full registered grid.
set -x
cd /workspace/work
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -V

if [ ! -d verl-e3 ]; then
  git clone https://github.com/volcengine/verl.git verl-e3 || exit 1
fi
cd verl-e3
git checkout a4b9354968705a441fb7da7367924b15b902e771 || exit 1
git checkout -- .
git rev-parse HEAD
git status --porcelain

RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"
curl -sL "$RAW/probe_cuda_map.py" -o /workspace/work/probe_cuda_map.py || exit 1

if [ ! -d /workspace/venv-e3 ]; then
  python3 -m venv --system-site-packages /workspace/venv-e3 || exit 1
fi
/workspace/venv-e3/bin/pip install -q -e . || exit 1
/workspace/venv-e3/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1
/workspace/venv-e3/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" || exit 1

cd /workspace/work
A15_OUT=/workspace/results/probe_cuda_map.json /workspace/venv-e3/bin/python probe_cuda_map.py --device cuda || exit 1
echo PROBE_MAP_OK
