#!/bin/bash
# Coverage check (amendment 12 step 3): identical to
# stage_train_diag.sh (amendment 11, 30 steps, instrumentation v3)
# except reward_fn_d.py (unconditional 0.3 + 0.7 * correctness)
# replaces reward_fn_c.py. This is a coverage measurement, not a
# training run: the readout is whether uniform 0.3 groups occur in
# the logged tensor values, and the phantom share on those groups.
set -x
PY=/workspace/venv-e3/bin/python
export PATH="/workspace/venv-e3/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
cd /workspace/work/verl-e3
mkdir -p /workspace/results/advlogs /workspace/results/tensors
RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"

git checkout -- verl/trainer/ppo/core_algos.py verl/utils/groupwise.py
curl -sf "$RAW/instrument_patch_diag.py" -o /workspace/work/instrument_patch_diag.py || exit 1
curl -sf "$RAW/reward_fn_d.py" -o /workspace/work/reward_fn_d.py || exit 1
PYTHONPATH=/workspace/work $PY /workspace/work/instrument_patch_diag.py verl/trainer/ppo/core_algos.py || exit 1

export E3_ADV_LOG="/workspace/results/advlogs/coverage-s1.csv"
export E3_DUMP_DIR="/workspace/results/tensors"
rm -f "$E3_ADV_LOG"
echo "call,fn,groups,uniform_groups,sum_abs_adv_uniform,sum_abs_adv_total,uniform_values" > "$E3_ADV_LOG"
date -u +"TRAIN_START coverage s1 %Y-%m-%dT%H:%M:%SZ"
$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo_vectorized \
  algorithm.use_kl_in_reward=False \
  data.train_files=/workspace/data/mix_train.parquet \
  data.val_files=/workspace/data/heldout.parquet \
  data.train_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.seed=1 \
  custom_reward_function.path=/workspace/work/reward_fn_d.py \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  trainer.logger='["console"]' \
  trainer.project_name=phantom-e3 \
  trainer.experiment_name="coverage-s1" \
  trainer.default_local_dir="/workspace/results/ckpt/coverage-s1" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.test_freq=1000 \
  trainer.save_freq=1000 \
  trainer.total_training_steps=30 \
  trainer.resume_mode=disable
rc=$?
date -u +"TRAIN_END coverage s1 rc=${rc} %Y-%m-%dT%H:%M:%SZ"
echo TRAIN_DONE
