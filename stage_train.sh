#!/bin/bash
# Trains the pilot pair: stock seed 1, then fp64 seed 1 (RUNBOOK.md).
set -x
PY=/workspace/venv-e3/bin/python
cd /workspace/work/verl-e3
mkdir -p /workspace/results/advlogs

train_arm() {
  local ARM="$1" SEED="$2"
  if [ "$ARM" = "stock" ]; then cp /workspace/work/core_algos_stock.py verl/trainer/ppo/core_algos.py; cp /workspace/work/groupwise_stock.py verl/utils/groupwise.py;
  else cp /workspace/work/core_algos_fp64.py verl/trainer/ppo/core_algos.py; cp /workspace/work/groupwise_fp64.py verl/utils/groupwise.py; fi
  export E3_ADV_LOG="/workspace/results/advlogs/${ARM}-s${SEED}.csv"
  rm -f "$E3_ADV_LOG"
  echo "call,fn,groups,uniform_groups,sum_abs_adv_uniform,sum_abs_adv_total" > "$E3_ADV_LOG"
  date -u +"TRAIN_START ${ARM} s${SEED} %Y-%m-%dT%H:%M:%SZ"
  $PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=/workspace/data/mix_train.parquet \
    data.val_files=/workspace/data/heldout.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.seed=$SEED \
    custom_reward_function.path=/workspace/work/reward_fn.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    trainer.logger='["console"]' \
    trainer.project_name=phantom-e3 \
    trainer.experiment_name="${ARM}-s${SEED}" \
    trainer.default_local_dir="/workspace/results/ckpt/${ARM}-s${SEED}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=1000 \
    trainer.save_freq=50 \
    trainer.total_training_steps=250 \
    trainer.resume_mode=disable
  local rc=$?
  date -u +"TRAIN_END ${ARM} s${SEED} rc=${rc} %Y-%m-%dT%H:%M:%SZ"
}

train_arm stock 1
train_arm fp64 1
echo TRAIN_DONE
