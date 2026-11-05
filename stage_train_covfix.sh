#!/bin/bash
# Coverage check rerun (amendment 12 step 3, authorized
# 2026-11-05): identical to stage_train_cov.sh except the
# custom reward override is passed as
# reward.custom_reward_function.path / .name, which is the
# config path verl's get_custom_reward_fn actually reads
# (verl/trainer/ppo/reward.py). The first run passed
# custom_reward_function.path at the config root; the workers
# saw path None and fell back to the built-in binary scorer, so
# reward_fn_d never ran. Guard: after step 1, the watchdog
# reads critic/rewards/min from the trainer log; under
# reward_fn_d every reward is 0.3 or 1.0, so the minimum must
# exceed 0.29. If it does not, the trainer is killed, the stage
# fails with rc=97, and GUARD_FAIL.txt is written, so a second
# wiring failure is loud and cheap instead of silent.
set -x
PY=/workspace/venv-e3/bin/python
export PATH="/workspace/venv-e3/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
cd /workspace/work/verl-e3
mkdir -p /workspace/results/advlogs /workspace/results/tensors /workspace/results/logs
RAW="https://raw.githubusercontent.com/saloni-garg/phantom-e3-assets/main"

git checkout -- verl/trainer/ppo/core_algos.py verl/utils/groupwise.py
curl -sf "$RAW/instrument_patch_diag.py" -o /workspace/work/instrument_patch_diag.py || exit 1
curl -sf "$RAW/reward_fn_d.py" -o /workspace/work/reward_fn_d.py || exit 1
PYTHONPATH=/workspace/work $PY /workspace/work/instrument_patch_diag.py verl/trainer/ppo/core_algos.py || exit 1

export E3_ADV_LOG="/workspace/results/advlogs/covfix-s1.csv"
export E3_DUMP_DIR="/workspace/results/tensors"
rm -f "$E3_ADV_LOG"
echo "call,fn,groups,uniform_groups,sum_abs_adv_uniform,sum_abs_adv_total,uniform_values" > "$E3_ADV_LOG"
date -u +"TRAIN_START covfix s1 %Y-%m-%dT%H:%M:%SZ"
TRAINER_LOG=/workspace/results/logs/trainer_covfix.log
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
  reward.custom_reward_function.path=/workspace/work/reward_fn_d.py \
  reward.custom_reward_function.name=compute_score \
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
  trainer.experiment_name="covfix-s1" \
  trainer.default_local_dir="/workspace/results/ckpt/covfix-s1" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.test_freq=1000 \
  trainer.save_freq=1000 \
  trainer.total_training_steps=30 \
  trainer.resume_mode=disable \
  > "$TRAINER_LOG" 2>&1 &
TPID=$!
guard_fail=0
guard_note="not_checked"
while kill -0 "$TPID" 2>/dev/null; do
  if grep -q "step:1 " "$TRAINER_LOG" 2>/dev/null; then
    min=$(grep "step:1 " "$TRAINER_LOG" | head -1 | grep -o "critic/rewards/min:[0-9.]*" | head -1 | cut -d: -f2)
    if [ -n "$min" ]; then
      if awk -v m="$min" 'BEGIN{exit !(m > 0.29)}'; then
        guard_note="pass min=$min"
        echo "GUARD_PASS critic_rewards_min_step1=$min"
      else
        guard_fail=1
        guard_note="fail min=$min"
        echo "GUARD_FAIL critic_rewards_min_step1=$min"
        echo "step1 critic/rewards/min=$min (expected above 0.29 under reward_fn_d)" > /workspace/results/GUARD_FAIL.txt
        kill "$TPID" 2>/dev/null
        sleep 5
        pkill -f "verl.trainer.main_ppo" 2>/dev/null
      fi
      break
    fi
  fi
  sleep 15
done
wait "$TPID"
rc=$?
echo "GUARD_RESULT $guard_note"
if [ "$guard_fail" -eq 1 ]; then
  rc=97
fi
date -u +"TRAIN_END covfix s1 rc=${rc} %Y-%m-%dT%H:%M:%SZ"
echo TRAIN_DONE
exit $rc
