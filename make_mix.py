"""Build the E3 task mix from GSM8K (design frozen in RUNBOOK.md).

1. Run verl's own gsm8k preprocessing to get train/test parquets.
2. Profile every train prompt with the base policy (vLLM, n=8, temp 1.0)
   using the E3 reward (1.0 / 0.1 / 0.0).
3. Classify by pass rate: easy = 8/8 correct, hard = 0/8, middle = else.
4. Mix pool easy:middle:hard = 40:20:40 (640:320:640 target, seed 2027),
   written in verl parquet schema to /workspace/data/mix_train.parquet.
   Held-out: 500 test prompts to /workspace/data/heldout.parquet.
   Probe set: 128 mix prompts (text) to /workspace/data/probe_prompts.json.
"""
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, "/workspace/work")
from reward_fn import compute_score  # noqa: E402

DATA = "/workspace/data"
os.makedirs(DATA, exist_ok=True)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

subprocess.run(
    ["/workspace/venv-e3/bin/python", "examples/data_preprocess/gsm8k.py",
     "--local_save_dir", DATA],
    cwd="/workspace/work/verl-e3", check=True,
)

import pandas as pd  # noqa: E402

train = pd.read_parquet(f"{DATA}/train.parquet")
test = pd.read_parquet(f"{DATA}/test.parquet")
heldout = test.sample(n=500, random_state=2027)
heldout.to_parquet(f"{DATA}/heldout.parquet", index=False)

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
prompts = []
for msgs in train["prompt"]:
    prompts.append(tok.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True))
gts = [r["reward_model"]["ground_truth"] for r in train.to_dict("records")]

llm = LLM(model=MODEL, gpu_memory_utilization=0.85, max_model_len=1024, seed=2027)
sp = SamplingParams(n=8, temperature=1.0, top_p=1.0, max_tokens=512)
outs = llm.generate(prompts, sp)

rates = []
for out, gt in zip(outs, gts):
    hits = sum(1 for o in out.outputs if compute_score("openai/gsm8k", o.text, gt) >= 1.0)
    rates.append(hits / 8.0)
train["pass_rate"] = rates

easy = train[train.pass_rate >= 1.0]
hard = train[train.pass_rate <= 0.0]
middle = train[(train.pass_rate > 0.0) & (train.pass_rate < 1.0)]
print(f"buckets: easy={len(easy)} middle={len(middle)} hard={len(hard)}")


def take(df, n):
    return df.sample(n=min(n, len(df)), random_state=2027)


mix = pd.concat([take(easy, 640), take(middle, 320), take(hard, 640)])
mix = mix.sample(frac=1.0, random_state=2027).reset_index(drop=True)
mix.drop(columns=["pass_rate"]).to_parquet(f"{DATA}/mix_train.parquet", index=False)
counts = {
    "easy_avail": int(len(easy)), "middle_avail": int(len(middle)), "hard_avail": int(len(hard)),
    "mix_rows": int(len(mix)),
    "mix_easy": int((mix.pass_rate >= 1.0).sum()),
    "mix_middle": int(((mix.pass_rate > 0.0) & (mix.pass_rate < 1.0)).sum()),
    "mix_hard": int((mix.pass_rate <= 0.0).sum()),
}
probe_texts = []
for msgs in mix["prompt"].head(128):
    probe_texts.append(tok.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True))
json.dump(probe_texts, open(f"{DATA}/probe_prompts.json", "w"))
json.dump(counts, open("/workspace/results/mix_counts.json", "w"), indent=2)
print(json.dumps(counts, indent=2))
print("MIX_DONE")
