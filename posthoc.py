"""E3 post-hoc measurement (runs on the pod after training).

For each arm and each saved checkpoint step:
  - merge the FSDP checkpoint to HF format (verl model_merger),
  - mean per-token entropy of the checkpoint policy on fixed probe
    sequences (base-model greedy continuations of the 128 probe prompts),
  - mean per-token KL(checkpoint || base) on the same sequences.
Plus greedy held-out accuracy (500 prompts) for base and final
checkpoints via vLLM. Writes /workspace/results/posthoc.json.
"""
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, "/workspace/work")
from reward_fn import compute_score  # noqa: E402

RES = "/workspace/results"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PY = "/workspace/venv-e3/bin/python"
def main():
    probe_prompts = json.load(open("/workspace/data/probe_prompts.json"))

    import torch  # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    tok = AutoTokenizer.from_pretrained(MODEL)


    def load(path):
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).cuda().eval()
        return m


    base = load(MODEL)
    seqs = []
    with torch.no_grad():
        for p in probe_prompts:
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            out = base.generate(ids, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
            seqs.append(out[0])
    del base
    torch.cuda.empty_cache()


    def stats_for(path):
        m = load(path)
        b = load(MODEL)
        tot_kl = tot_ent = tot_tok = 0.0
        with torch.no_grad():
            for s in seqs:
                x = s.unsqueeze(0)
                lm = m(x).logits[0, :-1].float()
                lb = b(x).logits[0, :-1].float()
                pm = torch.log_softmax(lm, dim=-1)
                pb = torch.log_softmax(lb, dim=-1)
                kl = (pm.exp() * (pm - pb)).sum(dim=-1)
                ent = -(pm.exp() * pm).sum(dim=-1)
                tot_kl += float(kl.sum())
                tot_ent += float(ent.sum())
                tot_tok += kl.numel()
        del m, b
        torch.cuda.empty_cache()
        return {"kl_per_token": tot_kl / tot_tok, "entropy_per_token": tot_ent / tot_tok}


    def merge(ckpt_dir, target):
        if os.path.exists(os.path.join(target, "config.json")):
            return target
        actor = os.path.join(ckpt_dir, "actor")
        subprocess.run(
            [PY, "scripts/model_merger.py", "merge", "--backend", "fsdp",
             "--local_dir", actor, "--target_dir", target],
            cwd="/workspace/work/verl-e3", check=True,
        )
        return target


    def accuracy(path):
        import pandas as pd
        from vllm import LLM, SamplingParams
        held = pd.read_parquet("/workspace/data/heldout.parquet")
        prompts = [tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True)
                   for m in held["prompt"]]
        gts = [r["reward_model"]["ground_truth"] for r in held.to_dict("records")]
        llm = LLM(model=path, gpu_memory_utilization=0.85, max_model_len=1024, seed=2027)
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))
        hits = sum(1 for o, gt in zip(outs, gts)
                   if compute_score("openai/gsm8k", o.outputs[0].text, gt) >= 1.0)
        del llm
        torch.cuda.empty_cache()
        return hits / len(gts)


    results = {"base": stats_for(MODEL), "arms": {}}
    results["base"]["heldout_accuracy"] = accuracy(MODEL)
    for arm in ("stock", "fp64"):
        results["arms"][arm] = {}
        for ck in sorted(glob.glob(f"{RES}/ckpt/{arm}-s1/global_step_*"),
                         key=lambda p: int(p.rsplit("_", 1)[1])):
            step = int(ck.rsplit("_", 1)[1])
            merged = merge(ck, f"/workspace/work/merged/{arm}-{step}")
            results["arms"][arm][str(step)] = stats_for(merged)
            print(arm, step, results["arms"][arm][str(step)], flush=True)
        final = f"/workspace/work/merged/{arm}-250"
        results["arms"][arm]["heldout_accuracy"] = accuracy(final)

    json.dump(results, open(f"{RES}/posthoc.json", "w"), indent=2)
    print("POSTHOC_DONE")


if __name__ == "__main__":
    main()
