"""Amendment 15 probe: CUDA phase-map completion plus reduction ablation.

Registered design: PAPER_PLAN.md amendment 15 (2026-11-04). Execution
was separately authorized by Saloni ("let's launch?", 2026-11-04
23:47 PT). Estimator level only; no training.

Question 1 (map): on CUDA tensors, for which (group size, reward)
cells does verl's stock vectorized GRPO estimator return nonzero
advantages on uniform groups? Grid: group sizes 2 to 64 (all),
rewards 0.1 to 0.9 in steps of 0.1, 20 repetitions per cell (CUDA
index_add_ accumulation order is nondeterministic, so one call
proves nothing). Construction matches the E4 probe exactly: a single
uniform group per call, token_level_rewards (g, 4) zeros with the
last column set to the reward, mask all ones. The E4 anchor cells
(sizes 6 to 16 at rewards 0.1, 0.2, 0.3, 0.6, 0.7) are inside this
grid, and the batched anchor cell (64 groups of 8 at 0.3, stock
vectorized) is run unchanged.

Question 2 (ablation): three variants of the vectorized estimator's
grouped statistics on identical tensors, everything else fixed
(device, dtype float32, inputs, estimator formula):
  stock   : the pinned two-pass index_add_ group_mean_std.
  looped  : per-group slice mean/std (torch mean over each group's
            slice, centered sum of squares over the slice), the
            accumulation the scalar path effectively uses.
  fp64acc : the stock index_add_ structure with float64
            accumulators, results cast back to float32
            (dtype-only change).
Plus scalar_ref: the stock scalar estimator on the same grid, the
reference pattern for registered prediction (b).

Registered predictions (amendment 15): (a) stock reproduces the E4
probe values at overlapping cells; (b) the looped pattern tracks
the scalar pattern and differs from stock; if looped instead
reproduces the stock pattern broadly, the reduction-order
explanation fails; (c) fp64acc is exactly zero in every cell;
(d) the completed stock map is descriptive.

Output: JSON at $A15_OUT (default /workspace/results/probe_cuda_map.json)
plus summary lines on stdout. CPU pre-check mode: --device cpu
--sizes 6,8,12,16 --rewards 0.1,0.3 --reps 3.
No em or en dashes in this file.
"""
import argparse
import json
import os

import numpy as np
import torch

import verl.trainer.ppo.core_algos as ca
from verl.utils.groupwise import group_mean_std as stock_gms

scalar_fn = ca.compute_grpo_outcome_advantage
vectorized_fn = ca.compute_grpo_vectorized_outcome_advantage


def group_mean_std_looped(scores, gidx, eps=1e-6, device=None):
    target = scores.device if device is None else torch.device(device)
    scores = scores.reshape(-1).to(device=target, dtype=torch.float32)
    gidx = gidx.reshape(-1).to(device=target, dtype=torch.long)
    G = int(torch.max(gidx).item()) + 1 if gidx.numel() > 0 else 0
    mean = torch.zeros(G, device=target, dtype=torch.float32)
    std = torch.zeros(G, device=target, dtype=torch.float32)
    count = torch.zeros(G, device=target, dtype=torch.float32)
    for g in range(G):
        sel = scores[gidx == g]
        c = float(sel.numel())
        count[g] = c
        if c <= 1.0:
            mean[g] = 0.0
            std[g] = 1.0
            continue
        m = sel.mean()
        centered = sel - m
        var = (centered * centered).sum() / max(c - 1.0, 1.0)
        mean[g] = m
        std[g] = torch.sqrt(torch.clamp(var, min=float(eps)))
    return mean, std, count


def group_mean_std_fp64acc(scores, gidx, eps=1e-6, device=None):
    target = scores.device if device is None else torch.device(device)
    scores32 = scores.reshape(-1).to(device=target, dtype=torch.float32)
    gidx = gidx.reshape(-1).to(device=target, dtype=torch.long)
    G = int(torch.max(gidx).item()) + 1 if gidx.numel() > 0 else 0
    s = scores32.to(torch.float64)
    ones = torch.ones_like(s, dtype=torch.float64)
    count = torch.zeros(G, device=target, dtype=torch.float64).index_add_(0, gidx, ones)
    s1 = torch.zeros(G, device=target, dtype=torch.float64).index_add_(0, gidx, s)
    mean = s1 / count.clamp_min(1.0)
    centered = s - mean[gidx]
    var_num = torch.zeros(G, device=target, dtype=torch.float64).index_add_(0, gidx, centered * centered)
    denom = (count - 1.0).clamp_min(1.0)
    var = var_num / denom
    std = torch.sqrt(torch.clamp(var, min=float(eps)))
    single = count <= 1.0
    if torch.any(single):
        mean = mean.clone()
        std = std.clone()
        mean[single] = 0.0
        std[single] = 1.0
    return mean.to(torch.float32), std.to(torch.float32), count.to(torch.float32)


VARIANTS = {
    "stock": stock_gms,
    "looped": group_mean_std_looped,
    "fp64acc": group_mean_std_fp64acc,
}


def call(fn, value, n_rows, n_groups, device):
    per = n_rows // n_groups
    rewards = torch.zeros((n_rows, 4), dtype=torch.float32, device=device)
    rewards[:, -1] = value
    mask = torch.ones((n_rows, 4), dtype=torch.float32, device=device)
    index = np.repeat(np.arange(n_groups), per)
    adv, _ = fn(token_level_rewards=rewards, response_mask=mask, index=index)
    return float(adv.abs().max())


def cell(fn, value, g, device, reps):
    vals = [call(fn, value, g, 1, device) for _ in range(reps)]
    return {
        "max_abs_adv": max(vals),
        "reps_nonzero": sum(1 for v in vals if v > 0),
        "reps_material": sum(1 for v in vals if v >= 1e-2),
        "reps": reps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sizes", default=",".join(str(s) for s in range(2, 65)))
    ap.add_argument("--rewards", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=os.environ.get("A15_OUT", "/workspace/results/probe_cuda_map.json"))
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    rewards = [float(r) for r in args.rewards.split(",")]
    device = args.device
    out = {"device": device, "torch": torch.__version__, "cells": {}, "batched_anchor": {}}
    if device == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)

    series = {}
    for vname, gms in VARIANTS.items():
        ca.group_mean_std = gms
        series[vname] = vectorized_fn
        for v in rewards:
            for g in sizes:
                r = cell(vectorized_fn, v, g, device, args.reps)
                key = f"{vname}/{device}/v={v}/g={g}"
                out["cells"][key] = r
        print(f"variant {vname} done", flush=True)
    ca.group_mean_std = stock_gms
    for v in rewards:
        for g in sizes:
            r = cell(scalar_fn, v, g, device, args.reps)
            out["cells"][f"scalar_ref/{device}/v={v}/g={g}"] = r
    print("scalar_ref done", flush=True)
    # batched anchor, stock vectorized: 64 groups of 8 at 0.3
    vals = [call(vectorized_fn, 0.3, 64 * 8, 64, device) for _ in range(args.reps)]
    out["batched_anchor"] = {
        "max_abs_adv": max(vals),
        "reps_nonzero": sum(1 for x in vals if x > 0),
        "reps_material": sum(1 for x in vals if x >= 1e-2),
        "reps": args.reps,
    }

    def pattern(series_name):
        p = {}
        for v in rewards:
            for g in sizes:
                c = out["cells"][f"{series_name}/{device}/v={v}/g={g}"]
                p[(v, g)] = c["reps_nonzero"] > 0
        return p

    pats = {name: pattern(name) for name in ("stock", "looped", "fp64acc", "scalar_ref")}
    total = len(rewards) * len(sizes)

    def agree(a, b):
        return sum(1 for k in pats[a] if pats[a][k] == pats[b][k]) / total

    out["pattern_agreement"] = {
        "looped_vs_scalar": agree("looped", "scalar_ref"),
        "stock_vs_scalar": agree("stock", "scalar_ref"),
        "looped_vs_stock": agree("looped", "stock"),
        "cells": total,
        "fp64acc_nonzero_cells": sum(1 for k in pats["fp64acc"] if pats["fp64acc"][k]),
        "stock_nonzero_cells": sum(1 for k in pats["stock"] if pats["stock"][k]),
        "looped_nonzero_cells": sum(1 for k in pats["looped"] if pats["looped"][k]),
        "scalar_nonzero_cells": sum(1 for k in pats["scalar_ref"] if pats["scalar_ref"][k]),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["pattern_agreement"], indent=1))
    print(f"batched anchor: {out['batched_anchor']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
