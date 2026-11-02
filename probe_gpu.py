"""E4 probe: do verl's stock GRPO advantage functions leave a phantom
residue on GPU tensors, the way they do on CPU?

Runs the pinned tree's own stock functions (the _e3orig_ originals when
the tree is instrumented, else the plain names) on uniform fractional
reward groups across devices, paths, rewards, and group sizes, with
repetitions (CUDA index_add_ accumulation order is nondeterministic,
so one call proves nothing). Also runs a batched cell (64 groups of 8
at reward 0.3) that mimics a training step's shape.

Sanity anchors (CPU, pinned commit, matching the E3 on-pod toggle):
scalar at reward 0.3 must show max|A| = 0.0289, scalar worst on the
audit grid (rewards 0.1/0.3/0.6/0.7, sizes 8/16) must be 0.0561, and
vectorized worst on that grid must be 0.1061. If the anchors fail,
the tree state is wrong and the GPU numbers are meaningless. (The
E1 audit's -0.3482 vectorized figure came from an earlier verl
snapshot whose grouped-statistics implementation differed; it is
not an anchor for the pinned code.)

Output: JSON at $E4_PROBE_OUT (default /workspace/results/probe_gpu.json)
plus one summary line per cell on stdout.
"""
import json
import os

import numpy as np
import torch

import verl.trainer.ppo.core_algos as ca

scalar_fn = getattr(ca, "_e3orig_compute_grpo_outcome_advantage",
                    ca.compute_grpo_outcome_advantage)
vectorized_fn = getattr(ca, "_e3orig_compute_grpo_vectorized_outcome_advantage",
                        ca.compute_grpo_vectorized_outcome_advantage)

REWARDS = (0.1, 0.2, 0.3, 0.6, 0.7)
GROUP_SIZES = (6, 8, 10, 12, 16)
REPS = 20


def call(fn, value, n_rows, n_groups, device):
    per = n_rows // n_groups
    rewards = torch.zeros((n_rows, 4), dtype=torch.float32, device=device)
    rewards[:, -1] = value
    mask = torch.ones((n_rows, 4), dtype=torch.float32, device=device)
    index = np.repeat(np.arange(n_groups), per)
    adv, _ = fn(token_level_rewards=rewards, response_mask=mask, index=index)
    return float(adv.abs().max())


def cell(fn, value, g, device):
    vals = [call(fn, value, g, 1, device) for _ in range(REPS)]
    return {
        "max_abs_adv": max(vals),
        "reps_nonzero": sum(1 for v in vals if v > 0),
        "reps_material": sum(1 for v in vals if v >= 1e-2),
        "reps": REPS,
    }


def main():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    out = {"devices": devices, "cells": {}, "batched": {}}
    for name, fn in (("scalar", scalar_fn), ("vectorized", vectorized_fn)):
        for device in devices:
            for v in REWARDS:
                for g in GROUP_SIZES:
                    r = cell(fn, v, g, device)
                    key = f"{name}/{device}/v={v}/g={g}"
                    out["cells"][key] = r
                    print(f"{key}: max|A|={r['max_abs_adv']:.4f} "
                          f"nonzero {r['reps_nonzero']}/{REPS} "
                          f"material {r['reps_material']}/{REPS}", flush=True)
            b = [call(fn, 0.3, 512, 64, device) for _ in range(REPS)]
            out["batched"][f"{name}/{device}/v=0.3/64x8"] = {
                "max_abs_adv": max(b),
                "reps_nonzero": sum(1 for x in b if x > 0),
                "reps_material": sum(1 for x in b if x >= 1e-2),
                "reps": REPS,
            }
            print(f"{name}/{device} batched 64x8 v=0.3: max|A|={max(b):.4f} "
                  f"nonzero {out['batched'][f'{name}/{device}/v=0.3/64x8']['reps_nonzero']}/{REPS}",
                  flush=True)
    path = os.environ.get("E4_PROBE_OUT", "/workspace/results/probe_gpu.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
