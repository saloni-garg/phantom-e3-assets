"""E1 probe against the verl tree in the current working directory.

Imports the tree's own GRPO advantage functions and calls them on uniform
fractional-reward groups (the E3 toggle verification from RUNBOOK.md):
stock must reproduce the audit values, the float64-patched tree must
return exactly zero.
"""
import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
)

for name, fn in (("scalar", compute_grpo_outcome_advantage),
                 ("vectorized", compute_grpo_vectorized_outcome_advantage)):
    worst = 0.0
    for v in (0.1, 0.3, 0.6, 0.7):
        for n in (8, 16):
            rewards = torch.zeros((n, 4), dtype=torch.float32)
            rewards[:, -1] = v
            mask = torch.ones((n, 4), dtype=torch.float32)
            adv, _ = fn(rewards, mask, np.zeros(n, dtype=int))
            m = float(adv.abs().max())
            worst = max(worst, m)
            print(f"{name} reward={v} n={n}: max|A| = {m:.4f}")
    print(f"{name} WORST: {worst:.4f}")
