"""Apply the float64 group-statistics patch to a stock verl core_algos.py.

This is the E3 patched arm: identical code to stock except that GRPO group
mean/std are computed in float64 in both the scalar and vectorized
estimators. Every replacement is exact and asserted; the script fails
loudly if the stock source does not match.

Usage: apply_patch_fp64.py <path-to-core_algos.py>
"""
import sys

SCALAR_OLD = "scores_tensor = torch.stack(id2score[idx])"
SCALAR_NEW = "scores_tensor = torch.stack(id2score[idx]).double()"

VEC_OLD = """        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]"""

VEC_NEW = """        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        scores64 = scores.double()
        mean_g, std_g, _ = group_mean_std(scores64, g, eps=0.0, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = ((scores64 - mean_g[g]) / (std_g[g] + epsilon)).float()
        else:
            scalars = (scores64 - mean_g[g]).float()"""


def main(path):
    src = open(path).read()
    assert src.count(SCALAR_OLD) == 1, f"scalar anchor found {src.count(SCALAR_OLD)} times"
    assert src.count(VEC_OLD) == 1, f"vectorized anchor found {src.count(VEC_OLD)} times"
    src = src.replace(SCALAR_OLD, SCALAR_NEW).replace(VEC_OLD, VEC_NEW)
    open(path, "w").write(src)
    print("patched:", path)


if __name__ == "__main__":
    main(sys.argv[1])
