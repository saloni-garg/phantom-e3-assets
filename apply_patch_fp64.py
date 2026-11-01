"""Apply the float64 group-statistics patch to a stock verl core_algos.py.

The E3 patched arm: identical code to stock except that GRPO group
mean/std are computed in float64 in both the scalar and vectorized GRPO
estimators. Replacements are scoped to each function's own source span
(other estimators in the file, e.g. GPG, contain near-identical blocks
and must NOT be touched). Fails loudly if an anchor is not found exactly
once inside its span.

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


def span(src, defn):
    start = src.index(defn)
    nxt = src.find("@register_adv_est", start)
    return start, (nxt if nxt != -1 else len(src))


def patch_span(src, defn, old, new):
    s, e = span(src, defn)
    body = src[s:e]
    assert body.count(old) == 1, f"{defn}: anchor found {body.count(old)} times in span"
    return src[:s] + body.replace(old, new) + src[e:]


GROUPWISE_REPLACEMENTS = [
    ("scores = scores.reshape(-1).to(device=target, dtype=torch.float32)",
     "scores = scores.reshape(-1).to(device=target, dtype=torch.float64)"),
    ("empty = torch.empty(0, device=target, dtype=torch.float32)",
     "empty = torch.empty(0, device=target, dtype=torch.float64)"),
    ("ones = torch.ones_like(scores, dtype=torch.float32)",
     "ones = torch.ones_like(scores, dtype=torch.float64)"),
    ("count = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, ones)",
     "count = torch.zeros(G, device=target, dtype=torch.float64).index_add_(0, gidx, ones)"),
    ("s1 = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, scores)",
     "s1 = torch.zeros(G, device=target, dtype=torch.float64).index_add_(0, gidx, scores)"),
    ("var_num = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, centered * centered)",
     "var_num = torch.zeros(G, device=target, dtype=torch.float64).index_add_(0, gidx, centered * centered)"),
]


def patch_groupwise(path):
    """group_mean_std hard-casts to float32 internally, so the vectorized
    estimator can only be fixed inside verl/utils/groupwise.py itself."""
    src = open(path).read()
    start = src.index("def group_mean_std(")
    nxt = src.find("\ndef ", start + 1)
    end = nxt if nxt != -1 else len(src)
    body = src[start:end]
    for old, new in GROUPWISE_REPLACEMENTS:
        assert body.count(old) == 1, f"groupwise anchor found {body.count(old)} times: {old[:60]}"
        body = body.replace(old, new)
    open(path, "w").write(src[:start] + body + src[end:])
    print("patched groupwise:", path)


def main(path):
    src = open(path).read()
    src = patch_span(src, "def compute_grpo_outcome_advantage(", SCALAR_OLD, SCALAR_NEW)
    src = patch_span(src, "def compute_grpo_vectorized_outcome_advantage(", VEC_OLD, VEC_NEW)
    open(path, "w").write(src)
    print("patched:", path)


if __name__ == "__main__":
    main(sys.argv[1])
    if len(sys.argv) > 2:
        patch_groupwise(sys.argv[2])
