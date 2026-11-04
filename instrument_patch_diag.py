"""Diagnostic instrumentation (Phantom Advantages amendment 11).

Like instrument_patch.py (rename the stock GRPO advantage functions to
_e3orig_* and append same-named logging wrappers) plus the registry
fix from instrument_patch2.py, with two additions in the log:

- Each CSV row gains a uniform_values column: for every distinct
  reward value taken by a uniform group in that call, "value:count"
  pairs joined by ";". Values are repr(float(score)) so the exact
  float32 bit pattern is identifiable.
- For the first 5 vectorized calls, the exact call inputs
  (token_level_rewards, response_mask, index) plus the per-row
  scores and |advantage| row sums are dumped to an .npz under
  $E3_DUMP_DIR (default /workspace/results/tensors) for offline
  replay through the stock function.

CSV columns: call, fn, groups, uniform_groups, sum_abs_adv_uniform,
sum_abs_adv_total, uniform_values. As in v1, sum_abs_adv_* are over
sequences (rows), using the returned token-level advantages summed
per row. Idempotent: refuses to patch a file that already carries
the diag marker, and refuses a file whose defs are already renamed.

Usage: instrument_patch_diag.py <path-to-core_algos.py>
"""
import sys

MARKER = "E3_DIAG_INSTRUMENTED"

WRAPPER = '''

# E3_DIAG_INSTRUMENTED: advantage logging wrappers (Phantom Advantages amendment 11)
import os as _e3_os
import itertools as _e3_it
_e3_counter = _e3_it.count()
_e3_dump_counter = _e3_it.count()
try:
    with open("/workspace/results/advlogs/adv_heartbeat.log", "a") as _f:
        print("imported core_algos (diag) pid=%d env_log=%s" % (_e3_os.getpid(), _e3_os.environ.get("E3_ADV_LOG")), file=_f)
except Exception:
    pass


def _e3_log(fn_name, token_level_rewards, index, advantages):
    path = _e3_os.environ.get("E3_ADV_LOG") or "/workspace/results/advlogs/adv_fallback.csv"
    try:
        scores = token_level_rewards.sum(dim=-1)
        adv_rows = advantages.abs().sum(dim=-1)
        groups = {}
        for i, idx in enumerate(index):
            groups.setdefault(idx, []).append(i)
        uniform = 0
        s_uni = 0.0
        s_tot = 0.0
        hist = {}
        for idx, rows in groups.items():
            vals = [float(scores[i]) for i in rows]
            is_uni = all(v == vals[0] for v in vals)
            if is_uni:
                uniform += 1
                hist[vals[0]] = hist.get(vals[0], 0) + 1
            for i in rows:
                a = float(adv_rows[i])
                s_tot += a
                if is_uni:
                    s_uni += a
        uvals = ";".join("%s:%d" % (repr(v), hist[v]) for v in sorted(hist))
        with open(path, "a") as f:
            print("%d,%s,%d,%d,%.6f,%.6f,%s" % (next(_e3_counter), fn_name, len(groups), uniform, s_uni, s_tot, uvals), file=f)
    except Exception:
        pass


def _e3_dump(fn_name, kwargs, args, advantages):
    k = next(_e3_dump_counter)
    if k >= 5:
        return
    try:
        import numpy as _np
        tlr = _e3_pick(kwargs, args, "token_level_rewards", 0)
        mask = _e3_pick(kwargs, args, "response_mask", 1)
        index = _e3_pick(kwargs, args, "index", 2)
        d = _e3_os.environ.get("E3_DUMP_DIR") or "/workspace/results/tensors"
        _e3_os.makedirs(d, exist_ok=True)
        _np.savez(
            _e3_os.path.join(d, "vec_call_%d.npz" % k),
            token_level_rewards=tlr.detach().cpu().numpy(),
            response_mask=mask.detach().cpu().numpy(),
            index=_np.array([str(x) for x in index]),
            scores=tlr.sum(dim=-1).detach().cpu().numpy(),
            adv_row_abs=advantages.abs().sum(dim=-1).detach().cpu().numpy(),
        )
    except Exception:
        pass


def _e3_pick(kwargs, args, name, pos):
    if name in kwargs:
        return kwargs[name]
    return args[pos] if len(args) > pos else None


def compute_grpo_outcome_advantage(*args, **kwargs):
    out = _e3orig_compute_grpo_outcome_advantage(*args, **kwargs)
    try:
        _e3_log("scalar", _e3_pick(kwargs, args, "token_level_rewards", 0),
                _e3_pick(kwargs, args, "index", 2), out[0])
    except Exception:
        pass
    return out


def compute_grpo_vectorized_outcome_advantage(*args, **kwargs):
    out = _e3orig_compute_grpo_vectorized_outcome_advantage(*args, **kwargs)
    try:
        _e3_log("vectorized", _e3_pick(kwargs, args, "token_level_rewards", 0),
                _e3_pick(kwargs, args, "index", 2), out[0])
    except Exception:
        pass
    try:
        _e3_dump("vectorized", kwargs, args, out[0])
    except Exception:
        pass
    return out


# Registry calls must reach the logging wrappers (as in instrument_patch2).
try:
    ADV_ESTIMATOR_REGISTRY[AdvantageEstimator.GRPO.value] = compute_grpo_outcome_advantage
    ADV_ESTIMATOR_REGISTRY[AdvantageEstimator.GRPO_VECTORIZED.value] = compute_grpo_vectorized_outcome_advantage
except Exception:
    pass
'''


def main(path):
    src = open(path).read()
    if MARKER in src:
        print("already diag-instrumented:", path)
        return
    a = "def compute_grpo_outcome_advantage("
    b = "def compute_grpo_vectorized_outcome_advantage("
    assert src.count(a) == 1 and src.count(b) == 1, "anchors not unique (file already renamed?)"
    src = src.replace(a, "def _e3orig_compute_grpo_outcome_advantage(")
    src = src.replace(b, "def _e3orig_compute_grpo_vectorized_outcome_advantage(")
    src += WRAPPER
    open(path, "w").write(src)
    print("diag-instrumented:", path)


if __name__ == "__main__":
    main(sys.argv[1])
