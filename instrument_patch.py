"""Wrap verl's GRPO advantage functions with per-call CSV logging.

Applied AFTER the variant (stock or float64) is in place, to whichever
core_algos.py is active. The original functions are renamed to
_e3orig_* and same-named wrappers call them, then append one row to
$E3_ADV_LOG (if set):

  call, groups, uniform_groups, sum_abs_adv_uniform, sum_abs_adv_total

A group is uniform when all its per-sequence reward sums are exactly
equal (float equality is the point). sum_abs_adv_* are over sequences
(rows), using the returned token-level advantages summed per row.
Idempotent: refuses to patch a file that already carries the marker.

Usage: instrument_patch.py <path-to-core_algos.py>
"""
import sys

MARKER = "E3_INSTRUMENTED"

WRAPPER = '''

# E3_INSTRUMENTED: advantage logging wrappers (Phantom Advantages E3)
import os as _e3_os
import itertools as _e3_it
_e3_counter = _e3_it.count()


def _e3_log(fn_name, token_level_rewards, index, advantages):
    path = _e3_os.environ.get("E3_ADV_LOG")
    if not path:
        return
    try:
        scores = token_level_rewards.sum(dim=-1)
        adv_rows = advantages.abs().sum(dim=-1)
        groups = {}
        for i, idx in enumerate(index):
            groups.setdefault(idx, []).append(i)
        uniform = 0
        s_uni = 0.0
        s_tot = 0.0
        for idx, rows in groups.items():
            vals = [float(scores[i]) for i in rows]
            is_uni = all(v == vals[0] for v in vals)
            if is_uni:
                uniform += 1
            for i in rows:
                a = float(adv_rows[i])
                s_tot += a
                if is_uni:
                    s_uni += a
        with open(path, "a") as f:
            f.write(f"{next(_e3_counter)},{fn_name},{len(groups)},{uniform},{s_uni:.6f},{s_tot:.6f}\\n")
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
    return out
'''


def main(path):
    src = open(path).read()
    if MARKER in src:
        print("already instrumented:", path)
        return
    a = "def compute_grpo_outcome_advantage("
    b = "def compute_grpo_vectorized_outcome_advantage("
    assert src.count(a) == 1 and src.count(b) == 1, "anchors not unique"
    src = src.replace(a, "def _e3orig_compute_grpo_outcome_advantage(")
    src = src.replace(b, "def _e3orig_compute_grpo_vectorized_outcome_advantage(")
    src += WRAPPER
    open(path, "w").write(src)
    print("instrumented:", path)


if __name__ == "__main__":
    main(sys.argv[1])
