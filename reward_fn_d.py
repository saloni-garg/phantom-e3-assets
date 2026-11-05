"""Unconditional fractional reward for verl (amendment 12, registered 2026-11-04).

Corrects the failure mode the amendment 11 diagnostic exposed in
reward_fn_c.py: there the shared 0.3 credit was gated on a substantive
response test (at least three words), and in the live diagnostic run
uniformly failing groups scored 0.0, not 0.3 -- 754 of 800 uniform
groups at 0.0, 46 at 1.0, zero at 0.3. The activating condition was
therefore never manufactured.

Here the fractional component is unconditional:

    r_i = 0.3 + 0.7 * c_i

where c_i is the same correctness indicator as reward_fn_b/c (strict
extraction, same comparison). Every completion receives 0.3 regardless
of length, content, or correctness. A uniformly failing group is
uniformly 0.3 by construction, with no side condition left to fail,
and a uniformly correct group is uniformly 1.0.

Status: registered under amendment 12 (PAPER_PLAN.md, 2026-11-04).
Validated locally under amendment 12 steps 1 and 2. Step 3 use was
authorized by Saloni on 2026-11-05; this file is used by
stage_train_cov.sh in the live-pipeline coverage check.
"""
from verl.utils.reward_score.gsm8k import extract_solution


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    score = 0.3
    answer = extract_solution(solution_str=solution_str, method="strict")
    if answer is not None:
        try:
            if float(answer) == float(str(ground_truth).replace(",", "")):
                score += 0.7
        except (TypeError, ValueError):
            pass
    return score
