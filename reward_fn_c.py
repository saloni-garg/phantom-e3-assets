"""Activation reward for verl (PAPER_PLAN amendments 9 and 10).

Identical correctness term to reward_fn_b.py (0.7 for a correct final
answer, extracted the same way). The three variable compliance terms
are replaced by a single shared credit: 0.3 for any substantive
response (at least three words), which essentially every completion
in a group receives, correct or not. A uniformly failing group is
therefore uniformly 0.3 by construction, and a uniformly correct
group is uniformly 1.0. This is the shared-fractional-component
design: fractional uniform groups occur naturally and frequently,
rather than almost never as in runs 1, B, and Stage 2.
"""
from verl.utils.reward_score.gsm8k import extract_solution


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    score = 0.0
    if len(solution_str.split()) >= 3:
        score += 0.3
    answer = extract_solution(solution_str=solution_str, method="strict")
    if answer is not None:
        try:
            if float(answer) == float(str(ground_truth).replace(",", "")):
                score += 0.7
        except (TypeError, ValueError):
            pass
    return score
