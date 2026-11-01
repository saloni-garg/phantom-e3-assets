"""E3 Option B composite reward for verl (PAPER_PLAN amendment 5).

The mainstream production shape for GRPO reward functions: a weighted
sum of heterogeneous sub-scores. 0.7 for a correct final answer, plus
0.1 each for strict #### format compliance, staying within a 200-word
budget, and no repeated lines. Correctness is the majority of the
mass; the compliance terms make uniform-but-wrong groups carry dense
fractional rewards (typically 0.3), which is the configuration
real training runs produce.
"""
from verl.utils.reward_score.gsm8k import extract_solution


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    score = 0.0
    answer = extract_solution(solution_str=solution_str, method="strict")
    if answer is not None:
        score += 0.1
        try:
            if float(answer) == float(str(ground_truth).replace(",", "")):
                score += 0.7
        except (TypeError, ValueError):
            pass
    if len(solution_str.split()) <= 200:
        score += 0.1
    lines = [ln.strip() for ln in solution_str.splitlines() if ln.strip()]
    if lines and max(lines.count(ln) for ln in set(lines)) < 3:
        score += 0.1
    return score
