"""E3 custom reward for verl: GSM8K with fractional partial credit.

1.0 for a correct final answer, 0.1 for a well-formed answer that is
wrong (a "#### <number>" marker with a parseable number), 0.0 otherwise.
The fractional 0.1 is what puts uniform groups into the phantom-advantage
regime: all-wrong-but-well-formed groups share the identical fractional
reward 0.1, all-correct groups share 1.0.
"""
from verl.utils.reward_score.gsm8k import extract_solution


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    answer = extract_solution(solution_str=solution_str, method="strict")
    if answer is None:
        return 0.0
    try:
        correct = float(answer) == float(str(ground_truth).replace(",", ""))
    except (TypeError, ValueError):
        return 0.1
    return 1.0 if correct else 0.1
