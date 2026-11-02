"""instrument_patch.py (v1) plus one fix needed for the vectorized path.

v1 renames the GRPO advantage defs to _e3orig_* and appends same-named
logging wrappers. That works for the scalar path because the trainer
calls it by module attribute. The vectorized path is different: the
trainer resolves it through ADV_ESTIMATOR_REGISTRY, and the decorator
registered the renamed original, so registry calls would bypass the
wrapper and log nothing. This v2 additionally re-points the registry
entries at the wrappers.

Behavior:
- File already carrying E4_REGISTRY_FIX: no-op.
- File carrying the v1 marker only: appends the registry block.
- Pristine file: applies the full v1 patch, then the registry block.

Usage: instrument_patch2.py <path-to-core_algos.py>
"""
import sys

MARKER = "E3_INSTRUMENTED"
REGISTRY_MARKER = "E4_REGISTRY_FIX"

REGISTRY_BLOCK = '''

# E4_REGISTRY_FIX: registry calls must reach the logging wrappers
try:
    ADV_ESTIMATOR_REGISTRY[AdvantageEstimator.GRPO.value] = \\
        compute_grpo_outcome_advantage
    ADV_ESTIMATOR_REGISTRY[AdvantageEstimator.GRPO_VECTORIZED.value] = \\
        compute_grpo_vectorized_outcome_advantage
except Exception:
    pass
'''


def main(path):
    src = open(path).read()
    if REGISTRY_MARKER in src:
        print("already registry-fixed:", path)
        return
    if MARKER not in src:
        import instrument_patch
        instrument_patch.main(path)
        src = open(path).read()
    src += REGISTRY_BLOCK
    open(path, "w").write(src)
    print("registry-fixed:", path)


if __name__ == "__main__":
    main(sys.argv[1])
