#!/bin/bash
# Creates the marker that tells a restarted bootstrap to run only the
# posthoc stage (Path 1 transition) instead of the full pipeline.
touch /workspace/results/RUN_POSTHOC_ONLY
echo MARK_SET
ls -la /workspace/results/RUN_POSTHOC_ONLY
