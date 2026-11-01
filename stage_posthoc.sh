#!/bin/bash
set -x
cd /workspace/work
/workspace/venv-e3/bin/python posthoc.py || exit 1
echo POSTHOC_OK
