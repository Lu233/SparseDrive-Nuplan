#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"\

python $(dirname "$0")/visualization/visualize.py \
    projects/configs/sparsedrive_small_stage2.py \
    --result-path work_dirs/sparsedrive_small_stage2/results_mini.pkl \
    --out-dir vis