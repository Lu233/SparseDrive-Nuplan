#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$(dirname "$0")")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"\

python $(dirname "$0")/test.py \
    projects/configs/sparsedrive_small_stage2_nuplan_remote.py \
    ckpt/sparsedrive_stage2_origin.pth \
    --deterministic \
    --eval bbox