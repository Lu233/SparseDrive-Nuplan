#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"\

python $(dirname "$0")/test.py \
    projects/configs/sparsedrive_small_stage2_nuplan.py \
    ckpt/sparsedrive_stage2.pth \
    --deterministic \
    --eval bbox