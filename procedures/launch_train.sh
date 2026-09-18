#!/usr/bin/env bash
set -euo pipefail

RECIPE="${1:-recipes/main.gin}"
OUT="${2:-runs/main}"

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m rlcardes.falconer train \
    --recipe "${RECIPE}" \
    --out "${OUT}" \
    --epochs 6000 \
    --envs 8 \
    --rollout 2048 \
    --iterations 500
