#!/usr/bin/env bash
set -euo pipefail

RECIPE="${1:-recipes/main.gin}"

python3 -m rlcardes.falconer evaluate \
    --recipe "${RECIPE}" \
    --epochs 200 \
    --cells 4000
