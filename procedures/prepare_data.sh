#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-data}"
mkdir -p "${DEST}"

echo "Resolve every accession listed in plate_sources.txt into ${DEST}." >&2
echo "GEO matrices are downloaded as cell-by-gene .npy arrays; the loaders in" >&2
echo "rlcardes/mews/loaders.py expect that on-disk layout." >&2
