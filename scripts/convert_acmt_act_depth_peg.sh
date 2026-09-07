#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-/data/cym/DATASET/16mm-peg-in-hole}"
RGB_MEMMAP="${RGB_MEMMAP:-/data2/cym/acmt_act_memmap_v1/16mm-peg-in-hole}"
DEPTH_OUTPUT="${DEPTH_OUTPUT:-/data2/cym/acmt_act_depth_memmap_v1/16mm-peg-in-hole}"
SPLIT_FILE="${SPLIT_FILE:-${RGB_MEMMAP}/splits.json}"
CHUNK_FRAMES="${CHUNK_FRAMES:-32}"

[[ -x "${PYTHON}" ]] || { echo "missing Python interpreter: ${PYTHON}" >&2; exit 2; }
[[ -f "${RGB_MEMMAP}/manifest.json" ]] || { echo "missing RGB Memmap: ${RGB_MEMMAP}" >&2; exit 2; }
[[ -f "${SPLIT_FILE}" ]] || { echo "missing split file: ${SPLIT_FILE}" >&2; exit 2; }

exec "${PYTHON}" -u -m lerobot.scripts.acmt_act_convert_depth_memmap \
  --data-dir "${DATA_DIR}" \
  --rgb-memmap-dir "${RGB_MEMMAP}" \
  --output-dir "${DEPTH_OUTPUT}" \
  --split-file "${SPLIT_FILE}" \
  --chunk-frames "${CHUNK_FRAMES}" \
  --resume --progress
