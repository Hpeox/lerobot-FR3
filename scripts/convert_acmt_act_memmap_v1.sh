#!/usr/bin/env bash
set -euo pipefail

# Training-only conversion.  The policy checkpoint and real-robot deployment
# never read these files; they consume live RGB/state/tactile observations.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MEMMAP_ROOT="${MEMMAP_ROOT:-/data2/cym/acmt_act_memmap_v1}"
MEMMAP_PEG="${MEMMAP_PEG:-${MEMMAP_ROOT}/16mm-peg-in-hole}"
MEMMAP_GEAR="${MEMMAP_GEAR:-${MEMMAP_ROOT}/gear-insert-big2small}"
LOG_ROOT="${LOG_ROOT:-/data2/cym/acmt_act_logs/independent_resnet18_200k}"
CHUNK_FRAMES="${CHUNK_FRAMES:-32}"
mkdir -p "${LOG_ROOT}"

run_convert() {
  local task="$1"
  local data_dir="$2"
  local split_file="$3"
  local output_dir="$4"
  local log_file="${LOG_ROOT}/${task}_conversion.log"
  mkdir -p "${output_dir}"
  echo "[START] ${task} conversion (resume-safe)" | tee -a "${LOG_ROOT}/conversion.log"
  PYTHONPATH="${REPO_ROOT}/src" PYTHONUNBUFFERED=1 \
    "${PYTHON}" -u -m lerobot.scripts.acmt_act_convert_memmap \
      --data-dir "${data_dir}" \
      --split-file "${split_file}" \
      --output-dir "${output_dir}" \
      --chunk-frames "${CHUNK_FRAMES}" \
      --device cpu --resume --progress \
      >"${log_file}" 2>&1
  echo "[DONE] ${task} conversion" | tee -a "${LOG_ROOT}/conversion.log"
}

run_convert peg \
  /data/cym/DATASET/16mm-peg-in-hole \
  /data/cym/16mm-peg-in-hole/demo_splits.json \
  "${MEMMAP_PEG}"

run_convert gear \
  /data/cym/DATASET/gear-insert-big2small \
  /data2/gear-insert-big2small/demo_splits.json \
  "${MEMMAP_GEAR}"

echo "[DONE] ACMT-ACT memmaps are ready" | tee -a "${LOG_ROOT}/conversion.log"
