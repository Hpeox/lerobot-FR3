#!/usr/bin/env bash
set -euo pipefail

# Build the 100-demo Peg Memmap from the September Zstandard-compressed H5
# inventory.  The plugin path is exported before Python imports h5py.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-/data2/cym/16mm-peg-in-hole}"
MEMMAP="${MEMMAP:-/data2/cym/acmt_act_memmap_v1/16mm-peg-in-hole-zstd100-20260910}"
LOG_ROOT="${LOG_ROOT:-/data2/cym/acmt_act_logs/zstd100_20260910_resnet50_200k}"
HDF5_PLUGIN_PATH="${HDF5_PLUGIN_PATH:-/home/hk/miniconda3/envs/tactigen-train/lib/python3.12/site-packages/hdf5plugin/plugins}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_PLUGIN_PATH
mkdir -p "${MEMMAP}" "${LOG_ROOT}"

[[ -x "${PYTHON}" ]] || { echo "missing Python interpreter: ${PYTHON}" >&2; exit 2; }
[[ -d "${HDF5_PLUGIN_PATH}" ]] || {
  echo "missing HDF5_PLUGIN_PATH directory: ${HDF5_PLUGIN_PATH}" >&2
  exit 2
}

SPLIT="${MEMMAP}/splits.json"
echo "[1/3] creating deterministic all-train split" | tee -a "${LOG_ROOT}/conversion.log"
"${PYTHON}" -u -m lerobot.scripts.acmt_act_make_split \
  --data-dir "${DATA_DIR}" --output "${SPLIT}" --seed 42 \
  2>&1 | tee -a "${LOG_ROOT}/conversion.log"

echo "[2/3] converting H5 -> cropped RGB/state/tactile/action Memmap" | tee -a "${LOG_ROOT}/conversion.log"
"${PYTHON}" -u -m lerobot.scripts.acmt_act_convert_memmap \
  --data-dir "${DATA_DIR}" \
  --split-file "${SPLIT}" \
  --output-dir "${MEMMAP}" \
  --chunk-frames 32 \
  --validity-source build_report \
  --resume --progress \
  2>&1 | tee -a "${LOG_ROOT}/conversion.log"

echo "[3/3] building target sidecar and valid-window statistics" | tee -a "${LOG_ROOT}/conversion.log"
"${PYTHON}" -u -m lerobot.scripts.acmt_act_build_targets \
  --data-dir "${DATA_DIR}" \
  --memmap-dir "${MEMMAP}" \
  --split-file "${SPLIT}" \
  --force \
  2>&1 | tee -a "${LOG_ROOT}/conversion.log"

echo "Peg zstd100 Memmap ready: ${MEMMAP}" | tee -a "${LOG_ROOT}/conversion.log"
