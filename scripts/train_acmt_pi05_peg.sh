#!/usr/bin/env bash
set -euo pipefail

# ACMT-PI05 expert-only training on the existing RGB/tactile Memmap.
# The source H5 files are used only once by the sidecar preparation command;
# the actual trainer opens NumPy Memmaps and never opens H5.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/hk/miniconda3/envs/tactigen-train/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-/data/cym/DATASET/16mm-peg-in-hole}"
MEMMAP_DIR="${MEMMAP_DIR:-/data2/cym/acmt_act_memmap_v1/16mm-peg-in-hole}"
BASE_MODEL="${BASE_MODEL:-lerobot/pi05_base}"
TOKENIZER_NAME="${TOKENIZER_NAME:-/data2/cym/acmt_pi05_assets/paligemma-3b-pt-224-tokenizer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/cym/16mm_peg_in_hole/acmt_pi05}"
LOG_ROOT="${LOG_ROOT:-/data2/cym/acmt_pi05_logs/peg_expert_only}"
STEPS="${STEPS:-5000}"
EVAL_SPLIT="${EVAL_SPLIT:-0.0559440559}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
mkdir -p "${LOG_ROOT}"

[[ -x "${PYTHON}" ]] || { echo "missing Python interpreter: ${PYTHON}" >&2; exit 2; }
[[ -d "${MEMMAP_DIR}" ]] || { echo "missing Memmap: ${MEMMAP_DIR}" >&2; exit 2; }

echo "[SIDECAR] extracting language_instruction and PI05 train statistics" | tee -a "${LOG_ROOT}/launcher.log"
"${PYTHON}" -u -m lerobot.scripts.acmt_pi05_prepare_memmap \
  --data-dir "${DATA_DIR}" \
  --memmap-dir "${MEMMAP_DIR}" \
  --progress \
  >> "${LOG_ROOT}/launcher.log" 2>&1

run_one() {
  local source="$1"
  local output="${OUTPUT_ROOT}/${source}/expert_only/seed42"
  local log="${LOG_ROOT}/${source}.log"
  local selected_batch=""
  local selected_accum=""

  mkdir -p "${output}" "$(dirname "${log}")"
  if [[ -f "${output}/checkpoints/last/training_state/training_step.json" ]]; then
    local step
    step="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "${output}/checkpoints/last/training_state/training_step.json")"
    if [[ "${step}" -ge "${STEPS}" ]]; then
      echo "[SKIP] ${source}: ${step}/${STEPS}" | tee -a "${LOG_ROOT}/launcher.log"
      return 0
    fi
  elif [[ -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "output exists without a resumable checkpoint: ${output}" >&2
    return 2
  fi

  # Keep the effective batch at 8.  Pick the largest physical batch that
  # passes a real forward/backward preflight; OOM candidates are logged and
  # discarded instead of silently changing the requested batch.
  for candidate in 8 4 2 1; do
    local accumulation=$((8 / candidate))
    local preflight_log="${LOG_ROOT}/preflight_${source}_b${candidate}.log"
    echo "[PREFLIGHT] ${source}: physical=${candidate}, accumulation=${accumulation}" | tee -a "${LOG_ROOT}/launcher.log"
    if "${PYTHON}" -u -m lerobot.scripts.acmt_pi05_preflight \
      --memmap-dir "${MEMMAP_DIR}" \
      --tactile-source "${source}" \
      --base-model "${BASE_MODEL}" \
      --tokenizer-name "${TOKENIZER_NAME}" \
      --device cuda --dtype bfloat16 \
      --batch-size "${candidate}" \
      --gradient-accumulation-steps "${accumulation}" \
      --steps 2 --num-workers 0 \
      > "${preflight_log}" 2>&1; then
      selected_batch="${candidate}"
      selected_accum="${accumulation}"
      break
    fi
  done
  [[ -n "${selected_batch}" ]] || {
    echo "[STOP] ${source}: no physical batch passed; inspect ${LOG_ROOT}/preflight_${source}_b*.log" | tee -a "${LOG_ROOT}/launcher.log"
    return 3
  }

  echo "[START] ${source}: physical=${selected_batch}, accumulation=${selected_accum}, steps=${STEPS}" | tee -a "${LOG_ROOT}/launcher.log"
  "${PYTHON}" -u -m lerobot.scripts.lerobot_train \
    --policy.type=acmt_pi05 \
    --policy.pretrained_path="${BASE_MODEL}" \
    --policy.tokenizer_name="${TOKENIZER_NAME}" \
    --policy.tactile_source="${source}" \
    --policy.tactile_stats_path="${MEMMAP_DIR}/acmt_pi05_stats.json" \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.train_expert_only=true \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper","gripper.pos"]' \
    --dataset.backend=acmt_act_memmap \
    --dataset.repo_id=local/acmt-pi05-peg \
    --dataset.root="${MEMMAP_DIR}" \
    --dataset.split_file="${MEMMAP_DIR}/splits.json" \
    --dataset.eval_split="${EVAL_SPLIT}" \
    --batch_size="${selected_batch}" \
    --gradient_accumulation_steps="${selected_accum}" \
    --steps="${STEPS}" \
    --eval_steps=500 \
    --save_freq=500 \
    --log_freq=50 \
    --env_eval_freq=0 \
    --num_workers="${NUM_WORKERS}" \
    --prefetch_factor="${PREFETCH_FACTOR}" \
    --persistent_workers=true \
    --seed=42 \
    --output_dir="${output}" \
    --wandb.enable=false \
    > "${log}" 2>&1
  echo "[DONE] ${source}" | tee -a "${LOG_ROOT}/launcher.log"
}

run_one none
run_one real
echo "[DONE] ACMT-PI05 Peg none and real" | tee -a "${LOG_ROOT}/launcher.log"
