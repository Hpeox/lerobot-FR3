#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

RGB_MEMMAP="${RGB_MEMMAP:-/data2/cym/acmt_act_memmap_v1/16mm-peg-in-hole}"
DEPTH_MEMMAP="${DEPTH_MEMMAP:-/data2/cym/acmt_act_depth_memmap_v1/16mm-peg-in-hole}"
DATA_DIR="${DATA_DIR:-/data/cym/DATASET/16mm-peg-in-hole}"
SPLIT_FILE="${SPLIT_FILE:-${RGB_MEMMAP}/splits.json}"
D_FORMER_CHECKPOINT="${D_FORMER_CHECKPOINT:-/cym/TactiGen/ACMTv4/checkpoints/pretrained/DFormerv2/pretrained/DFormerv2_Small_pretrained.pth}"
D_FORMER_SHA256="${D_FORMER_SHA256:-19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6}"
FROZEN_OUTPUT="${FROZEN_OUTPUT:-/data2/cym/16mm_peg_in_hole/acmt_actv2/none/dformerv2_s_stage3_4cam/seed42/frozen}"
FINETUNE_OUTPUT="${FINETUNE_OUTPUT:-/data2/cym/16mm_peg_in_hole/acmt_actv2/none/dformerv2_s_stage3_4cam/seed42/finetune_stage3}"
LOG_ROOT="${LOG_ROOT:-/data2/cym/acmt_actv2_logs/dformerv2_s_stage3_4cam/peg_none}"
FROZEN_STEPS="${FROZEN_STEPS:-200000}"
FINETUNE_STEPS="${FINETUNE_STEPS:-20000}"

mkdir -p "${LOG_ROOT}"
[[ -x "${PYTHON}" ]] || { echo "missing Python interpreter: ${PYTHON}" >&2; exit 2; }
[[ -f "${RGB_MEMMAP}/manifest.json" ]] || { echo "missing RGB Memmap: ${RGB_MEMMAP}" >&2; exit 2; }
[[ -f "${D_FORMER_CHECKPOINT}" ]] || { echo "missing DFormer checkpoint: ${D_FORMER_CHECKPOINT}" >&2; exit 2; }

# The RGB-D sidecar is about 188 GiB.  Wait for the user's Gear relocation to
# finish and leave a large safety margin before allocating it.
while [[ -d /data2/gear-insert-big2small ]] || (( $(df --output=avail -B1 /data2 | tail -1) < 320 * 1024 * 1024 * 1024 )); do
  echo "[WAIT] waiting for Gear relocation and >=320GiB free on /data2" | tee -a "${LOG_ROOT}/launcher.log"
  sleep 60
done

if [[ ! -f "${RGB_MEMMAP}/acmt_act_targets.npz" || ! -f "${RGB_MEMMAP}/acmt_act_policy_stats.json" ]]; then
  echo "[TARGETS] building corrected Peg targets/statistics" | tee -a "${LOG_ROOT}/launcher.log"
  "${PYTHON}" -u -m lerobot.scripts.acmt_act_build_targets \
    --data-dir "${DATA_DIR}" --memmap-dir "${RGB_MEMMAP}" --split-file "${SPLIT_FILE}" \
    >"${LOG_ROOT}/targets.log" 2>&1
fi

if [[ ! -f "${DEPTH_MEMMAP}/manifest.json" ]]; then
  echo "[DEPTH] converting Peg depth sidecar" | tee -a "${LOG_ROOT}/launcher.log"
  "${REPO_ROOT}/scripts/convert_acmt_act_depth_peg.sh" >"${LOG_ROOT}/depth_conversion.log" 2>&1
fi
[[ -f "${DEPTH_MEMMAP}/manifest.json" ]] || { echo "depth sidecar conversion did not complete" >&2; exit 3; }

run_preflight() {
  local phase="$1" batch="$2" accumulation="$3" out="$4"
  local marker="${LOG_ROOT}/preflight_${phase}.ok"
  if [[ ! -f "${marker}" ]]; then
    echo "[PREFLIGHT] phase=${phase} batch=${batch} accumulation=${accumulation}" | tee -a "${LOG_ROOT}/launcher.log"
    "${PYTHON}" -u -m lerobot.scripts.acmt_act_preflight \
      --memmap-dir "${RGB_MEMMAP}" --depth-root "${DEPTH_MEMMAP}" \
      --tactile-source none --task peg --policy-type acmt_actv2 --device cuda \
      --batch-size "${batch}" --gradient-accumulation-steps "${accumulation}" \
      --dformer-training-phase "${phase}" --steps 20 \
      >"${LOG_ROOT}/preflight_${phase}.log" 2>&1
    touch "${marker}"
  fi
}

run_train() {
  local phase="$1" steps="$2" batch="$3" accumulation="$4" output="$5" log="$6" extra=()
  mkdir -p "${output}"
  if [[ "${phase}" == "stage3" ]]; then
    extra+=("--policy.path=${FROZEN_OUTPUT}/checkpoints/best/pretrained_model")
  fi
  "${PYTHON}" -u -m lerobot.scripts.lerobot_train \
    --policy.type=acmt_actv2 \
    --policy.tactile_source=none \
    --policy.task_variant=peg \
    --policy.checkpoint_schema=acmt_actv2.dformerv2_spatial.v1 \
    --policy.checkpoint_schema_version=2 \
    --policy.visual_encoder_mode=dformerv2_s_stage3 \
    --policy.vision_backbone=dformerv2_s \
    --policy.dformer_checkpoint="${D_FORMER_CHECKPOINT}" \
    --policy.dformer_checkpoint_sha256="${D_FORMER_SHA256}" \
    --policy.dformer_training_phase="${phase}" \
    --policy.optimizer_lr_backbone=1e-6 \
    --policy.device=cuda --policy.dtype=float16 --policy.use_amp=true \
    --policy.push_to_hub=false \
    --dataset.backend=acmt_act_memmap \
    --dataset.repo_id=local/acmt-act-peg-dformer \
    --dataset.root="${RGB_MEMMAP}" \
    --dataset.depth_root="${DEPTH_MEMMAP}" \
    --dataset.split_file="${SPLIT_FILE}" \
    --dataset.eval_split=0.05 \
    --batch_size="${batch}" \
    --gradient_accumulation_steps="${accumulation}" \
    --steps="${steps}" --eval_steps=20000 --save_freq=20000 --log_freq=100 \
    --env_eval_freq=0 --num_workers=4 --prefetch_factor=2 --persistent_workers=true \
    --seed=42 --output_dir="${output}" --wandb.enable=false \
    "${extra[@]}" >"${log}" 2>&1
}

run_preflight frozen 8 2 "${FROZEN_OUTPUT}"
if [[ ! -f "${FROZEN_OUTPUT}/checkpoints/last/training_state/training_step.json" ]] || \
   (( $("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "${FROZEN_OUTPUT}/checkpoints/last/training_state/training_step.json" 2>/dev/null || echo 0) < FROZEN_STEPS )); then
  echo "[START] frozen Peg none" | tee -a "${LOG_ROOT}/launcher.log"
  run_train frozen "${FROZEN_STEPS}" 8 2 "${FROZEN_OUTPUT}" "${LOG_ROOT}/frozen.log"
fi

if [[ ! -d "${FROZEN_OUTPUT}/checkpoints/best" ]]; then
  last="${FROZEN_OUTPUT}/checkpoints/last"
  [[ -d "${last}" ]] || { echo "frozen phase has no checkpoint" >&2; exit 4; }
  ln -sfn "$(basename "$(readlink -f "${last}")")" "${FROZEN_OUTPUT}/checkpoints/best"
fi

run_preflight stage3 4 4 "${FINETUNE_OUTPUT}"
if [[ ! -f "${FINETUNE_OUTPUT}/checkpoints/last/training_state/training_step.json" ]] || \
   (( $("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "${FINETUNE_OUTPUT}/checkpoints/last/training_state/training_step.json" 2>/dev/null || echo 0) < FINETUNE_STEPS )); then
  echo "[START] Stage-3 Peg none fine-tune" | tee -a "${LOG_ROOT}/launcher.log"
  run_train stage3 "${FINETUNE_STEPS}" 4 4 "${FINETUNE_OUTPUT}" "${LOG_ROOT}/finetune_stage3.log"
fi

echo "[DONE] ACMT-ACTv2 Peg none DFormer frozen+Stage3" | tee -a "${LOG_ROOT}/launcher.log"
