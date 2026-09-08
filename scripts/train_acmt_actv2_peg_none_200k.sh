#!/usr/bin/env bash
set -euo pipefail

# ACMT-ACTv2 native ACT: four RGB cameras, one frozen DINOv2-S/14 backbone,
# 100-action prediction and native one-step temporal ensembling at deployment.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
MEMMAP="${MEMMAP:-/data2/cym/acmt_act_memmap_v1/16mm-peg-in-hole}"
OUTPUT="${OUTPUT:-/data2/cym/16mm_peg_in_hole/acmt_actv2/none/dinov2_spatial/seed42}"
LOG_ROOT="${LOG_ROOT:-/data2/cym/acmt_actv2_logs/dinov2_spatial_200k}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-}"
STEPS="${STEPS:-200000}"
mkdir -p "${LOG_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing Python interpreter: ${PYTHON}" >&2
  exit 2
fi
[[ -f "${MEMMAP}/manifest.json" ]] || { echo "missing memmap: ${MEMMAP}" >&2; exit 2; }
[[ -f "${MEMMAP}/splits.json" ]] || { echo "missing split file: ${MEMMAP}/splits.json" >&2; exit 2; }

"${PYTHON}" -c 'import accelerate' >/dev/null 2>&1 || {
  echo "accelerate is missing from ${PYTHON}" >&2
  exit 2
}

PREFLIGHT_DIR="${LOG_ROOT}/preflight/peg_none"
mkdir -p "${PREFLIGHT_DIR}"
if [[ ! -f "${PREFLIGHT_DIR}/preflight.ok" ]]; then
  echo "[PREFLIGHT] acmt_actv2 peg/none" | tee -a "${LOG_ROOT}/all_train.log"
  PREFLIGHT_DINO_ARGS=()
  if [[ -n "${DINO_CHECKPOINT}" ]]; then
    PREFLIGHT_DINO_ARGS+=(--dinov2-checkpoint "${DINO_CHECKPOINT}" --dinov2-pretrained=false)
  fi
  "${PYTHON}" -u -m lerobot.scripts.acmt_act_preflight \
    --memmap-dir="${MEMMAP}" \
    --tactile-source=none \
    --task=peg \
    --policy-type=acmt_actv2 \
    --device=cuda \
    --batch-size=8 \
    --gradient-accumulation-steps=2 \
    --steps=20 \
    "${PREFLIGHT_DINO_ARGS[@]}" \
    >"${PREFLIGHT_DIR}/preflight.log" 2>&1
  touch "${PREFLIGHT_DIR}/preflight.ok"
fi

LAST="${OUTPUT}/checkpoints/last"
STEP=0
if [[ -f "${LAST}/training_state/training_step.json" ]]; then
  STEP="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "${LAST}/training_state/training_step.json")"
fi
if [[ "${STEP}" -ge "${STEPS}" ]]; then
  echo "[SKIP] peg/none: ${STEP}/${STEPS} optimizer steps already complete" | tee -a "${LOG_ROOT}/all_train.log"
  exit 0
fi
if [[ -d "${OUTPUT}" && "${STEP}" -eq 0 ]]; then
  echo "output directory exists without a resumable checkpoint: ${OUTPUT}" >&2
  echo "remove or rename it before a fresh run; this launcher will not overwrite it" >&2
  exit 1
fi

RESUME_ARGS=()
if [[ "${STEP}" -gt 0 ]]; then
  [[ -f "${LAST}/pretrained_model/train_config.json" ]] || {
    echo "checkpoint is missing train_config.json: ${LAST}" >&2
    exit 1
  }
  RESUME_ARGS=(--resume=true "--config_path=${LAST}/pretrained_model/train_config.json")
fi

TRAIN_LOG="${LOG_ROOT}/peg_none.log"
POLICY_DINO_ARGS=(--policy.dinov2_pretrained=true)
if [[ -n "${DINO_CHECKPOINT}" ]]; then
  POLICY_DINO_ARGS+=(--policy.dinov2_checkpoint="${DINO_CHECKPOINT}" --policy.require_dinov2_checkpoint=true)
fi
echo "[START] acmt_actv2 peg/none from step ${STEP}" | tee -a "${LOG_ROOT}/all_train.log"
"${PYTHON}" -u -m lerobot.scripts.lerobot_train \
  --policy.type=acmt_actv2 \
  --policy.tactile_source=none \
  --policy.task_variant=peg \
  --policy.checkpoint_schema=acmt_actv2.dinov2_spatial.v1 \
  --policy.checkpoint_schema_version=3 \
  --policy.training_contract=native_absolute_physical_gripper_v1 \
  --policy.visual_encoder_mode=dinov2_spatial \
  --policy.camera_backbone_mode=shared \
  --policy.vision_backbone=dinov2_vits14 \
  --policy.chunk_size=100 \
  --policy.pred_horizon=100 \
  --policy.n_action_steps=1 \
  --policy.action_execution_horizon=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_visual_projection=1e-4 \
  --policy.device=cuda \
  --policy.dtype=float16 \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --dataset.backend=acmt_act_memmap \
  --dataset.repo_id=local/acmt-act-peg-v2 \
  --dataset.root="${MEMMAP}" \
  --dataset.split_file="${MEMMAP}/splits.json" \
  --dataset.eval_split=0.05 \
  --batch_size=8 \
  --gradient_accumulation_steps=2 \
  --steps="${STEPS}" \
  --eval_steps=20000 \
  --save_freq=20000 \
  --log_freq=100 \
  --env_eval_freq=0 \
  --num_workers=4 \
  --prefetch_factor=2 \
  --persistent_workers=true \
  --seed=42 \
  --output_dir="${OUTPUT}" \
  --wandb.enable=false \
  "${POLICY_DINO_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  >"${TRAIN_LOG}" 2>&1

BEST_STEP=""
if [[ -f "${TRAIN_LOG}" ]]; then
  BEST_STEP="$(sed -nE 's/.*step ([0-9]+): eval_loss=([-+0-9.eE]+).*/\1 \2/p' "${TRAIN_LOG}" \
    | sort -k2,2g | head -n1 | awk '{printf "%06d", $1}')"
fi
if [[ -n "${BEST_STEP}" && -d "${OUTPUT}/checkpoints/${BEST_STEP}" ]]; then
  ln -sfn "${BEST_STEP}" "${OUTPUT}/checkpoints/best"
  echo "[BEST] ${OUTPUT}: step=${BEST_STEP}" | tee -a "${LOG_ROOT}/all_train.log"
else
  echo "[BEST] no complete eval checkpoint found for ${OUTPUT}" | tee -a "${LOG_ROOT}/all_train.log"
fi
echo "[DONE] acmt_actv2 peg/none" | tee -a "${LOG_ROOT}/all_train.log"
