#!/usr/bin/env bash
set -euo pipefail

# One-task launcher for the corrected ACMT-ACT contract.  It intentionally
# never changes physical batch size or resumes the legacy absolute-action run.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MEMMAP="${MEMMAP:-/data2/cym/acmt_act_memmap_v1/16mm-peg-in-hole}"
OUTPUT="${OUTPUT:-/data2/cym/16mm_peg_in_hole/acmt_act/none/independent_resnet50/seed42}"
LOG_ROOT="${LOG_ROOT:-/data2/cym/acmt_act_logs/corrected_peg_none_200k}"
STEPS="${STEPS:-200000}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
mkdir -p "${LOG_ROOT}"

[[ -x "${PYTHON}" ]] || { echo "missing Python interpreter: ${PYTHON}" >&2; exit 2; }
[[ -f "${MEMMAP}/manifest.json" ]] || { echo "missing Memmap: ${MEMMAP}" >&2; exit 2; }
[[ -f "${MEMMAP}/splits.json" ]] || { echo "missing split file: ${MEMMAP}/splits.json" >&2; exit 2; }
[[ -f "${MEMMAP}/acmt_act_targets.npz" ]] || { echo "missing target sidecar: ${MEMMAP}/acmt_act_targets.npz" >&2; exit 2; }
[[ -f "${MEMMAP}/acmt_act_policy_stats.json" ]] || { echo "missing residual stats: ${MEMMAP}/acmt_act_policy_stats.json" >&2; exit 2; }

PREFLIGHT_DIR="${LOG_ROOT}/preflight"
mkdir -p "${PREFLIGHT_DIR}"
if [[ ! -f "${PREFLIGHT_DIR}/preflight.ok" ]]; then
  "${PYTHON}" -u -m lerobot.scripts.acmt_act_preflight \
    --memmap-dir="${MEMMAP}" --tactile-source=none --task=peg \
    --policy-type=acmt_act --device=cuda --batch-size=16 --steps=20 \
    >"${PREFLIGHT_DIR}/preflight.log" 2>&1
  touch "${PREFLIGHT_DIR}/preflight.ok"
fi

LAST="${OUTPUT}/checkpoints/last"
STEP=0
if [[ -f "${LAST}/training_state/training_step.json" ]]; then
  STEP="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "${LAST}/training_state/training_step.json")"
fi
if [[ "${STEP}" -ge "${STEPS}" ]]; then
  echo "[SKIP] peg/none already complete: ${STEP}/${STEPS}" | tee -a "${LOG_ROOT}/launcher.log"
  exit 0
fi
if [[ -d "${OUTPUT}" && "${STEP}" -eq 0 ]]; then
  echo "output exists without a resumable corrected checkpoint: ${OUTPUT}" >&2
  echo "archive the legacy output before starting this launcher" >&2
  exit 1
fi

RESUME_ARGS=()
if [[ "${STEP}" -gt 0 ]]; then
  RESUME_ARGS=(--resume=true "--config_path=${LAST}/pretrained_model/train_config.json")
fi

"${PYTHON}" -u -m lerobot.scripts.lerobot_train \
  --policy.type=acmt_act \
  --policy.tactile_source=none \
  --policy.task_variant=peg \
  --policy.checkpoint_schema=acmt_act.v3 \
  --policy.checkpoint_schema_version=3 \
  --policy.training_contract=residual_joint_physical_gripper_visual_goal_v1 \
  --policy.camera_backbone_mode=independent \
  --policy.vision_backbone=resnet50 \
  --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V2 \
  --policy.device=cuda \
  --policy.dtype=float16 \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --dataset.backend=acmt_act_memmap \
  --dataset.repo_id=local/acmt-act-peg-corrected \
  --dataset.root="${MEMMAP}" \
  --dataset.split_file="${MEMMAP}/splits.json" \
  --dataset.eval_split=0.05 \
  --batch_size=16 \
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
  "${RESUME_ARGS[@]}"

# Keep ``last`` as the exact resume pointer, and expose the lowest validation
# loss as ``best`` for deployment.  The training loop writes one eval line at
# every 20k checkpoint; selecting it only after completion avoids ever using
# test data or changing the optimizer trajectory.
select_best_checkpoint() {
  local best_step=""
  local log_file="${LOG_ROOT}/launcher.log"
  if [[ -f "${log_file}" ]]; then
    best_step="$(sed -nE 's/.*step ([0-9]+): eval_loss=([-+0-9.eE]+).*/\1 \2/p' "${log_file}" \
      | sort -k2,2g | head -n1 | awk '{printf "%06d", $1}')"
  fi
  if [[ -n "${best_step}" && -d "${OUTPUT}/checkpoints/${best_step}" ]]; then
    ln -sfn "${best_step}" "${OUTPUT}/checkpoints/best"
    echo "[BEST] ${OUTPUT}: step=${best_step}" | tee -a "${LOG_ROOT}/launcher.log"
  else
    echo "[BEST] ${OUTPUT}: no complete eval checkpoint found" | tee -a "${LOG_ROOT}/launcher.log"
  fi
}

select_best_checkpoint
