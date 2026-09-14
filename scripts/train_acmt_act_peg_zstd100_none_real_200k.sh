#!/usr/bin/env bash
set -euo pipefail

# Full-train (no validation/test) ACMT-ACT v3 run for the 100-demo Peg
# inventory.  The only published pointer is checkpoints/last; this launcher
# never creates or selects a best checkpoint.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MEMMAP="${MEMMAP:-/data2/cym/acmt_act_memmap_v1/16mm-peg-in-hole-zstd100-20260910}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/cym/16mm_peg_in_hole/acmt_act/zstd100_20260910}"
LOG_ROOT="${LOG_ROOT:-/data2/cym/acmt_act_logs/zstd100_20260910_resnet50_200k}"
STEPS="${STEPS:-200000}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
mkdir -p "${LOG_ROOT}"

[[ -x "${PYTHON}" ]] || { echo "missing Python interpreter: ${PYTHON}" >&2; exit 2; }
[[ -f "${MEMMAP}/manifest.json" ]] || { echo "missing Memmap: ${MEMMAP}" >&2; exit 2; }
[[ -f "${MEMMAP}/splits.json" ]] || { echo "missing all-train split: ${MEMMAP}/splits.json" >&2; exit 2; }
[[ -f "${MEMMAP}/acmt_act_targets.npz" ]] || { echo "missing target sidecar" >&2; exit 2; }
[[ -f "${MEMMAP}/acmt_act_policy_stats.json" ]] || { echo "missing residual-action stats" >&2; exit 2; }

run_preflight() {
  local source="$1"
  local marker="${LOG_ROOT}/preflight_${source}.ok"
  [[ -f "${marker}" ]] && return 0
  echo "[PREFLIGHT] peg/${source}: 20 steps, physical batch 16" | tee -a "${LOG_ROOT}/all_train.log"
  "${PYTHON}" -u -m lerobot.scripts.acmt_act_preflight \
    --memmap-dir="${MEMMAP}" --tactile-source="${source}" --task=peg \
    --policy-type=acmt_act --device=cuda --batch-size=16 \
    --gradient-accumulation-steps=1 --steps=20 \
    >"${LOG_ROOT}/preflight_${source}.log" 2>&1
  touch "${marker}"
}

run_one() {
  local source="$1"
  local output="${OUTPUT_ROOT}/${source}/independent_resnet50/seed42"
  local log="${LOG_ROOT}/peg_${source}.log"
  local last="${output}/checkpoints/last"
  local step=0
  if [[ -f "${last}/training_state/training_step.json" ]]; then
    step="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "${last}/training_state/training_step.json")"
  fi
  if [[ "${step}" -ge "${STEPS}" ]]; then
    echo "[SKIP] peg/${source}: last=${step}/${STEPS}" | tee -a "${LOG_ROOT}/all_train.log"
    return 0
  fi
  if [[ -d "${output}" && "${step}" -eq 0 ]]; then
    echo "output exists without resumable checkpoint: ${output}" >&2
    echo "archive it explicitly; this launcher never overwrites a run" >&2
    return 1
  fi
  local -a resume_args=()
  if [[ "${step}" -gt 0 ]]; then
    [[ -f "${last}/pretrained_model/train_config.json" ]] || {
      echo "last checkpoint has no train_config.json: ${last}" >&2
      return 1
    }
    resume_args=(--resume=true "--config_path=${last}/pretrained_model/train_config.json")
  fi

  echo "[START] peg/${source} from ${step}/${STEPS}" | tee -a "${LOG_ROOT}/all_train.log"
  # A power-loss recovery must retain the previous progress log.  A fresh
  # run starts a new file; a resumed run appends to the checkpoint's log.
  if [[ "${step}" -gt 0 ]]; then
    exec 3>>"${log}"
  else
    exec 3>"${log}"
  fi
  "${PYTHON}" -u -m lerobot.scripts.lerobot_train \
    --policy.type=acmt_act \
    --policy.tactile_source="${source}" \
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
    --dataset.repo_id=local/acmt-act-peg-zstd100 \
    --dataset.root="${MEMMAP}" \
    --dataset.split_file="${MEMMAP}/splits.json" \
    --dataset.eval_split=0.0 \
    --batch_size=16 \
    --steps="${STEPS}" \
    --eval_steps=0 \
    --save_freq=20000 \
    --log_freq=100 \
    --env_eval_freq=0 \
    --num_workers=4 \
    --prefetch_factor=2 \
    --persistent_workers=true \
    --seed=42 \
    --output_dir="${output}" \
    --wandb.enable=false \
    "${resume_args[@]}" \
    >&3 2>&1
  exec 3>&-

  [[ -f "${last}/training_state/training_step.json" ]] || {
    echo "training ended without checkpoints/last: ${output}" >&2
    return 1
  }
  local final_step
  final_step="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "${last}/training_state/training_step.json")"
  [[ "${final_step}" -eq "${STEPS}" ]] || {
    echo "last checkpoint step ${final_step} != ${STEPS}: ${output}" >&2
    return 1
  }
  echo "[DONE] peg/${source}: last=${final_step}" | tee -a "${LOG_ROOT}/all_train.log"
}

run_preflight none
run_preflight real
run_one none
run_one real
echo "[DONE] Peg zstd100 none -> real full-train pipeline" | tee -a "${LOG_ROOT}/all_train.log"
