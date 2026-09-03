#!/usr/bin/env bash
set -euo pipefail

# ACMT-ACT v2: four independent ResNet18 backbones, physical/effective
# batch-size 16, no gradient accumulation, 200k optimizer updates per run.
# The four runs are intentionally serialized so one 5090 is never shared by
# competing jobs.  A completed run is skipped; an interrupted run resumes
# from its last LeRobot checkpoint after validating the schema/task/mode.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MEMMAP_ROOT="${MEMMAP_ROOT:-/data2/cym/acmt_act_memmap_v1}"
MEMMAP_PEG="${MEMMAP_PEG:-${MEMMAP_ROOT}/16mm-peg-in-hole}"
MEMMAP_GEAR="${MEMMAP_GEAR:-${MEMMAP_ROOT}/gear-insert-big2small}"
OUTPUT_PEG="${OUTPUT_PEG:-/data2/cym/16mm_peg_in_hole/acmt_act}"
OUTPUT_GEAR="${OUTPUT_GEAR:-/data2/cym/gear_insert_big2small/acmt_act}"
LOG_ROOT="${LOG_ROOT:-/data2/cym/acmt_act_logs/independent_resnet18_200k}"
STEPS="${STEPS:-200000}"
mkdir -p "${LOG_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing Python interpreter: ${PYTHON}" >&2
  exit 2
fi

"${PYTHON}" -c 'import accelerate' >/dev/null 2>&1 || {
  echo "accelerate is missing from ${PYTHON}; install lerobot[training] before starting 200k training" >&2
  exit 2
}

run_one() {
  local task="$1"
  local source="$2"
  local output="$3"
  local log="${LOG_ROOT}/${task}_${source}.log"
  local root=""

  if [[ "${task}" == "peg" ]]; then
    root="${MEMMAP_PEG}"
  else
    root="${MEMMAP_GEAR}"
  fi
  [[ -f "${root}/manifest.json" ]] || { echo "missing memmap: ${root}" >&2; return 1; }
  [[ -f "${root}/splits.json" ]] || { echo "missing split file: ${root}/splits.json" >&2; return 1; }

  local preflight_dir="${LOG_ROOT}/preflight/${task}_${source}"
  mkdir -p "${preflight_dir}"
  if [[ ! -f "${preflight_dir}/preflight.ok" ]]; then
    echo "[PREFLIGHT] ${task}/${source}" | tee -a "${LOG_ROOT}/all_train.log"
    "${PYTHON}" -u -m lerobot.scripts.acmt_act_preflight \
      --memmap-dir="${root}" --tactile-source="${source}" --task="${task}" \
      --device=cuda --batch-size=16 --steps=20 \
      >"${preflight_dir}/preflight.log" 2>&1
    touch "${preflight_dir}/preflight.ok"
  fi

  local last="${output}/checkpoints/last"
  local step=0
  if [[ -f "${last}/training_state/training_step.json" ]]; then
    step="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "${last}/training_state/training_step.json")"
  fi
  if [[ "${step}" -ge "${STEPS}" ]]; then
    echo "[SKIP] ${task}/${source}: ${step}/${STEPS} optimizer steps already complete" | tee -a "${LOG_ROOT}/all_train.log"
    return 0
  fi

  if [[ -d "${output}" && "${step}" -eq 0 ]]; then
    echo "output directory exists without a resumable checkpoint: ${output}" >&2
    echo "remove or rename it before a fresh run; the launcher will not overwrite it" >&2
    return 1
  fi

  local -a resume_args=()
  if [[ "${step}" -gt 0 ]]; then
    [[ -f "${last}/pretrained_model/train_config.json" ]] || { echo "checkpoint is missing train_config.json: ${last}" >&2; return 1; }
    resume_args=(--resume=true "--config_path=${last}/pretrained_model/train_config.json")
  fi

  echo "[START] ${task}/${source} from step ${step}" | tee -a "${LOG_ROOT}/all_train.log"
  "${PYTHON}" -u -m lerobot.scripts.lerobot_train \
    --policy.type=acmt_act \
    --policy.tactile_source="${source}" \
    --policy.task_variant="${task}" \
    --policy.checkpoint_schema=acmt_act.v2 \
    --policy.checkpoint_schema_version=2 \
    --policy.camera_backbone_mode=independent \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.device=cuda \
    --policy.dtype=float16 \
    --policy.use_amp=true \
    --policy.push_to_hub=false \
    --dataset.backend=acmt_act_memmap \
    --dataset.repo_id="local/acmt-act-${task}" \
    --dataset.root="${root}" \
    --dataset.split_file="${root}/splits.json" \
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
    --output_dir="${output}" \
    --wandb.enable=false \
    "${resume_args[@]}" \
    >"${log}" 2>&1
  echo "[DONE] ${task}/${source}" | tee -a "${LOG_ROOT}/all_train.log"
}

run_one peg none "${OUTPUT_PEG}/none/independent_resnet18/seed42"
run_one gear none "${OUTPUT_GEAR}/none/independent_resnet18/seed42"
run_one peg real "${OUTPUT_PEG}/real/independent_resnet18/seed42"
run_one gear real "${OUTPUT_GEAR}/real/independent_resnet18/seed42"
echo "[DONE] all ACMT-ACT v2 runs" | tee -a "${LOG_ROOT}/all_train.log"
