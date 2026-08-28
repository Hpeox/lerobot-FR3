# ACMT-DP Native-DP v4

This directory contains the inference-only LeRobot port of upstream ACMT-DP
commit `770a30f`. It is v4-only: v3 center480/DFormer policy checkpoints,
28-dimensional state checkpoints, and the old `generated` mode are rejected.

The policy type remains `acmt_dp` and the three runtime modes remain:

- `none`: four RGB views and four zero tactile frames;
- `real`: four RGB views and the two real Xense force fields;
- `tactigen`: the task-matched real scratch policy plus a frozen TactiGen
  generator. The generator receives only each successfully sent 8-D action,
  never an unexecuted predicted action.

The fixed camera order is `camera.cam1` (top), `camera.cam2` (side),
`camera.cam3` (wrist left), and `camera.cam4` (wrist right). Policy RGB is
provided as raw `480x640` frames. The model applies the native transform:
resize to 256, center crop to 224, clamp/round to uint8, divide by 255, and
ImageNet normalization. TactiGen wrist RGB-D retains the existing center480 to
128 preprocessing on its private branch.

FR3 runtime observations retain the physical RealSense IDs `camera.cam1` through
`camera.cam4`. The v4 processor adapts the current deployment to the semantic
policy views by reading them in the order `camera.cam3`, `camera.cam4`,
`camera.cam1`, `camera.cam2` and writing those values into the policy slots
`top`, `side`, `wrist_left`, and `wrist_right`. This is a policy-side input
permutation; RealSense topics, SensorHub ordering, and the checkpoint weights
are unchanged. The mapping assumes the current mounting contract and must be
updated with the processor configuration if the cameras are remounted.

Native v4 uses a shared scratch ResNet18, an 8-D Gaussian-normalized state,
the shared spatial force-field CNN (160-D per frame, no GRU), and a configurable
8-step or 100-step diffusion sampler. Its condition is four frames of four
camera features, state, and tactile features. `predict_action_chunk()` returns
`[B,16,8]`; `select_action()` returns the first `[B,8]` action. Training methods
are intentionally unavailable.

Online rollout uses LeRobot's ordinary synchronous inference path: each control
tick calls `select_action()`, which computes a fresh native 16-step plan and
returns its first `[B,8]` action. The rollout layer does not maintain an
ACMT-DP-specific rolling/action queue or background planner. The v4 policy
contract remains `control_hz=30`, but the effective loop rate is bounded by the
synchronous inference latency. For the production real-time setting use 8
denoising steps; 100 steps are available for slower synchronous experiments
and do not change the model weights. TactiGen starts with four zero frames, and
`reset()` clears all visual/state/tactile causal history. RTC is unsupported
for `tactigen`.

Example:

```bash
lerobot-rollout \
  --policy.path=outputs/acmt_dp/peg/tactigen/seed42/pretrained_model \
  --policy.tactile_source=tactigen \
  --inference.type=sync
```
Convert only v4 scratch checkpoints. The converter performs strict EMA state
mapping and atomic output replacement; it does not upload checkpoints:

```bash
lerobot-convert-acmt-dp-checkpoint \
  --task peg --mode none \
  --policy-checkpoint /path/to/native_dp_v4/none/scratch/seed42/best.pt \
  --output-root outputs/acmt_dp
```

## Native-DP v5

The v5 implementation follows upstream commit `f6306c9` and is selected with
`--policy.type=acmt_dp_v5`. It supports `none`, `real`, and `tactigen` tactile
modes; `tactigen` reuses a task-matched real policy checkpoint and embeds one
frozen TactiGen generator.

The independent policy type `acmt_dp_v5` loads only checkpoints with schema
`acmt_dp.native_dp_v5_hybrid`. It keeps the raw four-camera 4:3 input and
performs resize `240x320`, center crop `216x288`, and `[0,1] -> [-1,1]`
normalization online. Each camera has its own scratch ResNet18Conv with
GroupNorm and a 32-point SpatialSoftmax, producing 64 values; no BatchNorm is
present. The observation history is four frames with official first-frame
padding. The internal 19-step prediction is exposed as
`prediction_raw[:,3:19]` (`[B,16,8]`), and the runtime queue executes its first
eight actions.

Convert v5 checkpoints with the separate converter; it never accepts v3/v4
artifacts:

```bash
python -m lerobot.scripts.convert_acmt_dp_v5_checkpoint \
  --task peg --mode real \
  --policy-checkpoint /data2/cym/16mm_peg_in_hole/native_dp_v5/real/scratch/seed42/best.pt \
  --output-root outputs/acmt_dp_v5
```

Launch with `--policy.type=acmt_dp_v5` and choose
`--policy.tactile_source=none|real|tactigen`. A real v5 policy can run with
`real`, while `tactigen` additionally requires the task-matched frozen ACMT
generator inputs. Memmap files are training-only and are not read by this
runtime. DDIM uses `eta=0`; the policy keeps one initial noise tensor per
episode and clears it on `reset()`, so replanning does not randomly switch
between action modes. The 16/8 rollout keeps eight reserve commands and
blends timestamp-overlapping reserve/execution commands from `0.25` new-plan
weight at the boundary.
