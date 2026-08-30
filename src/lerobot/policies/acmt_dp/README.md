# ACMT-DP Native-DP v4/v5

This directory contains the inference-only LeRobot ports of ACMT-DP v4 and v5.
The v4 policy follows upstream commit `770a30f`; v3 center480/DFormer and
28-dimensional state checkpoints are rejected. The v5 policy uses the
independent `acmt_dp_v5` type and schema; both policy families reject the old
`generated` mode.
The policy type remains `acmt_dp` and the three runtime modes remain:

- `none`: four RGB views and four zero tactile frames;
- `real`: four RGB views and the two real Xense force fields;
- `tactigen`: the task-matched real scratch policy plus a frozen TactiGen
  generator. The generator receives only each successfully sent 8-D action,
  never an unexecuted predicted action.

The policy semantic slots are `camera.cam1` (top), `camera.cam2` (side),
`camera.cam3` (wrist left), and `camera.cam4` (wrist right). Policy RGB is
provided as raw `480x640` frames. The model applies the native transform:
resize to 256, center crop to 224, clamp/round to uint8, divide by 255, and
ImageNet normalization. TactiGen wrist RGB-D retains the existing center480 to
128 preprocessing on its private branch.

FR3 runtime observations retain the physical RealSense IDs `camera.cam1` through
`camera.cam4`. The v4 processor adapts the current deployment to the semantic
policy views by reading them in the order `camera.cam4`, `camera.cam3`,
`camera.cam2`, `camera.cam1` and writing those values into the policy slots
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
`acmt_dp.native_dp_v5_hybrid`. It keeps the raw four-camera 4:3 input.
The processor preserves the deployment camera contract by reading runtime `camera.cam4`, `camera.cam3`, `camera.cam2`, `camera.cam1` into policy semantic slots `top`, `side`, `wrist_left`, `wrist_right`; the serialized `source_camera_keys` field makes this permutation explicit. The policy postprocessor maps the opening action from `[0,1]` to the existing normalized gPO wire range `[255,3]`.
The model performs resize `240x320`, center crop `216x288`, and `[0,1] -> [-1,1]`
normalization online. Each camera has its own scratch ResNet18Conv with
GroupNorm and a 32-point SpatialSoftmax, producing 64 values; no BatchNorm is
present. The observation history is four frames with official first-frame
padding. The internal 19-step prediction is exposed as
`prediction_raw[:,3:19]` (`[B,16,8]`); the rolling v5 engine consumes one action per 30 Hz deadline.

Convert v5 checkpoints with the separate converter; it never accepts v3/v4
artifacts:

```bash
python -m lerobot.scripts.convert_acmt_dp_v5_checkpoint \
  --task gear --mode none \
  --policy-checkpoint /data/internal/ACMT-DP-gear-runs/gear_insert_big2small/native_dp_v5/none/scratch/seed42/best.pt \
  --output-root outputs/acmt_dp
```

Launch with `--policy.type=acmt_dp_v5` and choose
`--policy.tactile_source=none|real|tactigen`. A real v5 policy can run with
`real`, while `tactigen` additionally requires the task-matched frozen ACMT
generator inputs. Memmap files are training-only and are not read by this
runtime. DDIM uses `eta=0`; the policy creates initial noise with private seed
`42`, reuses it for all replans in an episode, and clears it on `reset()`. The
rolling 16/8 engine consumes one action at each 30 Hz deadline, starts a
background refill when eight actions remain, replaces only future deadlines,
skips expired actions, and clips/logs outputs to checkpoint action bounds.
Overlap blending is disabled.
