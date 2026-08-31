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

Native v4 uses a shared scratch ResNet18, an 8-D Gaussian-normalized state,
the shared spatial force-field CNN (160-D per frame, no GRU), and an 8-step
diffusion sampler. Its condition is four frames of four camera features,
state, and tactile features. `predict_action_chunk()` returns `[B,16,8]`;
`select_action()` returns the first `[B,8]` action. Training methods are
intentionally unavailable.

Online rollout keeps the existing LeRobot 16/8, 30 Hz runtime: eight commands
execute while eight remain as a replanning reserve. TactiGen starts with four
zero frames, and `reset()` clears all visual/state/tactile causal history.
RTC is unsupported for `tactigen`.

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
`acmt_dp.native_dp_v5_robomimic_hybrid`. It constructs the official Robomimic
0.2.0 `ObservationEncoder`: four independent scratch ResNet18Conv keys,
GroupNorm, per-key CropRandomizer, ReLU feature activation, and a 32-point
SpatialSoftmax producing 64 values per camera. It keeps the raw four-camera
4:3 input and performs resize `240x320`, center crop `216x288`, and
`[0,1] -> [-1,1]` normalization online; no BatchNorm is present. The
observation history is four frames with official first-frame padding. The
internal 19-step prediction is exposed as
`prediction_raw[:,3:19]` (`[B,16,8]`), and the runtime queue executes its first
eight actions.

Install the optional dependency before constructing a v5 policy:

```bash
pip install 'lerobot[acmt-dp]'
```

This pins `robomimic==0.2.0`; v5 imports remain available without the extra,
but model construction reports the installation command when the encoder is
needed.

Convert v5 checkpoints with the separate converter; it never accepts v3/v4
artifacts:

```bash
python -m lerobot.scripts.convert_acmt_dp_v5_checkpoint \
  --task peg --mode real \
  --policy-checkpoint /data2/cym/16mm_peg_in_hole/native_dp_v5/real/robomimic_official/seed42/best.pt \
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
