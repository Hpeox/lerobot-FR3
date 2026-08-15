# ACMT-DP policy

This directory contains the inference-only LeRobot port of ACMT-DP. Runtime
loading does not import the legacy `/cym/TactiGen/ACMT-DP` or ACMTv4 source
trees. `tactigen` checkpoints contain the task-matched real-policy weights plus
their full TactiGen force generator.

Select one of the policy modes by pairing its checkpoint directory with the
same `tactile_source` value:

```bash
lerobot-rollout \
  --policy.path=outputs/acmt_dp/peg/tactigen/seed42/pretrained_model \
  --policy.tactile_source=tactigen \
  --inference.type=sync
```

The v3 policy uses a four-frame tactile history. `none` feeds four zero frames,
`real` consumes the two `(35,20,3)` Xense fields, and `tactigen` starts with four
zero frames and then calls the embedded frozen generator once per successfully
sent action. The generator receives the previous four observations and the
actual `[B,8]` command; predicted future actions are never used.

Online rollout uses a `[16,8]` diffusion plan at 30 Hz: the first eight actions
are executed and the remaining eight provide a planning reserve. The sync
inference engine atomically replaces only unexecuted future actions when a new
plan is ready, drops expired timestamps, and returns no command (robot hold /
safe stop) if the reserve is exhausted before replanning finishes; it never
replays stale actions. RTC is rejected for ACMT-DP. Call `reset()` at every
episode boundary.

The default wrist cameras are `camera.cam1` and `camera.cam2`. Raw RGB-D frames
are expected as `[3,480,640]` and `[1,480,640]`; v3 crops columns `80:560` and
resizes to 128×128 with antialiased bilinear RGB and nearest-neighbour depth.
In addition to `observation.state`, all modes require
`observation.fr3.dq`, `observation.fr3.tau_J`, the FT300 wrench, and gripper
`gPO`. `real` requires both Xense force fields; `tactigen` requires
`observation.fr3.O_T_EE` with shape `(4, 4)`.

The `tactigen` policy checkpoint is always based on the task-matched `real`
policy checkpoint. The former independent `generated` policy checkpoint format
is deprecated and rejected.

Recreate checkpoints with:

```bash
lerobot-convert-acmt-dp-checkpoint --task all --mode all \
  --output-root outputs/acmt_dp
```

The converter overlays EMA floating-point tensors on the complete legacy state,
checks every target key and shape strictly, writes LeRobot processors, and
records source SHA256 values in `conversion_manifest.json`.
