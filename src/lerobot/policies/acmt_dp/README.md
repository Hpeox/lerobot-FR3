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

`tactigen` is causal and synchronous: the first call uses zero tactile input;
subsequent calls generate tactile input from the previous four-frame window and
the first action of the previous plan. Every call replans a `[B, 16, 8]` chunk
and `select_action()` returns only its first action. Call `reset()` at every
episode boundary.

The default wrist cameras are `camera.cam1` and `camera.cam2`. Their RGB and
depth frames are cropped to the training ROI `(y0, y1, x0, x1) =
(176, 304, 256, 384)`. In addition to `observation.state`, all modes require
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
