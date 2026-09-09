# ACMT-PI05 Peg

ACMT-PI05 uses the existing four-way RGB/tactile Memmap for training.  The
Memmap is a training backend only; deployment reads the live FR3 observation
stream.  The policy receives one instruction, the 8-D state and the two Xense
force fields.  `none` replaces both fields with zeros, while `real` reads them
from SensorHub.  A future `substitution` run uses the `real` checkpoint and
injects the causal ACMT output through the private runtime tactile key.

The live camera IDs are remapped to the training semantic order as follows:

```text
policy cam1/top         <- live cam4
policy cam2/side        <- live cam3
policy cam3/wrist_left  <- live cam1
policy cam4/wrist_right <- live cam2
```

For the persistent MainController, pass these optional arguments:

```bash
--inference-type=rtc \
--inference-rtc-execution-horizon=10 \
--policy-dtype=float16 \
--policy-tactile-source=real \
--rename-map='{"observation.images.camera.cam1.rgb":"observation.images.camera.cam3.rgb","observation.images.camera.cam2.rgb":"observation.images.camera.cam4.rgb","observation.images.camera.cam3.rgb":"observation.images.camera.cam2.rgb","observation.images.camera.cam4.rgb":"observation.images.camera.cam1.rgb"}'
```

RTC is required when relative actions are enabled: synchronous one-action
postprocessing would re-anchor queued relative actions to a later state.
The rollout process must load a local ACMT-PI05 checkpoint whose config has
`checkpoint_schema=acmt_pi05.tactile.v1`; it must not load a native `pi05` or a
v3/v4 ACMT checkpoint directly.

Before starting training, run:

```bash
PYTHON=/home/hk/miniconda3/envs/tactigen-train/bin/python
export PYTHONPATH=/data2/gello-deploy/LerobotFR3/src
"$PYTHON" -u -m lerobot.scripts.acmt_pi05_prepare_memmap \
  --data-dir /data/cym/DATASET/16mm-peg-in-hole \
  --memmap-dir /data2/cym/acmt_act_memmap_v1/16mm-peg-in-hole \
  --progress
```

Then run `bash scripts/train_acmt_pi05_peg.sh`.  It performs a real forward/backward
preflight, keeps effective batch size 8, trains `none` then `real` for 5000
optimizer steps, and stops before training if `lerobot/pi05_base` is not
available locally.

If the base checkpoint is not cached, populate it before launching (the
download must be performed on a host with Hugging Face access):

```bash
hf download lerobot/pi05_base --local-dir /data2/cym/acmt_pi05_assets/pi05_base
BASE_MODEL=/data2/cym/acmt_pi05_assets/pi05_base bash scripts/train_acmt_pi05_peg.sh
```
