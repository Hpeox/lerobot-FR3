# FR3 Robot and SensorHub Implementation Handover Report

## Document Status

| Item | Value |
| --- | --- |
| Report date | 2026-07-22 |
| Repository branch | `wip/FR3` |
| LeRobot version | `0.6.0` |
| Supported Python | `>=3.12` |
| Hardware acceptance | Not completed |

This report describes the current FR3 implementation. It is intended for the engineer who will
integrate the external processes, perform hardware acceptance, and own subsequent maintenance. The
byte-level protocol reference is the
[FR3 integration guide](docs/source/fr3.mdx).

## Executive Summary

The implementation adds a LeRobot `FR3` Robot backed by a managed SensorHub subprocess. SensorHub
attaches read-only to four RealSense shared-memory writers, one dual-Xense writer, one FT300S
writer, and an FGT1 ZMQ telemetry stream. It causally aligns these sources and publishes coherent
observations through a two-slot shared-memory mapping owned by SensorHub.

The Robot exposes the aligned data through the normal LeRobot observation interface and publishes
eight-field actions through a latest-only ZMQ `PUB` socket. Explicit dataset feature definitions
and an FR3 policy observation processor keep the Xense arrays numeric, convert array layouts for
policy input, normalize RGB data, and retain the native Z16 depth scale.

The implementation includes focused protocol, alignment, shared-memory, lifecycle, feature-schema,
and processor tests. It has not yet been accepted against the complete physical FR3 system.

## Scope and Ownership Boundaries

### Implemented and owned by `FR3`

- LeRobot Robot configuration, factory registration, and public Robot interface.
- Creation, supervision, and termination of one SensorHub subprocess per connected Robot instance.
- The SensorHub UDS control connection and aligned-observation shared-memory client.
- The local ZMQ command `PUB` socket connected to `tcp://192.168.1.37:6001`.
- Action validation, gripper clipping/conversion, command ABI packing, and publication.
- Observation feature declarations, dataset schemas, and deterministic policy layout conversion.

### External and not implemented here

- The run-level external controller. It does not exist yet and is future work.
- Four RealSense shared-memory writer processes.
- The dual-Xense and FT300S shared-memory writer processes.
- The FGT1 telemetry relay at `tcp://192.168.1.37:6000`.
- The remote command `SUB` process bound to `tcp://*:6001`.
- Remote robot safety checks, joint limits, command watchdog behavior, and timeout hold behavior.
- Starting, monitoring, restarting, or terminating any of the external processes above.

`FR3.disconnect()` only stops resources owned by the Robot. It must never signal or unlink an
upstream writer, telemetry relay, or remote controller resource.

## System Architecture

```text
External processes                                         LeRobot process

4 x RealSense SHM writers --\
Dual-Xense SHM writer -----+--> SensorHub subprocess --> /fr3_aligned_observation
FT300S SHM writer ---------+       ^       |                         |
FGT1 telemetry ZMQ PUB ----/       |       +-- UDS status/control    +--> FR3.get_observation()
                                   |                                      |
                                   +------------ managed by FR3 ----------+

Policy/teleoperator --> FR3.send_action() --> ZMQ PUB, latest-only --> remote command SUB
```

### Observation path

1. External writers and the telemetry relay are started and verified by an operator or temporary
   orchestration script.
2. `FR3.connect()` starts SensorHub with the configured source names, endpoints, timeouts, parent
   PID, and UDS socket path.
3. SensorHub defensively attaches to the already-running sources and validates their fixed ABIs.
4. Dedicated reader threads feed bounded sample caches. Robot and gripper telemetry use independent
   caches even though both arrive on the same FGT1 stream.
5. The causal aligner selects a complete, fresh set of required samples and publishes it through
   `/fr3_aligned_observation`.
6. SensorHub reports `READY` only after every required source has produced at least two advancing
   samples and the first aligned snapshot has been published.
7. `FR3.get_observation()` copies a stable shared-memory slot into Robot-owned memory before
   returning arrays, so later SensorHub writes cannot mutate a snapshot already given to a policy.

All modalities are mandatory. Startup fails within the configured timeout if a source is missing,
invalid, static, or cannot participate in an aligned snapshot.

### Action path

1. `FR3.send_action()` requires exactly seven joint position fields and one policy-scale gripper
   field.
2. It rejects booleans, non-numeric values, non-finite values, missing fields, and extra fields.
3. Joint targets are transported without local limiting. Safety limits remain a remote-controller
   responsibility.
4. `gripper.pos` is clipped to `[0, 1]` and converted to `uint8 gPO` using
   `floor(value * 255 + 0.5)`.
5. A fixed 112-byte command frame is sent non-blocking through the latest-only ZMQ `PUB` socket.
6. `send_action()` returns the eight policy-scale values actually submitted, including the clipped
   `gripper.pos` value.

PUB/SUB provides no delivery acknowledgement. The remote subscriber must be running before the
Robot begins publishing, and the remote real-time controller must enforce its own watchdog.

## Implemented Components

### FR3 Robot and configuration

- `FR3Config` is registered as `--robot.type=fr3`, validates all timeout/cache values, and provides
  the JSON-safe SensorHub configuration.
- `FR3` implements the LeRobot lifecycle and exposes explicit action, observation, visual-feature,
  and dataset-feature declarations.
- The Robot factory and package registration recognize the FR3 configuration.
- `connect()` is transactional: a startup failure closes local resources and terminates the managed
  SensorHub process group.
- `disconnect()` requests UDS `SHUTDOWN`, waits for graceful exit, and escalates to process-group
  `SIGTERM` and `SIGKILL` only when required.

### SensorHub

- Strict readers support the RealSense POSIX SHM v1 layouts, dual-Xense v2 layout, FT300S v2 layout,
  and FGT1 v1 telemetry frames.
- Xense and FT readers open `/dev/shm` mappings read-only instead of using Python ownership APIs,
  preventing accidental unlinking of externally owned shared memory.
- A bounded cache and causal aligner enforce freshness and camera skew limits while suppressing
  duplicate aligned output.
- The aligned writer owns one fixed two-slot seqlock mapping and records fatal state in its global
  header.
- Runtime supervision detects parent death, source stalls, sustained alignment failure, and reader
  errors. A fatal condition stops publication, records the failure, sends `FATAL`, and exits.
- There is intentionally no reader reconnection, SHM remapping, local recovery, or automatic
  SensorHub restart in this version.

### Features and policy processing

- The dataset adapter explicitly separates state arrays from camera data. Xense is stored as two
  independent numeric `float32` arrays shaped `(35, 20, 3)`; it is not encoded as image or video.
- Four RGB arrays use shape `(480, 640, 3)` and four depth arrays use shape `(480, 640, 1)`.
- The FR3 policy processor converts HWC to BCHW, BHWC to BCHW, and BTHWC to BTCHW.
- RGB is converted to `float32` and normalized to `[0, 1]`.
- Depth is converted to `float32` without dividing the Z16 values, preserving the depth scale.
- Xense is converted to channel-first layout without combining the two sensors or performing CNN
  fusion.
- The processor performs no sensor I/O, temporal alignment, safety control, or feature encoding.

### Generic LeRobot integration

- Recording and rollout use a Robot-provided observation dataset schema when available.
- Rollout visual-feature checks use the Robot's explicit visual key list, keeping Xense out of the
  camera path.
- Dataset frame construction accepts independent multidimensional numeric features while retaining
  the existing named-vector behavior.
- FR3 array layout preparation is applied on the inference path.
- The `fr3` optional dependency installs the existing `pyzmq` dependency group, and SensorHub is
  also exposed as the `lerobot-fr3-sensorhub` console script.

## Public Interfaces

### Configuration defaults

| Field | Default |
| --- | --- |
| `command_endpoint` | `tcp://192.168.1.37:6001` |
| `telemetry_endpoint` | `tcp://192.168.1.37:6000` |
| `observation_shm_name` | `/fr3_aligned_observation` |
| `sensorhub_socket_path` | `/run/user/<uid>/fr3_sensorhub.sock` |
| `realsense_shm_names` | `/realsense_cam1` through `/realsense_cam4` |
| `xense_shm_name` | `xense_sensor_frame` |
| `ft300s_shm_name` | `ft300_sensor_frame` |
| `sensorhub_start_timeout_s` | `10.0` |
| `sensorhub_stop_timeout_s` | `2.0` |
| `snapshot_read_timeout_ms` | `20` |
| `max_snapshot_age_ms` | `100` |
| `cache_horizon_s` | `0.5` |
| `camera_max_skew_ms` | `50` |
| `required_sample_max_age_ms` | `100` |
| `camera_xense_stall_timeout_ms` | `200` |
| `ft_robot_gripper_stall_timeout_ms` | `100` |
| `alignment_failure_timeout_ms` | `500` |

Every timeout and cache duration must be positive, and exactly four RealSense SHM names are required.

### Action schema

| Fields | Count | Type and units |
| --- | ---: | --- |
| `fr3_joint1.pos` ... `fr3_joint7.pos` | 7 | finite float, radians |
| `gripper.pos` | 1 | finite float, policy scale; clipped to `[0, 1]` |

The dataset action feature is one `float32` vector of shape `(8,)` with the fields in the order shown
above.

### Observation schema

| Fields | Shape/type | Meaning |
| --- | --- | --- |
| `fr3_joint1.pos` ... `fr3_joint7.pos` | scalar float | Joint position in radians |
| `fr3.dq` | `float32 (7,)` | Joint velocity |
| `fr3.tau_J` | `float32 (7,)` | Joint torque |
| `gripper.pos` | scalar float | Normalized `gPO / 255` |
| `gripper.gPO` | `uint8` | Raw gripper position feedback |
| `gripper.gCU` | `uint8` | Raw gripper current feedback |
| `ft300s.wrench` | `float32 (6,)` | FT300S wrench |
| `xense.sensor0.force_field` | `float32 (35, 20, 3)` | First tactile force field |
| `xense.sensor1.force_field` | `float32 (35, 20, 3)` | Second tactile force field |
| `camera.cam1.rgb` ... `camera.cam4.rgb` | `uint8 (480, 640, 3)` | RGB images |
| `camera.cam1.depth` ... `camera.cam4.depth` | `uint16 (480, 640, 1)` | Z16 depth images |

The dataset schema groups the seven joint positions and normalized gripper position into
`observation.state` with shape `(8,)`. Velocity, torque, raw gripper values, wrench, and Xense fields
remain independent dataset features.

### Lifecycle methods

| Method | Contract |
| --- | --- |
| `connect()` | Starts SensorHub, waits for `READY`, opens aligned SHM, then opens command PUB |
| `get_observation()` | Checks SensorHub health and returns one coherent, fresh, owned snapshot |
| `send_action()` | Validates, clips/converts, publishes one command, and returns the submitted action |
| `disconnect()` | Stops only Robot-owned resources and is safe to call repeatedly |

## Protocol Summary

### Command ABI v1

- Transport: single-part ZMQ `PUB/SUB`, little-endian, exactly 112 bytes.
- Publisher: connects to `tcp://192.168.1.37:6001` with `SNDHWM=1`, `CONFLATE=1`, and `LINGER=0`.
- Subscriber: must bind to `tcp://*:6001`, subscribe to `b""`, and use equivalent latest-only
  behavior.
- Header: magic `FRCMD1\0\0`, ABI version `1`, header size `48`, total size `112`, flags `0`, sequence,
  realtime nanoseconds, and monotonic nanoseconds.
- Payload: seven `float64` joint targets, one `uint8 gPO`, and seven zero padding bytes.

### FGT1 telemetry ABI

- SensorHub accepts only 504-byte `FGT1` version 1 frames.
- Robot source `2` requires valid-mask bit `2` and maps `q=floats[8:15]`,
  `dq=floats[15:22]`, and `tau_J=floats[22:29]`.
- Gripper source `3` requires valid-mask bit `4` and supplies `gPO` and `gCU`.
- Robot and gripper sequences/caches are independent. GELLO source `1` is ignored.
- The multiplexed telemetry subscriber intentionally does not use `CONFLATE`, because conflation
  across sources could starve either the robot or gripper cache. It uses `RCVHWM=100`.

### AlignedObservation SHM ABI v1

- Linux x86-64, little-endian, 8-byte aligned, two-slot seqlock mapping.
- Magic `FR3OBS1\0`, global header `320` bytes, slot header `160` bytes.
- Slot stride `6,161,080` bytes; total mapping size `12,322,480` bytes.
- Snapshot `L` uses slot `L % 2` and is stable only when its seqlock equals `2 * L` before and after
  the reader's copy.
- Payload order is four RGB arrays, four depth arrays, two Xense arrays, wrench, `q`, `dq`, `tau_J`,
  normalized gripper position, `gPO`, `gCU`, and padding.

### SensorHub UDS protocol

- Transport: `AF_UNIX/SOCK_SEQPACKET`; packet size is at most 512 bytes.
- Every JSON packet contains exactly `protocol_version`, `type`, `sequence`, `timestamp_ns`,
  `status_code`, and `message`.
- Supported types: `READY`, `HEALTH`, `FATAL`, `PING`, `PONG`, and `SHUTDOWN`.
- Diagnostic messages are truncated to at most 256 UTF-8 bytes before encoding so a long exception
  cannot prevent delivery of a fatal packet.

See the [FR3 integration guide](docs/source/fr3.mdx) and
[`protocols.py`](src/lerobot/robots/fr3/protocols.py) for the authoritative command and telemetry
definitions. See
[`aligned_shm.py`](src/lerobot/robots/fr3/sensorhub/aligned_shm.py) for the authoritative aligned
mapping layout.

## Operations Runbook

### 1. Install dependencies

From the repository root:

```bash
uv sync --locked --extra fr3
```

The FR3 extra supplies `pyzmq`. The repository requires Python 3.12 or newer.

### 2. Start and verify external prerequisites

Start these components before starting LeRobot:

1. RealSense writers for `/realsense_cam1`, `/realsense_cam2`, `/realsense_cam3`, and
   `/realsense_cam4`.
2. Dual-Xense writer for `xense_sensor_frame`.
3. FT300S writer for `ft300_sensor_frame`.
4. FGT1 telemetry relay publishing at `tcp://192.168.1.37:6000`.
5. Remote command subscriber bound to `tcp://*:6001`, with its watchdog active.

The local SHM files can be checked with:

```bash
ls -l \
  /dev/shm/realsense_cam1 /dev/shm/realsense_cam2 \
  /dev/shm/realsense_cam3 /dev/shm/realsense_cam4 \
  /dev/shm/xense_sensor_frame /dev/shm/ft300_sensor_frame
```

Existence alone is insufficient: the producer sequences must be advancing and the mapping sizes,
encodings, and offsets must match their expected ABIs. SensorHub performs these checks defensively
and will fail startup if they do not hold.

### 3. Start LeRobot

Select the FR3 Robot through the normal CLI configuration. For example:

```bash
uv run lerobot-rollout \
  --robot.type=fr3 \
  --robot.id=fr3 \
  ...
```

The exact remaining rollout or recording options depend on the policy and dataset. Do not start
`lerobot-fr3-sensorhub` separately during normal operation; `FR3.connect()` owns that subprocess.

### 4. Confirm readiness

A successful `connect()` means SensorHub has attached to every source, observed at least two
advancing samples from each, and published an aligned snapshot. It does not prove that the physical
sensors are calibrated, that values are physically plausible, or that command motion is safe.

### 5. Shutdown

Use the normal LeRobot shutdown path so `FR3.disconnect()` runs. It sends UDS `SHUTDOWN`, closes the
command socket and aligned SHM client, and waits for SensorHub. Forced signals are limited to the
managed SensorHub process group. Shut down external writers and controllers separately.

### Diagnostic guide

| Symptom | Likely cause or first check |
| --- | --- |
| SensorHub UDS timeout | SensorHub failed before control socket creation, invalid socket directory, or process launch failure |
| `required upstream writers were not ready` | Missing SHM object, incorrect mapping size/layout, or inaccessible source |
| `readers did not produce advancing coherent samples` | Static sequence, missing telemetry source, or no valid alignment |
| `camera/Xense required source stalled` | Camera or Xense publication stopped for more than the configured stall timeout |
| `FT/robot/gripper required source stalled` | FT or FGT1 robot/gripper publication stopped |
| `causal alignment failed continuously past timeout` | Sources are individually live but timestamps/skew cannot form a valid snapshot |
| stale aligned snapshot | SensorHub is no longer publishing quickly enough or has entered a fatal state |
| no remote motion | Check subscriber startup order, subscription filter, ABI, watchdog state, and ZMQ endpoint |
| clipped gripper warning | Policy produced `gripper.pos` outside `[0, 1]`; the submitted value was clipped |

## Validation Status

### Completed software validation

| Check | Result |
| --- | --- |
| Ruff over FR3 and touched integration modules | Passed |
| `git diff --check` | Passed |
| Python compile validation | Passed during implementation |
| Dependency lock validation | `uv lock --check` passed during implementation |
| FR3 tests in the restricted environment | `32 passed, 1 skipped in 0.29s` |
| FR3 tests in an unrestricted environment | `33 passed in 0.34s` |

The restricted environment skips the ZMQ IPC case because IPC socket creation is unavailable there.
The same case passed in the unrestricted run.

Focused tests cover:

- Command and telemetry golden ABIs, gripper conversion/clipping, invalid inputs, and masks.
- UDS schema validation, size bounds, and independent multiplexed telemetry sources.
- Causal alignment, duplicate suppression, shared-memory sizing, seqlock round trips, owned snapshots,
  stale reads, and fatal state.
- Strict RealSense layout checks and fixed Xense/FT offsets.
- Robot action schema, feature declarations, managed lifecycle, and idempotent disconnect.
- Online/offline layout conversion, RGB normalization, Z16 preservation, and numeric Array3D dataset
  handling.

### Validation still required

- End-to-end operation with all production sensor writers and the production FGT1 relay.
- Verification that every external producer uses the exact ABI, timestamps, and sequence semantics
  assumed by the readers.
- Safe command delivery to the real remote subscriber and FR3 controller.
- Joint/gripper direction, unit, range, latency, and watchdog verification on hardware.
- Observation timestamp/skew characterization under sustained load.
- Long-duration process, shared-memory, CPU, and memory stability testing.
- Record/rollout acceptance with the intended production policy and dataset configuration.

No statement in this report should be interpreted as completed real-hardware acceptance.

## Known Limitations and Risks

- There is no run-level external controller. Operators must manually establish and verify startup
  order until one is implemented.
- SensorHub does not reconnect readers, remap replaced SHM objects, recover locally, or restart
  itself after a fatal condition.
- Every modality is required; there is no degraded mode for a missing camera or tactile sensor.
- ZMQ PUB/SUB can drop commands and has a slow-subscriber startup window. There is no application
  acknowledgement in command ABI v1.
- The Robot does not impose FR3 joint limits or velocity/acceleration limits. Safety remains with
  the remote real-time controller.
- The default endpoints and SHM names are deployment-specific and must be reviewed before using a
  different network or host layout.
- Shared-memory ABIs assume Linux x86-64, little-endian operation and fixed array dimensions.

## Recommended Next Steps

1. Freeze and independently review the command, FGT1, RealSense, Xense, FT300S, and aligned SHM ABI
   definitions against every production producer and consumer.
2. Execute staged hardware acceptance: read-only observation validation first, disabled-motion
   command capture second, and supervised low-risk motion only after watchdog and limits are proven.
3. Measure source rate, timestamp quality, alignment success, observation age, command latency, and
   resource use during an extended soak test.
4. Implement the external controller contract described in the FR3 integration guide: start and
   verify external components, start LeRobot, supervise the complete run, and terminate all owned
   processes on fatal failure.
5. Add deployment-specific launch configuration and an operator acceptance checklist once the
   production process topology is fixed.
6. Run broader LeRobot tests, then review any follow-up code changes against the relevant protocol
    and operations documentation so the implementation stays aligned with the documented interfaces.

## Handover Checklist

- [ ] Incoming owner has reviewed the ownership boundaries and agrees on the future controller's
      responsibilities.
- [ ] All external ABI implementations have been compared with the definitions in this repository.
- [ ] Production endpoints, SHM names, permissions, and UDS runtime directory have been confirmed.
- [ ] Remote command watchdog and safe hold behavior have been demonstrated independently.
- [ ] Read-only sensor and telemetry acceptance has passed on the target machine.
- [ ] Supervised action acceptance has passed on the physical FR3 and gripper.
- [ ] Record and rollout workflows have passed with production policy preprocessing.
- [ ] Soak-test results and any tuned timeout values have been recorded.
- [ ] The implementation, protocol definitions, and operations documentation remain consistent.

