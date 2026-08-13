# FR3 Controlled Rollout Hardware Integration Test

## Status and authorization

This document is a future operator-run integration procedure. It was prepared on a workstation
without connecting to a NUC, Franka, gripper, FCI session, or any hardware-reachable command
endpoint. Do not execute the NUC or hardware steps without a new explicit authorization and an
operator responsible for physical safety.

The implementation baseline used while writing this document was:

- LeRobotFR3 branch `wip/FR3`, baseline commit `5e02c53acecb0fb8e8d14f71e83a3baed2b5a7ec`
- zmq_franka_gello branch `main`, baseline commit `b0063a23ac70e533e2db7522966e627d4c4184cb`

Record the actual branches, commits, dirty state, Python environments, configurations, endpoints,
and process versions again at test time. The current routed reverse-control procedure in
`zmq_franka_gello/docs/reverse_validation/NUC_routed_C2_to_C4_plan.md` is authoritative for the NUC
stack. `zmq_franka_gello/docs/NUC_reverse_test_prompt.md` is useful historical structure but is
superseded for the routed production stack.

## Already verified locally

The workstation-local test evidence covers:

- FR3 reset/home configuration validation, randomized target bounds, exact home target, local target
  logging, and the existing synchronous `_reset_joints()` retry/`RESETTING` contract with mocks.
- Controlled config/factory, application-level `ACK`, lifecycle `STATUS`, strictly increasing command
  sequence validation, phase-local blocking/polling behavior, blind phase-boundary drain, no deferred
  command execution, STOP/ABORT/completion transitions, SHUTDOWN/FAIL_STOP behavior, stale exception
  propagation, and the RTC previous-episode quiescence boundary.
- Dataset-disabled behavior and mocked episode save/clear/single-finalize ownership. A real
  `LeRobotDataset`/video session was not run because the workstation environment lacked the optional
  `datasets`, `av`, and `pyarrow` packages.
- FRCMD1 and FGT1 ABI compatibility between both repositories, router fan-out, reset gripper OPEN,
  fake reverse-control `RESETTING 0 -> 1 -> 0`, JUMP_HOLD recovery, failure paths, and gripper dry-run
  components.

The local fork does not provide a complete command-to-control-to-telemetry localhost runtime. The
following still require the staged NUC and/or hardware tests below: a complete LeRobot
`_reset_joints()` routed closed loop, real SensorHub sources, NUC process timing, real FCI behavior,
real arm motion, and real gripper motion.

## Prerequisites and stop authority

Before any NUC or hardware run:

1. Obtain explicit authorization for the exact round.
2. Place a trained policy and its processors on the workstation and validate all configuration
   files offline.
3. Confirm the configured deterministic home vector and randomized delta bounds:

   ```text
   q_home = (
       0.1416057646,
       0.3408541381,
       -0.0186031274,
       -1.5938080549,
       0.0486696586,
       1.8890386820,
       0.0432172865,
   )
   delta_i ~ Uniform(-0.01, 0.01)
   ```

4. Confirm there is exactly one routed FRCMD1 owner, one reverse Franka controller, one Robotiq
   server, and one FCI/control session. Never run a legacy and reverse controller together.
5. Confirm all RealSense, Xense, FT300S, robot, and gripper telemetry sources required by SensorHub
   are present and advancing before policy control.
6. Keep the Franka stop mechanism available. The operator must retain immediate stop authority and
   must personally approve every step marked as physical motion.
7. Do not use automatic confirmation flags for the first hardware run. Do not intentionally produce
   an FCI fault or an unsafe target to test failure handling.

Immediately stop the test on unexpected arm direction or amplitude, unexpected gripper behavior,
non-finite command or telemetry, excessive or unexplained torque, missing or out-of-order
`RESETTING` transitions, unexplained process/session restart, telemetry loss, FCI error, inability to
stop the robot, or any operator concern. Gripper OPEN during `RESET_JOINT` is expected motion, not an
error by itself.

## Startup and shutdown order

Use the exact commands and routed endpoints from the authoritative NUC documentation at test time.
The required ownership order is:

1. NUC: verify no conflicting robot controller or command publisher exists.
2. NUC: start the routed local IPC owners, reverse Franka server, Robotiq server, FRCMD1 router, and
   FGT1 telemetry relay in the documented order. The operator performs all hardware safety gates.
3. Workstation: start and validate RealSense, Xense, and FT300S producers without starting policy
   motion.
4. Workstation: start the external MainController, which starts LeRobot as a persistent Controlled
   child. Confirm the LeRobot control UDS is the expected local pathname.
5. Confirm advancing finite robot and gripper FGT1 telemetry, SensorHub READY, and Controlled
   `STATUS READY` with phase `WAIT_INITIALIZE` before any lifecycle command.
6. Run only the authorized round.
7. Prefer explicit `SHUTDOWN` for a successful session. For a session already judged unsafe or
   unusable, use the defined `FAIL_STOP` path. Stop remaining processes in reverse ownership order.

## Protocol evidence to capture

For every round, preserve timestamped logs and packet captures sufficient to correlate:

- UDS command sequence, operation, `ACK accepted=true/false`, and worker `STATUS`/phase.
- The invariant `finish old-phase work -> blind drain -> commit new phase -> publish STATUS -> accept
  fresh input`.
- FR3 local `q_reset` or `q_home` log, FRCMD1 sequence/flags/joint payload/gripper byte, FGT1 robot q,
  FGT1 `RESETTING`, and gripper FGT1.
- NUC process IDs/start times, FCI/control-session continuity, reverse-control mode/reason, and any
  watchdog or JUMP_HOLD state.
- Dataset episode metadata and reload evidence when optional recording is enabled.

An `ACK` only proves that a parsed command was accepted or rejected in the phase in which LeRobot
read it. It does not prove that reset, rollout, or shutdown completed. The later `STATUS` and
robot/gripper telemetry provide operation outcome evidence.

## Test rounds

### 1. Idle connection and fail-safe baseline

Purpose: establish the routed runtime and telemetry without policy or reset motion.

Operator steps:

1. Complete the startup order and leave Controlled in `WAIT_INITIALIZE`.
2. Send no FRCMD1 action or reset command.
3. Observe bounded idle telemetry and process/session continuity.

Pass criteria: no hardware command is sent, finite advancing robot/gripper FGT1 is present,
`RESETTING=0`, and no process or FCI session restarts. This round must not move the arm or gripper.

### 2. Randomized INITIALIZE and repeated initialization

Purpose: validate the full LeRobotFR3 reset handshake and the formal reset motion semantics.

Physical motion: **arm motion and gripper OPEN**.

Operator steps:

1. Issue one legal `INITIALIZE` while LeRobot is in `WAIT_INITIALIZE`.
2. Record the locally logged sampled `q_reset` before `_reset_joints()`.
3. Correlate the FRCMD1 reset frame with the exact logged seven-joint target, reset flag, and
   `gripper_gPO=0`.
4. Observe ordered robot FGT1 `RESETTING 0 -> 1 -> 0`, gripper OPEN, and achieved q telemetry.
5. Confirm `STATUS INITIALIZED` with phase `WAIT_START` is published only after reset completion and
   the old transport backlog drain.
6. Complete or abort the associated rollout, then repeat from `WAIT_INITIALIZE` several times.

Pass criteria: every sampled joint lies within its configured `q_home +/- 0.01` range, the wire target
matches the local log exactly, reset retries are byte-identical for one FRCMD1 sequence, retries stop
after `RESETTING=1`, completion reaches `RESETTING=0`, and each new initialization samples
independently. Gripper OPEN is required expected behavior.

### 3. START, normal action, STOP, and duration completion

Purpose: validate Base-like policy control while the long-lived process and inference engine remain
alive.

Physical motion: **policy-driven arm and gripper motion**.

Operator steps:

1. From a completed initialization, issue `START` and confirm accepted `ACK`, then `STATUS STARTED`.
2. Observe finite normal FRCMD1 action frames and corresponding telemetry for a bounded interval.
3. Issue `STOP`; separately run a configured duration-completion case.
4. Confirm inference pauses and an optional dataset saves only a non-empty completed episode.

Pass criteria: STOP and completion both transition to `WAIT_INITIALIZE`; neither allows another START
without a new INITIALIZE; the process, policy, robot, SensorHub, and inference engine remain alive;
and no old phase input executes after the transition.

### 4. ABORT and optional dataset discard

Purpose: validate local rollout cancellation without terminating the session.

Physical motion: **policy-driven motion before ABORT; no return-home motion caused by ABORT**.

Issue ABORT during RUNNING. Pass requires immediate rollout exit, inference pause, partial dataset
buffer clear with no saved partial episode, transition to `WAIT_INITIALIZE`, and a mandatory new
INITIALIZE before the next START.

### 5. No deferred command execution at phase boundaries

Purpose: validate blind stale-input cleanup rather than command scheduling.

Physical motion: **the active INITIALIZE reset still moves the arm and opens the gripper**.

During a deliberately observable synchronous INITIALIZING interval, place ordinary lifecycle input
in the UDS transport backlog. After reset returns, confirm LeRobot blindly drains the old packets
before committing `WAIT_START` and does not parse, acknowledge, or execute them in the new phase.
Repeat around a RUNNING exit. A fresh command sent only after the destination STATUS must remain
available and execute normally. A stale SHUTDOWN or FAIL_STOP packet has no cross-phase delivery
guarantee and need not be retained by the blind drain.

Pass criteria: no stale command crosses a phase; no destination STATUS is published before the old
backlog drain; fresh post-STATUS input is not deleted. Do not add or infer any deferred command queue.

### 6. FAIL_STOP idempotence and no-new-motion teardown

Purpose: validate fatal termination after LeRobot actually reads a legal FAIL_STOP.

Physical motion: **none initiated by FAIL_STOP**.

Use normal strictly increasing command sequences for repeated requests, such as 100, 101, and 102.
Duplicate or decreasing sequences must be rejected by the normal sequence rule. Once one legal
FAIL_STOP is read, capture the accepted ACK and `STATUS FAIL_STOPPING`, then verify that LeRobot sends
no return-to-home RESET_JOINT, no new gripper command, and no subsequent policy action. The session
must tear down with a non-success result. Repetition must not produce motion or require exactly-once
delivery.

### 7. Graceful SHUTDOWN to deterministic home

Purpose: validate the only termination path that intentionally adds robot motion.

Physical motion: **arm motion to exact q_home and gripper OPEN**.

After a bounded rollout, issue a legal SHUTDOWN. Correlate the local deterministic `q_home` log with
the exact reset FRCMD1 payload and achieved telemetry. Confirm `gripper_gPO=0`, gripper OPEN,
`RESETTING 0 -> 1 -> 0`, inference stop, dataset/video finalization, UDS close, SensorHub/robot
disconnect, and successful process exit.

Pass criteria: the reset target equals configured q_home exactly, not a randomized target; the home
motion completes once; gripper OPEN occurs as designed; and resource teardown succeeds.

### 8. Graceful home failure result

Purpose: verify that a failed return-to-home cannot be reported as successful shutdown.

Do not intentionally create an FCI or physical fault. Exercise this only with an approved safe
failure-injection method at a layer that cannot generate unintended motion, or record it as untested
until such a method exists. Pass requires the original home failure to be reported, no second
recovery motion, continued resource teardown, and a final non-success result.

### 9. RTC episode isolation

Purpose: ensure an in-flight previous-rollout inference cannot populate the next episode after its
START reset boundary.

Run a policy with observable RTC latency, STOP or ABORT while an inference is in flight, perform a
new INITIALIZE, then START again. Correlate queue/debug evidence and first actions. Pass requires the
old inference to quiesce before the new engine/interpolator/observation-cache reset and no old action
or observation to appear in the new rollout. No RTC quiescence is required merely for concurrent
INITIALIZE arm reset motion.

### 10. Disconnect, timeout, stale telemetry, and internal fatal paths

Purpose: validate no-new-motion teardown for faults detected locally by LeRobot.

Use only separately approved, safe fault injection. Cover UDS EOF/socket failure, SensorHub fatal,
reset timeout/failure, aligned observation stale/max-age, policy/inference failure, and dataset
failure where possible. Do not convert stale/max-age into ABORT. Pass requires original error
propagation, no return-to-home or new gripper command, no partial episode save, resource teardown,
and a non-success/fatal result.

## Dataset recording and replay

Run the normal lifecycle once without `--dataset.*` and once with a temporary standard rollout
dataset. Dataset presence must not change phase transitions. With recording enabled, verify frame
schema, STOP/completion save, ABORT/FAIL_STOP/RUNNING-SHUTDOWN clear, no partial save on fatal, one
session finalization owner, video completion, and normal `LeRobotDataset` reload/inspection. Do not
change Episodic behavior while investigating a Controlled dataset failure.

## Final pass/fail report

The operator report must include:

- exact commands, environments, commits, configs, endpoint ownership, and bounded durations;
- the local A/B evidence reused and every NUC/hardware item actually executed;
- UDS ACK/STATUS and phase-transition ordering evidence;
- q_reset/q_home log-to-FRCMD1-to-FGT1 correlation;
- RESETTING, JUMP_HOLD, process/session, torque, cadence, and gripper evidence;
- dataset/video artifacts and reload results when enabled;
- all skipped or blocked checks without claiming them as passed;
- every stop condition encountered and the final session exit result.

This document does not authorize its own execution. At creation time, no NUC connection, Franka or
gripper call, FCI session, real reset, real policy motion, or hardware-reachable command was made.
