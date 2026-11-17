# Deferred Problems

## Intermittent controller-manager crash during shutdown

- **Status:** Deferred by the user on 2026-11-16; not resolved.
- **Priority:** Safety-relevant. Deferral is not approval of hardware safety.
- **Scope:** Physical OpenArm bringup using `me6404_bringup`.

### Symptoms and evidence

After a trajectory completed successfully, stopping bringup with Ctrl-C
intermittently caused `ros2_control_node` to segfault inside
`JointTrajectoryController::update()`.

The observed shutdown started right-arm hardware deactivation but did not log
completed deactivation for either arm. The user reported motor LEDs remaining
green and used the established E-stop/recovery procedure. LED color alone is
not independent confirmation of torque-enable state.

Example log:
`/home/lukec/.ros/log/2026-11-16-23-51-42-101497-wolfgang-kubuntu-916239/launch.log`

Installed versions at investigation:

- ROS 2 Jazzy
- `controller_manager`: 4.50.0
- `joint_trajectory_controller`: 4.43.1

### Likely cause

The evidence strongly matches the shutdown synchronization race described in
[ros2_control issue #3716](https://github.com/ros-controls/ros2_control/issues/3716).
The issue was open when investigated.

The manager's pre-shutdown callback deactivates controllers and releases their
interfaces while the separate control-loop thread can still execute updates.
The trajectory controller's deactivation clears interface references and resets
its trajectory object, which its update function uses. The exact invalid access
in this crash has not been confirmed with a core dump/debug symbols.

The OpenArm driver's `on_deactivate()` sends `disable_all()` three times.
A process crash can interrupt that sequence or prevent the other arm's hardware
deactivation from being reached. Successful trajectory execution does not prove
successful shutdown.

### Current limitation

Do not rely on Ctrl-C or process exit as confirmation that motors are disabled.
Treat uncertain motor state as enabled and follow the established stop/E-stop
and recovery procedure. Removing holding torque can let the arms fall under
gravity; account for this before disabling power or motors.

Do not relaunch solely to disable motors: the current driver enables motors and
commands a return to zero during activation.

### Candidate workaround — not implemented or validated

While the controller manager is still running:

1. End the trial and verify that the arms are stationary.
2. Use normal controller switching to deactivate both arm and both gripper
   controllers, then verify their inactive states.
3. With the arms supported against gravity, deactivate both hardware components.
4. Verify actual motor-disable state before stopping the launch.

Normal controller switching synchronizes with the update loop, unlike the
implicated shutdown path. Controller deactivation alone does not disable motors.
A launch shutdown hook may run too late to avoid the race.

### Revisit when

- A reviewed upstream fix is available, or an explicit orderly-shutdown helper
  is requested.
- First verify controller/hardware transition ordering, failures, and timeouts
  using fake hardware.
- Any physical verification requires separate user authorization and the
  established hardware safety procedure. Fake-hardware success is not physical
  safety validation.

### Separate cleanup error observed in the same log

`publish_motor_gains.py` calls `rclpy.shutdown()` after its context has already
shut down, producing `rcl_shutdown already called`. This is distinct from the
C++ controller-manager crash. An idempotent shutdown call is a candidate fix;
no change has been made for this error.
