#!/usr/bin/env python3
"""Execute one checked CSV action goal; this executable can move hardware."""

import argparse
import csv
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]
ACTION = "/left_joint_trajectory_controller/follow_joint_trajectory"
COMMAND_RECORD_TOPIC = "/me6404/commanded_trajectory"
# Radians and N*m from the pinned OpenArm v1 MuJoCo model. These checks do
# not establish the limits or safety of a different physical configuration.
POSITION_LIMITS = [
    (-3.490659, 1.3962629999999998),
    (-3.3161253267948965, 0.17453267320510335),
    (-1.570796, 1.570796),
    (0.0, 2.443461),
    (-1.570796, 1.570796),
    (-0.785398, 0.785398),
    (-1.570796, 1.570796),
]
EFFORT_LIMITS = [40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0]
STATE_MAX_AGE_S = 0.25
START_POSITION_TOLERANCE_RAD = 0.1
CONNECTION_TIMEOUT_S = 5.0


def finite_values(row: dict[str, str], quantity: str) -> list[float]:
    values = [float(row[f"J{i}_{quantity}"]) for i in range(1, 8)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Nonfinite CSV values in {quantity}.")
    return values


def load_trajectory(
    csv_path: Path, gravity_scale: float, selection: str
) -> JointTrajectory:
    """Read commands, not measured motion or total PD torque; never publish."""
    if not math.isfinite(gravity_scale) or not 0.0 <= gravity_scale <= 1.0:
        raise ValueError("--gravity-scale must be finite and within [0, 1].")
    labels: set[str] = set()
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "trajectory" not in reader.fieldnames:
            raise ValueError("CSV must contain a trajectory column.")
        for row in reader:
            label = row["trajectory"]
            if label is not None and label != "":
                labels.add(label)
            if label == selection:
                rows.append(row)
    if selection not in labels:
        available = (
            ", ".join(map(repr, sorted(labels))) if len(labels) > 0 else "(none)"
        )
        raise ValueError(
            f"Unknown trajectory {selection!r}. Available CSV labels: {available}."
        )
    if len(rows) < 2:
        raise ValueError(
            f"CSV has no usable {selection} trial (need at least two rows)."
        )

    metadata_path = csv_path.with_name(
        csv_path.name.removesuffix("_results.csv") + "_metadata.yaml"
    )
    with metadata_path.open() as stream:
        metadata: dict[str, Any] = yaml.safe_load(stream)
    if (
        metadata["results_file"] != csv_path.name
        or metadata["joint_order"] != JOINT_NAMES
    ):
        raise ValueError("CSV filename or joint ordering does not match metadata.")
    trial = metadata["trials"][selection]
    if (
        trial["completed"] is not True
        or trial["simulation_checks"]["passing"] is not True
        or trial["samples"] != len(rows)
    ):
        raise ValueError(
            "Selected trial must be complete and have passing simulation checks."
        )
    config = cast(dict[str, object], metadata["config"])
    # Match the MuJoCo command ceilings; legacy exports have only one limit tier.
    velocity_key = (
        "absolute_velocity_limits"
        if "absolute_velocity_limits" in config
        else "velocity_limits"
    )
    acceleration_key = (
        "absolute_acceleration_limits"
        if "absolute_acceleration_limits" in config
        else "acceleration_limits"
    )
    velocity_limits = [
        float(value) for value in cast(Sequence[float | str], config[velocity_key])
    ]
    acceleration_limits = [
        float(value) for value in cast(Sequence[float | str], config[acceleration_key])
    ]
    if (
        len(velocity_limits) != 7
        or len(acceleration_limits) != 7
        or not all(
            math.isfinite(value) and value > 0
            for value in (
                *velocity_limits,
                *acceleration_limits,
            )
        )
    ):
        raise ValueError("Metadata must contain finite, positive limits.")

    points: list[JointTrajectoryPoint] = []
    previous_time_ns = -1
    previous_positions: list[float] | None = None
    for row in rows:
        time_s = float(row["time_s"])
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("CSV timestamps must be finite and nonnegative.")
        if time_s >= 2**31:
            raise ValueError("CSV timestamp exceeds ROS Duration range.")
        time_ns = round(time_s * 1_000_000_000)
        if time_ns <= previous_time_ns or time_ns >= 2**31 * 1_000_000_000:
            raise ValueError(
                "CSV timestamps must increase strictly and fit ROS Duration."
            )
        if previous_time_ns == -1 and time_ns != 0:
            raise ValueError(
                "Selected trial must start at time zero; refusing to retime it."
            )
        positions = finite_values(row, "q_command_rad")
        gravity = finite_values(row, "torque_gravity_ff_Nm")
        velocities = finite_values(row, "dq_reference_rad_s")
        accelerations = finite_values(row, "ddq_reference_rad_s2")
        efforts = [gravity_scale * value for value in gravity]
        for i, name in enumerate(JOINT_NAMES):
            lower, upper = POSITION_LIMITS[i]
            if not lower <= positions[i] <= upper:
                raise ValueError(f"{name}: position outside pinned model limits.")
            if abs(efforts[i]) > EFFORT_LIMITS[i]:
                raise ValueError(f"{name}: scaled gravity effort outside motor limits.")
            if abs(velocities[i]) > velocity_limits[i] + 1e-9:
                raise ValueError(
                    f"{name}: reference velocity exceeds configured limit."
                )
            if abs(accelerations[i]) > acceleration_limits[i] + 1e-9:
                raise ValueError(
                    f"{name}: reference acceleration exceeds configured limit."
                )
            if previous_positions is not None:
                segment_velocity = abs(positions[i] - previous_positions[i]) / (
                    (time_ns - previous_time_ns) * 1e-9
                )
                if segment_velocity > velocity_limits[i] + 1e-9:
                    raise ValueError(
                        f"{name}: position segment exceeds velocity limit."
                    )
        seconds, nanoseconds = divmod(time_ns, 1_000_000_000)
        points.append(
            JointTrajectoryPoint(
                positions=positions,
                effort=efforts,
                time_from_start=Duration(sec=seconds, nanosec=nanoseconds),
            )
        )
        previous_time_ns = time_ns
        previous_positions = positions
    return JointTrajectory(joint_names=JOINT_NAMES, points=points)


def check_start_state(
    state: JointState,
    now_ns: int,
    initial_positions: Sequence[float],
) -> None:
    """Require valid state within 0.1 rad of each start angle; speed is manual."""
    stamp_ns = state.header.stamp.sec * 1_000_000_000 + state.header.stamp.nanosec
    age_s = (now_ns - stamp_ns) * 1e-9
    if stamp_ns <= 0 or not 0.0 <= age_s <= STATE_MAX_AGE_S:
        raise ValueError("Joint state timestamp is missing, stale, or in the future.")
    # ROS generated accessors have broad sequence/array unions rather than
    # field-specific annotations. Narrow the known JointState field types here.
    names = cast(Sequence[str], state.name)
    positions = cast(Sequence[float], state.position)
    velocities = cast(Sequence[float], state.velocity)
    if (
        len(set(names)) != len(names)
        or len(positions) != len(names)
        or len(velocities) != len(names)
    ):
        raise ValueError(
            "Joint state must include unique names, positions, and velocities."
        )
    for name, target in zip(JOINT_NAMES, initial_positions, strict=True):
        if name not in names:
            raise ValueError(f"Joint state is missing {name}.")
        index = names.index(name)
        position, velocity = positions[index], velocities[index]
        if not math.isfinite(position) or not math.isfinite(velocity):
            raise ValueError(f"{name}: nonfinite joint state.")
        if not (
            target - START_POSITION_TOLERANCE_RAD
            <= position
            <= target + START_POSITION_TOLERANCE_RAD
        ):
            raise ValueError(
                f"{name}: position {position:.6f} rad is outside "
                f"starting pose {target:.6f} ± {START_POSITION_TOLERANCE_RAD:g} rad."
            )


def execute_trajectory(
    node: Node,
    client: ActionClient,
    message: JointTrajectory,
    command_record: Publisher,
) -> None:
    """Send once, require acceptance, and wait for the controller's result.

    Timeouts/interrupts do not cancel a goal or disable motors. No automatic
    retries: after sending, a lost response leaves physical command status unknown.
    """
    feedback_received = False

    def receive_feedback(_feedback: object) -> None:
        nonlocal feedback_received
        if not feedback_received:
            node.get_logger().info("Controller execution feedback received.")
            feedback_received = True

    # Zero header stamp means start on acceptance, without a delayed start.
    goal_future = client.send_goal_async(
        FollowJointTrajectory.Goal(trajectory=message),
        feedback_callback=receive_feedback,
    )
    rclpy.spin_until_future_complete(
        node, goal_future, timeout_sec=CONNECTION_TIMEOUT_S
    )
    if not goal_future.done():
        raise RuntimeError(
            "Timed out waiting for goal acceptance; command status is unknown. "
            "Do not retry blindly."
        )
    goal_handle = goal_future.result()
    if goal_handle is None:
        raise RuntimeError("No goal response; command status is unknown.")
    if not goal_handle.accepted:
        raise RuntimeError(f"Controller rejected trajectory goal at {ACTION}.")
    goal_id = bytes(goal_handle.goal_id.uuid).hex()
    node.get_logger().info(f"Controller accepted trajectory goal {goal_id}.")
    # Diagnostic copy only: never republish to the controller's command topic.
    # Transient-local durability lets rosbag discover this new topic afterward.
    command_record.publish(message)
    result_future = goal_handle.get_result_async()
    last = cast(Sequence[JointTrajectoryPoint], message.points)[-1].time_from_start
    duration_s = last.sec + last.nanosec * 1e-9
    rclpy.spin_until_future_complete(
        node, result_future, timeout_sec=duration_s + CONNECTION_TIMEOUT_S
    )
    if not result_future.done():
        raise RuntimeError(
            f"Timed out waiting for result of goal {goal_id}; execution status is "
            "unknown. No cancel or motor-disable command was sent."
        )
    response = result_future.result()
    if response is None:
        raise RuntimeError(
            f"No result for goal {goal_id}; execution status is unknown."
        )
    result = response.result
    summary = (
        f"Goal {goal_id}: status={response.status}, "
        f"error_code={result.error_code}, error_string={result.error_string!r}."
    )
    if (
        response.status != GoalStatus.STATUS_SUCCEEDED
        or result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
    ):
        raise RuntimeError(f"Controller reported unsuccessful execution. {summary}")
    node.get_logger().info(
        f"Controller reported successful execution. {summary} "
        "This is not independent physical safety validation."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--gravity-scale", type=float, required=True)
    parser.add_argument(
        "--trajectory", required=True, help="Exact CSV trajectory label."
    )
    args, ros_args = parser.parse_known_args()
    try:
        message = load_trajectory(args.csv, args.gravity_scale, args.trajectory)
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))

    print(
        f"READY TO COMMAND HARDWARE: {args.trajectory}, {len(message.points)} points, "
        f"gravity scale {args.gravity_scale:g}.\n"
        "Simulation metadata is not hardware authorization and does not verify "
        "this gravity scale, current gains, or physical clearance.\n"
        "You must visually confirm the arm is stationary; speed is NOT checked "
        "automatically. Each joint must be within ±0.1 rad of the CSV starting pose. "
        "An accepted pose mismatch can still cause an immediate position change.\n"
        "This client is not a safety monitor. Exiting does not cancel motion or "
        "disable motors; confirm your stop/E-stop procedure is ready.\n"
        "Type GO only after reviewing the exact trajectory/configuration and scale, "
        "and physical clearance."
    )
    try:
        if input("Authorize this physical trial [GO]: ").strip() != "GO":
            print("Cancelled; no goal sent.")
            return
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled; no goal sent.")
        return

    rclpy.init(args=ros_args)
    node = Node("publish_csv_trajectory")
    client = ActionClient(node, FollowJointTrajectory, ACTION)
    command_record = node.create_publisher(
        JointTrajectory,
        COMMAND_RECORD_TOPIC,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    latest_states: list[JointState] = []

    def receive_state(state: JointState) -> None:
        latest_states[:] = [state]

    node.create_subscription(
        JointState, "/joint_states", receive_state, qos_profile_sensor_data
    )
    try:
        deadline = time.monotonic() + CONNECTION_TIMEOUT_S
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if latest_states and client.server_is_ready():
                break
        else:
            raise ValueError(
                "Timed out waiting for joint states and FollowJointTrajectory server."
            )
        if not latest_states:
            raise ValueError("No joint state received.")
        first_point = cast(Sequence[JointTrajectoryPoint], message.points)[0]
        check_start_state(
            latest_states[0],
            node.get_clock().now().nanoseconds,
            cast(Sequence[float], first_point.positions),
        )
        execute_trajectory(node, client, message, command_record)
    except (ValueError, RuntimeError, KeyboardInterrupt) as error:
        node.get_logger().error(
            f"Stopped: {error}. Exiting does not cancel motion or disable motors."
        )
        raise SystemExit(1) from error
    finally:
        client.destroy()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
