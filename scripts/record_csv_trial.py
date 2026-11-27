#!/usr/bin/env python3
"""Record a CSV hardware trial; stopping this script does NOT stop the robot."""

import argparse
import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMAND_TOPIC = "/left_joint_trajectory_controller/joint_trajectory"
ACTION = "/left_joint_trajectory_controller/follow_joint_trajectory"
COMMAND_RECORD_TOPIC = "/me6404/commanded_trajectory"
REQUIRED_TOPICS = (
    "/joint_states",
    "/left_joint_trajectory_controller/controller_state",
    f"{ACTION}/_action/status",
    f"{ACTION}/_action/feedback",
)
TOPICS = (
    *REQUIRED_TOPICS,
    COMMAND_RECORD_TOPIC,
    COMMAND_TOPIC,
    "/dynamic_joint_states",
    "/right_joint_trajectory_controller/joint_trajectory",
    "/right_joint_trajectory_controller/controller_state",
    "/left_motor_gains/kp",
    "/left_motor_gains/kd",
    "/right_motor_gains/kp",
    "/right_motor_gains/kd",
    "/robot_description",
    "/tf",
    "/tf_static",
    "/diagnostics",
    "/rosout",
    "/clock",
)


def copy_inputs(
    csv_path: Path, output_root: Path, selection: str
) -> tuple[Path, Path, float]:
    """Snapshot both inputs without renaming files referenced by metadata."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", selection) is None:
        raise ValueError("Trajectory labels must use only letters, digits, _ or -.")
    metadata_path = csv_path.with_name(
        csv_path.name.removesuffix("_results.csv") + "_metadata.yaml"
    )
    if not csv_path.is_file() or not metadata_path.is_file():
        raise ValueError("CSV and matching *_metadata.yaml must both exist.")
    metadata_bytes = metadata_path.read_bytes()
    try:
        metadata = cast(object, yaml.safe_load(metadata_bytes))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {metadata_path.name}.") from error
    match metadata:
        case dict() as mapping:
            fields = cast(dict[str, object], mapping)
        case _:
            raise ValueError(f"{metadata_path.name} must contain a YAML mapping.")
    if "design_model" in fields:
        model = fields["design_model"]
        if not isinstance(model, str) or model not in ("calibrated", "nominal"):
            raise ValueError("design_model must be 'calibrated' or 'nominal'.")
        output_root = output_root / model
    trial_root = output_root / selection
    trial_root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().astimezone().strftime("run_%Y%m%d_%H%M%S_%f_")
    run_dir = Path(mkdtemp(prefix=prefix, dir=trial_root))
    saved_csv = Path(shutil.copy2(csv_path, run_dir))
    _ = (run_dir / metadata_path.name).write_bytes(metadata_bytes)
    labels: set[str] = set()
    times: list[float] = []
    with saved_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"trajectory", "time_s"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("CSV must contain trajectory and time_s columns.")
        for row in reader:
            label = row["trajectory"]
            if label is not None and label != "":
                labels.add(label)
            if label == selection:
                times.append(float(row["time_s"]))
    if selection not in labels:
        available = (
            ", ".join(map(repr, sorted(labels))) if len(labels) > 0 else "(none)"
        )
        raise ValueError(
            f"Unknown trajectory {selection!r}. Available CSV labels: {available}."
        )
    if len(times) < 2 or not all(math.isfinite(t) and t >= 0 for t in times):
        raise ValueError(
            "Selected CSV trial needs at least two finite nonnegative times."
        )
    return run_dir, saved_csv, max(times)


def stop_process(process: subprocess.Popen) -> None:
    """SIGINT the child group and allow rosbag to flush metadata."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("Child did not exit after SIGINT; terminating. Bag may need recovery.")
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def wait_for_recorder(
    recorder: subprocess.Popen, node_name: str, timeout: float
) -> None:
    """Check DDS subscriptions, not a fixed sleep, before allowing publication."""
    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient

    rclpy.init()
    node = rclpy.create_node(f"{node_name}_check")
    client = ActionClient(node, FollowJointTrajectory, ACTION)
    try:
        deadline = time.monotonic() + timeout
        missing = list(REQUIRED_TOPICS)
        while time.monotonic() < deadline:
            if recorder.poll() is not None:
                raise RuntimeError(
                    "Recorder exited before ready; inspect recorder.log."
                )
            rclpy.spin_once(node, timeout_sec=0.1)
            missing = [
                topic
                for topic in REQUIRED_TOPICS
                if not any(
                    endpoint.node_name == node_name
                    for endpoint in node.get_subscriptions_info_by_topic(topic)
                )
            ]
            if not missing and client.server_is_ready():
                return
        raise RuntimeError(
            f"Recording readiness timed out; missing recorder subscriptions: {missing}. "
            "The left FollowJointTrajectory action server must also be ready."
        )
    finally:
        client.destroy()
        node.destroy_node()
        rclpy.try_shutdown()


def monitor(process: subprocess.Popen, recorder: subprocess.Popen) -> int:
    while process.poll() is None:
        if recorder.poll() is not None:
            raise RuntimeError(
                "Recorder exited during publication. Robot command status is unknown; "
                "use your established stop procedure if needed."
            )
        time.sleep(0.1)
    return process.returncode


def record_trial(args: argparse.Namespace) -> Path:
    run_dir, saved_csv, duration = copy_inputs(
        args.csv.resolve(), args.output_dir.resolve(), args.trajectory
    )
    node_name = f"csv_recorder_{run_dir.name}"
    # The diagnostic command copy is latched; it is not a hardware input.
    qos_path = run_dir / "recording_qos.yaml"
    qos_path.write_text(
        f"{COMMAND_RECORD_TOPIC}:\n"
        "  reliability: reliable\n  durability: transient_local\n  history: keep_last\n  depth: 1\n"
        "/joint_states:\n"
        "  reliability: best_effort\n  durability: volatile\n  history: keep_last\n  depth: 100\n"
        "/robot_description:\n"
        "  reliability: reliable\n  durability: transient_local\n  history: keep_last\n  depth: 1\n"
        "/tf_static:\n"
        "  reliability: reliable\n  durability: transient_local\n  history: keep_last\n  depth: 100\n"
    )
    publish_command = [
        "ros2",
        "run",
        "me6404_bringup",
        "publish_csv_trajectory.py",
        "--csv",
        str(saved_csv),
        "--trajectory",
        args.trajectory,
        "--gravity-scale",
        str(args.gravity_scale),
    ]
    record_command = [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
        str(run_dir / "bag"),
        "--node-name",
        node_name,
        "--disable-keyboard-controls",
        "--include-unpublished-topics",
        "--include-hidden-topics",
        "--qos-profile-overrides-path",
        str(qos_path),
        "--topics",
        *TOPICS,
    ]
    manifest = {
        "source_csv": str(args.csv.resolve()),
        "csv": saved_csv.name,
        "trajectory": args.trajectory,
        "gravity_scale": args.gravity_scale,
        "trajectory_duration_s": duration,
        "command_transport": "FollowJointTrajectory action",
        "post_roll_s": args.post_roll,
        "publish_command": publish_command,
        "record_command": record_command,
        "topics_requested": TOPICS,
        "status": "starting",
    }
    manifest_path = run_dir / "trial.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saving trial to {run_dir}", flush=True)
    print("Stopping recording does NOT stop motion or disable motors.", flush=True)
    publisher = None
    recorder = None
    try:
        with (run_dir / "recorder.log").open("w") as log:
            recorder = subprocess.Popen(
                record_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_recorder(recorder, node_name, args.ready_timeout)
            # Keep stdin/stdout attached: never bypass or automate the GO prompt.
            publisher = subprocess.Popen(publish_command, start_new_session=True)
            code = monitor(publisher, recorder)
            manifest["publisher_exit_code"] = code
            if code:
                raise RuntimeError(f"Publisher exited with status {code}.")
            # Action publisher stays alive until the controller reports a result.
            deadline = time.monotonic() + args.post_roll
            print("Action publisher exited; recording post-roll.")
            while time.monotonic() < deadline:
                if recorder.poll() is not None:
                    raise RuntimeError("Recorder exited early; inspect recorder.log.")
                time.sleep(0.1)
            manifest["status"] = "recording_window_finished"
    except BaseException as error:
        manifest["status"] = "interrupted_or_failed"
        manifest["error"] = str(error) or type(error).__name__
        raise
    finally:
        try:
            if publisher is not None:
                stop_process(publisher)
        finally:
            try:
                if recorder is not None:
                    stop_process(recorder)
                    manifest["recorder_exit_code"] = recorder.returncode
            finally:
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if recorder is not None and recorder.returncode not in (0, -signal.SIGINT):
        raise RuntimeError(
            "Recorder did not close cleanly; inspect recorder.log and bag."
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--trajectory", required=True, help="Exact CSV trajectory label."
    )
    parser.add_argument("--gravity-scale", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "hardware_out")
    parser.add_argument("--post-roll", type=float, default=2.0)
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    args = parser.parse_args()
    if not math.isfinite(args.gravity_scale) or not 0 <= args.gravity_scale <= 1:
        parser.error("--gravity-scale must be within [0, 1].")
    if not math.isfinite(args.post_roll) or args.post_roll < 0:
        parser.error("--post-roll must be finite and nonnegative.")
    if not math.isfinite(args.ready_timeout) or args.ready_timeout <= 0:
        parser.error("--ready-timeout must be finite and positive.")
    try:
        print(f"Saved: {record_trial(args)}")
    except (KeyboardInterrupt, RuntimeError, OSError, ValueError) as error:
        parser.exit(1, f"\nRecording stopped: {error}\nThis does NOT stop the robot.\n")


if __name__ == "__main__":
    main()
