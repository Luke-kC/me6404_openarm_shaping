"""Offline CSV/message checks only: never initialize ROS or publish commands."""

import csv
import importlib.util
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest
import yaml

TRIAL_POSITIONS = {
    "unshaped": 0.1,
    "shaped": 0.05,
    "baseline": 0.12,
    "time_invariant_zvd": 0.08,
    "tvist_zvd": 0.04,
}


@pytest.fixture(scope="module")
def replay() -> ModuleType:
    pytest.importorskip("rclpy", reason="Source ROS Jazzy to test real ROS messages.")
    script = (
        Path(__file__).resolve().parents[1]
        / "ros2_ws/src/me6404_bringup/scripts/publish_csv_trajectory.py"
    )
    spec = importlib.util.spec_from_file_location("publish_csv_trajectory", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_trial(
    root: Path, replay: ModuleType, *, labels: tuple[str, ...] = tuple(TRIAL_POSITIONS)
) -> tuple[Path, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    for label in labels:
        for index in range(3):
            row = {"trajectory": label, "time_s": str(index * 0.1)}
            for joint in range(1, 8):
                row[f"J{joint}_q_command_rad"] = str(TRIAL_POSITIONS[label])
                row[f"J{joint}_q_rad"] = "999"  # Must never replay measured motion.
                row[f"J{joint}_torque_requested_Nm"] = "999"
                row[f"J{joint}_torque_feedback_Nm"] = "999"
                row[f"J{joint}_torque_gravity_ff_Nm"] = str(joint * 0.5)
                row[f"J{joint}_dq_reference_rad_s"] = "0.0"
                row[f"J{joint}_ddq_reference_rad_s2"] = "0.0"
            rows.append(row)
    path = root / "run_results.csv"
    write_rows(path, rows)
    metadata = {
        "results_file": path.name,
        "joint_order": replay.JOINT_NAMES,
        "config": {
            "velocity_limits": [1.0] * 7,
            "acceleration_limits": [2.0] * 7,
            "simulation_assessment": {
                "position_tolerance_rad": 0.01,
                "velocity_tolerance_rad_s": 0.01,
            },
        },
        "trials": {
            label: {
                "completed": True,
                "samples": 3,
                "simulation_checks": {"passing": True},
            }
            for label in labels
        },
    }
    (root / "run_metadata.yaml").write_text(yaml.safe_dump(metadata))
    return path, rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("selection", list(TRIAL_POSITIONS))
@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0])
def test_message_mapping(
    tmp_path: Path, replay: ModuleType, selection: str, scale: float
) -> None:
    path, _ = write_trial(tmp_path, replay)
    message = replay.load_trajectory(path, scale, selection)
    assert list(message.joint_names) == replay.JOINT_NAMES
    assert len(message.points) == 3
    for index, point in enumerate(message.points):
        assert list(point.positions) == [TRIAL_POSITIONS[selection]] * 7
        assert list(point.effort) == [scale * joint * 0.5 for joint in range(1, 8)]
        assert len(point.velocities) == len(point.accelerations) == 0
        assert point.time_from_start.sec == 0
        assert point.time_from_start.nanosec == index * 100_000_000


def test_unknown_label_does_not_fall_back(tmp_path: Path, replay: ModuleType) -> None:
    path, _ = write_trial(tmp_path, replay, labels=("baseline", "tvist_zvd"))
    with pytest.raises(
        ValueError, match="Available CSV labels: 'baseline', 'tvist_zvd'"
    ):
        replay.load_trajectory(path, 1.0, "tvist_ZVD")


@pytest.mark.parametrize("scale", [-0.01, 1.01, math.nan, math.inf])
def test_invalid_gravity_scale(
    tmp_path: Path, replay: ModuleType, scale: float
) -> None:
    with pytest.raises(ValueError, match="gravity-scale"):
        replay.load_trajectory(tmp_path / "unused.csv", scale, "unshaped")


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("time_s", "0", "increase strictly"),
        ("time_s", "nan", "finite"),
        ("J1_q_command_rad", "nan", "Nonfinite"),
        ("J1_q_command_rad", "2", "position"),
        ("J7_torque_gravity_ff_Nm", "8", "effort"),
        ("J1_dq_reference_rad_s", "2", "velocity"),
        ("J1_ddq_reference_rad_s2", "3", "acceleration"),
        ("J1_q_command_rad", "0.3", "segment"),
    ],
)
def test_invalid_commands_rejected(
    tmp_path: Path, replay: ModuleType, column: str, value: str, reason: str
) -> None:
    path, rows = write_trial(tmp_path, replay)
    rows[1][column] = value
    write_rows(path, rows)
    with pytest.raises(ValueError, match=reason):
        replay.load_trajectory(path, 1.0, "unshaped")


@pytest.mark.parametrize(
    "failure", ["missing", "failed", "incomplete", "samples", "joint_order"]
)
def test_requires_matching_passing_metadata(
    tmp_path: Path, replay: ModuleType, failure: str
) -> None:
    path, _ = write_trial(tmp_path, replay)
    metadata_path = tmp_path / "run_metadata.yaml"
    if failure == "missing":
        metadata_path.unlink()
    else:
        metadata = yaml.safe_load(metadata_path.read_text())
        trial = metadata["trials"]["tvist_zvd"]
        if failure == "failed":
            trial["simulation_checks"]["passing"] = False
        elif failure == "incomplete":
            trial["completed"] = False
        elif failure == "samples":
            trial["samples"] = 4
        else:
            metadata["joint_order"].reverse()
        metadata_path.write_text(yaml.safe_dump(metadata))
    with pytest.raises((ValueError, FileNotFoundError)):
        replay.load_trajectory(path, 1.0, "tvist_zvd")


def test_start_state_checks(replay: ModuleType) -> None:
    from sensor_msgs.msg import JointState

    state = JointState(
        name=list(reversed(replay.JOINT_NAMES)),
        position=[0.2] * 7,
        velocity=[0.1] * 7,
    )
    state.header.stamp.sec = 10
    # Finite velocity magnitude is not gated.
    replay.check_start_state(state, 10_100_000_000, [0.2] * 7)
    with pytest.raises(ValueError, match="stale"):
        replay.check_start_state(state, 11_000_000_000, [0.2] * 7)
    with pytest.raises(ValueError, match="future"):
        replay.check_start_state(state, 9_900_000_000, [0.2] * 7)
    state.velocity = [math.nan] + [0.0] * 6
    with pytest.raises(ValueError, match="nonfinite"):
        replay.check_start_state(state, 10_100_000_000, [0.2] * 7)
    state.velocity = [0.0] * 7
    state.position = [math.nan] + [0.2] * 6
    with pytest.raises(ValueError, match="nonfinite"):
        replay.check_start_state(state, 10_100_000_000, [0.2] * 7)


@pytest.mark.parametrize("joint", [0, 6])
@pytest.mark.parametrize("offset", [-0.100001, -0.1, -0.05, 0.0, 0.05, 0.1, 0.100001])
def test_start_position_tolerance(
    replay: ModuleType, joint: int, offset: float
) -> None:
    from sensor_msgs.msg import JointState

    target = [0.05 * i for i in range(7)]
    measured = target.copy()
    measured[joint] += offset
    state = JointState(
        name=list(reversed(replay.JOINT_NAMES)),
        position=list(reversed(measured)),
        velocity=[0.1] * 7,  # No stationary-speed threshold.
    )
    state.header.stamp.sec = 10
    if abs(offset) <= 0.1:
        replay.check_start_state(state, 10_100_000_000, target)
    else:
        with pytest.raises(ValueError, match=f"openarm_left_joint{joint + 1}"):
            replay.check_start_state(state, 10_100_000_000, target)


def test_limits_match_pinned_model(replay: ModuleType) -> None:
    root = Path(__file__).resolve().parents[1]
    model = ET.parse(root / "third_party/openarm_mujoco/v1/openarm_bimanual.xml")
    for name, limits in zip(replay.JOINT_NAMES, replay.POSITION_LIMITS, strict=True):
        joint = model.find(f".//worldbody//joint[@name='{name}']")
        assert joint is not None
        assert list(limits) == [float(value) for value in joint.attrib["range"].split()]
    for i, limit in enumerate(replay.EFFORT_LIMITS, start=1):
        motor = model.find(f".//actuator/motor[@name='left_joint{i}_ctrl']")
        assert motor is not None
        default = model.find(f".//default[@class='{motor.attrib['class']}']/motor")
        assert default is not None
        assert [float(value) for value in default.attrib["forcerange"].split()] == [
            -limit,
            limit,
        ]


def test_cancel_does_not_initialize_ros(
    tmp_path: Path,
    replay: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, _ = write_trial(tmp_path, replay)
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish_csv_trajectory",
            "--csv",
            str(path),
            "--gravity-scale",
            "1",
            "--trajectory",
            "tvist_zvd",
        ],
    )
    monkeypatch.setattr("builtins.input", lambda _: "NO")
    monkeypatch.setattr(
        replay.rclpy, "init", lambda **_: pytest.fail("ROS must not start")
    )
    replay.main()
    prompt = capsys.readouterr().out
    assert "tvist_zvd" in prompt
    assert "visually confirm the arm is stationary" in prompt
    assert "within ±0.1 rad of the CSV starting pose" in prompt
    assert "speed is NOT checked automatically" in prompt
