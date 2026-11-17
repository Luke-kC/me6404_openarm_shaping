"""Wrapper checks with mocked processes only; never start ROS or move hardware."""

import argparse
import importlib
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def wrapper(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("record_csv_trial")


@pytest.fixture
def fake_ros(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    ros = MagicMock()
    action = MagicMock()
    monkeypatch.setitem(sys.modules, "rclpy", ros)
    monkeypatch.setitem(sys.modules, "rclpy.action", action)
    monkeypatch.setitem(sys.modules, "control_msgs", MagicMock())
    monkeypatch.setitem(sys.modules, "control_msgs.action", MagicMock())
    return ros, action.ActionClient


def inputs(tmp_path: Path) -> argparse.Namespace:
    source = tmp_path / "source"
    source.mkdir()
    csv_path = source / "sim_results.csv"
    csv_path.write_text(
        "trajectory,time_s\nunshaped,0\nshaped,0\nunshaped,0.001\nshaped,0.002\n"
    )
    (source / "sim_metadata.yaml").write_text("results_file: sim_results.csv\n")
    return argparse.Namespace(
        csv=csv_path,
        output_dir=tmp_path / "hardware_out",
        trajectory="shaped",
        gravity_scale=0.5,
        post_roll=0.0,
        ready_timeout=1.0,
    )


@pytest.mark.parametrize("selection", ["shaped", "unshaped"])
def test_copies_and_unique_directories(
    wrapper: ModuleType, tmp_path: Path, selection: str
) -> None:
    args = inputs(tmp_path)
    args.trajectory = selection
    first, copied, duration = wrapper.copy_inputs(
        args.csv, args.output_dir, args.trajectory
    )
    second, _, _ = wrapper.copy_inputs(args.csv, args.output_dir, args.trajectory)
    assert first != second
    assert first.parent == second.parent == args.output_dir / selection
    assert duration == (0.002 if selection == "shaped" else 0.001)
    assert copied.read_bytes() == args.csv.read_bytes()
    assert (first / "sim_metadata.yaml").read_bytes() == (
        args.csv.parent / "sim_metadata.yaml"
    ).read_bytes()
    assert copied.name == args.csv.name


def test_record_before_publish_and_finalize(
    wrapper: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = inputs(tmp_path)
    events: list[str] = []
    recorder = MagicMock(pid=1234)
    recorder.poll.return_value = None
    recorder.returncode = 0
    publisher = MagicMock()
    publisher.poll.return_value = 0
    publisher.returncode = 0
    commands: list[list[str]] = []

    def popen(command: list[str], **kwargs: object) -> MagicMock:
        commands.append(command)
        if command[1] == "bag":
            events.append("record")
            assert kwargs["start_new_session"] is True
            return recorder
        events.append("publish")
        assert "stdin" not in kwargs  # Original interactive GO prompt is retained.
        assert "stdout" not in kwargs
        return publisher

    def ready(*_args: object) -> None:
        events.append("ready")

    monkeypatch.setattr(wrapper.subprocess, "Popen", popen)
    monkeypatch.setattr(wrapper, "wait_for_recorder", ready)
    kill = MagicMock()
    monkeypatch.setattr(wrapper.os, "killpg", kill)
    folder = wrapper.record_trial(args)
    assert events == ["record", "ready", "publish"]
    assert commands[0][commands[0].index("--storage") + 1] == "mcap"
    assert "--include-unpublished-topics" in commands[0]
    assert "--include-hidden-topics" in commands[0]
    assert str(folder / "sim_results.csv") in commands[1]
    assert commands[1][-2:] == ["--gravity-scale", "0.5"]
    kill.assert_called_once_with(1234, signal.SIGINT)
    recorder.wait.assert_called_once_with(timeout=15)
    manifest = json.loads((folder / "trial.json").read_text())
    assert manifest["status"] == "recording_window_finished"
    assert manifest["trajectory_duration_s"] == 0.002
    assert "/joint_states" in manifest["topics_requested"]
    assert wrapper.COMMAND_RECORD_TOPIC in manifest["topics_requested"]
    assert f"{wrapper.ACTION}/_action/feedback" in manifest["topics_requested"]
    assert manifest["command_transport"] == "FollowJointTrajectory action"
    assert (
        "/left_joint_trajectory_controller/controller_state"
        in manifest["topics_requested"]
    )


@pytest.mark.parametrize("failure", [RuntimeError("not ready"), KeyboardInterrupt()])
def test_readiness_failure_never_starts_publisher_and_closes_recorder(
    wrapper: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    recorder = MagicMock(pid=1234)
    recorder.poll.return_value = None
    recorder.returncode = 0
    popen = MagicMock(return_value=recorder)
    monkeypatch.setattr(wrapper.subprocess, "Popen", popen)
    monkeypatch.setattr(wrapper, "wait_for_recorder", MagicMock(side_effect=failure))
    kill = MagicMock()
    monkeypatch.setattr(wrapper.os, "killpg", kill)
    args = inputs(tmp_path)
    with pytest.raises(type(failure)):
        wrapper.record_trial(args)
    assert popen.call_count == 1
    kill.assert_called_once_with(1234, signal.SIGINT)
    folder = next((args.output_dir / args.trajectory).iterdir())
    assert (
        json.loads((folder / "trial.json").read_text())["status"]
        == "interrupted_or_failed"
    )
    assert (folder / "sim_results.csv").is_file()


def test_readiness_checks_recorder_and_controller(
    wrapper: ModuleType, fake_ros: tuple[MagicMock, MagicMock]
) -> None:
    ros, client_factory = fake_ros
    node = ros.create_node.return_value

    def subscriptions(topic: str) -> list[SimpleNamespace]:
        endpoints = [SimpleNamespace(node_name="test_recorder")]
        return endpoints

    node.get_subscriptions_info_by_topic.side_effect = subscriptions
    recorder = MagicMock()
    recorder.poll.return_value = None
    wrapper.wait_for_recorder(recorder, "test_recorder", 1.0)
    assert node.get_subscriptions_info_by_topic.call_count == 4
    client_factory.return_value.server_is_ready.assert_called_once()
    client_factory.return_value.destroy.assert_called_once()
    node.destroy_node.assert_called_once()
    ros.try_shutdown.assert_called_once()


def test_dead_recorder_prevents_readiness(
    wrapper: ModuleType, fake_ros: tuple[MagicMock, MagicMock]
) -> None:
    ros, _ = fake_ros
    recorder = MagicMock()
    recorder.poll.return_value = 1
    with pytest.raises(RuntimeError, match="Recorder exited"):
        wrapper.wait_for_recorder(recorder, "test_recorder", 1.0)
    ros.try_shutdown.assert_called_once()


def test_action_server_not_ready_prevents_publication(
    wrapper: ModuleType,
    fake_ros: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ros, client_factory = fake_ros
    ros.create_node.return_value.get_subscriptions_info_by_topic.return_value = [
        SimpleNamespace(node_name="test_recorder")
    ]
    client_factory.return_value.server_is_ready.return_value = False
    monkeypatch.setattr(wrapper.time, "monotonic", MagicMock(side_effect=[0, 0, 2]))
    recorder = MagicMock()
    recorder.poll.return_value = None
    with pytest.raises(RuntimeError, match="action server must also be ready"):
        wrapper.wait_for_recorder(recorder, "test_recorder", 1.0)
    client_factory.return_value.send_goal_async.assert_not_called()


def test_monitor_detects_recording_failure(wrapper: ModuleType) -> None:
    publisher = MagicMock()
    publisher.poll.return_value = None
    recorder = MagicMock()
    recorder.poll.return_value = 1
    with pytest.raises(RuntimeError, match="command status is unknown"):
        wrapper.monitor(publisher, recorder)


def test_slow_recorder_shutdown_escalates(
    wrapper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = MagicMock(pid=1234)
    recorder.poll.return_value = None
    recorder.wait.side_effect = [subprocess.TimeoutExpired("record", 15), 0]
    kill = MagicMock()
    monkeypatch.setattr(wrapper.os, "killpg", kill)
    wrapper.stop_process(recorder)
    assert [call.args for call in kill.call_args_list] == [
        (1234, signal.SIGINT),
        (1234, signal.SIGTERM),
    ]
