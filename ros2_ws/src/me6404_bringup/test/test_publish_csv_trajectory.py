"""Action execution tests with mocked ROS operations; never create a ROS node."""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="module")
def publisher() -> ModuleType:
    pytest.importorskip("rclpy")
    path = Path(__file__).resolve().parents[1] / "scripts/publish_csv_trajectory.py"
    spec = importlib.util.spec_from_file_location("publish_csv_trajectory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def action_case(
    publisher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    node = MagicMock()
    client = MagicMock()
    handle = client.send_goal_async.return_value.result.return_value
    handle.accepted = True
    handle.goal_id.uuid = list(range(16))
    response = SimpleNamespace(
        status=publisher.GoalStatus.STATUS_SUCCEEDED,
        result=publisher.FollowJointTrajectory.Result(),
    )
    handle.get_result_async.return_value.result.return_value = response
    message = publisher.JointTrajectory(
        joint_names=publisher.JOINT_NAMES,
        points=[
            publisher.JointTrajectoryPoint(
                positions=[0.0] * 7,
                effort=[0.1] * 7,
                time_from_start=publisher.Duration(sec=0),
            ),
            publisher.JointTrajectoryPoint(
                positions=[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                effort=[0.2] * 7,
                time_from_start=publisher.Duration(sec=2),
            ),
        ],
    )
    spin = MagicMock()
    monkeypatch.setattr(publisher.rclpy, "spin_until_future_complete", spin)
    return SimpleNamespace(
        node=node,
        client=client,
        handle=handle,
        response=response,
        message=message,
        mirror=MagicMock(),
        spin=spin,
    )


def execute(publisher: ModuleType, case: SimpleNamespace) -> None:
    publisher.execute_trajectory(case.node, case.client, case.message, case.mirror)


def test_accepted_goal_waits_for_success_and_records_copy(
    publisher: ModuleType, action_case: SimpleNamespace
) -> None:
    case = action_case
    execute(publisher, case)
    case.client.send_goal_async.assert_called_once()
    sent = case.client.send_goal_async.call_args.args[0].trajectory
    assert sent == case.message
    case.mirror.publish.assert_called_once_with(case.message)
    case.handle.get_result_async.assert_called_once()
    assert case.spin.call_count == 2
    assert (
        case.spin.call_args_list[0].kwargs["timeout_sec"]
        == publisher.CONNECTION_TIMEOUT_S
    )
    assert (
        case.spin.call_args_list[1].kwargs["timeout_sec"]
        == 2 + publisher.CONNECTION_TIMEOUT_S
    )
    output = " ".join(
        call.args[0] for call in case.node.get_logger().info.call_args_list
    )
    assert "accepted trajectory goal" in output
    assert "successful execution" in output
    case.handle.cancel_goal_async.assert_not_called()


def test_rejected_goal_is_not_retried_or_mirrored(
    publisher: ModuleType, action_case: SimpleNamespace
) -> None:
    case = action_case
    case.handle.accepted = False
    with pytest.raises(RuntimeError, match="rejected"):
        execute(publisher, case)
    case.client.send_goal_async.assert_called_once()
    case.mirror.publish.assert_not_called()
    case.handle.get_result_async.assert_not_called()


@pytest.mark.parametrize("stage", ["acceptance", "result"])
def test_timeout_reports_unknown_status_without_retry_or_cancel(
    publisher: ModuleType, action_case: SimpleNamespace, stage: str
) -> None:
    case = action_case
    future = (
        case.client.send_goal_async.return_value
        if stage == "acceptance"
        else case.handle.get_result_async.return_value
    )
    future.done.return_value = False
    with pytest.raises(RuntimeError, match="status is unknown"):
        execute(publisher, case)
    case.client.send_goal_async.assert_called_once()
    case.handle.cancel_goal_async.assert_not_called()


@pytest.mark.parametrize("failure", ["aborted", "canceled", "error-code"])
def test_unsuccessful_result_reports_controller_details(
    publisher: ModuleType, action_case: SimpleNamespace, failure: str
) -> None:
    case = action_case
    if failure == "aborted":
        case.response.status = publisher.GoalStatus.STATUS_ABORTED
    elif failure == "canceled":
        case.response.status = publisher.GoalStatus.STATUS_CANCELED
    else:
        case.response.result.error_code = -1
    case.response.result.error_string = "test controller failure"
    with pytest.raises(RuntimeError, match="test controller failure"):
        execute(publisher, case)
    case.client.send_goal_async.assert_called_once()


def test_feedback_is_reported_without_reissuing_goal(
    publisher: ModuleType, action_case: SimpleNamespace
) -> None:
    case = action_case
    execute(publisher, case)
    callback = case.client.send_goal_async.call_args.kwargs["feedback_callback"]
    callback(object())
    callback(object())
    logs = [call.args[0] for call in case.node.get_logger().info.call_args_list]
    assert logs.count("Controller execution feedback received.") == 1
    case.client.send_goal_async.assert_called_once()


def test_no_go_never_initializes_ros_or_sends_goal(
    publisher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish_csv_trajectory.py",
            "--csv",
            "test_results.csv",
            "--trajectory",
            "shaped",
            "--gravity-scale",
            "1",
        ],
    )
    monkeypatch.setattr(
        publisher,
        "load_trajectory",
        MagicMock(return_value=SimpleNamespace(points=[object()])),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "NO")
    init = MagicMock()
    client = MagicMock()
    monkeypatch.setattr(publisher.rclpy, "init", init)
    monkeypatch.setattr(publisher, "ActionClient", client)
    publisher.main()
    init.assert_not_called()
    client.assert_not_called()


@pytest.mark.parametrize("state_case", ["valid", "stale", "off-start"])
def test_main_preserves_start_state_gate_before_action(
    publisher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    action_case: SimpleNamespace,
    state_case: str,
) -> None:
    case = action_case
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish_csv_trajectory.py",
            "--csv",
            "test_results.csv",
            "--trajectory",
            "shaped",
            "--gravity-scale",
            "1",
        ],
    )
    monkeypatch.setattr(publisher, "load_trajectory", lambda *_args: case.message)
    monkeypatch.setattr("builtins.input", lambda _prompt: "GO")
    monkeypatch.setattr(publisher, "Node", MagicMock(return_value=case.node))
    monkeypatch.setattr(publisher, "ActionClient", MagicMock(return_value=case.client))
    monkeypatch.setattr(publisher.rclpy, "init", MagicMock())
    monkeypatch.setattr(publisher.rclpy, "try_shutdown", MagicMock())
    monkeypatch.setattr(publisher.rclpy, "ok", lambda: True)
    monkeypatch.setattr(publisher.rclpy, "spin_once", MagicMock())
    case.node.get_clock.return_value.now.return_value.nanoseconds = 1_000_000_000
    state = publisher.JointState(
        name=publisher.JOINT_NAMES, position=[0.0] * 7, velocity=[0.0] * 7
    )
    state.header.stamp.sec = 1 if state_case != "stale" else 0
    if state_case == "off-start":
        state.position[1] = -0.5

    def subscribe(
        _type: object, _topic: str, callback: Callable[[object], None], _qos: object
    ) -> MagicMock:
        callback(state)
        return MagicMock()

    case.node.create_subscription.side_effect = subscribe
    execute_mock = MagicMock()
    monkeypatch.setattr(publisher, "execute_trajectory", execute_mock)
    if state_case == "valid":
        publisher.main()
        execute_mock.assert_called_once()
    else:
        with pytest.raises(SystemExit):
            publisher.main()
        execute_mock.assert_not_called()
    case.client.destroy.assert_called_once()
    assert (
        case.node.create_publisher.call_args.args[1] == publisher.COMMAND_RECORD_TOPIC
    )
