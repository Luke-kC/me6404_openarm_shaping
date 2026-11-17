"""Check gain extraction and timer publishing without initializing ROS."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="module")
def reporter() -> ModuleType:
    pytest.importorskip("rclpy")
    path = Path(__file__).resolve().parents[1] / "scripts/publish_motor_gains.py"
    spec = importlib.util.spec_from_file_location("publish_motor_gains", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robot_description() -> str:
    systems = []
    for side, offset in (("left", 0), ("right", 10)):
        parameters = "".join(
            f'<param name="{quantity}{joint}">{offset + joint * scale}</param>'
            for quantity, scale in (("kp", 1.0), ("kd", 0.1))
            for joint in range(1, 8)
        )
        systems.append(
            f'<ros2_control name="openarm_{side}_hardware_interface"><hardware>'
            f'{parameters}<param name="kp_hand">999</param>'
            '<param name="kd_hand">999</param></hardware></ros2_control>'
        )
    return f"<robot>{''.join(systems)}</robot>"


def test_extract_loaded_gains_without_gripper(reporter: ModuleType) -> None:
    gains = reporter.extract_gains(robot_description())
    for side, offset in (("left", 0), ("right", 10)):
        assert gains[side]["kp"] == [offset + i for i in range(1, 8)]
        assert gains[side]["kd"] == [offset + i * 0.1 for i in range(1, 8)]


def test_timer_publishes_four_numeric_arrays(reporter: ModuleType) -> None:
    publishers = {
        (side, quantity): MagicMock()
        for side in ("left", "right")
        for quantity in ("kp", "kd")
    }
    gains = reporter.extract_gains(robot_description())
    node = SimpleNamespace(gains=gains, gain_publishers=publishers)
    reporter.MotorGainPublisher.publish_gains(node)
    for (side, quantity), publisher in publishers.items():
        publisher.publish.assert_called_once()
        message = publisher.publish.call_args.args[0]
        assert list(message.data) == gains[side][quantity]
        assert len(message.data) == 7


@pytest.mark.parametrize(
    "replacement", ["", '<param name="kp1">nan</param>', '<param name="kp1">-1</param>']
)
def test_invalid_or_missing_gain_rejected(
    reporter: ModuleType, replacement: str
) -> None:
    description = robot_description().replace(
        '<param name="kp1">1.0</param>', replacement
    )
    with pytest.raises(ValueError):
        reporter.extract_gains(description)


def test_no_publish_without_description(reporter: ModuleType) -> None:
    publisher = MagicMock()
    node = SimpleNamespace(gains={}, gain_publishers={("left", "kp"): publisher})
    reporter.MotorGainPublisher.publish_gains(node)
    publisher.publish.assert_not_called()


def test_invalid_update_clears_old_gains(reporter: ModuleType) -> None:
    from std_msgs.msg import String

    logger = MagicMock()
    node = SimpleNamespace(
        gains=reporter.extract_gains(robot_description()),
        get_logger=lambda: logger,
    )
    reporter.MotorGainPublisher.receive_description(node, String(data="<robot/>"))
    assert node.gains == {}
    logger.warning.assert_called_once()


def test_one_hz_timer_and_latched_description_subscription(
    reporter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rclpy.qos import DurabilityPolicy

    # Replace every Node/ROS operation; no context or ROS node is initialized.
    monkeypatch.setattr(reporter.Node, "__init__", MagicMock(return_value=None))
    publisher = MagicMock()
    subscription = MagicMock()
    timer = MagicMock()
    monkeypatch.setattr(reporter.Node, "create_publisher", publisher)
    monkeypatch.setattr(reporter.Node, "create_subscription", subscription)
    monkeypatch.setattr(reporter.Node, "create_timer", timer)
    monkeypatch.setattr(reporter.Node, "get_logger", MagicMock())
    node = reporter.MotorGainPublisher()
    timer.assert_called_once_with(1.0, node.publish_gains)
    assert publisher.call_count == 4
    assert {call.args[1] for call in publisher.call_args_list} == {
        f"/{side}_motor_gains/{quantity}"
        for side in ("left", "right")
        for quantity in ("kp", "kd")
    }
    assert subscription.call_args.args[1] == "/robot_description"
    qos = subscription.call_args.args[3]
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
