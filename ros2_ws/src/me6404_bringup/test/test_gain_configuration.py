"""Expand descriptions and construct launch actions without starting ROS nodes."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest
import yaml

xacro = pytest.importorskip("xacro")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION_ROOT = PACKAGE_ROOT.parent / "openarm_description"
UPSTREAM_GAINS = (
    DESCRIPTION_ROOT / "assets/robot/openarm_v1.0/config/arm/control_gains.yaml"
)
UPSTREAM_XACRO = (
    DESCRIPTION_ROOT / "assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro"
)


def expand_local(gains_path: Path, fake: str = "false") -> str:
    return xacro.process_file(
        str(PACKAGE_ROOT / "urdf/openarm_v10.urdf.xacro"),
        mappings={
            "control_gains_file": str(gains_path),
            "use_fake_hardware": fake,
            "left_can_interface": "can_arm_l",
            "right_can_interface": "can_arm_r",
        },
    ).toxml()


def test_default_gain_file_reaches_hardware() -> None:
    gains_path = PACKAGE_ROOT / "config/control_gains.yaml"
    gains = yaml.safe_load(gains_path.read_text())
    robot = ET.fromstring(expand_local(gains_path))
    for side in ("left", "right"):
        hardware = robot.find(
            f"ros2_control[@name='openarm_{side}_hardware_interface']/hardware"
        )
        assert hardware is not None
        parameters = {
            param.attrib["name"]: param.text for param in hardware.findall("param")
        }
        for joint in range(1, 8):
            for gain in ("kp", "kd"):
                value = parameters[f"{gain}{joint}"]
                assert value is not None
                assert float(value) == gains[f"joint{joint}"][gain]
        for gain in ("kp", "kd"):
            value = parameters[f"{gain}_hand"]
            assert value is not None
            assert float(value) == gains["hand"][gain]


@pytest.mark.parametrize("fake", ["false", "true"])
def test_local_description_matches_upstream(fake: str) -> None:
    upstream = xacro.process_file(
        str(UPSTREAM_XACRO),
        mappings={
            "arm_type": "v1.0",
            "bimanual": "true",
            "ros2_control": "true",
            "use_fake_hardware": fake,
            "left_can_interface": "can_arm_l",
            "right_can_interface": "can_arm_r",
        },
    ).toxml()
    # Compare geometry/interfaces using identical gains, allowing the user to
    # edit the project-local YAML independently afterward.
    local = expand_local(UPSTREAM_GAINS, fake)
    assert ET.canonicalize(local, strip_text=True) == ET.canonicalize(
        upstream, strip_text=True
    )


def test_gain_override_reaches_both_hardware_interfaces(tmp_path: Path) -> None:
    gains = yaml.safe_load(UPSTREAM_GAINS.read_text())
    gains["joint1"]["kp"] = 71.0
    gains["joint4"]["kd"] = 1.5
    gains["hand"]["kp"] = 4.0
    override = tmp_path / "test_gains.yaml"
    override.write_text(yaml.safe_dump(gains))
    robot = ET.fromstring(expand_local(override))
    interfaces = robot.findall("ros2_control")
    assert len(interfaces) == 2
    for interface, can in zip(interfaces, ("can_arm_l", "can_arm_r"), strict=True):
        hardware = interface.find("hardware")
        assert hardware is not None
        parameters = {
            param.attrib["name"]: param.text for param in hardware.findall("param")
        }
        assert hardware.findtext("plugin") == "openarm_hardware/OpenArmHW"
        assert parameters["kp1"] == "71.0"
        assert parameters["kd4"] == "1.5"
        assert parameters["kp_hand"] == "4.0"
        assert parameters["can_interface"] == can


@pytest.fixture
def local_launch(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = PACKAGE_ROOT / "launch/openarm.bimanual.launch.py"
    spec = importlib.util.spec_from_file_location("local_openarm_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream_share = module.get_package_share_directory

    def package_share(name: str) -> str:
        return str(PACKAGE_ROOT) if name == "me6404_bringup" else upstream_share(name)

    monkeypatch.setattr(module, "get_package_share_directory", package_share)
    return module


def test_launch_defaults_and_actions(local_launch: ModuleType) -> None:
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, TimerAction
    from launch_ros.actions import Node

    description = local_launch.generate_launch_description()
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    assert context.launch_configurations["use_fake_hardware"] == "true"
    assert context.launch_configurations["left_can_interface"] == "can_arm_l"
    assert context.launch_configurations["right_can_interface"] == "can_arm_r"
    assert context.launch_configurations["control_gains_file"] == str(
        PACKAGE_ROOT / "config/control_gains.yaml"
    )
    context.launch_configurations["use_fake_hardware"] = "false"
    # Constructing Node actions does not execute them or communicate with hardware.
    actions = local_launch.launch_setup(context)
    assert sum(isinstance(action, Node) for action in actions) == 5
    bridge = actions[-2]
    assert isinstance(bridge, Node)
    assert bridge.condition is not None and bridge.condition.evaluate(context)
    context.launch_configurations["foxglove"] = "false"
    assert not bridge.condition.evaluate(context)
    assert isinstance(actions[-1], TimerAction)
    context.launch_configurations["arm_type"] = "v2.0"
    with pytest.raises(ValueError, match="v1.0 only"):
        local_launch.launch_setup(context)
