"""Project-local OpenArm v1 bringup with a separate motor-gain YAML."""

from pathlib import Path
from typing import cast
from xml.dom.minidom import Document

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.action import Action
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

V1_ARM_TYPES = ("v1.0", "v10", "v1_0", "openarm_v1.0", "openarm_v10", "openarm_v1_0")


def launch_setup(context: LaunchContext) -> list[Action]:
    """Construct actions and expand Xacro; do not start processes here."""
    arm_type = LaunchConfiguration("arm_type").perform(context)
    if arm_type not in V1_ARM_TYPES:
        raise ValueError("The project-local gain configuration supports v1.0 only.")
    share = Path(get_package_share_directory("me6404_bringup"))
    description_share = Path(get_package_share_directory("openarm_description"))
    gains_file = (
        Path(LaunchConfiguration("control_gains_file").perform(context))
        .expanduser()
        .resolve()
    )
    controllers_file = (
        Path(
            get_package_share_directory(
                LaunchConfiguration("runtime_config_package").perform(context)
            )
        )
        / "config/controllers"
        / LaunchConfiguration("controllers_file").perform(context)
    )
    if not gains_file.is_file() or not controllers_file.is_file():
        raise FileNotFoundError("Motor gain YAML or controller YAML does not exist.")
    document = xacro.process_file(
        str(share / "urdf/openarm_v10.urdf.xacro"),
        mappings={
            "control_gains_file": str(gains_file),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware").perform(
                context
            ),
            "left_can_interface": LaunchConfiguration("left_can_interface").perform(
                context
            ),
            "right_can_interface": LaunchConfiguration("right_can_interface").perform(
                context
            ),
        },
    )
    # process_file returns a DOM document; Xacro's inferred return union is broader.
    robot_description = cast(Document, document).toxml()
    parameters = {
        "robot_description": ParameterValue(robot_description, value_type=str)
    }
    controller = LaunchConfiguration("robot_controller").perform(context)
    if controller not in ("joint_trajectory_controller", "forward_position_controller"):
        raise ValueError(f"Unsupported robot controller: {controller}")
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[parameters],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="both",
            parameters=[parameters, str(controllers_file)],
        ),
        Node(
            package="me6404_bringup",
            executable="publish_motor_gains.py",
            name="publish_motor_gains",
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="log",
            arguments=["-d", str(description_share / "rviz/bimanual.rviz")],
        ),
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            condition=IfCondition(LaunchConfiguration("foxglove")),
            parameters=[
                {
                    "address": "127.0.0.1",
                    "port": 8765,
                    "capabilities": ["connectionGraph", "assets"],
                }
            ],
        ),
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=[
                        "joint_state_broadcaster",
                        f"left_{controller}",
                        f"right_{controller}",
                        "left_gripper_controller",
                        "right_gripper_controller",
                        "-c",
                        "/controller_manager",
                    ],
                )
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("me6404_bringup"))
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "arm_type", default_value="v1.0", choices=list(V1_ARM_TYPES)
            ),
            DeclareLaunchArgument(
                "use_fake_hardware", default_value="true", choices=["true", "false"]
            ),
            DeclareLaunchArgument(
                "robot_controller",
                default_value="joint_trajectory_controller",
                choices=["joint_trajectory_controller", "forward_position_controller"],
            ),
            DeclareLaunchArgument("left_can_interface", default_value="can_arm_l"),
            DeclareLaunchArgument("right_can_interface", default_value="can_arm_r"),
            DeclareLaunchArgument(
                "runtime_config_package", default_value="me6404_bringup"
            ),
            DeclareLaunchArgument(
                "controllers_file", default_value="me6404_controllers.yaml"
            ),
            DeclareLaunchArgument(
                "control_gains_file",
                default_value=str(share / "config/control_gains.yaml"),
                description="Motor Kp/Kd YAML shared by both arms and grippers.",
            ),
            DeclareLaunchArgument(
                "foxglove", default_value="true", choices=["true", "false"]
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
