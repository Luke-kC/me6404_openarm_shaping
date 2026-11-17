#!/usr/bin/env python3
"""Report launch-configured J1--J7 motor gains, not motor readback, at 1 Hz."""

import math
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray, String


def extract_gains(description: str) -> dict[str, dict[str, list[float]]]:
    """Extract both arms from the running URDF; never substitute YAML defaults."""
    robot = ET.fromstring(description)
    gains: dict[str, dict[str, list[float]]] = {}
    for side in ("left", "right"):
        hardware = robot.find(
            f"ros2_control[@name='openarm_{side}_hardware_interface']/hardware"
        )
        if hardware is None:
            raise ValueError(f"No {side} arm hardware in robot description.")
        parameters = {
            param.attrib["name"]: param.text for param in hardware.findall("param")
        }
        gains[side] = {}
        for quantity in ("kp", "kd"):
            values: list[float] = []
            for joint in range(1, 8):
                text = parameters.get(f"{quantity}{joint}")
                if text is None:
                    raise ValueError(
                        f"No {side} {quantity}{joint} in robot description "
                        "(fake hardware may omit motor gains)."
                    )
                value = float(text)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {side} {quantity}{joint}: {text}")
                values.append(value)
            gains[side][quantity] = values
    return gains


class MotorGainPublisher(Node):
    def __init__(self) -> None:
        super().__init__("publish_motor_gains")
        self.gains: dict[str, dict[str, list[float]]] = {}
        self.gain_publishers: dict[tuple[str, str], Publisher] = {
            (side, quantity): self.create_publisher(
                Float64MultiArray, f"/{side}_motor_gains/{quantity}", 1
            )
            for side in ("left", "right")
            for quantity in ("kp", "kd")
        }
        # robot_state_publisher uses transient-local delivery so late subscribers
        # can receive the already-published robot description.
        self.description_subscription = self.create_subscription(
            String,
            "/robot_description",
            self.receive_description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.publish_timer = self.create_timer(1.0, self.publish_gains)
        self.get_logger().info(
            "Waiting for /robot_description; gain array indices 0--6 are J1--J7."
        )

    def receive_description(self, message: String) -> None:
        try:
            self.gains = extract_gains(message.data)
        except (ET.ParseError, ValueError, KeyError) as error:
            self.gains = {}  # Never keep reporting stale gains after an invalid update.
            self.get_logger().warning(f"Motor gains unavailable: {error}")

    def publish_gains(self) -> None:
        for side, quantities in self.gains.items():
            for quantity, values in quantities.items():
                self.gain_publishers[side, quantity].publish(
                    Float64MultiArray(data=values)
                )


def main() -> None:
    rclpy.init()
    node = MotorGainPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
