#!/usr/bin/env bash
set -euo pipefail

# Apply minimal OpenArm v1.0 compatibility patches for ME6404:
#   1. Default to Classic CAN instead of CAN FD.
#   2. Forward can_fd into the bimanual hardware Xacro.
#   3. Remove the unused MoveIt bringup dependency.
#
# Safe to run multiple times against the pinned upstream revisions.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

DESCRIPTION_DIR="$REPO_ROOT/ros2_ws/src/openarm_description"
ROS2_DIR="$REPO_ROOT/ros2_ws/src/openarm_ros2"

python3 - "$DESCRIPTION_DIR" "$ROS2_DIR" <<'PY'
from pathlib import Path
import sys

description_dir = Path(sys.argv[1])
ros2_dir = Path(sys.argv[2])

urdf_dir = description_dir / "assets/robot/openarm_v1.0/urdf"

files = {
    "top_level": urdf_dir / "openarm_v10.urdf.xacro",
    "robot": urdf_dir / "robot/openarm_robot.xacro",
    "bimanual": (
        urdf_dir / "ros2_control/openarm.bimanual.ros2_control.xacro"
    ),
    "bringup": ros2_dir / "openarm_bringup/package.xml",
}


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    """Apply one replacement, accepting an already-patched file."""
    if old in text:
        if text.count(old) != 1:
            raise RuntimeError(f"Unexpected multiple matches in {path}")
        return text.replace(old, new, 1)

    if new in text:
        return text

    raise RuntimeError(
        f"Unexpected upstream content in {path}. "
        "Check the pinned submodule revision."
    )


# Read all files before making changes.
for path in files.values():
    if not path.is_file():
        raise SystemExit(f"Missing upstream file: {path}")

original = {name: path.read_text() for name, path in files.items()}
patched = original.copy()

# 1. Set the top-level v1.0 default to Classic CAN.
patched["top_level"] = replace_once(
    patched["top_level"],
    '<xacro:arg name="can_fd" default="true" />',
    '<xacro:arg name="can_fd" default="false" />',
    files["top_level"],
)

# 2. Forward can_fd to the bimanual hardware macro.
patched["robot"] = replace_once(
    patched["robot"],
    'right_arm_prefix="${right_arm_prefix}"',
    'right_arm_prefix="${right_arm_prefix}"\n'
    '             can_fd="${can_fd}"',
    files["robot"],
)

# 3. Set the bimanual hardware macro default to Classic CAN.
patched["bimanual"] = replace_once(
    patched["bimanual"],
    "can_fd:=^|true",
    "can_fd:=^|false",
    files["bimanual"],
)

# 4. Remove the unused MoveIt dependency.
patched["bringup"] = replace_once(
    patched["bringup"],
    "  <exec_depend>openarm_bimanual_moveit_config</exec_depend>\n",
    "",
    files["bringup"],
) if "openarm_bimanual_moveit_config" in patched["bringup"] else patched["bringup"]

# Write only changed files.
for name, path in files.items():
    if patched[name] != original[name]:
        path.write_text(patched[name])
        print(f"Patched: {path.relative_to(path.parents[3])}")
    else:
        print(f"Already patched: {path.name}")

print("OpenArm compatibility patches applied successfully.")
PY
