#!/usr/bin/env bash
set -euo pipefail

# Configure OpenArm v1.0 bimanual hardware for Classic CAN.
# Required because the pinned upstream launch does not expose can_fd.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

XACRO="$REPO_ROOT/ros2_ws/src/openarm_description/assets/robot/openarm_v1.0/urdf/ros2_control/openarm.bimanual.ros2_control.xacro"

if [[ ! -f "$XACRO" ]]; then
    echo "Error: OpenArm Xacro not found. Initialize Git submodules first." >&2
    exit 1
fi

python3 - "$XACRO" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
content = path.read_text()

original = "can_fd:=^|true"
patched = "can_fd:=^|false"

if patched in content:
    print("Classic CAN patch already applied.")
elif content.count(original) == 1:
    path.write_text(content.replace(original, patched, 1))
    print("Classic CAN patch applied successfully.")
else:
    raise SystemExit(
        "Error: Unexpected upstream Xacro content. "
        "Check the pinned openarm_description revision."
    )
PY
