#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export NVM_DIR="${NVM_DIR:-$HOME/.config/nvm}"
. "$NVM_DIR/nvm.sh" --no-use
nvm use --silent 24

npx repomix --ignore "third_party/**/meshes/**,third_party/**/assets/**,third_party/**/media/**,third_party/**/web/**,ros2_ws/src/**/meshes/**,ros2_ws/src/**/assets/**,ros2_ws/src/**/media/**,ros2_ws/src/**/web/**,repomix-output.*"
