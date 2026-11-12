#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec uv run --frozen --no-sync pybind11-stubgen mujoco -o typings
