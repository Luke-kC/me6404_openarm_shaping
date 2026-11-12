#!/usr/bin/env bash
set -euo pipefail

npx repomix --include "src/**,tests/**,AGENTS.md,README.md,pyproject.toml,.gitmodules,third_party/openarm_mujoco/v1/openarm_bimanual.xml"
