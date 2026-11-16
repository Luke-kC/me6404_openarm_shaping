#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

npx repomix --ignore "third_party/**/meshes/**,third_party/**/assets/**,third_party/**/media/**,third_party/**/web/**,repomix-output.*"
