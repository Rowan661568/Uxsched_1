#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zm/project/UXSched"
OUT="${1:-${ROOT}/build-hb/hb_xqueue_probe}"

mkdir -p "$(dirname "${OUT}")"
c++ -std=c++17 -O2 -Wall -Wextra \
  -I"${ROOT}/platforms/cuda/hal/include" \
  -I"${ROOT}/include" \
  "${ROOT}/tools/hb_xqueue_probe.cpp" \
  -ldl \
  -o "${OUT}"

printf '%s\n' "${OUT}"
