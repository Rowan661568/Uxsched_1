#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zm/project/UXSched"
BUILD_DIR="${1:-${ROOT}/build-hb-cu128}"
OUT="${2:-${BUILD_DIR}/hb_metadata_registry_probe}"

mkdir -p "$(dirname "${OUT}")"

c++ -std=c++17 -O2 -Wall -Wextra \
  -I"${ROOT}/platforms/cuda/hal/include" \
  -I"${ROOT}/preempt/include" \
  -I"${ROOT}/protocol/include" \
  -I"${ROOT}/utils/include" \
  "${ROOT}/tools/hb_metadata_registry_probe.cpp" \
  -L"${BUILD_DIR}/platforms/cuda" \
  -L"${BUILD_DIR}/preempt" \
  -Wl,-rpath,"${BUILD_DIR}/platforms/cuda:${BUILD_DIR}/preempt" \
  -lhalcuda \
  -Wl,--no-as-needed \
  -lpreempt \
  -Wl,--as-needed \
  -ldl \
  -o "${OUT}"

printf '%s\n' "${OUT}"
