#!/usr/bin/env bash
# Run ResNet50 inference with XSched transparent CUDA scheduling (single GPU).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XSCHED_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
XSCHED_INSTALL="${XSCHED_INSTALL:-${XSCHED_ROOT}/output}"

if [[ ! -d "${XSCHED_INSTALL}/lib" ]]; then
  echo "ERROR: XSched not built at ${XSCHED_INSTALL}" >&2
  echo "  cd ${XSCHED_ROOT} && git submodule update --init --recursive" >&2
  echo "  make cuda INSTALL_PATH=${XSCHED_INSTALL}" >&2
  exit 1
fi

# LCL: in-process scheduler (no xserver needed for single-process trials on one GPU).
# Use XSCHED_SCHEDULER=GLB and start xserver for multi-process priority scheduling.
export XSCHED_SCHEDULER="${XSCHED_SCHEDULER:-LCL}"
export XSCHED_POLICY="${XSCHED_POLICY:-HPF}"
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_LEVEL="${XSCHED_AUTO_XQUEUE_LEVEL:-1}"
export XSCHED_AUTO_XQUEUE_THRESHOLD="${XSCHED_AUTO_XQUEUE_THRESHOLD:-16}"
export XSCHED_AUTO_XQUEUE_BATCH_SIZE="${XSCHED_AUTO_XQUEUE_BATCH_SIZE:-8}"
export XSCHED_AUTO_XQUEUE_PRIORITY="${XSCHED_AUTO_XQUEUE_PRIORITY:-0}"

export LD_LIBRARY_PATH="${XSCHED_INSTALL}/lib:${LD_LIBRARY_PATH:-}"

echo "XSched install: ${XSCHED_INSTALL}"
echo "  XSCHED_SCHEDULER=${XSCHED_SCHEDULER}"
echo "  XSCHED_AUTO_XQUEUE_PRIORITY=${XSCHED_AUTO_XQUEUE_PRIORITY}"
echo ""

exec python3 "${SCRIPT_DIR}/resnet50_infer.py" "$@"
