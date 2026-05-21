#!/usr/bin/env bash
# ResNet50 inference without XSched (baseline timings).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/resnet50_infer.py" "$@"
