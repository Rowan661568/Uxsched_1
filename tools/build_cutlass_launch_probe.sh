#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zm/project/UXSched"
BUILD_DIR="${ROOT}/build-cutlass-cu128"
CUTLASS_ROOT_VALUE="${CUTLASS_ROOT:-/home/zm/project/cutlass}"
CUDA_HOME_VALUE="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_COMPILER_VALUE="${CUDACXX:-/usr/local/cuda-12.8/bin/nvcc}"
ARCH="120-real;120-virtual"

usage() {
  printf 'Usage: %s [--build-dir DIR] [--cutlass-root DIR] [--cuda-home DIR] [--cuda-compiler PATH] [--arch 120-real;120-virtual]\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --cutlass-root)
      CUTLASS_ROOT_VALUE="$2"
      shift 2
      ;;
    --cuda-home)
      CUDA_HOME_VALUE="$2"
      shift 2
      ;;
    --cuda-compiler)
      CUDA_COMPILER_VALUE="$2"
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${ARCH}" != "120-real;120-virtual" ]]; then
  printf 'only CUDA architecture 120-real;120-virtual is supported by this probe\n' >&2
  exit 2
fi

if [[ ! -f "${CUTLASS_ROOT_VALUE}/include/cutlass/cutlass.h" ]]; then
  printf 'CUTLASS_ROOT_UNAVAILABLE: %s\n' "${CUTLASS_ROOT_VALUE}" >&2
  exit 2
fi

if [[ ! -x "${CUDA_COMPILER_VALUE}" ]]; then
  printf 'CUDA_COMPILER_UNAVAILABLE: %s\n' "${CUDA_COMPILER_VALUE}" >&2
  exit 2
fi

mkdir -p "${BUILD_DIR}"

CUTLASS_REVISION="$(git -C "${CUTLASS_ROOT_VALUE}" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
NVCC_VERSION="$("${CUDA_COMPILER_VALUE}" --version | tr '\n' ' ' | sed 's/[[:space:]][[:space:]]*/ /g')"

CONFIGURE_CMD=(
  cmake -S "${ROOT}/benchmarks/cutlass" -B "${BUILD_DIR}"
  -DCUTLASS_ROOT="${CUTLASS_ROOT_VALUE}"
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"
  -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER_VALUE}"
  -DCMAKE_BUILD_TYPE=Release
)
BUILD_CMD=(
  cmake --build "${BUILD_DIR}" --target cutlass_launch_probe cutlass_realtime_worker -j2
)

{
  printf 'CUTLASS_ROOT=%s\n' "${CUTLASS_ROOT_VALUE}"
  printf 'CUTLASS_REVISION=%s\n' "${CUTLASS_REVISION}"
  printf 'CUDA_HOME=%s\n' "${CUDA_HOME_VALUE}"
  printf 'CUDACXX=%s\n' "${CUDA_COMPILER_VALUE}"
  printf 'NVCC_VERSION=%s\n' "${NVCC_VERSION}"
  printf 'CMAKE_CUDA_ARCHITECTURES=%s\n' "${ARCH}"
  printf 'CUTLASS_MODE=NATIVE_SM120\n'
  printf 'PROBE_BINARY=%s\n' "${BUILD_DIR}/cutlass_launch_probe"
  printf 'REALTIME_WORKER_BINARY=%s\n' "${BUILD_DIR}/cutlass_realtime_worker"
} > "${BUILD_DIR}/build_info.env"

printf '%q ' "${CONFIGURE_CMD[@]}" > "${BUILD_DIR}/configure_command.txt"
printf '\n' >> "${BUILD_DIR}/configure_command.txt"
printf '%q ' "${BUILD_CMD[@]}" > "${BUILD_DIR}/build_command.txt"
printf '\n' >> "${BUILD_DIR}/build_command.txt"

"${CONFIGURE_CMD[@]}" > "${BUILD_DIR}/configure_stdout.log" 2> "${BUILD_DIR}/configure_stderr.log"
"${BUILD_CMD[@]}" > "${BUILD_DIR}/build_stdout.log" 2> "${BUILD_DIR}/build_stderr.log"

printf 'build_pass=1\n' > "${BUILD_DIR}/status.env"
printf '%s\n' "${BUILD_DIR}/cutlass_launch_probe"
printf '%s\n' "${BUILD_DIR}/cutlass_realtime_worker"
