#!/usr/bin/env bash
set -u

ROOT="/home/zm/project/UXSched"
OUT_DIR=""
UXSCHED_BUILD="${ROOT}/build-hb-cu128"
CUTLASS_BUILD="${ROOT}/build-cutlass-cu128"
CUTLASS_ROOT_VALUE="${CUTLASS_ROOT:-/home/zm/project/cutlass}"
CUDA_HOME_VALUE="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_LIB="${XSCHED_CUDA_LIB:-/usr/lib/wsl/lib/libcuda.so.1}"
SPLIT_SIZES="32,52,64,128"
M=2048
N=2048
K=2048
WARMUP=5
HP_REQUESTS=200
HP_PERIOD_US=30000
LP_DURATION_MS=8000
REPEAT=5
COOLDOWN_SEC=5
VERIFIED_KERNEL_FILE="${ROOT}/benchmarks/cutlass/verified_kernel_sm120_fp32_simt.txt"
CPU_AFFINITY=""
PRE_RUN_IDLE_SEC=0
ENABLE_GPU_TELEMETRY=0
TELEMETRY_INTERVAL_SEC=1

usage() {
  printf 'Usage: %s --output-dir DIR [--split-sizes CSV] [--repeat N] [--m N] [--n N] [--k N] [--warmup N] [--hp-requests N] [--hp-period-us US] [--lp-duration-ms MS] [--cooldown-sec N] [--uxsched-build DIR] [--cutlass-build DIR] [--verified-kernel-file PATH] [--cpu-affinity LIST] [--pre-run-idle-sec N] [--enable-gpu-telemetry] [--telemetry-interval-sec N]\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --split-sizes) SPLIT_SIZES="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --hp-requests) HP_REQUESTS="$2"; shift 2 ;;
    --hp-period-us) HP_PERIOD_US="$2"; shift 2 ;;
    --lp-duration-ms) LP_DURATION_MS="$2"; shift 2 ;;
    --cooldown-sec) COOLDOWN_SEC="$2"; shift 2 ;;
    --uxsched-build) UXSCHED_BUILD="$2"; shift 2 ;;
    --cutlass-build) CUTLASS_BUILD="$2"; shift 2 ;;
    --cutlass-root) CUTLASS_ROOT_VALUE="$2"; shift 2 ;;
    --cuda-home) CUDA_HOME_VALUE="$2"; shift 2 ;;
    --cuda-lib) CUDA_LIB="$2"; shift 2 ;;
    --verified-kernel-file) VERIFIED_KERNEL_FILE="$2"; shift 2 ;;
    --cpu-affinity) CPU_AFFINITY="$2"; shift 2 ;;
    --pre-run-idle-sec) PRE_RUN_IDLE_SEC="$2"; shift 2 ;;
    --enable-gpu-telemetry) ENABLE_GPU_TELEMETRY=1; shift ;;
    --telemetry-interval-sec) TELEMETRY_INTERVAL_SEC="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  printf 'missing required --output-dir\n' >&2
  usage >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

quote_command() {
  local first=1
  for arg in "$@"; do
    if [[ "${first}" -eq 0 ]]; then printf ' '; fi
    printf '%q' "$arg"
    first=0
  done
  printf '\n'
}

write_config() {
  local cutlass_rev
  cutlass_rev="$(git -C "${CUTLASS_ROOT_VALUE}" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
  {
    printf 'root=%s\n' "${ROOT}"
    printf 'git_head=%s\n' "$(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
    printf 'cutlass_root=%s\n' "${CUTLASS_ROOT_VALUE}"
    printf 'cutlass_revision=%s\n' "${cutlass_rev}"
    printf 'cuda_home=%s\n' "${CUDA_HOME_VALUE}"
    printf 'cuda_lib=%s\n' "${CUDA_LIB}"
    printf 'uxsched_build=%s\n' "${UXSCHED_BUILD}"
    printf 'cutlass_build=%s\n' "${CUTLASS_BUILD}"
    printf 'split_sizes=%s\n' "${SPLIT_SIZES}"
    printf 'm=%s\nn=%s\nk=%s\n' "${M}" "${N}" "${K}"
    printf 'warmup=%s\nhp_requests=%s\nhp_period_us=%s\n' "${WARMUP}" "${HP_REQUESTS}" "${HP_PERIOD_US}"
    printf 'lp_duration_ms=%s\nrepeat=%s\ncooldown_sec=%s\n' "${LP_DURATION_MS}" "${REPEAT}" "${COOLDOWN_SEC}"
    printf 'systems=uxsched_native_hp_lp,uxsched_hb_fixed_hp_lp\n'
    printf 'verified_kernel_file=%s\n' "${VERIFIED_KERNEL_FILE}"
    printf 'cpu_affinity=%s\npre_run_idle_sec=%s\n' "${CPU_AFFINITY}" "${PRE_RUN_IDLE_SEC}"
    printf 'enable_gpu_telemetry=%s\ntelemetry_interval_sec=%s\n' "${ENABLE_GPU_TELEMETRY}" "${TELEMETRY_INTERVAL_SEC}"
  } > "${OUT_DIR}/metadata.env"

  python3 - "$OUT_DIR/config.json" <<PY
import json, sys
cfg = {
  "split_sizes": "${SPLIT_SIZES}",
  "repeat": int("${REPEAT}"),
  "m": int("${M}"),
  "n": int("${N}"),
  "k": int("${K}"),
  "warmup": int("${WARMUP}"),
  "hp_requests": int("${HP_REQUESTS}"),
  "hp_period_us": int("${HP_PERIOD_US}"),
  "lp_duration_ms": int("${LP_DURATION_MS}"),
  "cooldown_sec": int("${COOLDOWN_SEC}"),
  "systems": ["uxsched_native_hp_lp", "uxsched_hb_fixed_hp_lp"],
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, sort_keys=True)
    f.write("\\n")
PY
}

build_compare_command() {
  local split="$1"
  local repeat="$2"
  local run_dir="$3"
  local cmd=(
    bash "${ROOT}/tools/run_cutlass_realtime_compare.sh"
    --output-dir "${run_dir}"
    --uxsched-build "${UXSCHED_BUILD}"
    --cutlass-build "${CUTLASS_BUILD}"
    --cutlass-root "${CUTLASS_ROOT_VALUE}"
    --cuda-home "${CUDA_HOME_VALUE}"
    --cuda-lib "${CUDA_LIB}"
    --m "${M}" --n "${N}" --k "${K}"
    --warmup "${WARMUP}"
    --hp-requests "${HP_REQUESTS}"
    --hp-period-us "${HP_PERIOD_US}"
    --lp-duration-ms "${LP_DURATION_MS}"
    --split-blocks "${split}"
    --repeat 1
    --systems uxsched_native_hp_lp,uxsched_hb_fixed_hp_lp
    --cooldown-sec "${COOLDOWN_SEC}"
    --verified-kernel-file "${VERIFIED_KERNEL_FILE}"
  )
  if [[ -n "${CPU_AFFINITY}" ]]; then
    cmd+=(--cpu-affinity "${CPU_AFFINITY}")
  fi
  if [[ "${PRE_RUN_IDLE_SEC}" != "0" ]]; then
    cmd+=(--pre-run-idle-sec "${PRE_RUN_IDLE_SEC}")
  fi
  if [[ "${ENABLE_GPU_TELEMETRY}" -eq 1 ]]; then
    cmd+=(--enable-gpu-telemetry --telemetry-interval-sec "${TELEMETRY_INTERVAL_SEC}")
  fi
  printf '%s\0' "${cmd[@]}"
}

write_config

IFS=',' read -r -a splits <<< "${SPLIT_SIZES}"
run_status=0
: > "${OUT_DIR}/commands.txt"

for ((r = 0; r < REPEAT; ++r)); do
  order=()
  split_count="${#splits[@]}"
  for ((i = 0; i < split_count; ++i)); do
    idx=$(( (split_count - (r % split_count) + i) % split_count ))
    order+=("${splits[$idx]}")
  done
  printf 'repeat_%s_split_order=%s\n' "${r}" "$(IFS=,; printf '%s' "${order[*]}")" >> "${OUT_DIR}/metadata.env"
  for split in "${order[@]}"; do
    run_dir="${OUT_DIR}/split_${split}/repeat_${r}"
    mkdir -p "${run_dir}"
    mapfile -d '' -t cmd < <(build_compare_command "${split}" "${r}" "${run_dir}")
    quote_command "${cmd[@]}" >> "${OUT_DIR}/commands.txt"
    printf 'split_blocks=%s\nrepeat=%s\n' "${split}" "${r}" > "${run_dir}/sweep_case.env"
    if ! "${cmd[@]}" > "${run_dir}/runner_stdout.log" 2> "${run_dir}/runner_stderr.log"; then
      run_status=1
      printf 'status=FAILED\n' > "${run_dir}/sweep_status.env"
    else
      printf 'status=COMPLETE\n' > "${run_dir}/sweep_status.env"
    fi
    if [[ "${COOLDOWN_SEC}" != "0" ]]; then
      sleep "${COOLDOWN_SEC}"
    fi
  done
done

python3 "${ROOT}/tools/aggregate_cutlass_split_size_sweep.py" --result-dir "${OUT_DIR}" \
  > "${OUT_DIR}/aggregate_stdout.log" 2> "${OUT_DIR}/aggregate_stderr.log" || run_status=1

python3 "${ROOT}/tools/plot_cutlass_split_size_sweep.py" --result-dir "${OUT_DIR}" --formats png,pdf,svg --dpi 300 \
  > "${OUT_DIR}/plot_stdout.log" 2> "${OUT_DIR}/plot_stderr.log" || run_status=1

printf 'status=%s\n' "$([[ "${run_status}" -eq 0 ]] && printf COMPLETE || printf FAILED)" > "${OUT_DIR}/status.env"
exit "${run_status}"
