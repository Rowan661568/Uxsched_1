#!/usr/bin/env bash
set -u

ROOT="/home/zm/project/UXSched"
OUT_DIR=""
UXSCHED_BUILD="${ROOT}/build-hb-cu128"
CUTLASS_BUILD="${ROOT}/build-cutlass-cu128"
CUTLASS_ROOT_VALUE="${CUTLASS_ROOT:-/home/zm/project/cutlass}"
CUDA_HOME_VALUE="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_LIB="${XSCHED_CUDA_LIB:-/usr/lib/wsl/lib/libcuda.so.1}"
M=2048
N=2048
K=2048
WARMUP=5
HP_REQUESTS=20
HP_PERIOD_US=30000
LP_DURATION_MS=1500
SPLIT_BLOCKS=52
REPEAT=1
SYSTEMS="standalone_hp,uxsched_native_hp_lp,uxsched_hb_fixed_hp_lp"
VERIFIED_KERNEL_FILE="${ROOT}/benchmarks/cutlass/verified_kernel_sm120_fp32_simt.txt"
BARRIER_TIMEOUT_MS=180000
COOLDOWN_SEC=5
CPU_AFFINITY=""
CPU_AFFINITY_EFFECTIVE=""
PRE_RUN_IDLE_SEC=0
ENABLE_GPU_TELEMETRY=0
TELEMETRY_INTERVAL_SEC=1
PIDS=()
XSERVER_PID=""
TELEMETRY_PID=""

usage() {
  printf 'Usage: %s --output-dir DIR [--uxsched-build DIR] [--cutlass-build DIR] [--m N] [--n N] [--k N] [--warmup N] [--hp-requests N] [--hp-period-us US] [--lp-duration-ms MS] [--split-blocks N] [--verified-kernel-file PATH] [--repeat N] [--systems LIST] [--cooldown-sec N] [--cpu-affinity LIST] [--pre-run-idle-sec N] [--enable-gpu-telemetry] [--telemetry-interval-sec N]\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --uxsched-build) UXSCHED_BUILD="$2"; shift 2 ;;
    --cutlass-build) CUTLASS_BUILD="$2"; shift 2 ;;
    --cutlass-root) CUTLASS_ROOT_VALUE="$2"; shift 2 ;;
    --cuda-home) CUDA_HOME_VALUE="$2"; shift 2 ;;
    --cuda-lib) CUDA_LIB="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --hp-requests) HP_REQUESTS="$2"; shift 2 ;;
    --hp-period-us) HP_PERIOD_US="$2"; shift 2 ;;
    --lp-duration-ms) LP_DURATION_MS="$2"; shift 2 ;;
    --split-blocks) SPLIT_BLOCKS="$2"; shift 2 ;;
    --verified-kernel-file) VERIFIED_KERNEL_FILE="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --systems) SYSTEMS="$2"; shift 2 ;;
    --cooldown-sec) COOLDOWN_SEC="$2"; shift 2 ;;
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

WORKER="${CUTLASS_BUILD}/cutlass_realtime_worker"
HB_SHIM="${UXSCHED_BUILD}/platforms/cuda/libshimcuda.so"
XSERVER="${UXSCHED_BUILD}/service/xserver"
HB_LIB_PATH="${UXSCHED_BUILD}/platforms/cuda:${UXSCHED_BUILD}/preempt:/usr/lib/wsl/lib"

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

cleanup() {
  local pid
  stop_gpu_telemetry
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  if [[ -n "${XSERVER_PID}" ]] && kill -0 "${XSERVER_PID}" 2>/dev/null; then
    kill "${XSERVER_PID}" 2>/dev/null || true
    wait "${XSERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

apply_runner_affinity() {
  if [[ -z "${CPU_AFFINITY}" ]]; then
    CPU_AFFINITY_EFFECTIVE="$(taskset -pc $$ 2>/dev/null | sed 's/.*: //' || true)"
    return 0
  fi
  {
    printf 'cpu_affinity_requested=%s\n' "${CPU_AFFINITY}"
    taskset -pc "${CPU_AFFINITY}" $$ 2>&1
  } > "${OUT_DIR}/runner_affinity.txt"
  CPU_AFFINITY_EFFECTIVE="$(taskset -pc $$ 2>/dev/null | sed 's/.*: //' || true)"
  printf 'cpu_affinity_effective=%s\n' "${CPU_AFFINITY_EFFECTIVE}" >> "${OUT_DIR}/runner_affinity.txt"
  if [[ -z "${CPU_AFFINITY_EFFECTIVE}" ]]; then
    printf 'warning=CPU_AFFINITY_QUERY_FAILED\n' >> "${OUT_DIR}/runner_affinity.txt"
  fi
}

write_environment_snapshot() {
  local file="${OUT_DIR}/environment_snapshot.txt"
  {
    printf '[uname]\n'
    uname -a 2>&1 || true
    printf '\n[/etc/os-release]\n'
    sed -n '1,40p' /etc/os-release 2>&1 || true
    printf '\n[/proc/version]\n'
    sed -n '1p' /proc/version 2>&1 || true
    printf '\n[WSL]\n'
    if [[ -f /proc/sys/kernel/osrelease ]]; then
      sed -n '1p' /proc/sys/kernel/osrelease 2>&1 || true
    fi
    if command -v cmd.exe >/dev/null 2>&1; then
      cmd.exe /c ver 2>&1 || true
    fi
    printf '\n[nvidia-smi]\n'
    nvidia-smi 2>&1 || true
    printf '\n[nvidia-smi-query-initial]\n'
    nvidia-smi --query-gpu=name,driver_version,pstate,clocks.current.sm,clocks.current.graphics,temperature.gpu,power.draw,utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>&1 || printf 'N/A\n'
    printf '\n[nvcc]\n'
    if [[ -x "${CUDA_HOME_VALUE}/bin/nvcc" ]]; then
      "${CUDA_HOME_VALUE}/bin/nvcc" --version 2>&1 || true
    else
      printf 'N/A\n'
    fi
    printf '\n[cpuinfo]\n'
    grep -m1 'model name' /proc/cpuinfo 2>&1 || true
    printf '\n[lscpu]\n'
    lscpu 2>&1 || true
    printf '\n[loadavg]\n'
    cat /proc/loadavg 2>&1 || true
    printf '\n[cpu-governor]\n'
    if compgen -G '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor' >/dev/null; then
      for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        printf '%s=' "${governor}"
        cat "${governor}" 2>&1 || true
      done
    else
      printf 'N/A\n'
    fi
  } > "${file}"
}

start_gpu_telemetry() {
  local dir="$1"
  TELEMETRY_PID=""
  if [[ "${ENABLE_GPU_TELEMETRY}" -ne 1 ]]; then
    return 0
  fi
  local file="${dir}/gpu_telemetry.csv"
  local query='timestamp,pstate,clocks.current.sm,clocks.current.graphics,temperature.gpu,power.draw,utilization.gpu,utilization.memory'
  printf 'host_timestamp_unix_ns,%s\n' "${query}" > "${file}"
  (
    while true; do
      local now
      now="$(date +%s%N)"
      local row
      row="$(nvidia-smi --query-gpu="${query}" --format=csv,noheader,nounits 2>/dev/null || printf 'N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A')"
      printf '%s,%s\n' "${now}" "${row}" >> "${file}"
      sleep "${TELEMETRY_INTERVAL_SEC}"
    done
  ) &
  TELEMETRY_PID=$!
  if [[ -n "${CPU_AFFINITY}" ]]; then
    taskset -pc "${CPU_AFFINITY}" "${TELEMETRY_PID}" > "${dir}/gpu_telemetry_affinity.txt" 2>&1 || {
      printf 'warning=TELEMETRY_AFFINITY_FAILED\n' >> "${dir}/gpu_telemetry_affinity.txt"
    }
  fi
}

stop_gpu_telemetry() {
  if [[ -n "${TELEMETRY_PID}" ]] && kill -0 "${TELEMETRY_PID}" 2>/dev/null; then
    kill "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
  fi
  TELEMETRY_PID=""
}

write_metadata() {
  local file="${OUT_DIR}/metadata.env"
  local cutlass_rev
  cutlass_rev="$(git -C "${CUTLASS_ROOT_VALUE}" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
  {
    printf 'root=%s\n' "${ROOT}"
    printf 'git_head=%s\n' "$(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
    printf 'cutlass_root=%s\n' "${CUTLASS_ROOT_VALUE}"
    printf 'cutlass_revision=%s\n' "${cutlass_rev}"
    printf 'cuda_home=%s\n' "${CUDA_HOME_VALUE}"
    printf 'uxsched_build=%s\n' "${UXSCHED_BUILD}"
    printf 'cutlass_build=%s\n' "${CUTLASS_BUILD}"
    printf 'worker=%s\n' "${WORKER}"
    printf 'hb_shim=%s\n' "${HB_SHIM}"
    printf 'xserver=%s\n' "${XSERVER}"
    printf 'm=%s\nn=%s\nk=%s\n' "${M}" "${N}" "${K}"
    printf 'warmup=%s\nhp_requests=%s\nhp_period_us=%s\n' "${WARMUP}" "${HP_REQUESTS}" "${HP_PERIOD_US}"
    printf 'lp_duration_ms=%s\nsplit_blocks=%s\nrepeat=%s\n' "${LP_DURATION_MS}" "${SPLIT_BLOCKS}" "${REPEAT}"
    printf 'cooldown_sec=%s\n' "${COOLDOWN_SEC}"
    printf 'pre_run_idle_sec=%s\n' "${PRE_RUN_IDLE_SEC}"
    printf 'cpu_affinity_requested=%s\n' "${CPU_AFFINITY}"
    printf 'cpu_affinity_effective=%s\n' "${CPU_AFFINITY_EFFECTIVE}"
    printf 'enable_gpu_telemetry=%s\n' "${ENABLE_GPU_TELEMETRY}"
    printf 'telemetry_interval_sec=%s\n' "${TELEMETRY_INTERVAL_SEC}"
    printf 'systems=%s\n' "${SYSTEMS}"
    printf 'verified_kernel_file=%s\n' "${VERIFIED_KERNEL_FILE}"
  } > "${file}"
  if [[ -x "${CUDA_HOME_VALUE}/bin/nvcc" ]]; then
    "${CUDA_HOME_VALUE}/bin/nvcc" --version > "${OUT_DIR}/nvcc_version.txt" 2>&1 || true
  fi
  nvidia-smi > "${OUT_DIR}/nvidia_smi.txt" 2>&1 || true
  write_environment_snapshot
}

read_verified_kernel() {
  if [[ ! -f "${VERIFIED_KERNEL_FILE}" ]]; then
    return 1
  fi
  local kernel
  kernel="$(grep -v '^[[:space:]]*#' "${VERIFIED_KERNEL_FILE}" | grep -v '^[[:space:]]*$' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | head -n 1)"
  if [[ -z "${kernel}" || "${kernel}" == "*" ]]; then
    return 1
  fi
  printf '%s\n' "${kernel}"
}

write_worker_env() {
  local file="$1"
  local role="$2"
  local strategy="$3"
  local priority="$4"
  local system="$5"
  {
    printf 'role=%s\n' "${role}"
    printf 'system=%s\n' "${system}"
    printf 'LD_PRELOAD=%s\n' "${HB_SHIM}"
    printf 'LD_LIBRARY_PATH=%s\n' "${HB_LIB_PATH}"
    printf 'XSCHED_CUDA_LIB=%s\n' "${CUDA_LIB}"
    printf 'CUXTRA_CUDA_LIB=%s\n' "${CUDA_LIB}"
    printf 'XSCHED_SCHEDULER=GLB\n'
    printf 'XSCHED_AUTO_XQUEUE=ON\n'
    printf 'XSCHED_AUTO_XQUEUE_LEVEL=1\n'
    printf 'XSCHED_AUTO_XQUEUE_PRIORITY=%s\n' "${priority}"
    printf 'UXSCHED_CUDA_RUNTIME_STRATEGY=%s\n' "${strategy}"
    printf 'UXSCHED_HB_SPLIT_BLOCKS=%s\n' "${SPLIT_BLOCKS}"
  } > "${file}"
}

start_xserver() {
  local dir="$1"
  XSERVER_PID=""
  mkdir -p "${dir}/xserver"
  quote_command env -u LD_PRELOAD -u XSCHED_POLICY "${XSERVER}" HPF 50000 > "${dir}/xserver/command.txt"
  if [[ ! -x "${XSERVER}" ]]; then
    printf 'status=BLOCKED\nreason=XSERVER_UNAVAILABLE\n' > "${dir}/xserver/status.env"
    return 1
  fi
  env -u LD_PRELOAD -u XSCHED_POLICY "${XSERVER}" HPF 50000 > "${dir}/xserver/xserver.log" 2>&1 &
  XSERVER_PID=$!
  sleep 1
  if ! kill -0 "${XSERVER_PID}" 2>/dev/null; then
    printf 'status=FAILED\nreason=XSERVER_START_FAILED\n' > "${dir}/xserver/status.env"
    return 1
  fi
  printf 'status=RUNNING\npid=%s\n' "${XSERVER_PID}" > "${dir}/xserver/status.env"
  return 0
}

stop_xserver() {
  local dir="$1"
  if [[ -n "${XSERVER_PID}" ]] && kill -0 "${XSERVER_PID}" 2>/dev/null; then
    kill "${XSERVER_PID}" 2>/dev/null || true
    wait "${XSERVER_PID}" 2>/dev/null || true
    printf 'status=STOPPED\n' >> "${dir}/xserver/status.env"
  fi
  XSERVER_PID=""
}

wait_for_ready() {
  local barrier_dir="$1"
  local roles_csv="$2"
  local timeout_s=$((BARRIER_TIMEOUT_MS / 1000))
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    local all_ready=1
    IFS=',' read -r -a roles <<< "${roles_csv}"
    local role
    for role in "${roles[@]}"; do
      if [[ ! -f "${barrier_dir}/${role}.ready" ]]; then
        all_ready=0
      fi
    done
    if [[ "${all_ready}" -eq 1 ]]; then
      return 0
    fi
    if (( $(date +%s) - start_ts > timeout_s )); then
      return 1
    fi
    sleep 0.05
  done
}

start_worker() {
  local role="$1"
  local system="$2"
  local strategy="$3"
  local priority="$4"
  local barrier_dir="$5"
  local dir="$6"
  local verified_kernel="$7"
  local preload="$8"

  mkdir -p "${dir}"
  local requests_arg=("--requests" "${HP_REQUESTS}")
  local duration_arg=("--duration-ms" "0")
  if [[ "${role}" == "lp" ]]; then
    requests_arg=("--requests" "0")
    duration_arg=("--duration-ms" "${LP_DURATION_MS}")
  fi
  local cmd=(
    "${WORKER}"
    --role "${role}"
    --m "${M}" --n "${N}" --k "${K}"
    --warmup "${WARMUP}"
    "${requests_arg[@]}"
    "${duration_arg[@]}"
    --hp-period-us "${HP_PERIOD_US}"
    --stream explicit
    --correctness
    --barrier-dir "${barrier_dir}"
    --barrier-timeout-ms "${BARRIER_TIMEOUT_MS}"
    --output "${dir}/output.jsonl"
  )
  local launch_prefix=()
  if [[ -n "${CPU_AFFINITY}" ]]; then
    launch_prefix=(taskset -c "${CPU_AFFINITY}")
  fi
  quote_command "${launch_prefix[@]}" "${cmd[@]}" > "${dir}/command.txt"
  write_worker_env "${dir}/env.txt" "${role}" "${strategy}" "${priority}" "${system}"

  if [[ "${preload}" == "1" ]]; then
    if [[ "${strategy}" == "HB_FIXED" ]]; then
      "${launch_prefix[@]}" env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
        LD_PRELOAD="${HB_SHIM}" \
        LD_LIBRARY_PATH="${HB_LIB_PATH}" \
        XSCHED_CUDA_LIB="${CUDA_LIB}" \
        CUXTRA_CUDA_LIB="${CUDA_LIB}" \
        XSCHED_SCHEDULER=GLB \
        XSCHED_AUTO_XQUEUE=ON \
        XSCHED_AUTO_XQUEUE_LEVEL=1 \
        XSCHED_AUTO_XQUEUE_PRIORITY="${priority}" \
        UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
        UXSCHED_CUDART_TRACE=1 \
        UXSCHED_XQUEUE_TRACE=1 \
        UXSCHED_HB_STRICT=0 \
        UXSCHED_HB_SPLIT_BLOCKS="${SPLIT_BLOCKS}" \
        UXSCHED_HB_VERIFIED_KERNELS="${verified_kernel}" \
        "${cmd[@]}" > "${dir}/combined.log" 2>&1 &
    else
      "${launch_prefix[@]}" env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
        LD_PRELOAD="${HB_SHIM}" \
        LD_LIBRARY_PATH="${HB_LIB_PATH}" \
        XSCHED_CUDA_LIB="${CUDA_LIB}" \
        CUXTRA_CUDA_LIB="${CUDA_LIB}" \
        XSCHED_SCHEDULER=GLB \
        XSCHED_AUTO_XQUEUE=ON \
        XSCHED_AUTO_XQUEUE_LEVEL=1 \
        XSCHED_AUTO_XQUEUE_PRIORITY="${priority}" \
        UXSCHED_CUDA_RUNTIME_STRATEGY=NATIVE \
        UXSCHED_CUDART_TRACE=1 \
        UXSCHED_XQUEUE_TRACE=1 \
        "${cmd[@]}" > "${dir}/combined.log" 2>&1 &
    fi
  else
    "${launch_prefix[@]}" env -u LD_PRELOAD -u XSCHED_POLICY -u HB_TASK_PRIORITY "${cmd[@]}" > "${dir}/combined.log" 2>&1 &
  fi
  local pid=$!
  PIDS+=("${pid}")
  printf '%s\n' "${pid}" > "${dir}/pid.txt"
  {
    printf 'cpu_affinity_requested=%s\n' "${CPU_AFFINITY}"
    taskset -pc "${pid}" 2>&1 || true
  } > "${dir}/cpu_affinity.txt"
  printf 'combined_into_stdout_log=1\n' > "${dir}/stderr.log"
  ln -sf combined.log "${dir}/stdout.log"
}

count_before_marker() {
  local file="$1"
  local pattern="$2"
  awk -v pat="${pattern}" '
    /UXSCHED_CUTLASS_PHASE=MEASUREMENT_START/ { seen=1 }
    !seen && $0 ~ pat { count++ }
    END { print count + 0 }
  ' "${file}" 2>/dev/null
}

count_after_marker() {
  local file="$1"
  local pattern="$2"
  awk -v pat="${pattern}" '
    /UXSCHED_CUTLASS_PHASE=MEASUREMENT_START/ { seen=1; next }
    seen && $0 ~ pat { count++ }
    END { print count + 0 }
  ' "${file}" 2>/dev/null
}

extract_backend_stats() {
  local repeat_dir="$1"
  local lp_log="${repeat_dir}/lp/combined.log"
  local hp_log="${repeat_dir}/hp/combined.log"
  local out="${repeat_dir}/uxsched_backend_stats.env"
  if [[ ! -f "${lp_log}" ]]; then
    lp_log="/dev/null"
  fi
  if [[ ! -f "${hp_log}" ]]; then
    hp_log="/dev/null"
  fi
  local transform_before transform_after parent_delta child_delta transformed_delta fallback_delta no_xqueue_delta
  transform_before="$(count_before_marker "${lp_log}" 'transform_succeeded')"
  transform_after="$(count_after_marker "${lp_log}" 'transform_succeeded')"
  parent_delta="$(count_after_marker "${lp_log}" 'parent_launch_submitted')"
  child_delta="$(count_after_marker "${lp_log}" 'child_launch_submitted')"
  transformed_delta="${child_delta}"
  fallback_delta="$(count_after_marker "${lp_log}" 'backend_selected=NATIVE|runtime_launch_fallback|fallback')"
  no_xqueue_delta="$(count_after_marker "${lp_log}" 'NO_XQUEUE')"
  {
    printf 'runtime_launch_intercepted_count=%s\n' "$(grep -h 'runtime_launch_intercepted' "${lp_log}" "${hp_log}" 2>/dev/null | wc -l)"
    printf 'runtime_sync_intercepted_count=%s\n' "$(grep -h 'runtime_.*sync' "${lp_log}" "${hp_log}" 2>/dev/null | wc -l)"
    printf 'runtime_hb_metadata_bridge_pass=%s\n' "$(grep -q 'runtime_hb_function_registered' "${lp_log}" 2>/dev/null && printf 1 || printf 0)"
    printf 'hb_transform_count_before_measurement=%s\n' "${transform_before}"
    printf 'hb_transform_count_after_measurement=%s\n' "$((transform_before + transform_after))"
    printf 'hb_transform_count_delta=%s\n' "${transform_after}"
    printf 'hb_parent_launch_count_delta=%s\n' "${parent_delta}"
    printf 'hb_child_launch_count_delta=%s\n' "${child_delta}"
    printf 'hb_transformed_launch_count_delta=%s\n' "${transformed_delta}"
    printf 'hb_fallback_count_delta=%s\n' "${fallback_delta}"
    printf 'hb_no_xqueue_count_delta=%s\n' "${no_xqueue_delta}"
    printf 'hp_hb_transform_count=%s\n' "$(grep -h 'transform_succeeded' "${hp_log}" 2>/dev/null | wc -l)"
    printf 'global_scheduler_log_pass=%s\n' "$(grep -q 'scheduler created with policy HPF' "${repeat_dir}/xserver/xserver.log" 2>/dev/null && printf 1 || printf 0)"
    printf 'local_fallback_count=%s\n' "$(grep -h 'local scheduler' "${lp_log}" "${hp_log}" 2>/dev/null | wc -l)"
  } > "${out}"
}

finish_workers() {
  local repeat_dir="$1"
  local status=0
  local pid
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  PIDS=()
  printf 'worker_return_status=%s\n' "${status}" > "${repeat_dir}/status.env"
  return "${status}"
}

run_case() {
  local system="$1"
  local repeat_idx="$2"
  local repeat_dir="${OUT_DIR}/${system}/repeat_${repeat_idx}"
  local barrier_dir="${repeat_dir}/barrier"
  local verified_kernel=""
  local rc=0
  mkdir -p "${barrier_dir}"
  if [[ "${PRE_RUN_IDLE_SEC}" != "0" ]]; then
    printf 'pre_run_idle_sec=%s\n' "${PRE_RUN_IDLE_SEC}" > "${repeat_dir}/pre_run_idle.env"
    sleep "${PRE_RUN_IDLE_SEC}"
  fi
  start_gpu_telemetry "${repeat_dir}"

  if [[ "${system}" == "uxsched_hb_fixed_hp_lp" ]]; then
    verified_kernel="$(read_verified_kernel || true)"
    if [[ -z "${verified_kernel}" ]]; then
      printf 'status=BLOCKED\nreason=VERIFIED_KERNEL_UNAVAILABLE\n' > "${repeat_dir}/status.env"
      stop_gpu_telemetry
      return 1
    fi
    printf '%s\n' "${verified_kernel}" > "${repeat_dir}/verified_kernel.txt"
  fi

  if [[ "${system}" == "standalone_hp" ]]; then
    start_worker hp "${system}" NATIVE 10 "${barrier_dir}" "${repeat_dir}/hp" "" 0
    if ! wait_for_ready "${barrier_dir}" "hp"; then
      printf 'status=FAILED\nreason=READY_TIMEOUT\n' > "${repeat_dir}/status.env"
      stop_gpu_telemetry
      return 1
    fi
    date +%s%N > "${barrier_dir}/start"
    finish_workers "${repeat_dir}" || rc=1
  else
    if ! start_xserver "${repeat_dir}"; then
      printf 'status=FAILED\nreason=XSERVER_FAILED\n' > "${repeat_dir}/status.env"
      stop_gpu_telemetry
      return 1
    fi
    start_worker lp "${system}" "$([[ "${system}" == "uxsched_hb_fixed_hp_lp" ]] && printf HB_FIXED || printf NATIVE)" -10 "${barrier_dir}" "${repeat_dir}/lp" "${verified_kernel}" 1
    start_worker hp "${system}" NATIVE 10 "${barrier_dir}" "${repeat_dir}/hp" "" 1
    if ! wait_for_ready "${barrier_dir}" "hp,lp"; then
      printf 'status=FAILED\nreason=READY_TIMEOUT\n' > "${repeat_dir}/status.env"
      stop_gpu_telemetry
      cleanup
      return 1
    fi
    date +%s%N > "${barrier_dir}/start"
    finish_workers "${repeat_dir}" || rc=1
    stop_xserver "${repeat_dir}"
  fi

  stop_gpu_telemetry
  extract_backend_stats "${repeat_dir}"
  if [[ "${rc}" -eq 0 ]]; then
    printf 'status=COMPLETE\n' >> "${repeat_dir}/status.env"
  else
    printf 'status=FAILED\n' >> "${repeat_dir}/status.env"
  fi
  return "${rc}"
}

apply_runner_affinity
write_metadata

if [[ ! -x "${WORKER}" ]]; then
  printf 'status=BLOCKED\nreason=WORKER_UNAVAILABLE\nworker=%s\n' "${WORKER}" > "${OUT_DIR}/status.env"
  exit 1
fi
if [[ ! -x "${HB_SHIM}" ]]; then
  printf 'status=BLOCKED\nreason=HB_SHIM_UNAVAILABLE\nhb_shim=%s\n' "${HB_SHIM}" > "${OUT_DIR}/status.env"
  exit 1
fi

run_status=0
IFS=',' read -r -a system_list <<< "${SYSTEMS}"
system_count="${#system_list[@]}"
for ((r = 0; r < REPEAT; ++r)); do
  order=()
  for ((i = 0; i < system_count; ++i)); do
    idx=$(( (system_count - (r % system_count) + i) % system_count ))
    order+=("${system_list[$idx]}")
  done
  printf 'repeat_%s_case_order=%s\n' "${r}" "$(IFS=,; printf '%s' "${order[*]}")" >> "${OUT_DIR}/metadata.env"
  for ((i = 0; i < system_count; ++i)); do
    system="${order[$i]}"
    if ! run_case "${system}" "${r}"; then
      run_status=1
    fi
    if (( i + 1 < system_count )) && [[ "${COOLDOWN_SEC}" != "0" ]]; then
      sleep "${COOLDOWN_SEC}"
    fi
  done
done

python3 "${ROOT}/tools/summarize_cutlass_realtime_compare.py" --result-dir "${OUT_DIR}" \
  > "${OUT_DIR}/summarize_stdout.log" 2> "${OUT_DIR}/summarize_stderr.log" || run_status=1

printf 'status=%s\n' "$([[ "${run_status}" -eq 0 ]] && printf COMPLETE || printf FAILED)" > "${OUT_DIR}/status.env"
exit "${run_status}"
