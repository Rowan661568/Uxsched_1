#!/usr/bin/env bash
set -u

ROOT="/home/zm/project/UXSched"
OUT_DIR=""
BUILD_DIR="${ROOT}/build-cutlass-cu128"
UXSCHED_BUILD="${ROOT}/build-hb-cu128"
CUTLASS_ROOT_VALUE="${CUTLASS_ROOT:-/home/zm/project/cutlass}"
CUDA_HOME_VALUE="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_COMPILER_VALUE="${CUDACXX:-/usr/local/cuda-12.8/bin/nvcc}"
CUDA_LIB="${XSCHED_CUDA_LIB:-/usr/lib/wsl/lib/libcuda.so.1}"
M=2048
N=2048
K=2048
ITERATIONS=1
WARMUP=0
STREAM_MODE="explicit"
SPLIT_BLOCKS=64
VERIFIED_KERNELS="${UXSCHED_HB_VERIFIED_KERNELS:-*}"
BUILD_PROBE=0
XSERVER_PID=""

usage() {
  printf 'Usage: %s --output-dir DIR [--build-probe] [--build-dir DIR] [--uxsched-build DIR] [--m N] [--n N] [--k N] [--iterations N] [--warmup N] [--stream default|explicit] [--split-blocks N] [--verified-kernels LIST]\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --build-probe)
      BUILD_PROBE=1
      shift
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --uxsched-build)
      UXSCHED_BUILD="$2"
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
    --cuda-lib)
      CUDA_LIB="$2"
      shift 2
      ;;
    --m)
      M="$2"
      shift 2
      ;;
    --n)
      N="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --stream)
      STREAM_MODE="$2"
      shift 2
      ;;
    --split-blocks)
      SPLIT_BLOCKS="$2"
      shift 2
      ;;
    --verified-kernels)
      VERIFIED_KERNELS="$2"
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

if [[ -z "${OUT_DIR}" ]]; then
  printf 'missing required --output-dir\n' >&2
  usage >&2
  exit 2
fi

PROBE="${BUILD_DIR}/cutlass_launch_probe"
HB_SHIM="${UXSCHED_BUILD}/platforms/cuda/libshimcuda.so"
XSERVER="${UXSCHED_BUILD}/service/xserver"
HB_LIB_PATH="${UXSCHED_BUILD}/platforms/cuda:${UXSCHED_BUILD}/preempt:/usr/lib/wsl/lib"

mkdir -p "${OUT_DIR}"

quote_command() {
  local first=1
  for arg in "$@"; do
    if [[ "${first}" -eq 0 ]]; then
      printf ' '
    fi
    printf '%q' "$arg"
    first=0
  done
  printf '\n'
}

write_env_file() {
  local file="$1"
  local strategy="$2"
  {
    printf 'ROOT=%s\n' "${ROOT}"
    printf 'CUTLASS_ROOT=%s\n' "${CUTLASS_ROOT_VALUE}"
    printf 'CUDA_HOME=%s\n' "${CUDA_HOME_VALUE}"
    printf 'CUDACXX=%s\n' "${CUDA_COMPILER_VALUE}"
    printf 'XSCHED_CUDA_LIB=%s\n' "${CUDA_LIB}"
    printf 'CUXTRA_CUDA_LIB=%s\n' "${CUDA_LIB}"
    printf 'PROBE=%s\n' "${PROBE}"
    printf 'HB_SHIM=%s\n' "${HB_SHIM}"
    printf 'XSERVER=%s\n' "${XSERVER}"
    printf 'UXSCHED_BUILD=%s\n' "${UXSCHED_BUILD}"
    printf 'LD_LIBRARY_PATH_CASE=%s\n' "${HB_LIB_PATH}"
    printf 'CMAKE_CUDA_ARCHITECTURES=120-real;120-virtual\n'
    printf 'CUTLASS_MODE=NATIVE_SM120\n'
    printf 'M=%s\nN=%s\nK=%s\n' "${M}" "${N}" "${K}"
    printf 'ITERATIONS=%s\nWARMUP=%s\nSTREAM_MODE=%s\n' "${ITERATIONS}" "${WARMUP}" "${STREAM_MODE}"
    printf 'UXSCHED_CUDA_RUNTIME_STRATEGY=%s\n' "${strategy}"
    printf 'UXSCHED_HB_SPLIT_BLOCKS=%s\n' "${SPLIT_BLOCKS}"
    printf 'UXSCHED_HB_VERIFIED_KERNELS=%s\n' "${VERIFIED_KERNELS}"
  } > "${file}"
}

build_probe_if_requested() {
  if [[ "${BUILD_PROBE}" -eq 0 ]]; then
    return 0
  fi
  local dir="${OUT_DIR}/build_cutlass_launch_probe"
  mkdir -p "${dir}"
  quote_command "${ROOT}/tools/build_cutlass_launch_probe.sh" \
    --build-dir "${BUILD_DIR}" \
    --cutlass-root "${CUTLASS_ROOT_VALUE}" \
    --cuda-home "${CUDA_HOME_VALUE}" \
    --cuda-compiler "${CUDA_COMPILER_VALUE}" \
    > "${dir}/command.txt"
  if "${ROOT}/tools/build_cutlass_launch_probe.sh" \
      --build-dir "${BUILD_DIR}" \
      --cutlass-root "${CUTLASS_ROOT_VALUE}" \
      --cuda-home "${CUDA_HOME_VALUE}" \
      --cuda-compiler "${CUDA_COMPILER_VALUE}" \
      > "${dir}/stdout.log" 2> "${dir}/stderr.log"; then
    printf 'return_code=0\nstatus=BUILT\n' > "${dir}/status.env"
  else
    local rc=$?
    printf 'return_code=%s\nstatus=FAILED\n' "${rc}" > "${dir}/status.env"
    return "${rc}"
  fi
}

audit_probe_binary() {
  local dir="${OUT_DIR}/probe_binary"
  mkdir -p "${dir}"
  if [[ ! -x "${PROBE}" ]]; then
    printf 'status=BLOCKED\nreason=PROBE_UNAVAILABLE\n' > "${dir}/status.env"
    return 0
  fi

  ldd "${PROBE}" > "${dir}/ldd.txt" 2> "${dir}/ldd.stderr" || true
  readelf -Ws "${PROBE}" > "${dir}/dynamic_symbols.txt" 2> "${dir}/dynamic_symbols.stderr" || true
  nm -D "${PROBE}" >> "${dir}/dynamic_symbols.txt" 2>> "${dir}/dynamic_symbols.stderr" || true
  if [[ -x "${CUDA_HOME_VALUE}/bin/cuobjdump" ]]; then
    "${CUDA_HOME_VALUE}/bin/cuobjdump" --list-elf "${PROBE}" \
      > "${dir}/cuobjdump_elf.txt" 2> "${dir}/cuobjdump_elf.stderr" || true
    "${CUDA_HOME_VALUE}/bin/cuobjdump" --dump-ptx "${PROBE}" \
      > "${dir}/cuobjdump_ptx.txt" 2> "${dir}/cuobjdump_ptx.stderr" || true
  else
    printf 'CUOBJDUMP_UNAVAILABLE: %s/bin/cuobjdump\n' "${CUDA_HOME_VALUE}" \
      > "${dir}/cuobjdump_ptx.txt"
    : > "${dir}/cuobjdump_elf.txt"
  fi

  local shared_cudart=0
  local ptx_available=0
  local sass_available=0
  grep -q 'libcudart.so' "${dir}/ldd.txt" 2>/dev/null && shared_cudart=1
  grep -q 'Fatbin ptx code' "${dir}/cuobjdump_ptx.txt" 2>/dev/null && ptx_available=1
  grep -q 'sm_120' "${dir}/cuobjdump_elf.txt" 2>/dev/null && sass_available=1
  {
    printf 'status=COMPLETE\n'
    printf 'shared_cudart=%s\n' "${shared_cudart}"
    printf 'sm120_sass_available=%s\n' "${sass_available}"
    printf 'compute120_ptx_available=%s\n' "${ptx_available}"
  } > "${dir}/status.env"
}

start_xserver() {
  local dir="${OUT_DIR}/xserver"
  mkdir -p "${dir}"
  write_env_file "${dir}/env.txt" "NATIVE"
  quote_command env -u LD_PRELOAD -u XSCHED_POLICY "${XSERVER}" HPF 50000 > "${dir}/command.txt"
  if [[ ! -x "${XSERVER}" ]]; then
    printf 'status=BLOCKED\nreason=XSERVER_UNAVAILABLE\n' > "${dir}/status.env"
    return 0
  fi
  env -u LD_PRELOAD -u XSCHED_POLICY "${XSERVER}" HPF 50000 \
    > "${dir}/stdout.log" 2> "${dir}/stderr.log" &
  XSERVER_PID=$!
  printf '%s\n' "${XSERVER_PID}" > "${dir}/pid.txt"
  sleep 1
  if kill -0 "${XSERVER_PID}" 2>/dev/null; then
    printf 'status=RUNNING\n' > "${dir}/status.env"
  else
    printf 'status=FAILED_TO_START\n' > "${dir}/status.env"
  fi
}

stop_xserver() {
  if [[ -n "${XSERVER_PID}" ]] && kill -0 "${XSERVER_PID}" 2>/dev/null; then
    kill "${XSERVER_PID}" 2>/dev/null || true
    wait "${XSERVER_PID}" 2>/dev/null || true
    printf 'status=STOPPED\n' >> "${OUT_DIR}/xserver/status.env"
  fi
}

trap stop_xserver EXIT

extract_case_evidence() {
  local dir="$1"
  grep -hE '\[UXSCHED-(HB|XQUEUE|CUDART)\].*(split_count|backend_selected=HB_SPLIT|transform_succeeded|transformed_module_loaded|parent_launch_submitted|child_launch_submitted|child_launch_completed|split_group_completed|parent_launch_completed|HIGH_PRIORITY_PASSTHROUGH|backend_selected=NATIVE|NO_XQUEUE|KERNEL_NOT_VERIFIED|PTX_UNAVAILABLE|runtime_strategy|runtime_)' \
    "${dir}/stdout.log" "${dir}/stderr.log" > "${dir}/split_trace.log" || true
  if [[ ! -s "${dir}/split_trace.log" ]]; then
    printf 'NO_SPLIT_TRACE_OBSERVED\n' > "${dir}/split_trace.log"
  fi

  grep -hE '\[UXSCHED-HB\].*(child_launch_completed|split_group_completed)' \
    "${dir}/stdout.log" "${dir}/stderr.log" > "${dir}/child_completion.log" || true
  if [[ ! -s "${dir}/child_completion.log" ]]; then
    printf 'NO_CHILD_COMPLETION_OBSERVED\n' > "${dir}/child_completion.log"
  fi

  grep -hE '\[UXSCHED-HB\].*(parent_launch_completed|parent_launch_submitted)' \
    "${dir}/stdout.log" "${dir}/stderr.log" > "${dir}/parent_completion.log" || true
  if [[ ! -s "${dir}/parent_completion.log" ]]; then
    printf 'NO_PARENT_COMPLETION_OBSERVED\n' > "${dir}/parent_completion.log"
  fi

  count_logs() {
    local pattern="$1"
    grep -hE "${pattern}" "${dir}/stdout.log" "${dir}/stderr.log" 2>/dev/null | wc -l | tr -d ' '
  }

  grep -hE '\[UXSCHED-CUDART\].*(runtime_fatbin_registered|runtime_function_registered|runtime_launch_intercepted|runtime_launch_function_resolved|runtime_hb_module_registered|runtime_hb_function_registered|runtime_backend_selected|runtime_launch_fallback|runtime_sync_intercepted)' \
    "${dir}/stdout.log" "${dir}/stderr.log" > "${dir}/runtime_registration.log" || true
  if [[ ! -s "${dir}/runtime_registration.log" ]]; then
    printf 'NO_RUNTIME_REGISTRATION_TRACE_OBSERVED\n' > "${dir}/runtime_registration.log"
  fi

  {
    printf 'uxsched_hb_transform_count=%s\n' "$(count_logs '\[UXSCHED-HB\].*transform_succeeded')"
    printf 'uxsched_hb_parent_launch_count=%s\n' "$(count_logs '\[UXSCHED-HB\].*parent_launch_submitted')"
    printf 'uxsched_hb_child_launch_count=%s\n' "$(count_logs '\[UXSCHED-HB\].*child_launch_submitted')"
    printf 'uxsched_hb_transformed_launch_count=%s\n' "$(count_logs '\[UXSCHED-HB\].*child_launch_submitted.*transformed_function=')"
    printf 'uxsched_hb_fallback_count=%s\n' "$(count_logs '\[UXSCHED-HB\].*backend_selected=NATIVE reason=')"
    printf 'uxsched_hb_no_xqueue_count=%s\n' "$(count_logs '\[UXSCHED-HB\].*reason=NO_XQUEUE')"
    printf 'runtime_launch_intercepted_count=%s\n' "$(count_logs '\[UXSCHED-CUDART\].*runtime_launch_intercepted')"
    printf 'runtime_function_resolved_count=%s\n' "$(count_logs '\[UXSCHED-CUDART\].*runtime_launch_function_resolved')"
    printf 'runtime_launch_fallback_count=%s\n' "$(count_logs '\[UXSCHED-CUDART\].*runtime_launch_fallback')"
    printf 'runtime_sync_intercepted_count=%s\n' "$(count_logs '\[UXSCHED-CUDART\].*runtime_sync_intercepted')"
    printf 'runtime_hb_module_registered_count=%s\n' "$(count_logs '\[UXSCHED-CUDART\].*runtime_hb_module_registered.*result=OK')"
    printf 'runtime_hb_function_registered_count=%s\n' "$(count_logs '\[UXSCHED-CUDART\].*runtime_hb_function_registered.*result=OK')"
    printf 'runtime_hb_registration_failed_count=%s\n' "$(count_logs '\[UXSCHED-CUDART\].*runtime_hb_.*registered.*result=(RUNTIME_|[^O])')"
  } > "${dir}/uxsched_backend_stats.env"
}

json_bool_value() {
  local file="$1"
  local key="$2"
  grep -ho "\"${key}\":[^,}]*" "${file}" 2>/dev/null | tail -n 1 | cut -d ':' -f 2 | tr -d ' "' || true
}

env_value() {
  local file="$1"
  local key="$2"
  awk -F= -v key="${key}" '$1 == key { value=$2 } END { if (value == "") value=0; print value }' "${file}" 2>/dev/null
}

case_has_no_cuda_error() {
  local dir="$1"
  ! grep -qiE 'illegal memory access|invalid argument|segmentation fault|core dumped|CUDA error|cuda error' \
    "${dir}/stdout.log" "${dir}/stderr.log" 2>/dev/null
}

run_case() {
  local name="$1"
  local mode="$2"
  local uxsched="$3"
  local strategy="$4"
  local dir="${OUT_DIR}/${name}"
  local output_jsonl="${dir}/output.jsonl"
  mkdir -p "${dir}"
  : > "${output_jsonl}"
  write_env_file "${dir}/env.txt" "${strategy}"

  local cmd=("${PROBE}" --mode "${mode}" --m "${M}" --n "${N}" --k "${K}" \
    --iterations "${ITERATIONS}" --warmup "${WARMUP}" --stream "${STREAM_MODE}" \
    --correctness --output "${output_jsonl}")
  quote_command "${cmd[@]}" > "${dir}/command.txt"

  if [[ ! -x "${PROBE}" ]]; then
    printf 'return_code=2\nstatus=BLOCKED\nreason=PROBE_UNAVAILABLE\n' > "${dir}/status.env"
    extract_case_evidence "${dir}"
    return 0
  fi

  if [[ "${uxsched}" == "1" ]]; then
    if [[ ! -f "${HB_SHIM}" ]]; then
      printf 'return_code=2\nstatus=BLOCKED\nreason=HB_SHIM_UNAVAILABLE\n' > "${dir}/status.env"
      extract_case_evidence "${dir}"
      return 0
    fi
    (
      export CUDA_HOME="${CUDA_HOME_VALUE}"
      export PATH="${CUDA_HOME_VALUE}/bin:${PATH}"
      export LD_PRELOAD="${HB_SHIM}"
      export LD_LIBRARY_PATH="${HB_LIB_PATH}:${LD_LIBRARY_PATH:-}"
      export XSCHED_CUDA_LIB="${CUDA_LIB}"
      export CUXTRA_CUDA_LIB="${CUDA_LIB}"
      export XSCHED_SCHEDULER=GLB
      export XSCHED_AUTO_XQUEUE=ON
      export XSCHED_AUTO_XQUEUE_LEVEL=1
      export XSCHED_AUTO_XQUEUE_PRIORITY=-10
      export UXSCHED_CUDA_RUNTIME_STRATEGY="${strategy}"
      export UXSCHED_CUDART_TRACE=1
      export UXSCHED_XQUEUE_TRACE=1
      export UXSCHED_HB_STRICT=0
      export UXSCHED_HB_SPLIT_BLOCKS="${SPLIT_BLOCKS}"
      if [[ -n "${VERIFIED_KERNELS}" ]]; then
        export UXSCHED_HB_VERIFIED_KERNELS="${VERIFIED_KERNELS}"
      fi
      "${cmd[@]}"
    ) > "${dir}/stdout.log" 2> "${dir}/stderr.log"
  else
    (
      unset LD_PRELOAD
      unset XSCHED_POLICY
      unset HB_TASK_PRIORITY
      unset UXSCHED_CUDA_RUNTIME_STRATEGY
      unset UXSCHED_HB_VERIFIED_KERNELS
      export CUDA_HOME="${CUDA_HOME_VALUE}"
      export PATH="${CUDA_HOME_VALUE}/bin:${PATH}"
      "${cmd[@]}"
    ) > "${dir}/stdout.log" 2> "${dir}/stderr.log"
  fi

  local rc=$?
  local json_status
  local json_reason
  json_status="$(grep -ho '"status":"[^"]*"' "${output_jsonl}" "${dir}/stdout.log" 2>/dev/null | tail -n 1 | cut -d '"' -f 4 || true)"
  json_reason="$(grep -ho '"reason":"[^"]*"' "${output_jsonl}" "${dir}/stdout.log" 2>/dev/null | tail -n 1 | cut -d '"' -f 4 || true)"
  {
    printf 'return_code=%s\n' "${rc}"
    printf 'status=%s\n' "${json_status:-UNKNOWN}"
    printf 'reason=%s\n' "${json_reason:-}"
  } > "${dir}/status.env"
  extract_case_evidence "${dir}"

  if [[ "${mode}" == "runtime" && "${uxsched}" == "1" && "${strategy}" == "HB_FIXED" ]]; then
    local hb_transform hb_parent hb_child hb_transformed hb_fallback hb_noxq runtime_intercept
    hb_transform="$(env_value "${dir}/uxsched_backend_stats.env" uxsched_hb_transform_count)"
    hb_parent="$(env_value "${dir}/uxsched_backend_stats.env" uxsched_hb_parent_launch_count)"
    hb_child="$(env_value "${dir}/uxsched_backend_stats.env" uxsched_hb_child_launch_count)"
    hb_transformed="$(env_value "${dir}/uxsched_backend_stats.env" uxsched_hb_transformed_launch_count)"
    hb_fallback="$(env_value "${dir}/uxsched_backend_stats.env" uxsched_hb_fallback_count)"
    hb_noxq="$(env_value "${dir}/uxsched_backend_stats.env" uxsched_hb_no_xqueue_count)"
    runtime_intercept="$(env_value "${dir}/uxsched_backend_stats.env" runtime_launch_intercepted_count)"
    if [[ "${runtime_intercept}" == "0" && "${hb_transform}" == "0" && "${hb_parent}" == "0" &&
          "${hb_child}" == "0" && "${hb_transformed}" == "0" &&
          "${hb_fallback}" == "0" && "${hb_noxq}" == "0" ]]; then
      printf 'runtime_launch_not_intercepted=1\n' >> "${dir}/status.env"
    else
      printf 'runtime_launch_not_intercepted=0\n' >> "${dir}/status.env"
    fi
  fi
}

write_summary() {
  local summary="${OUT_DIR}/cutlass_probe_summary.env"
  local runtime_native_pass runtime_uxsched_native_pass runtime_hb_correctness
  local hb_transform hb_parent hb_child hb_transformed hb_fallback hb_noxq
  local runtime_intercept runtime_resolved runtime_fallback runtime_sync runtime_hb_module runtime_hb_function runtime_hb_reg_failed
  runtime_native_pass=0
  runtime_uxsched_native_pass=0
  runtime_hb_correctness=0

  [[ "$(json_bool_value "${OUT_DIR}/cutlass_native_runtime/output.jsonl" correctness_pass)" == "true" ]] &&
    case_has_no_cuda_error "${OUT_DIR}/cutlass_native_runtime" && runtime_native_pass=1
  [[ "$(json_bool_value "${OUT_DIR}/cutlass_uxsched_native_runtime/output.jsonl" correctness_pass)" == "true" ]] &&
    case_has_no_cuda_error "${OUT_DIR}/cutlass_uxsched_native_runtime" && runtime_uxsched_native_pass=1
  [[ "$(json_bool_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/output.jsonl" correctness_pass)" == "true" ]] &&
    case_has_no_cuda_error "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime" && runtime_hb_correctness=1

  hb_transform="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" uxsched_hb_transform_count)"
  hb_parent="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" uxsched_hb_parent_launch_count)"
  hb_child="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" uxsched_hb_child_launch_count)"
  hb_transformed="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" uxsched_hb_transformed_launch_count)"
  hb_fallback="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" uxsched_hb_fallback_count)"
  hb_noxq="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" uxsched_hb_no_xqueue_count)"
  runtime_intercept="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" runtime_launch_intercepted_count)"
  runtime_resolved="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" runtime_function_resolved_count)"
  runtime_fallback="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" runtime_launch_fallback_count)"
  runtime_sync="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" runtime_sync_intercepted_count)"
  runtime_hb_module="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" runtime_hb_module_registered_count)"
  runtime_hb_function="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" runtime_hb_function_registered_count)"
  runtime_hb_reg_failed="$(env_value "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/uxsched_backend_stats.env" runtime_hb_registration_failed_count)"

  local runtime_backend=0
  local runtime_not_intercepted=0
  if [[ "${hb_transform}" =~ ^[0-9]+$ && "${hb_parent}" =~ ^[0-9]+$ &&
        "${hb_child}" =~ ^[0-9]+$ && "${hb_transformed}" =~ ^[0-9]+$ &&
        "${hb_fallback}" =~ ^[0-9]+$ && "${hb_noxq}" =~ ^[0-9]+$ ]]; then
    if (( hb_transform > 0 && hb_parent > 0 && hb_child > 1 && hb_transformed > 0 &&
          hb_fallback == 0 && hb_noxq == 0 )); then
      runtime_backend=1
    fi
    if (( runtime_intercept == 0 && hb_transform == 0 && hb_parent == 0 && hb_child == 0 && hb_transformed == 0 &&
          hb_fallback == 0 && hb_noxq == 0 )); then
      runtime_not_intercepted=1
    fi
  fi

  local shared_cudart ptx_available sass_available
  shared_cudart="$(env_value "${OUT_DIR}/probe_binary/status.env" shared_cudart)"
  ptx_available="$(env_value "${OUT_DIR}/probe_binary/status.env" compute120_ptx_available)"
  sass_available="$(env_value "${OUT_DIR}/probe_binary/status.env" sm120_sass_available)"

  local discovered_kernel
  discovered_kernel="$(grep -hEo 'kernel_name=[^ ]+' "${OUT_DIR}/cutlass_uxsched_hb_fixed_runtime/runtime_registration.log" 2>/dev/null | tail -n 1 | cut -d= -f2- || true)"
  if [[ -z "${discovered_kernel}" ]]; then
    discovered_kernel="<unknown>"
  fi

  local probe_pass=0
  if [[ "${runtime_hb_correctness}" == "1" && "${runtime_backend}" == "1" ]]; then
    probe_pass=1
  fi

  local module_reg_pass=0
  local function_reg_pass=0
  local metadata_bridge_pass=0
  if [[ "${runtime_hb_module}" =~ ^[0-9]+$ && "${runtime_hb_module}" -gt 0 &&
        "${runtime_hb_reg_failed}" == "0" ]]; then
    module_reg_pass=1
  fi
  if [[ "${runtime_hb_function}" =~ ^[0-9]+$ && "${runtime_hb_function}" -gt 0 &&
        "${runtime_hb_reg_failed}" == "0" ]]; then
    function_reg_pass=1
  fi
  if [[ "${module_reg_pass}" == "1" && "${function_reg_pass}" == "1" ]]; then
    metadata_bridge_pass=1
  fi

  {
    printf 'cuda_toolkit_12_8=1\n'
    printf 'native_sm120_build_pass=%s\n' "$([[ -x "${PROBE}" ]] && printf 1 || printf 0)"
    printf 'cutlass_dependency_pass=%s\n' "$([[ -f "${CUTLASS_ROOT_VALUE}/include/cutlass/cutlass.h" ]] && printf 1 || printf 0)"
    printf 'shared_cudart_link_pass=%s\n' "${shared_cudart}"
    printf 'sm120_sass_available=%s\n' "${sass_available}"
    printf 'compute120_ptx_available=%s\n' "${ptx_available}"
    printf 'runtime_native_correctness_pass=%s\n' "${runtime_native_pass}"
    printf 'runtime_uxsched_native_correctness_pass=%s\n' "${runtime_uxsched_native_pass}"
    printf 'runtime_hb_fixed_backend_exercised=%s\n' "${runtime_backend}"
    printf 'runtime_launch_not_intercepted=%s\n' "${runtime_not_intercepted}"
    printf 'runtime_launch_intercepted_count=%s\n' "${runtime_intercept}"
    printf 'runtime_function_resolved_count=%s\n' "${runtime_resolved}"
    printf 'runtime_launch_fallback_count=%s\n' "${runtime_fallback}"
    printf 'runtime_sync_intercepted_count=%s\n' "${runtime_sync}"
    printf 'runtime_hb_module_registered_count=%s\n' "${runtime_hb_module}"
    printf 'runtime_hb_function_registered_count=%s\n' "${runtime_hb_function}"
    printf 'runtime_hb_registration_failed_count=%s\n' "${runtime_hb_reg_failed}"
    printf 'runtime_hb_module_registration_pass=%s\n' "${module_reg_pass}"
    printf 'runtime_hb_function_registration_pass=%s\n' "${function_reg_pass}"
    printf 'runtime_hb_metadata_bridge_pass=%s\n' "${metadata_bridge_pass}"
    printf 'runtime_hb_fixed_correctness_pass=%s\n' "${runtime_hb_correctness}"
    printf 'driver_mode_available=0\n'
    printf 'driver_native_correctness_pass=0\n'
    printf 'driver_uxsched_native_correctness_pass=0\n'
    printf 'driver_hb_fixed_backend_exercised=0\n'
    printf 'driver_hb_fixed_correctness_pass=0\n'
    printf 'hb_transform_count=%s\n' "${hb_transform}"
    printf 'hb_parent_launch_count=%s\n' "${hb_parent}"
    printf 'hb_child_launch_count=%s\n' "${hb_child}"
    printf 'hb_transformed_launch_count=%s\n' "${hb_transformed}"
    printf 'hb_fallback_count=%s\n' "${hb_fallback}"
    printf 'hb_no_xqueue_count=%s\n' "${hb_noxq}"
    printf 'discovered_cutlass_kernel_name=%s\n' "${discovered_kernel}"
    printf 'cutlass_probe_pass=%s\n' "${probe_pass}"
  } > "${summary}"
}

build_probe_if_requested || exit $?
audit_probe_binary

start_xserver
run_case cutlass_native_runtime runtime 0 NATIVE
run_case cutlass_uxsched_native_runtime runtime 1 NATIVE
run_case cutlass_uxsched_hb_fixed_runtime runtime 1 HB_FIXED
run_case cutlass_native_driver driver 0 NATIVE
run_case cutlass_uxsched_native_driver driver 1 NATIVE
run_case cutlass_uxsched_hb_fixed_driver driver 1 HB_FIXED
write_summary

printf 'summary=%s\n' "${OUT_DIR}/cutlass_probe_summary.env"
