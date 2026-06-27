#!/usr/bin/env bash
set -u

ROOT="/home/zm/project/UXSched"
HB_REPO="/home/zm/project/hummingbird"
HB_BUILD="/home/zm/project/hummingbird/build-lite"
CUDA_LIB="/usr/lib/wsl/lib/libcuda.so.1"
OUT_DIR=""
CORRECTNESS_SPLIT_BLOCKS=4096
UXSCHED_HB_SPLIT_BLOCKS=512

usage() {
  printf 'Usage: %s --output-dir PATH [--cuda-lib PATH]\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --cuda-lib)
      CUDA_LIB="$2"
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

BENCH="${HB_BUILD}/benchmarks/hb_open_resnet_like_eval"
RUNTIME_BENCH="${HB_BUILD}/benchmarks/hb_open_resnet_like_runtime_eval"
HB_SHIM="${ROOT}/build-hb/platforms/cuda/libshimcuda.so"
XSERVER="${ROOT}/build-hb/service/xserver"
PROBE="${ROOT}/build-hb/hb_xqueue_probe"
HB_LIB_PATH="${ROOT}/build-hb/platforms/cuda:${ROOT}/build-hb/preempt:/usr/lib/wsl/lib"
VERIFIED_KERNELS="hb_open_resnet_conv2d_kernel,hb_open_resnet_relu_kernel,hb_open_resnet_residual_add_kernel,hb_open_resnet_checksum_kernel"
XSERVER_PID=""

mkdir -p "${OUT_DIR}"

quote_command() {
  local first=1
  for arg in "$@"; do
    if [[ "${first}" -eq 0 ]]; then
      printf ' '
    fi
    printf '%q' "${arg}"
    first=0
  done
  printf '\n'
}

write_common_env() {
  local file="$1"
  {
    printf 'ROOT=%s\n' "${ROOT}"
    printf 'HB_REPO=%s\n' "${HB_REPO}"
    printf 'HB_BUILD=%s\n' "${HB_BUILD}"
    printf 'BENCH=%s\n' "${BENCH}"
    printf 'RUNTIME_BENCH=%s\n' "${RUNTIME_BENCH}"
    printf 'HB_SHIM=%s\n' "${HB_SHIM}"
    printf 'XSERVER=%s\n' "${XSERVER}"
    printf 'PROBE=%s\n' "${PROBE}"
    printf 'CUDA_LIB=%s\n' "${CUDA_LIB}"
    printf 'LD_LIBRARY_PATH_BASE=%s\n' "${HB_LIB_PATH}"
    printf 'VERIFIED_KERNELS=%s\n' "${VERIFIED_KERNELS}"
    printf 'CORRECTNESS_INPUT_MODE=deterministic_cuda_memset_1\n'
    printf 'CORRECTNESS_RANDOM_SEED=not_used_deterministic_workload\n'
    printf 'CORRECTNESS_SPLIT_BLOCKS=%s\n' "${CORRECTNESS_SPLIT_BLOCKS}"
    printf 'UXSCHED_HB_SPLIT_BLOCKS=%s\n' "${UXSCHED_HB_SPLIT_BLOCKS}"
  } > "${file}"
}

build_probe() {
  local dir="${OUT_DIR}/hb_xqueue_probe_build"
  mkdir -p "${dir}"
  write_common_env "${dir}/env.txt"
  quote_command "${ROOT}/tools/build_hb_xqueue_probe.sh" "${PROBE}" > "${dir}/command.txt"
  if "${ROOT}/tools/build_hb_xqueue_probe.sh" "${PROBE}" \
      > "${dir}/stdout.log" 2> "${dir}/stderr.log"; then
    printf 'return_code=0\nstatus=BUILT\n' > "${dir}/status.txt"
  else
    local rc=$?
    printf 'return_code=%s\nstatus=FAILED\n' "${rc}" > "${dir}/status.txt"
    return "${rc}"
  fi
}

start_xserver() {
  local dir="${OUT_DIR}/xserver"
  mkdir -p "${dir}"
  write_common_env "${dir}/env.txt"
  quote_command env -u LD_PRELOAD -u XSCHED_POLICY "${XSERVER}" HPF 50000 > "${dir}/command.txt"
  env -u LD_PRELOAD -u XSCHED_POLICY "${XSERVER}" HPF 50000 \
    > "${dir}/stdout.log" 2> "${dir}/stderr.log" &
  XSERVER_PID=$!
  printf '%s\n' "${XSERVER_PID}" > "${dir}/pid.txt"
  sleep 1
  if kill -0 "${XSERVER_PID}" 2>/dev/null; then
    printf 'status=RUNNING\n' > "${dir}/status.txt"
  else
    printf 'status=FAILED_TO_START\n' > "${dir}/status.txt"
  fi
}

stop_xserver() {
  if [[ -n "${XSERVER_PID}" ]] && kill -0 "${XSERVER_PID}" 2>/dev/null; then
    kill "${XSERVER_PID}" 2>/dev/null || true
    wait "${XSERVER_PID}" 2>/dev/null || true
    printf 'status=STOPPED\n' > "${OUT_DIR}/xserver/status.txt"
  fi
}

trap stop_xserver EXIT

write_case_status() {
  local dir="$1"
  local rc="$2"
  local jsonl="$3"
  {
    printf 'return_code=%s\n' "${rc}"
    if [[ -f "${jsonl}" ]] && grep -q '"cuda_available":false' "${jsonl}"; then
      printf 'status=BLOCKED\n'
      printf 'reason=CUDA_UNAVAILABLE\n'
    elif [[ "${rc}" -eq 0 ]]; then
      printf 'status=RAN\n'
    else
      printf 'status=FAILED\n'
    fi
  } > "${dir}/status.txt"
}

extract_evidence() {
  local dir="$1"
  local jsonl="$2"
  if [[ -f "${jsonl}" ]]; then
    grep -o '"checksum":[^,}]*' "${jsonl}" > "${dir}/checksum.txt" || true
  fi
  grep -hE '^checksum=' "${dir}/stdout.log" "${dir}/stderr.log" >> "${dir}/checksum.txt" || true
  if [[ ! -s "${dir}/checksum.txt" ]]; then
    printf 'NO_CHECKSUM_OBSERVED\n' > "${dir}/checksum.txt"
  fi

  if [[ -f "${jsonl}" ]]; then
    grep -o '"output_hash":"[^"]*"' "${jsonl}" > "${dir}/output_hash.txt" || true
    grep -o '"output_element_count":[^,}]*' "${jsonl}" > "${dir}/output_element_count.txt" || true
  fi
  grep -hE '^output_hash=' "${dir}/stdout.log" "${dir}/stderr.log" >> "${dir}/output_hash.txt" || true
  grep -hE '^output_element_count=' "${dir}/stdout.log" "${dir}/stderr.log" >> "${dir}/output_element_count.txt" || true
  if [[ ! -s "${dir}/output_hash.txt" ]]; then
    printf 'NO_OUTPUT_HASH_OBSERVED\n' > "${dir}/output_hash.txt"
  fi
  if [[ ! -s "${dir}/output_element_count.txt" ]]; then
    printf 'NO_OUTPUT_ELEMENT_COUNT_OBSERVED\n' > "${dir}/output_element_count.txt"
  fi

  grep -hE '\[UXSCHED-(HB|XQUEUE)\].*(split_count|backend_selected=HB_SPLIT|transform_succeeded|transformed_module_loaded|parent_launch_submitted|child_launch_submitted|child_launch_completed|split_group_completed|parent_launch_completed|lp_in_flight_threshold|HIGH_PRIORITY_PASSTHROUGH|backend_selected=NATIVE|NO_XQUEUE|KERNEL_NOT_VERIFIED|PTX_UNAVAILABLE)' \
    "${dir}/stdout.log" "${dir}/stderr.log" > "${dir}/split_trace.log" || true
  if [[ ! -s "${dir}/split_trace.log" ]]; then
    printf 'NO_SPLIT_TRACE_OBSERVED\n' > "${dir}/split_trace.log"
  fi

  grep -hE '\[UXSCHED-HB\].*(parent_launch_submitted|child_launch_submitted|backend_selected=HB_SPLIT|capability=splittable)' \
    "${dir}/stdout.log" "${dir}/stderr.log" > "${dir}/transformed_launch_evidence.log" || true
  if [[ ! -s "${dir}/transformed_launch_evidence.log" ]]; then
    printf 'NO_TRANSFORMED_LAUNCH_OBSERVED\n' > "${dir}/transformed_launch_evidence.log"
  fi

  grep -hE '\[UXSCHED-HB\].*(child_launch_completed|split_group_completed)' \
    "${dir}/stdout.log" "${dir}/stderr.log" > "${dir}/child_completion.log" || true
  if [[ ! -s "${dir}/child_completion.log" ]]; then
    printf 'NO_CHILD_COMPLETION_OBSERVED\n' > "${dir}/child_completion.log"
  fi

  grep -hE '\[UXSCHED-HB\].*parent_launch_completed|hb_open_resnet_like.*wrote|LP-only correctness|split correctness oracle|no CUDA device|^mismatches=0' \
    "${dir}/stdout.log" "${dir}/stderr.log" > "${dir}/parent_completion.log" || true
  if [[ ! -s "${dir}/parent_completion.log" ]]; then
    printf 'NO_PARENT_COMPLETION_OBSERVED\n' > "${dir}/parent_completion.log"
  fi

  count_logs() {
    local pattern="$1"
    grep -hE "${pattern}" "${dir}/stdout.log" "${dir}/stderr.log" 2>/dev/null | wc -l | tr -d ' '
  }

  {
    printf 'uxsched_hb_transform_count=%s\n' \
      "$(count_logs '\[UXSCHED-HB\].*transform_succeeded')"
    printf 'uxsched_hb_parent_launch_count=%s\n' \
      "$(count_logs '\[UXSCHED-HB\].*parent_launch_submitted')"
    printf 'uxsched_hb_child_launch_count=%s\n' \
      "$(count_logs '\[UXSCHED-HB\].*child_launch_submitted')"
    printf 'uxsched_hb_transformed_launch_count=%s\n' \
      "$(count_logs '\[UXSCHED-HB\].*child_launch_submitted.*transformed_function=')"
    printf 'uxsched_hb_fallback_count=%s\n' \
      "$(count_logs '\[UXSCHED-HB\].*backend_selected=NATIVE reason=')"
    printf 'uxsched_hb_no_xqueue_count=%s\n' \
      "$(count_logs '\[UXSCHED-HB\].*reason=NO_XQUEUE')"
  } > "${dir}/uxsched_backend_stats.env"

  local workload_split_policy=""
  local workload_fixed_split_blocks="0"
  local workload_lp_split_launched="0"
  local workload_lp_split_completed="0"
  if [[ -f "${jsonl}" ]]; then
    workload_split_policy="$(grep -ho '"split_policy":"[^"]*"' "${jsonl}" | tail -n 1 | cut -d '"' -f 4 || true)"
    workload_fixed_split_blocks="$(grep -ho '"fixed_split_blocks":[^,}]*' "${jsonl}" | tail -n 1 | cut -d ':' -f 2 || true)"
    workload_lp_split_launched="$(grep -ho '"lp_split_launched":[^,}]*' "${jsonl}" | tail -n 1 | cut -d ':' -f 2 || true)"
    workload_lp_split_completed="$(grep -ho '"lp_split_completed":[^,}]*' "${jsonl}" | tail -n 1 | cut -d ':' -f 2 || true)"
  fi
  {
    printf 'workload_split_policy=%s\n' "${workload_split_policy:-}"
    printf 'workload_fixed_split_blocks=%s\n' "${workload_fixed_split_blocks:-0}"
    printf 'workload_lp_split_launched=%s\n' "${workload_lp_split_launched:-0}"
    printf 'workload_lp_split_completed=%s\n' "${workload_lp_split_completed:-0}"
  } > "${dir}/workload_internal_stats.env"
}

env_value() {
  local file="$1"
  local key="$2"
  if [[ ! -f "${file}" ]]; then
    printf '0\n'
    return
  fi
  awk -F= -v key="${key}" '$1 == key { value=$2 } END { if (value == "") value=0; print value }' "${file}"
}

case_status_value() {
  local name="$1"
  local key="$2"
  env_value "${OUT_DIR}/${name}/status.txt" "${key}"
}

case_stat_value() {
  local name="$1"
  local key="$2"
  env_value "${OUT_DIR}/${name}/uxsched_backend_stats.env" "${key}"
}

last_json_value() {
  local name="$1"
  local key="$2"
  local file="${OUT_DIR}/${name}/output.jsonl"
  if [[ ! -f "${file}" ]]; then
    printf '\n'
    return
  fi
  grep -ho "\"${key}\":[^,}]*" "${file}" | tail -n 1 | cut -d ':' -f 2- | tr -d '"' || true
}

last_probe_value() {
  local name="$1"
  local key="$2"
  grep -hE "^${key}=" "${OUT_DIR}/${name}/stdout.log" "${OUT_DIR}/${name}/stderr.log" 2>/dev/null |
    tail -n 1 | cut -d '=' -f 2- || true
}

case_has_no_cuda_error() {
  local name="$1"
  ! grep -qiE 'illegal memory access|invalid argument|segmentation fault|core dumped|CUDA error|cuda error' \
    "${OUT_DIR}/${name}/stdout.log" "${OUT_DIR}/${name}/stderr.log" 2>/dev/null
}

case_ran_ok() {
  local name="$1"
  [[ "$(case_status_value "${name}" "status")" == "RAN" ]] && case_has_no_cuda_error "${name}"
}

int_gt() {
  local value="${1:-0}"
  local threshold="${2:-0}"
  [[ "${value}" =~ ^[0-9]+$ ]] && (( value > threshold ))
}

int_eq() {
  local value="${1:-0}"
  local expected="${2:-0}"
  [[ "${value}" =~ ^[0-9]+$ ]] && (( value == expected ))
}

hb_split_case_pass() {
  local name="$1"
  case_ran_ok "${name}" &&
    int_gt "$(case_stat_value "${name}" uxsched_hb_transform_count)" 0 &&
    int_gt "$(case_stat_value "${name}" uxsched_hb_parent_launch_count)" 0 &&
    int_gt "$(case_stat_value "${name}" uxsched_hb_child_launch_count)" 1 &&
    int_gt "$(case_stat_value "${name}" uxsched_hb_transformed_launch_count)" 0 &&
    int_eq "$(case_stat_value "${name}" uxsched_hb_no_xqueue_count)" 0
}

parent_completion_pass_for_case() {
  local name="$1"
  hb_split_case_pass "${name}" &&
    grep -q 'split_group_completed' "${OUT_DIR}/${name}/child_completion.log" &&
    grep -q 'parent_launch_completed' "${OUT_DIR}/${name}/parent_completion.log"
}

write_gate1_summary() {
  local summary="${OUT_DIR}/gate1_summary.env"

  local native_checksum uxsched_native_checksum hb_fixed_checksum
  local native_hash uxsched_native_hash hb_fixed_hash
  local native_count uxsched_native_count hb_fixed_count
  native_checksum="$(last_json_value native_correctness checksum)"
  uxsched_native_checksum="$(last_json_value uxsched_native_correctness checksum)"
  hb_fixed_checksum="$(last_json_value uxsched_hb_fixed_correctness checksum)"
  native_hash="$(last_json_value native_correctness output_hash)"
  uxsched_native_hash="$(last_json_value uxsched_native_correctness output_hash)"
  hb_fixed_hash="$(last_json_value uxsched_hb_fixed_correctness output_hash)"
  native_count="$(last_json_value native_correctness output_element_count)"
  uxsched_native_count="$(last_json_value uxsched_native_correctness output_element_count)"
  hb_fixed_count="$(last_json_value uxsched_hb_fixed_correctness output_element_count)"

  local native_correctness_pass=0
  local uxsched_native_correctness_pass=0
  local hb_fixed_correctness_pass=0
  local checksum_match=0
  local output_hash_match=0
  local output_element_count_match=0

  if case_ran_ok native_correctness && [[ -n "${native_hash}" && -n "${native_count}" ]]; then
    native_correctness_pass=1
  fi
  if case_ran_ok uxsched_native_correctness && [[ -n "${uxsched_native_hash}" && -n "${uxsched_native_count}" ]]; then
    uxsched_native_correctness_pass=1
  fi
  if hb_split_case_pass uxsched_hb_fixed_correctness &&
      int_eq "$(case_stat_value uxsched_hb_fixed_correctness uxsched_hb_fallback_count)" 0 &&
      [[ -n "${hb_fixed_hash}" && -n "${hb_fixed_count}" ]]; then
    hb_fixed_correctness_pass=1
  fi
  if [[ -n "${native_checksum}" && "${native_checksum}" == "${uxsched_native_checksum}" &&
        "${native_checksum}" == "${hb_fixed_checksum}" ]]; then
    checksum_match=1
  fi
  if [[ -n "${native_hash}" && "${native_hash}" == "${uxsched_native_hash}" &&
        "${native_hash}" == "${hb_fixed_hash}" ]]; then
    output_hash_match=1
  fi
  if [[ -n "${native_count}" && "${native_count}" == "${uxsched_native_count}" &&
        "${native_count}" == "${hb_fixed_count}" ]]; then
    output_element_count_match=1
  fi

  local event_sync_pass=0
  local stream_sync_pass=0
  local context_sync_pass=0
  local same_stream_ordering_pass=0
  local parent_completion_pass=0
  if hb_split_case_pass sync_event_hb_fixed_probe &&
      [[ "$(last_probe_value sync_event_hb_fixed_probe event_sync_pass)" == "1" ]]; then
    event_sync_pass=1
  fi
  if hb_split_case_pass sync_stream_hb_fixed_probe &&
      [[ "$(last_probe_value sync_stream_hb_fixed_probe stream_sync_pass)" == "1" ]]; then
    stream_sync_pass=1
  fi
  if hb_split_case_pass sync_context_hb_fixed_probe &&
      [[ "$(last_probe_value sync_context_hb_fixed_probe context_sync_pass)" == "1" ]]; then
    context_sync_pass=1
  fi
  if hb_split_case_pass sync_same_stream_ordering_hb_fixed_probe &&
      [[ "$(last_probe_value sync_same_stream_ordering_hb_fixed_probe same_stream_ordering_pass)" == "1" ]]; then
    same_stream_ordering_pass=1
  fi
  if parent_completion_pass_for_case sync_parent_completion_hb_fixed_probe &&
      [[ "$(last_probe_value sync_parent_completion_hb_fixed_probe parent_completion_probe)" == "1" ]]; then
    parent_completion_pass=1
  fi

  local hp_passthrough_pass=0
  if case_ran_ok probe_hb_fixed_hp_passthrough &&
      grep -q 'reason=HIGH_PRIORITY_PASSTHROUGH' "${OUT_DIR}/probe_hb_fixed_hp_passthrough/split_trace.log" &&
      int_eq "$(case_stat_value probe_hb_fixed_hp_passthrough uxsched_hb_child_launch_count)" 0 &&
      int_eq "$(case_stat_value probe_hb_fixed_hp_passthrough uxsched_hb_transformed_launch_count)" 0; then
    hp_passthrough_pass=1
  fi

  local ptx_unavailable_fallback_pass=0
  if case_ran_ok probe_fallback_ptx_unavailable &&
      grep -q 'reason=PTX_UNAVAILABLE' "${OUT_DIR}/probe_fallback_ptx_unavailable/split_trace.log"; then
    ptx_unavailable_fallback_pass=1
  fi

  local kernel_not_verified_fallback_pass=0
  if case_ran_ok probe_fallback_kernel_not_verified &&
      grep -q 'reason=KERNEL_NOT_VERIFIED' "${OUT_DIR}/probe_fallback_kernel_not_verified/split_trace.log" &&
      grep -q 'function=hb_xqueue_probe_kernel' "${OUT_DIR}/probe_fallback_kernel_not_verified/split_trace.log"; then
    kernel_not_verified_fallback_pass=1
  fi

  local no_local_scheduler_fallback=0
  if ! grep -RqiE 'local scheduler fallback|LOCAL_SCHEDULER|fallback.*local' "${OUT_DIR}" 2>/dev/null; then
    no_local_scheduler_fallback=1
  fi

  local no_cuda_runtime_error=0
  if ! grep -RqiE 'illegal memory access|invalid argument|segmentation fault|core dumped|CUDA error|cuda error' \
      "${OUT_DIR}" 2>/dev/null; then
    no_cuda_runtime_error=1
  fi

  local gate1_pass=0
  if [[ "${native_correctness_pass}" == "1" &&
        "${uxsched_native_correctness_pass}" == "1" &&
        "${hb_fixed_correctness_pass}" == "1" &&
        "${checksum_match}" == "1" &&
        "${output_hash_match}" == "1" &&
        "${output_element_count_match}" == "1" &&
        "${event_sync_pass}" == "1" &&
        "${stream_sync_pass}" == "1" &&
        "${context_sync_pass}" == "1" &&
        "${same_stream_ordering_pass}" == "1" &&
        "${parent_completion_pass}" == "1" &&
        "${hp_passthrough_pass}" == "1" &&
        "${ptx_unavailable_fallback_pass}" == "1" &&
        "${kernel_not_verified_fallback_pass}" == "1" &&
        "${no_cuda_runtime_error}" == "1" &&
        "${no_local_scheduler_fallback}" == "1" ]]; then
    gate1_pass=1
  fi

  {
    printf 'native_correctness_pass=%s\n' "${native_correctness_pass}"
    printf 'uxsched_native_correctness_pass=%s\n' "${uxsched_native_correctness_pass}"
    printf 'hb_fixed_correctness_pass=%s\n' "${hb_fixed_correctness_pass}"
    printf 'checksum_match=%s\n' "${checksum_match}"
    printf 'output_hash_match=%s\n' "${output_hash_match}"
    printf 'output_element_count_match=%s\n' "${output_element_count_match}"
    printf 'native_checksum=%s\n' "${native_checksum}"
    printf 'uxsched_native_checksum=%s\n' "${uxsched_native_checksum}"
    printf 'hb_fixed_checksum=%s\n' "${hb_fixed_checksum}"
    printf 'native_output_hash=%s\n' "${native_hash}"
    printf 'uxsched_native_output_hash=%s\n' "${uxsched_native_hash}"
    printf 'hb_fixed_output_hash=%s\n' "${hb_fixed_hash}"
    printf 'native_output_element_count=%s\n' "${native_count}"
    printf 'uxsched_native_output_element_count=%s\n' "${uxsched_native_count}"
    printf 'hb_fixed_output_element_count=%s\n' "${hb_fixed_count}"
    printf 'hb_transform_count=%s\n' "$(case_stat_value uxsched_hb_fixed_correctness uxsched_hb_transform_count)"
    printf 'hb_parent_launch_count=%s\n' "$(case_stat_value uxsched_hb_fixed_correctness uxsched_hb_parent_launch_count)"
    printf 'hb_child_launch_count=%s\n' "$(case_stat_value uxsched_hb_fixed_correctness uxsched_hb_child_launch_count)"
    printf 'hb_transformed_launch_count=%s\n' "$(case_stat_value uxsched_hb_fixed_correctness uxsched_hb_transformed_launch_count)"
    printf 'hb_fallback_count=%s\n' "$(case_stat_value uxsched_hb_fixed_correctness uxsched_hb_fallback_count)"
    printf 'hb_no_xqueue_count=%s\n' "$(case_stat_value uxsched_hb_fixed_correctness uxsched_hb_no_xqueue_count)"
    printf 'event_sync_pass=%s\n' "${event_sync_pass}"
    printf 'stream_sync_pass=%s\n' "${stream_sync_pass}"
    printf 'context_sync_pass=%s\n' "${context_sync_pass}"
    printf 'same_stream_ordering_pass=%s\n' "${same_stream_ordering_pass}"
    printf 'parent_completion_pass=%s\n' "${parent_completion_pass}"
    printf 'hp_passthrough_pass=%s\n' "${hp_passthrough_pass}"
    printf 'ptx_unavailable_fallback_pass=%s\n' "${ptx_unavailable_fallback_pass}"
    printf 'kernel_not_verified_fallback_pass=%s\n' "${kernel_not_verified_fallback_pass}"
    printf 'no_cuda_runtime_error=%s\n' "${no_cuda_runtime_error}"
    printf 'no_local_scheduler_fallback=%s\n' "${no_local_scheduler_fallback}"
    printf 'gate1_pass=%s\n' "${gate1_pass}"
  } > "${summary}"
}

run_case() {
  local name="$1"
  shift
  local dir="${OUT_DIR}/${name}"
  local jsonl="${dir}/output.jsonl"
  local rc=0
  mkdir -p "${dir}"
  write_common_env "${dir}/env.txt"
  {
    printf 'CASE=%s\n' "${name}"
    printf 'OUTPUT_JSONL=%s\n' "${jsonl}"
  } >> "${dir}/env.txt"
  quote_command "$@" --output "${jsonl}" > "${dir}/command.txt"
  "$@" --output "${jsonl}" > "${dir}/stdout.log" 2> "${dir}/stderr.log" || rc=$?
  printf '%s\n' "${rc}" > "${dir}/return_code.txt"
  write_case_status "${dir}" "${rc}" "${jsonl}"
  extract_evidence "${dir}" "${jsonl}"
  return 0
}

run_raw_case() {
  local name="$1"
  shift
  local dir="${OUT_DIR}/${name}"
  local jsonl="${dir}/output.jsonl"
  local rc=0
  mkdir -p "${dir}"
  write_common_env "${dir}/env.txt"
  {
    printf 'CASE=%s\n' "${name}"
    printf 'OUTPUT_JSONL=%s\n' "${jsonl}"
  } >> "${dir}/env.txt"
  quote_command "$@" > "${dir}/command.txt"
  "$@" > "${dir}/stdout.log" 2> "${dir}/stderr.log" || rc=$?
  printf '%s\n' "${rc}" > "${dir}/return_code.txt"
  write_case_status "${dir}" "${rc}" "${jsonl}"
  extract_evidence "${dir}" "${jsonl}"
  return 0
}

COMMON_ARGS=(
  --batch-size 8
  --channels 16
  --height 56
  --width 56
  --num-blocks 4
  --warmup 0
)

CORRECTNESS_ARGS=(
  "${COMMON_ARGS[@]}"
  --role lp
  --duration-ms 0
  --iterations 1
  --split-policy fixed
  --split-blocks "${CORRECTNESS_SPLIT_BLOCKS}"
  --correctness-mode lp-only
  --correctness-iterations 1
  --reset-state-each-iteration
  --dump-output-tensor
  --lp-correctness-sync-boundary iteration
)

build_probe || exit $?

run_case "native_correctness" \
  env -u LD_PRELOAD -u XSCHED_POLICY -u HB_TASK_PRIORITY \
    -u HB_SPLIT_KERNELS -u UXSCHED_CUDA_RUNTIME_STRATEGY \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    "${RUNTIME_BENCH}" "${CORRECTNESS_ARGS[@]}"

start_xserver

run_raw_case "probe_default_stream_hb_fixed_lp" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS=hb_xqueue_probe_kernel \
    "${PROBE}" --stream default --blocks 1024 --threads 1

run_raw_case "probe_explicit_stream_hb_fixed_lp" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS=hb_xqueue_probe_kernel \
    "${PROBE}" --stream explicit --blocks 1024 --threads 1

run_case "uxsched_native_correctness" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY -u HB_SPLIT_KERNELS \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=NATIVE \
    "${RUNTIME_BENCH}" "${CORRECTNESS_ARGS[@]}"

run_case "uxsched_hb_fixed_correctness" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY -u HB_SPLIT_KERNELS \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS="${VERIFIED_KERNELS}" \
    "${RUNTIME_BENCH}" "${CORRECTNESS_ARGS[@]}"

run_raw_case "sync_event_hb_fixed_probe" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS=hb_xqueue_probe_kernel \
    "${PROBE}" --stream explicit --sync event --blocks 1024 --threads 1

run_raw_case "sync_stream_hb_fixed_probe" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS=hb_xqueue_probe_kernel \
    "${PROBE}" --stream explicit --sync stream --blocks 1024 --threads 1

run_raw_case "sync_context_hb_fixed_probe" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS=hb_xqueue_probe_kernel \
    "${PROBE}" --stream explicit --sync context --blocks 1024 --threads 1

run_raw_case "sync_same_stream_ordering_hb_fixed_probe" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS=hb_xqueue_probe_kernel \
    "${PROBE}" --stream explicit --sync same-stream --blocks 1024 --threads 1

run_raw_case "sync_parent_completion_hb_fixed_probe" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS=hb_xqueue_probe_kernel \
    "${PROBE}" --stream explicit --sync parent --blocks 1024 --threads 1

run_raw_case "probe_hb_fixed_hp_passthrough" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS=hb_xqueue_probe_kernel \
    "${PROBE}" --stream explicit --blocks 1024 --threads 1

run_raw_case "probe_fallback_ptx_unavailable" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY -u HB_SPLIT_KERNELS \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS= \
    "${PROBE}" --stream explicit --blocks 1024 --threads 1

run_raw_case "probe_fallback_kernel_not_verified" \
  env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
    LD_LIBRARY_PATH="${HB_LIB_PATH}" \
    LD_PRELOAD="${HB_SHIM}" \
    XSCHED_CUDA_LIB="${CUDA_LIB}" \
    CUXTRA_CUDA_LIB="${CUDA_LIB}" \
    XSCHED_SCHEDULER=GLB \
    XSCHED_AUTO_XQUEUE=ON \
    XSCHED_AUTO_XQUEUE_LEVEL=1 \
    XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
    UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
    UXSCHED_XQUEUE_TRACE=1 \
    UXSCHED_HB_SPLIT_BLOCKS="${UXSCHED_HB_SPLIT_BLOCKS}" \
    UXSCHED_HB_STRICT=0 \
    UXSCHED_HB_VERIFIED_KERNELS=__not_verified__ \
    "${PROBE}" --stream explicit --blocks 1024 --threads 1

write_gate1_summary

printf 'Gate 1 smoke artifacts: %s\n' "${OUT_DIR}"
