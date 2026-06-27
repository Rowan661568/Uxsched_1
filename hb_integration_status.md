# HB Runtime Integration Status

Allowed status values in this file:

- IMPLEMENTED
- COMPILE VERIFIED
- RUNTIME VERIFIED
- CORRECTNESS VERIFIED
- PERFORMANCE VERIFIED
- NOT TESTED
- BLOCKED
- FAILED

## Current Runtime Strategy Status

| Item | Status | Notes |
| --- | --- | --- |
| Branch `feature/hummingbird-split-backend` | IMPLEMENTED | Continuing from `4146c1e`. |
| Re-audit document | IMPLEMENTED | `docs/hummingbird_runtime_reaudit.md`. |
| `CudaRuntimeStrategy` interface | COMPILE VERIFIED | Added under CUDA HAL runtime directory. |
| `NativeRuntimeStrategy` | COMPILE VERIFIED | Preserves original `CudaKernelLaunchCommand` path. |
| `HummingbirdRuntimeStrategy` | COMPILE VERIFIED | `HB_FIXED` delegates to fixed split implementation. |
| `UXSCHED_CUDA_RUNTIME_STRATEGY=NATIVE` | COMPILE VERIFIED | Runtime tests not run. |
| `UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED` | RUNTIME VERIFIED | Manual GPU result `results/hb_gate1_after_xqueue_fix_20260624_170107` observed transformed module load, 78 parent launches, 312 transformed child launches, fallback count 0, and `NO_XQUEUE` count 0. Correctness and synchronization Gate 1 checks are still pending rerun. |
| `UXSCHED_CUDA_RUNTIME_STRATEGY=HB_RUNTIME` | IMPLEMENTED | Explicit Native fallback: runtime not implemented yet. |
| `UXSCHED_CUDA_RUNTIME_STRATEGY=AUTO` | IMPLEMENTED | Explicit Native fallback: coordinator unavailable. |
| Per-device HB coordinator | BLOCKED | Not implemented. |
| HB state machine | BLOCKED | Not implemented. |
| Kernel profiler / SplitPlan cache | BLOCKED | Not implemented. |
| kernel-tick launcher | BLOCKED | Not implemented. |
| small bubble hints | BLOCKED | Not implemented. |
| large bubble / consolidation | BLOCKED | Not implemented. |
| GPU visibility | RUNTIME VERIFIED | User manual WSL Gate 1 run used RTX 5060 and recorded `cuda_available=true`; do not reuse earlier Codex tool-session GPU blocker as current status. |
| open_resnet_like GPU validation | FAILED | Native and UXSched split launch evidence exists, but prior non-correctness UXSched open_resnet_like cases returned 139 and did not provide checksum/hash evidence. Old open_resnet correctness is deferred and is not a blocker for CUTLASS planning. |
| CUDA stream to XQueue association fix | RUNTIME VERIFIED | Default-stream and explicit-stream Driver API probes both reached `HB_SPLIT` with transformed child launches and `NO_XQUEUE=0` in `results/hb_gate1_after_xqueue_fix_20260624_170107`. |
| CUTLASS realtime benchmark plan | IMPLEMENTED | `docs/cutlass_realtime_benchmark_plan.md` audits the current realtime benchmark and defines the CUTLASS replacement plan. |
| CUTLASS launch compatibility probe | COMPILE VERIFIED | `benchmarks/cutlass/cutlass_launch_probe.cu` builds with external CUTLASS revision `ad7b2f5`, CUDA 12.8, native SM120 SASS, and PTX. User manual GPU result verified Native and UXSched NATIVE correctness, but HB_FIXED backend was not exercised before the Runtime bridge. Runtime GPU compatibility must be rerun outside Codex. |
| CUTLASS CUDA Runtime bridge | COMPILE VERIFIED | Existing `libshimcuda.so` intercepts CUDA Runtime fatbin/function registration, `cudaLaunchKernel`, and Runtime stream/event/device synchronization APIs used by the CUTLASS probe. User GPU result confirmed Runtime API interception, PTX extraction, CUfunction resolution, and Runtime sync interception. The metadata bridge now explicitly registers Runtime `CUmodule -> PTX` and `CUfunction -> kernel name` into the existing HB backend registry; GPU rerun is required to verify that `function=<unknown>/PTX_UNAVAILABLE` is gone. |
| CUTLASS workload | NOT TESTED | Full HP/LP realtime workload is not implemented yet. CUTLASS correctness is mandatory before any P99 claim. |
| Persistent agent rules | IMPLEMENTED | Added `AGENTS.md` with UXSched-Hummingbird integration rules and CUTLASS realtime benchmark fairness/P99 proof requirements. |
| Gate 1 smoke runner | IMPLEMENTED | `tools/run_hb_gate1_smoke.sh` now records correctness, sync, HP passthrough, fallback artifacts, and writes `gate1_summary.env`; GPU rerun is required. |

## Completed

- Created Git branch `feature/hummingbird-split-backend`.
- Added `UXSCHED_ENABLE_HB_SPLIT` CMake option, default `OFF`.
- Added runtime backend mode selection:
  - `UXSCHED_CUDA_PREEMPT_BACKEND=NATIVE`
  - `UXSCHED_CUDA_PREEMPT_BACKEND=HB_SPLIT`
  - `UXSCHED_CUDA_PREEMPT_BACKEND=AUTO`
- Added runtime strategy mode selection:
  - `UXSCHED_CUDA_RUNTIME_STRATEGY=NATIVE`
  - `UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED`
  - `UXSCHED_CUDA_RUNTIME_STRATEGY=HB_RUNTIME`
  - `UXSCHED_CUDA_RUNTIME_STRATEGY=AUTO`
- Added `CudaRuntimeStrategy`, `NativeRuntimeStrategy`, and
  `HummingbirdRuntimeStrategy`.
- Added persistent project rules in `AGENTS.md`.
- Kept UXSched CUDA shim as the only CUDA hook entry.
- Routed module load/get/unload wrappers through UXSched HB-aware code:
  - `cuModuleLoad`
  - `cuModuleLoadData`
  - `cuModuleLoadDataEx`
  - `cuModuleGetFunction`
  - `cuModuleUnload`
- Added PTX offset transformation for verified kernels:
  - append `__hb_off_x/y/z`;
  - rewrite recognized `%ctaid.x/y/z` moves;
  - reject recognized grid-level sync tokens.
- Added hidden transformed module cache while keeping the application-visible
  module/function original for safe native fallback.
- Added fixed-size grid decomposition with default split size `512`.
- Added LP-only split selection based on negative queue priority environment.
- Added HP passthrough for non-negative priority.
- Added native fallback for missing PTX, unverified kernels, transform failure,
  missing XQueue, unsupported Lv2/Lv3 combination, unsupported axes, `extra`
  launch format, null `kernelParams`, and grids smaller than split size.
- Added `SplitCommandGroup` child completion tracking.
- Added per-XQueue LP split launch config `threshold=1, batch_size=1`.
- Added structured `[UXSCHED-HB]` logs.
- Added optional `[UXSCHED-XQUEUE]` diagnostics with `UXSCHED_XQUEUE_TRACE=1`.
- Added launch-time CUDA stream to XQueue auto-association for streams that did
  not pass through `XStreamCreate*`.
- Added default stream support through a per-context synthetic HwQueue handle.
- Added PTX-present unverified-kernel metadata so that fallback can report
  `KERNEL_NOT_VERIFIED`.
- Added minimal Driver API XQueue probe under `tools/hb_xqueue_probe.cpp`.
- Extended the Driver API probe with event, stream, context, same-stream
  ordering, and parent-completion synchronization modes.
- Built `halcuda` and `shimcuda` with HB enabled.
- Built `halcuda` and `shimcuda` with default HB disabled.
- Updated `tools/run_hb_gate1_smoke.sh` to preserve Gate 1 smoke artifacts,
  split UXSched backend stats from workload-internal split stats, and run
  default/explicit stream probes.
- Updated `tools/run_hb_gate1_smoke.sh` to run three correctness-mode cases
  (`native_correctness`, `uxsched_native_correctness`,
  `uxsched_hb_fixed_correctness`), five HB_FIXED synchronization probes, HP
  passthrough, separate `PTX_UNAVAILABLE` and `KERNEL_NOT_VERIFIED` fallback
  probes, and a final `gate1_summary.env`.
- Added `docs/cutlass_realtime_benchmark_plan.md` with the CUTLASS realtime
  benchmark audit and implementation plan.
- Added CUTLASS CUDA Runtime bridge inside the UXSched CUDA shim:
  - `__cudaRegisterFatBinary`
  - `__cudaRegisterFatBinaryEnd`
  - `__cudaRegisterFunction`
  - `__cudaUnregisterFatBinary`
  - `cudaLaunchKernel`
  - `cudaLaunchKernelExC` traced with safe Runtime fallback
  - Runtime stream/event/device synchronization wrappers used by the CUTLASS
    probe
- Added `UXSCHED_CUDART_TRACE=1` diagnostics for Runtime registration, launch
  interception, function resolution, backend selection, and fallback reasons.
- Updated the CUTLASS probe build to use shared cudart and
  `CMAKE_CUDA_ARCHITECTURES=120-real;120-virtual` with uncompressed fatbin PTX.
- Updated `tools/run_cutlass_launch_probe.sh` to save `ldd`, dynamic symbol,
  `cuobjdump` PTX/SASS evidence, Runtime registration logs, Runtime intercept
  counts, Runtime sync intercept counts, and discovered CUTLASS kernel name.
- Added HB metadata registration APIs shared by Driver hooks and Runtime bridge:
  `RegisterModuleMetadata`, `RegisterFunctionMetadata`,
  `UnregisterModuleMetadata`, and `LookupFunctionMetadata`.
- Runtime bridge now logs `runtime_hb_module_registered` and
  `runtime_hb_function_registered` before HB_FIXED dispatch.
- Added CPU-side metadata registry probe:
  `tools/hb_metadata_registry_probe.cpp` and
  `tools/build_hb_metadata_registry_probe.sh`.
- Added CUTLASS launch compatibility probe and runner:
  - `benchmarks/cutlass/CMakeLists.txt`;
  - `benchmarks/cutlass/cutlass_launch_probe.cu`;
  - `benchmarks/cutlass/cutlass_probe_common.h`;
  - `tools/build_cutlass_launch_probe.sh`;
  - `tools/run_cutlass_launch_probe.sh`.
- Built `xserver` and `xcli` in both `build-hb` and `build-native`.

## Partially Completed

- Completion group tracks all child command completion, but first CUDA launch
  error recovery is limited by current Lv1 `CudaQueueLv1` behavior, which asserts
  on failed CUDA launches.
- Module unload waits for all XQueues before unloading the hidden transformed
  module, but broader multi-threaded unload stress tests are still needed.
- Multi-dimensional grid splitting is implemented, but runtime validation has
  not been run on real GPU workloads yet.
- `cuLaunchKernelEx` remains native in stage 1.
- Manual Gate 1 remains incomplete for old open_resnet correctness, but
  HB_FIXED runtime is verified and correctness is deferred to the CUTLASS
  workload validation path.

## Not Completed

- Automatic split-size selection.
- Online kernel profiling.
- Bubble detection.
- Split-kernel consolidation.
- Kernel-tick scheduling.
- Hummingbird memory management or NVLink offload.
- CUDA Graph splitting.
- HB split combined with UXSched Lv2/Lv3.
- cuBLAS/cuDNN closed kernel splitting.
- CUTLASS realtime GEMM workload implementation and validation.
- CUTLASS Runtime launch GPU compatibility validation.
- CUTLASS Driver / `CudaHostAdapter` launch integration; current probe reports
  `cutlass_driver_launch_integration_blocked`.
- GPU runtime benchmark repeat runs.
- Per-device runtime coordinator, profiler, kernel-tick, bubble detection, and
  consolidation remain intentionally unimplemented.
- CUTLASS workload is planned but not implemented.

## Modified Files

- `CMakeLists.txt`
- `platforms/cuda/CMakeLists.txt`
- `platforms/RTX4060/CMakeLists.txt`
- `platforms/cuda/shim/include/xsched/cuda/shim/shim.h`
- `platforms/cuda/shim/src/intercept.cpp`
- `platforms/cuda/shim/src/shim.cpp`
- `platforms/cuda/hal/include/xsched/cuda/hal/common/handle.h`
- `platforms/cuda/hal/include/xsched/cuda/hal/hb_split/backend.h`
- `platforms/cuda/hal/include/xsched/cuda/hal/level1/cuda_queue.h`
- `platforms/cuda/hal/src/hb_split/backend.cpp`
- `platforms/cuda/hal/src/arch/arch.cpp`
- `platforms/cuda/hal/src/level1/cuda_queue.cpp`
- `platforms/cuda/hal/src/runtime/runtime_strategy.cpp`
- `tools/run_hb_gate1_smoke.sh`
- `benchmarks/cutlass/CMakeLists.txt`
- `benchmarks/cutlass/cutlass_launch_probe.cu`
- `benchmarks/cutlass/cutlass_probe_common.h`
- `tools/build_cutlass_launch_probe.sh`
- `tools/run_cutlass_launch_probe.sh`

## Added Files

- `platforms/cuda/hal/include/xsched/cuda/hal/hb_split/backend.h`
- `platforms/cuda/hal/src/hb_split/backend.cpp`
- `platforms/cuda/hal/include/xsched/cuda/hal/runtime/runtime_strategy.h`
- `platforms/cuda/hal/src/runtime/runtime_strategy.cpp`
- `docs/hummingbird_backend_design.md`
- `docs/hummingbird_backend_implementation.md`
- `docs/hummingbird_backend_test_plan.md`
- `docs/hummingbird_backend_results.md`
- `docs/hummingbird_runtime_reaudit.md`
- `docs/hummingbird_runtime_architecture.md`
- `docs/hummingbird_runtime_state_machine.md`
- `docs/hummingbird_runtime_profiler.md`
- `docs/hummingbird_runtime_bubble_detection.md`
- `docs/hummingbird_runtime_test_plan.md`
- `docs/hummingbird_runtime_results.md`
- `docs/codex_handoff.md`
- `AGENTS.md`
- `hb_integration_status.md`
- `tools/run_hb_gate1_smoke.sh`
- `tools/hb_xqueue_probe.cpp`
- `tools/build_hb_xqueue_probe.sh`

## Session Handoff

2026-06-24 NO_XQUEUE fix:

- Read manual result directory `results/hb_gate1_manual_20260624_163059`.
- Current real Gate 1 conclusion is FAIL/PARTIAL:
  - GPU was accessible in the user's WSL manual run.
  - Native open_resnet_like and UXSched Native ran.
  - HB_FIXED LP transformed PTX but actual launches fell back with
    `backend_selected=NATIVE reason=NO_XQUEUE`.
  - No transformed child launch was observed.
- Root cause: `XLaunchKernel` only looked up pre-existing XQueue mappings and
  default stream was hard-coded to `xqueue=nullptr`; the workload's explicit
  stream did not pass through the shim stream-create wrapper.
- Fix is COMPILE VERIFIED:
  - launch-time auto-association for missing managed CUDA streams;
  - per-context synthetic HwQueue handle for default stream;
  - `UXSCHED_XQUEUE_TRACE=1` diagnostics;
  - `KERNEL_NOT_VERIFIED` fallback metadata;
  - minimal default/explicit Driver API probe.
- Passed:
  - `tools/build_hb_xqueue_probe.sh build-hb/hb_xqueue_probe`
  - `cmake --build build-hb --target halcuda shimcuda -j2`
  - `cmake --build build-native --target halcuda shimcuda -j2`
  - `bash -n tools/run_hb_gate1_smoke.sh tools/build_hb_xqueue_probe.sh`
- Not run in Codex tool environment: GPU runtime validation.
- Required next task: user manual rerun of the minimal HB_FIXED LP probe and
  correctness-mode checksum/output-hash comparison.

2026-06-24 Gate 1 correctness/synchronization runner update:

- Read real GPU result directory `results/hb_gate1_after_xqueue_fix_20260624_170107`.
- Confirmed from artifacts:
  - `probe_default_stream_hb_fixed_lp`: transformed child launches and
    `NO_XQUEUE=0`;
  - `probe_explicit_stream_hb_fixed_lp`: transformed child launches and
    `NO_XQUEUE=0`;
  - `uxsched_hb_fixed_lp`: `uxsched_hb_transform_count=4`,
    `uxsched_hb_parent_launch_count=78`,
    `uxsched_hb_child_launch_count=312`,
    `uxsched_hb_transformed_launch_count=312`,
    `uxsched_hb_fallback_count=0`,
    `uxsched_hb_no_xqueue_count=0`.
- Existing `sync_event_boundary_probe` is not valid Gate 1 sync evidence:
  it had UXSched transform/child/parent counts all zero and only workload
  internal `fixed_split_blocks=16` / `lp_split_launched=1402`.
- Existing non-correctness open_resnet_like cases did not provide checksum,
  output hash, or output element-count comparison; UXSched open_resnet_like
  ordinary cases returned 139 in that artifact set.
- Added correctness-mode runner cases using
  `hb_open_resnet_like_runtime_eval --correctness-mode lp-only` with fixed
  dimensions, deterministic memset input/weight initialization, one correctness
  iteration, and `lp-correctness-sync-boundary=iteration`.
- Set workload correctness split blocks to `4096` so the workload does not
  provide the split evidence; UXSched HB_FIXED still uses
  `UXSCHED_HB_SPLIT_BLOCKS=512`.
- Added summary fields required for Gate 1, including checksum/hash/count
  comparison and synchronization pass flags.
- Codex did not run GPU validation; manual WSL rerun is still required.

2026-06-24 CUTLASS realtime benchmark audit/design:

- Added `docs/cutlass_realtime_benchmark_plan.md`.
- Audited `benchmarks/realtime_inference_latency.py`:
  - HP is PyTorch/TorchVision ResNet50 inference;
  - LP is PyTorch/TorchVision MobileNetV2 training;
  - HP and LP are independent child processes;
  - current HP requests are back-to-back after warmup, not fixed-period;
  - current latency samples use CPU wall time around `step()` plus stream
    synchronization, not CUDA events;
  - current percentile calculation sorts samples and linearly interpolates.
- Audited local CUTLASS availability:
  - no CUTLASS source or submodule was found locally;
  - CUDA toolkit and `nvcc 12.0` are present;
  - Hummingbird has optional CUTLASS include detection but no bundled CUTLASS.
- Decided to defer old open_resnet correctness runner repair and make CUTLASS
  workload correctness mandatory before any P99 claim.
- Designed a conservative first CUTLASS workload:
  - deterministic GEMM inputs;
  - CPU reference;
  - max absolute and relative error;
  - NaN/Inf detection;
  - output hash and element count;
  - CUDA event timing plus CPU request latency;
  - HP priority `10`, LP priority `-10`;
  - UXSched Native vs HB_FIXED fair comparison.
- No CUTLASS download, implementation, scheduler change, GPU benchmark, or P99
  performance claim was made.

2026-06-24 CUTLASS launch compatibility probe:

- External CUTLASS source is available at `/home/zm/project/cutlass`.
- CUTLASS revision: `ad7b2f5`.
- CUDA Toolkit for this phase: `/usr/local/cuda-12.8`.
- `nvcc`: `/usr/local/cuda-12.8/bin/nvcc`, release `12.8, V12.8.93`.
- Probe mode is native SM120 only:
  - `CMAKE_CUDA_ARCHITECTURES=120`;
  - `CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`;
  - `CUTLASS_MODE=NATIVE_SM120`.
- Forward PTX compatibility mode is canceled.
- Added a CUTLASS SIMT FP32 GEMM launch-path probe with deterministic inputs,
  closed-form CPU reference, checksum, output hash, output element count, max
  absolute and relative error, mismatch count, NaN/Inf counts, CUDA event time,
  and CPU request time.
- Runtime mode uses standard CUTLASS `device::Gemm::run(stream)`, which launches
  through CUDA Runtime. Whether it reaches the UXSched HB backend must be
  validated in a normal GPU WSL terminal.
- Driver / `CudaHostAdapter` mode is explicitly blocked because CUTLASS does not
  provide a generic official adapter that supplies `CUmodule`, `CUfunction`,
  CUTLASS kernel parameter layout, dynamic shared memory, and stream for this
  GEMM.
- Built:
  - `build-cutlass-cu128/cutlass_launch_probe`;
  - `build-hb-cu128/platforms/cuda/libhalcuda.so`;
  - `build-hb-cu128/platforms/cuda/libshimcuda.so`;
  - `build-hb-cu128/service/xserver`;
  - `build-hb-cu128/service/xcli`.
- Codex did not run Runtime GPU validation, HP/LP realtime benchmark, split-size
  sweep, or P99 experiment.

2026-06-24 handoff refresh:

- Rebuilt `docs/codex_handoff.md` using the required fixed structure.
- Rechecked current Git and source state before writing the handoff.
- Current source truth: `HB_RUNTIME` and `AUTO` are explicit Native fallback
  paths in `platforms/cuda/hal/src/runtime/runtime_strategy.cpp`.
- Current next gated task remains Gate 1 GPU validation of `HB_FIXED`.
- Build checks passed for `build-hb` and `build-native` targets
  `halcuda shimcuda`.
- Historical note: this refresh observed GPU access blocked inside that Codex
  tool session. It must not override the later user manual WSL GPU result.

2026-06-24 Gate 1 attempt:

- Rechecked the current Codex tool-session environment instead of reusing the
  previous blocker:
  - `/dev/dxg` is not visible.
  - `/usr/lib/wsl/lib/libcuda.so.1` exists.
  - `nvidia-smi` reports GPU access blocked by the operating system.
  - `torch.cuda.is_available()` is `False`; `torch.cuda.device_count()` is `0`.
- Rebuilt:
  - `cmake --build build-hb --target halcuda shimcuda -j2`
  - `cmake --build build-native --target halcuda shimcuda -j2`
  - `cmake --build build-hb --target xserver xcli -j2`
  - `cmake --build build-native --target xserver xcli -j2`
- Confirmed paths:
  - `build-hb/platforms/cuda/libhalcuda.so`
  - `build-hb/platforms/cuda/libshimcuda.so`
  - `build-native/platforms/cuda/libhalcuda.so`
  - `build-native/platforms/cuda/libshimcuda.so`
  - `build-hb/service/xserver`
  - `build-hb/service/xcli`
  - `build-native/service/xserver`
  - `build-native/service/xcli`
  - `/home/zm/project/hummingbird/build-lite/benchmarks/hb_open_resnet_like_eval`
  - `/home/zm/project/hummingbird/build-lite/benchmarks/hb_open_resnet_like_runtime_eval`
- Added and ran `tools/run_hb_gate1_smoke.sh`.
- Final artifact directory: `results/hb_gate1_20260624_162217`.
- Artifact results:
  - Native open_resnet_like: BLOCKED, `cuda_available=false`.
  - UXSched `NATIVE`: BLOCKED, `cuda_available=false`.
  - UXSched `HB_FIXED`: BLOCKED, `cuda_available=false`.
  - HP passthrough probe: BLOCKED, `cuda_available=false`.
  - fallback probe: BLOCKED, `cuda_available=false`.
  - event-boundary probe: BLOCKED, `cuda_available=false`.
  - xserver started with HPF, accepted UXSched clients, and stopped.
- No checksum, split trace, transformed CUfunction launch evidence, or child
  completion evidence was observed because no CUDA kernel was launched.
- Global Lv1 HPF smoke was not run because single-process GPU execution did not
  pass.

2026-06-24:

- Added `AGENTS.md` with permanent UXSched-Hummingbird integration rules.
- Created `docs/codex_handoff.md`.
- No source code behavior changed in this session.
- Hummingbird repository remained read-only.

## Build Results

Passed:

```bash
cmake -S . -B build-hb -DPLATFORM_CUDA=ON -DUXSCHED_ENABLE_HB_SPLIT=ON -DBUILD_TEST=OFF -DCMAKE_INSTALL_INCLUDEDIR=include
cmake --build build-hb --target halcuda shimcuda -j2
```

Passed:

```bash
cmake -S . -B build-native -DPLATFORM_CUDA=ON -DBUILD_TEST=OFF -DCMAKE_INSTALL_INCLUDEDIR=include
cmake --build build-native --target halcuda shimcuda -j2
```

Passed:

```bash
cmake --build build-hb --target xserver xcli -j2
cmake --build build-native --target xserver xcli -j2
```

## Test Results

Compile checks passed for the current NO_XQUEUE fix:

```bash
tools/build_hb_xqueue_probe.sh build-hb/hb_xqueue_probe
cmake --build build-hb --target halcuda shimcuda -j2
cmake --build build-native --target halcuda shimcuda -j2
bash -n tools/run_hb_gate1_smoke.sh tools/build_hb_xqueue_probe.sh
```

Manual GPU runtime result currently on record:

```text
results/hb_gate1_manual_20260624_163059
```

The latest manual run with the XQueue fix reached real UXSched transformed child
launches. Gate 1 still must be manually rerun with the updated correctness/sync
runner. Passing evidence must include:

```text
uxsched_hb_no_xqueue_count=0
uxsched_hb_child_launch_count > 1
transformed CUfunction child_launch_submitted logs
child_launch_completed / parent_launch_completed logs
Native, UXSched NATIVE, and HB_FIXED checksum/output_hash equality
event_sync_pass=1
stream_sync_pass=1
context_sync_pass=1
same_stream_ordering_pass=1
parent_completion_pass=1
gate1_pass=1
```

## Fallback Behavior

`STRICT=0` returns to the original UXSched native launch path whenever splitting
is unsupported. The original module and original `CUfunction` remain available
because transformed code is loaded into a hidden module.

## Known Risks

1. The PTX transformer only rewrites recognized `mov.u32 ..., %ctaid.*`
   patterns. Kernels compiled into different PTX forms will fall back native.
2. Runtime cannot prove block independence or non-persistence; verified kernel
   names are required before splitting.
3. `extra` launch format is not split because current command ownership for
   `extra_` is raw-pointer based.

## Interface Reserved for Future Work

- split-size policy can be added before `DecomposeGrid`;
- kernel-tick can be modeled as LaunchWorker pacing after split commands exist;
- bubble detection/consolidation can be added above backend selection without
  adding a second scheduler;
- CUTLASS support requires separate PTX transformability and correctness
  validation.

## CUTLASS Realtime Benchmark Status

2026-06-25:

- CUTLASS compatibility/correctness gate: RUNTIME VERIFIED and CORRECTNESS
  VERIFIED by user GPU testing.
- Runtime API interception, fatbin/PTX extraction, CUfunction resolution,
  Runtime synchronization bridge, and Runtime metadata registration into the HB
  backend: RUNTIME VERIFIED by user GPU testing.
- CUTLASS HP/LP realtime benchmark implementation: IMPLEMENTED and COMPILE
  VERIFIED.
- Added `benchmarks/cutlass/cutlass_realtime_worker.cu` for deterministic FP32
  SIMT GEMM HP/LP roles.
- Added `tools/run_cutlass_realtime_compare.sh` for repeat=1 smoke runs of
  `standalone_hp`, `uxsched_native_hp_lp`, and `uxsched_hb_fixed_hp_lp`.
- Added `tools/summarize_cutlass_realtime_compare.py` for P50/P95/P99,
  LP throughput, and HB counter-delta summaries.
- HB_FIXED realtime measurement requires an exact verified CUTLASS kernel name
  in `benchmarks/cutlass/verified_kernel_sm120_fp32_simt.txt`; wildcard
  verification is not allowed for formal measurement.
- The current fixed split setting for the formal CUTLASS realtime experiment is
  `UXSCHED_HB_SPLIT_BLOCKS=52`.
  - Source: Hummingbird hardware formula on RTX 5060 Laptop GPU, using 26 SMs
    and 2 active CUTLASS blocks per SM for the current SM120 FP32 SIMT GEMM.
  - User repeat=3 testing found split=52 reduced HB HP P99 by 7.76% versus
    split=64 and improved LP throughput by 10.54% versus split=64.
  - This is not automatic split selection, not runtime profiling, and not a
    global optimum for other GPUs or kernels.
- repeat=1 GPU smoke: NOT TESTED in Codex, waiting for user WSL GPU run.
- repeat=5 final P99 experiment: PERFORMANCE VERIFIED for the current RTX 5060
  Laptop GPU, CUDA 12.8, SM120 FP32 SIMT CUTLASS GEMM, M=N=K=2048 setup.
  - Selected result directory:
    `results/cutlass_realtime_compare_split52_repeat5_20260625_141255`.
  - UXSched + HB_FIXED versus UXSched Native paired-repeat metrics:
    HP P99 ratio `0.5024226243`, HP P99 reduction `49.7577375653%`,
    LP throughput retention `57.2883384624%`, LP throughput loss
    `42.7116615376%`.
  - The result remains scoped to this GPU/kernel/configuration and is not a
    claim of global optimality.
- Final report and figures: generated in the result directory by
  `tools/plot_cutlass_realtime_results.py`; not committed to Git.

## CUTLASS Standalone Stability Diagnostics

2026-06-25:

- The current repeat=5 CUTLASS HP/LP result remains the formal result for
  UXSched Native versus UXSched + HB_FIXED under identical contention.
- Standalone HP is only a context view in that result. Its P99 varies strongly
  across repeats while mean and P95 remain stable.
- Added `tools/analyze_cutlass_standalone_stability.py` to analyze existing
  Standalone JSONL files without removing or smoothing outliers.
- Analysis of
  `results/cutlass_realtime_compare_split52_repeat5_20260625_141255` found:
  - abnormal Standalone repeat: `0`;
  - slowest request: repeat `0`, request `196`;
  - latency: `4907.229 us`;
  - CUDA event time: `4621.792 us`;
  - release lateness: `102 us`;
  - initial classification: `GPU_EXECUTION_JITTER`.
- Added optional runner controls for future manual stability testing:
  `--cpu-affinity`, `--pre-run-idle-sec`, `--enable-gpu-telemetry`, and
  `--telemetry-interval-sec`.
- The next recommended check is a Standalone-only 1000-request repeat=5 manual
  run with longer warmup, cooldown, pre-run idle, and GPU telemetry. That run
  must remain separate from the existing 200-request three-system result.
