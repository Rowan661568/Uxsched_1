# Codex Development Handoff

## Source of truth

The source-of-truth priority is:

1. Current Git working tree and committed source code
2. `AGENTS.md`
3. `hb_integration_status.md`
4. Runtime and test documents
5. Previous chat text

If documents conflict with code, report the conflict and follow the code.

Current source-code check: `platforms/cuda/hal/src/runtime/runtime_strategy.cpp`
shows `HB_RUNTIME` and `AUTO` return explicit Native fallback, so any document
describing them as complete conflicts with code and must be treated as stale.

## Repository state

* Repository: `/home/zm/project/UXSched`
* Current branch: `feature/hummingbird-split-backend`
* Current CUTLASS Runtime bridge work starts from:
  `157662a add CUTLASS launch compatibility probe`
* HEAD commit at the start of the current NO_XQUEUE fix:
  `d8697de record Gate 1 GPU smoke blocker`
* Baseline commit: `4146c1e add CUDA Hummingbird split backend`
* Working tree status: clean except untracked build and result directories after
  committing this documentation/test-runner update
* Untracked files:
  * `build-hb/`
  * `build-native/`
  * `results/`

## Relevant commits

Chronological order:

* `4146c1e add CUDA Hummingbird split backend`
  * Added the initial compile-verified fixed split backend under UXSched's CUDA
    hook path.
* `038347d refactor CUDA backend into runtime strategies`
  * Added `CudaRuntimeStrategy`, `NativeRuntimeStrategy`, and
    `HummingbirdRuntimeStrategy`; moved `cuLaunchKernel` backend selection out
    of `shim.cpp`.
* `310bb42 tighten HB fixed runtime fallback semantics`
  * Prevented unimplemented `HB_RUNTIME`/`AUTO` from triggering PTX transform
    at module-load time.
* `cf48dae record HB runtime GPU gate status`
  * Recorded that GPU access is blocked and Gate 1 cannot run in the current
    environment.
* `f1a9c86 document UXSched Hummingbird agent rules`
  * Added `AGENTS.md`, created this handoff file, and updated status rules.
* `f605e8d refresh Codex handoff status`
  * Refreshed the handoff before the current Gate 1 attempt.
* `d8697de record Gate 1 GPU smoke blocker`
  * Added the Gate 1 smoke runner and recorded the earlier Codex tool-session
    GPU visibility blocker.
* `157662a add CUTLASS launch compatibility probe`
  * Added the first CUTLASS SIMT FP32 GEMM launch-path probe and runner.
  * User manual GPU result showed Native and UXSched NATIVE correctness pass,
    but Runtime launch did not enter UXSched HB_FIXED.

## 2026-06-24 CUTLASS Runtime Bridge Update

Root cause from local binary/source audit:

* `build-cutlass-cu128/cutlass_launch_probe` links shared cudart:
  `libcudart.so.12 => /usr/local/cuda-12.8/lib64/libcudart.so.12`.
* The probe dynamically imports `cudaLaunchKernel`,
  `__cudaRegisterFatBinary`, `__cudaRegisterFatBinaryEnd`,
  `__cudaRegisterFunction`, and `__cudaUnregisterFatBinary`.
* `cuobjdump --list-elf` shows `sm_120` cubin, and
  `cuobjdump --dump-ptx` shows CUTLASS PTX for `sm_120`.
* Before this update, `libshimcuda.so` did not export CUDA Runtime symbols, so
  CUTLASS Runtime launches bypassed the UXSched HB_FIXED Driver path.

Implemented bridge:

* Added UXSched-owned CUDA Runtime interception inside the existing
  `libshimcuda.so`; no second hook library is introduced.
* Intercepts Runtime fatbin/function registration and `cudaLaunchKernel`.
* Intercepts Runtime stream/event/device synchronization used by the CUTLASS
  probe and routes them to existing UXSched Driver shim synchronization
  wrappers.
* Builds a host-stub to CUTLASS kernel metadata map from
  `__cudaRegisterFunction`.
* Extracts plain PTX from the Runtime fatbin, then reuses existing
  `hb_split::XModuleLoadDataEx`, `XModuleGetFunction`, and
  `TryLaunchKernelFixed`.
* `cudaLaunchKernelExC` is traced and falls back to real cudart because the
  current CUTLASS probe does not use it.
* Fallbacks call the true CUDA Runtime launch. They do not bypass XQueue, fake
  backend stats, remove verified-kernel checks, or directly launch transformed
  kernels outside UXSched.
* Added `UXSCHED_CUDART_TRACE=1` logs for fatbin registration, function
  registration, launch interception, function resolution, backend selection,
  and fallback reasons.

Build and static checks passed:

```bash
cmake -S . -B build-hb-cu128
cmake --build build-hb-cu128 --target shimcuda -j2
tools/build_cutlass_launch_probe.sh --build-dir build-cutlass-cu128 --cutlass-root /home/zm/project/cutlass --cuda-home /usr/local/cuda-12.8 --cuda-compiler /usr/local/cuda-12.8/bin/nvcc
bash -n tools/build_cutlass_launch_probe.sh
bash -n tools/run_cutlass_launch_probe.sh
git diff --check
```

Static evidence:

* `nm -D build-hb-cu128/platforms/cuda/libshimcuda.so` now exports
  `__cudaRegisterFatBinary`, `__cudaRegisterFatBinaryEnd`,
  `__cudaRegisterFunction`, `__cudaUnregisterFatBinary`,
  `cudaLaunchKernel`, `cudaLaunchKernelExC`, `cudaStreamSynchronize`,
  `cudaEventRecord`, `cudaEventSynchronize`, and related Runtime sync
  wrappers.
* `ldd build-cutlass-cu128/cutlass_launch_probe` shows shared `libcudart.so.12`.
* `readelf -Ws` / `nm -D` show the probe imports the Runtime registration and
  launch symbols dynamically.
* `cuobjdump --list-elf` shows `cutlass_launch_probe.1.sm_120.cubin`.
* `cuobjdump --dump-ptx` shows CUTLASS PTX text.

GPU validation was not run in Codex. The user must rerun
`tools/run_cutlass_launch_probe.sh` from a normal WSL GPU terminal.

## 2026-06-25 CUTLASS Runtime Metadata Bridge Update

User GPU result after `9969773` confirmed:

* Runtime API interception works: `runtime_launch_intercepted_count=1`.
* Runtime fatbin/PTX extraction works: `ptx_available=1`, `ptx_bytes=106384`.
* Runtime CUfunction resolution works: `runtime_function_resolved_count=1`.
* Runtime synchronization bridge works: `runtime_sync_intercepted_count=8`.
* Native and UXSched NATIVE CUTLASS correctness pass.
* HB_FIXED correctness fallback still produced correct output, but the HB
  backend did not transform because it saw
  `function=<unknown> reason=PTX_UNAVAILABLE`.

Root cause:

* The HB backend registry is `g_modules` and `g_functions` in
  `platforms/cuda/hal/src/hb_split/backend.cpp`.
* Driver API module load wrappers register `CUmodule -> ModuleInfo`; Driver API
  `cuModuleGetFunction` registers `CUfunction -> FunctionInfo` only when the
  requested kernel name exists in `ModuleInfo.records`.
* Runtime bridge could load a module and resolve a `CUfunction`, but did not
  explicitly assert/register `CUmodule -> PTX` and `CUfunction -> kernel name`
  metadata into the HB registry before calling `TryLaunchKernelFixed`.
* `XModuleGetFunction` returns the real Driver result even if metadata was not
  registered, so Runtime bridge logged function resolution success while the HB
  backend later failed lookup.

Fix:

* Added public internal metadata APIs in `hb_split`:
  `RegisterModuleMetadata`, `RegisterFunctionMetadata`,
  `UnregisterModuleMetadata`, and `LookupFunctionMetadata`.
* Driver wrappers and Runtime bridge now share the same registration helpers.
* Runtime bridge now loads Runtime PTX with the real Driver API, then explicitly
  registers module metadata and function metadata before HB_FIXED dispatch.
* HB metadata owns a PTX string copy for the module lifetime; Runtime fatbin
  temporary buffers are not referenced by HB registry.
* Runtime unregister unloads modules via `XModuleUnload`, which removes original
  function mappings, module metadata, and hidden transformed modules.
* Added Runtime logs:
  `runtime_hb_module_registered` and `runtime_hb_function_registered`.
* Added CPU-side fake-handle registry probe:
  `tools/hb_metadata_registry_probe.cpp` and
  `tools/build_hb_metadata_registry_probe.sh`.

Local non-GPU checks passed:

```bash
cmake --build build-hb-cu128 --target halcuda shimcuda -j2
tools/build_hb_metadata_registry_probe.sh build-hb-cu128
build-hb-cu128/hb_metadata_registry_probe
bash -n tools/build_hb_metadata_registry_probe.sh
bash -n tools/build_cutlass_launch_probe.sh
bash -n tools/run_cutlass_launch_probe.sh
```

## Implemented and verified

| Item | Implemented | Compile verified | Runtime verified | Correctness verified | Performance verified |
| --- | --- | --- | --- | --- | --- |
| `UXSCHED_ENABLE_HB_SPLIT` CMake option | yes | yes | no | no | no |
| UXSched-only CUDA hook path | yes | yes | no | no | no |
| `CudaRuntimeStrategy` interface | yes | yes | no | no | no |
| `NativeRuntimeStrategy` | yes | yes | no | no | no |
| `HummingbirdRuntimeStrategy` shell | yes | yes | no | no | no |
| `NATIVE` strategy path | yes | yes | no | no | no |
| `HB_FIXED` strategy path | yes | yes | yes | no | no |
| `HB_RUNTIME` mode | fallback only | yes | no | no | no |
| `AUTO` mode | fallback only | yes | no | no | no |
| PTX transformation | yes | yes | no | no | no |
| Hidden transformed module | yes | yes | no | no | no |
| Fixed grid decomposition | yes | yes | no | no | no |
| `SplitCommandGroup` child tracking | yes | yes | partial | no | no |
| Per-XQueue threshold change to `1,1` | yes | yes | no | no | no |
| Gate 1 smoke runner | yes | yes | partial | no | no |
| CUDA stream to XQueue trace and auto-association | yes | yes | yes | no | no |
| GPU runtime benchmarks | no | no | no | no | no |

Compilation success is not runtime verification. No performance numbers are
claimed.

## Current known state

* NativeRuntimeStrategy:
  * COMPILE VERIFIED.
  * Source: `platforms/cuda/hal/src/runtime/runtime_strategy.cpp`,
    `NativeRuntimeStrategy::SubmitKernel`.
  * Builds a `CudaKernelLaunchCommand` and either direct-launches or submits to
    XQueue.
* HummingbirdRuntimeStrategy:
  * COMPILE VERIFIED.
  * Source: `platforms/cuda/hal/src/runtime/runtime_strategy.cpp`,
    `HummingbirdRuntimeStrategy::SubmitKernel`.
  * `HB_FIXED` delegates to `hb_split::TryLaunchKernelFixed`.
  * `HB_RUNTIME` and `AUTO` explicitly fallback Native.
* HB_FIXED:
  * RUNTIME VERIFIED for real transformed child launches; Gate 1 remains
    FAIL/PARTIAL pending correctness and synchronization rerun.
  * Source: `platforms/cuda/hal/src/hb_split/backend.cpp`,
    `TryLaunchKernelFixed`, `SubmitSplitCommands`.
  * The manual GPU run in
    `results/hb_gate1_after_xqueue_fix_20260624_170107` observed transformed
    module load, 78 parent launches, 312 transformed child launches,
    fallback count 0, and `NO_XQUEUE` count 0.
* HB_RUNTIME:
  * IMPLEMENTED as Native fallback only.
  * Source: `HummingbirdRuntimeStrategy::SubmitKernel`, fallback reason
    `HB_RUNTIME_NOT_IMPLEMENTED_YET`.
* AUTO:
  * IMPLEMENTED as Native fallback only.
  * Source: `HummingbirdRuntimeStrategy::SubmitKernel`, fallback reason
    `AUTO_RUNTIME_COORDINATOR_UNAVAILABLE`.
* PTX transformation:
  * COMPILE VERIFIED, NOT TESTED on GPU.
  * Source: `TransformKernelPtx`, `TransformModulePtx`.
* hidden transformed module:
  * COMPILE VERIFIED, NOT TESTED on GPU.
  * Source: `TransformModulePtx`, `XModuleGetFunction`, `XModuleUnload`.
* grid decomposition:
  * COMPILE VERIFIED, NOT TESTED on GPU.
  * Source: `DecomposeGrid`, `DecomposeBox`.
* SplitCommandGroup:
  * RUNTIME VERIFIED for child/parent completion log emission in the latest
    manual split run; synchronization semantics still need dedicated Gate 1
    probes.
  * Source: `SplitCommandGroup` and state listener inside
    `SubmitSplitCommands`.
  * It tracks child completion and clears child ownership; it is not an
    application-visible parent completion primitive.
* per-XQueue threshold:
  * COMPILE VERIFIED, NOT TESTED on GPU.
  * Source: `SetLpSplitThresholdOnce`.
* HP passthrough:
  * RUNTIME VERIFIED for selection logging in the latest manual run, but the
    updated runner still must verify transformed/child counts are zero and the
    process exits without CUDA/runtime errors.
  * Source: `TryLaunchKernelFixed`, `IsLowPriority`, log reason
    `HIGH_PRIORITY_PASSTHROUGH`.
* fallback:
  * RUNTIME VERIFIED for minimal Driver API fallback probes:
    `PTX_UNAVAILABLE` and `KERNEL_NOT_VERIFIED` both exited normally in
    `results/hb_gate1_after_xqueue_fix_20260624_170107`.
  * Sources: `TryLaunchKernelFixed`, `SubmitKernelWithRuntimeStrategy`.
  * PTX-present but unverified kernels now register `FunctionInfo` and fall back
    as `KERNEL_NOT_VERIFIED` instead of `<unknown>/PTX_UNAVAILABLE`.
* Gate 1 smoke runner:
  * IMPLEMENTED as `tools/run_hb_gate1_smoke.sh`.
  * Saves per-case `command.txt`, `env.txt`, stdout/stderr, JSONL, return code,
    checksum extraction, split trace extraction, transformed launch evidence,
    child completion evidence, parent completion evidence, and xserver logs.
  * Writes UXSched backend stats separately from workload-internal split stats.
  * Now runs Native, UXSched NATIVE, and UXSched HB_FIXED correctness-mode cases;
    default/explicit Driver API XQueue probes; event, stream, context,
    same-stream ordering, and parent-completion sync probes; HP passthrough; and
    separate `PTX_UNAVAILABLE` / `KERNEL_NOT_VERIFIED` fallback probes.
  * Writes `gate1_summary.env`.
* profiler:
  * BLOCKED, not implemented.
* kernel-tick:
  * BLOCKED, not implemented.
* small-bubble detection:
  * BLOCKED, not implemented.
* large-bubble consolidation:
  * BLOCKED, not implemented.
* CUTLASS workload:
  * NOT TESTED, not implemented.
  * Design/audit plan added in `docs/cutlass_realtime_benchmark_plan.md`.
  * CUTLASS correctness is mandatory before any P99 performance claim.

## Known gaps and risks

* `platforms/cuda/hal/src/hb_split/backend.cpp::TransformKernelPtx`
  * Rewrites only recognized `mov.u32 ..., %ctaid.x/y/z` PTX forms.
  * Does not prove full block independence.
* `platforms/cuda/hal/src/hb_split/backend.cpp::ContainsCrossBlockSync`
  * Detects only a few grid-level synchronization tokens:
    `grid.sync`, `griddepcontrol`, `barrier.cluster`.
* `platforms/cuda/hal/src/hb_split/backend.cpp::SubmitSplitCommands`
  * Submits all child commands immediately to XQueue.
  * No kernel-tick or bubble-aware pacing exists.
* `platforms/cuda/hal/src/hb_split/backend.cpp::SplitCommandGroup`
  * Tracks child completion via listeners but does not integrate as a parent
    command in XQueue.
* `platforms/cuda/hal/src/runtime/runtime_strategy.cpp::HummingbirdRuntimeStrategy::SubmitKernel`
  * `HB_RUNTIME` and `AUTO` are explicit Native fallbacks.
* `platforms/cuda/shim/src/shim.cpp::XLaunchKernelEx`
  * `cuLaunchKernelEx` remains native and is not split.
* `platforms/cuda/hal/src/hb_split/backend.cpp::TryLaunchKernelFixed`
  * `extra` launch format falls back Native.
* `platforms/cuda/hal/src/hb_split/backend.cpp::SetLpSplitThresholdOnce`
  * Threshold is applied per XQueue after first split attempt and has not been
    GPU validated under Global HPF.

## Current blocker

Blocker type: old open_resnet correctness and synchronization evidence is
incomplete, but HB_FIXED runtime split execution is verified and correctness is
deferred to CUTLASS workload validation.

Latest manual result directory:

```text
results/hb_gate1_after_xqueue_fix_20260624_170107
```

Observed latest manual GPU result:

* GPU was accessible in the user's normal WSL session.
* Default-stream and explicit-stream Driver API probes reached `HB_SPLIT`.
* `uxsched_hb_fixed_lp` recorded `uxsched_hb_transform_count=4`,
  `uxsched_hb_parent_launch_count=78`, `uxsched_hb_child_launch_count=312`,
  `uxsched_hb_transformed_launch_count=312`, `uxsched_hb_fallback_count=0`, and
  `uxsched_hb_no_xqueue_count=0`.
* `split_group_completed` and `parent_launch_completed` were observed.
* `sync_event_boundary_probe` is not valid Gate 1 sync evidence because its
  UXSched transform/parent/child/transformed counts were all zero; its
  `fixed_split_blocks=16` and `lp_split_launched=1402` fields were workload
  internal counters.
* Existing ordinary open_resnet_like UXSched cases did not emit checksum,
  output hash, or output element count and returned 139 in that artifact set.
* Current decision: do not block CUTLASS planning on the old
  open_resnet_like correctness runner; CUTLASS must provide its own correctness
  and synchronization validation.

## Next task

The next gated task is CUTLASS Phase 1 implementation: add the minimal CUTLASS
GEMM workload with deterministic inputs, CPU reference, error metrics, output
hash, and single-process correctness output. Do not run or claim P99 until
CUTLASS correctness and split evidence pass.

Do not start the complete Hummingbird runtime, coordinator, profiler,
kernel-tick, bubble detection, or consolidation. Do not claim CUTLASS P99
results until CUTLASS correctness passes and LP transformed child launches are
observed.

## Exact commands

Build commands:

```bash
cd /home/zm/project/UXSched
cmake -S . -B build-hb \
  -DPLATFORM_CUDA=ON \
  -DUXSCHED_ENABLE_HB_SPLIT=ON \
  -DBUILD_TEST=OFF \
  -DCMAKE_INSTALL_INCLUDEDIR=include
cmake --build build-hb --target halcuda shimcuda -j2
cmake -S . -B build-native \
  -DPLATFORM_CUDA=ON \
  -DBUILD_TEST=OFF \
  -DCMAKE_INSTALL_INCLUDEDIR=include
cmake --build build-native --target halcuda shimcuda -j2
```

Last run in this handoff session:

```text
build-hb: Built target halcuda; Built target shimcuda
build-native: Built target halcuda; Built target shimcuda
```

Additional service targets built for Global Lv1 smoke preparation:

```bash
cmake --build build-hb --target xserver xcli -j2
cmake --build build-native --target xserver xcli -j2
```

Confirmed paths:

```text
/home/zm/project/UXSched/build-hb/platforms/cuda/libhalcuda.so
/home/zm/project/UXSched/build-hb/platforms/cuda/libshimcuda.so
/home/zm/project/UXSched/build-native/platforms/cuda/libhalcuda.so
/home/zm/project/UXSched/build-native/platforms/cuda/libshimcuda.so
/home/zm/project/UXSched/build-hb/service/xserver
/home/zm/project/UXSched/build-hb/service/xcli
/home/zm/project/UXSched/build-native/service/xserver
/home/zm/project/UXSched/build-native/service/xcli
/home/zm/project/hummingbird/build-lite/benchmarks/hb_open_resnet_like_eval
/home/zm/project/hummingbird/build-lite/benchmarks/hb_open_resnet_like_runtime_eval
/home/zm/project/UXSched/build-hb/hb_xqueue_probe
```

Minimal manual retest after this fix:

```bash
cd /home/zm/project/UXSched
tools/build_hb_xqueue_probe.sh build-hb/hb_xqueue_probe

env -u LD_PRELOAD -u XSCHED_POLICY build-hb/service/xserver HPF 50000 \
  > /tmp/uxsched_hb_gate1_xserver.out 2> /tmp/uxsched_hb_gate1_xserver.err &
XSERVER_PID=$!

env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
  LD_LIBRARY_PATH=/home/zm/project/UXSched/build-hb/platforms/cuda:/home/zm/project/UXSched/build-hb/preempt:/usr/lib/wsl/lib \
  LD_PRELOAD=/home/zm/project/UXSched/build-hb/platforms/cuda/libshimcuda.so \
  XSCHED_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1 \
  CUXTRA_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1 \
  XSCHED_SCHEDULER=GLB \
  XSCHED_AUTO_XQUEUE=ON \
  XSCHED_AUTO_XQUEUE_LEVEL=1 \
  XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
  UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
  UXSCHED_XQUEUE_TRACE=1 \
  UXSCHED_HB_SPLIT_BLOCKS=512 \
  UXSCHED_HB_STRICT=0 \
  UXSCHED_HB_VERIFIED_KERNELS=hb_xqueue_probe_kernel \
  build-hb/hb_xqueue_probe --stream default --blocks 1024 --threads 1

env -u XSCHED_POLICY -u HB_TASK_PRIORITY \
  LD_LIBRARY_PATH=/home/zm/project/UXSched/build-hb/platforms/cuda:/home/zm/project/UXSched/build-hb/preempt:/usr/lib/wsl/lib \
  LD_PRELOAD=/home/zm/project/UXSched/build-hb/platforms/cuda/libshimcuda.so \
  XSCHED_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1 \
  CUXTRA_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1 \
  XSCHED_SCHEDULER=GLB \
  XSCHED_AUTO_XQUEUE=ON \
  XSCHED_AUTO_XQUEUE_LEVEL=1 \
  XSCHED_AUTO_XQUEUE_PRIORITY=-10 \
  UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED \
  UXSCHED_XQUEUE_TRACE=1 \
  UXSCHED_HB_SPLIT_BLOCKS=512 \
  UXSCHED_HB_STRICT=0 \
  UXSCHED_HB_VERIFIED_KERNELS=hb_xqueue_probe_kernel \
  build-hb/hb_xqueue_probe --stream explicit --blocks 1024 --threads 1

kill "${XSERVER_PID}"
wait "${XSERVER_PID}" 2>/dev/null || true
```

Reusable Gate 1 wrapper:

```bash
cd /home/zm/project/UXSched
bash tools/run_hb_gate1_smoke.sh --output-dir results/hb_gate1_<timestamp>
```

The runner now produces these primary cases:

```text
native_correctness
uxsched_native_correctness
uxsched_hb_fixed_correctness
sync_event_hb_fixed_probe
sync_stream_hb_fixed_probe
sync_context_hb_fixed_probe
sync_same_stream_ordering_hb_fixed_probe
sync_parent_completion_hb_fixed_probe
probe_hb_fixed_hp_passthrough
probe_fallback_ptx_unavailable
probe_fallback_kernel_not_verified
```

Historical artifact directory that exposed the original NO_XQUEUE blocker:

```text
results/hb_gate1_manual_20260624_163059
```

Manual artifact status:

```text
native_open_resnet_like_lp: RAN on GPU
uxsched_native_lp: RAN on GPU
uxsched_hb_fixed_lp: RAN but fell back Native with reason=NO_XQUEUE
uxsched_hb_fixed_hp_passthrough: RAN with HIGH_PRIORITY_PASSTHROUGH
sync_event_boundary_probe: workload-internal split counters only; no UXSched split trace
xserver: started with HPF, accepted clients, then stopped
```

Latest artifact directory after the XQueue fix:

```text
results/hb_gate1_after_xqueue_fix_20260624_170107
```

Latest artifact status:

```text
default stream probe: RAN with transformed child launches and NO_XQUEUE=0
explicit stream probe: RAN with transformed child launches and NO_XQUEUE=0
uxsched_hb_fixed_lp: RAN far enough to record 312 transformed child launches
sync_event_boundary_probe: invalid Gate 1 sync evidence, no UXSched split trace
ordinary open_resnet_like UXSched cases: no checksum/hash/count evidence and returned 139
```

Gate 1 remains FAIL/PARTIAL until a fresh manual run observes
`gate1_pass=1` in `gate1_summary.env`.

## Expected outputs

Expected build output:

```text
Built target halcuda
Built target shimcuda
```

Expected runtime logs for `HB_FIXED` Gate 1:

* `[UXSCHED-HB] transform_succeeded function=<kernel>`
* `[UXSCHED-HB] transformed_module_loaded transformed_module=<...>`
* `[UXSCHED-HB] backend_selected=HB_SPLIT`
* `[UXSCHED-HB] split_blocks=512`
* `[UXSCHED-HB] split_count=<N>` where `N > 1`
* `[UXSCHED-HB] child_launch_submitted ... transformed_function=<...>`
* `[UXSCHED-HB] child_launch_completed ...`
* `[UXSCHED-HB] xqueue=<...> lp_in_flight_threshold=1 batch_size=1`
* `[UXSCHED-XQUEUE] ... auto_create_attempted=1 create_result=0 ...`
* HP run logs include `HIGH_PRIORITY_PASSTHROUGH`
* Unsupported kernel logs include `backend_selected=NATIVE reason=<reason>`

Expected PASS conditions:

* Native, UXSched NATIVE, and `HB_FIXED` checksums/output hashes/element counts
  match.
* LP produces more than one real split launch.
* The transformed `CUfunction` is actually submitted.
* HP kernels are not split.
* Native fallback path completes successfully.
* Event, stream, context/device, same-stream ordering, and parent-completion
  synchronization checks pass.
* No local scheduler fallback is observed.

Expected result locations should be created by the chosen smoke script, not by
this handoff file. Do not claim numeric performance until repeat runs exist.

## Do not do next

Do not begin any of the following until CUTLASS Phase 1-4 correctness and sync
gates pass:

* per-device Hummingbird runtime coordinator;
* Hummingbird runtime state machine;
* kernel profiler or automatic `SplitPlan`;
* kernel-tick LP launcher;
* small-bubble detection or explicit bubble hint API;
* large-bubble detection;
* split-kernel consolidation;
* performance claims or repeat=3 benchmark summaries.

CUTLASS Phase 1 implementation is now allowed, but only as a correctness-first
GEMM workload. Do not download CUTLASS automatically; use a user-provided
`CUTLASS_ROOT` or a separately approved submodule step.

## Latest session update

2026-06-24 CUTLASS launch compatibility probe:

* External CUTLASS source is available at `/home/zm/project/cutlass`.
* CUTLASS revision used for the probe build: `ad7b2f5`.
* CUDA toolkit for this phase is `/usr/local/cuda-12.8`; `nvcc` is
  `/usr/local/cuda-12.8/bin/nvcc`, release `12.8, V12.8.93`.
* The phase uses native SM120 only:
  `CMAKE_CUDA_ARCHITECTURES=120` and `CUTLASS_MODE=NATIVE_SM120`.
* The older Forward PTX compatibility path is canceled.
* Added a single-process CUTLASS SIMT FP32 GEMM launch-path probe:
  `benchmarks/cutlass/cutlass_launch_probe.cu`.
* Added a standalone CUTLASS CMake project under `benchmarks/cutlass`.
* Added:
  * `tools/build_cutlass_launch_probe.sh`
  * `tools/run_cutlass_launch_probe.sh`
* Runtime mode uses standard CUTLASS device GEMM. Its audited launch path is
  `Gemm::run(stream)` to CUTLASS kernel launch to CUDA Runtime launch; UXSched
  can only split it if libcudart resolves to intercepted Driver API and module
  PTX/function metadata is visible.
* Driver / `CudaHostAdapter` mode is explicitly blocked for this probe because
  CUTLASS does not provide a generic official adapter that supplies `CUmodule`,
  `CUfunction`, kernel parameter layout, dynamic shared memory, and stream for
  this GEMM.
* Passed non-GPU checks:
  * `bash -n tools/build_cutlass_launch_probe.sh`
  * `bash -n tools/run_cutlass_launch_probe.sh`
  * `tools/build_cutlass_launch_probe.sh --build-dir build-cutlass-cu128 ...`
  * `cmake -S . -B build-hb-cu128 ...`
  * `cmake --build build-hb-cu128 --target halcuda shimcuda xserver xcli -j2`
  * `build-cutlass-cu128/cutlass_launch_probe --mode driver ...` returned the
    expected `cutlass_driver_launch_integration_blocked` JSON.
* Codex did not run Runtime GPU cases, did not run HP/LP realtime benchmarks,
  and did not make P99 claims.

2026-06-24 CUTLASS realtime benchmark audit/design:

* Audited `benchmarks/realtime_inference_latency.py` and its direct xserver,
  xcli, process, environment, logging, CSV/JSON, and percentile paths.
* Confirmed current HP workload is PyTorch/TorchVision ResNet50 inference and
  current LP workload is PyTorch/TorchVision MobileNetV2 training.
* Confirmed HP and LP are independent child processes, both create explicit
  PyTorch CUDA streams, and the current HP request loop is back-to-back rather
  than fixed-period.
* Confirmed existing script uses priorities `1` and `0`; CUTLASS design switches
  to HP `10` and LP `-10`.
* Confirmed no CUTLASS source/submodule is present locally; CUDA toolkit and
  `nvcc 12.0` are present.
* Confirmed Hummingbird has optional local CUTLASS include detection but no
  downloaded CUTLASS dependency.
* Added `docs/cutlass_realtime_benchmark_plan.md`.
* Old open_resnet correctness remains deferred; CUTLASS correctness is the new
  mandatory correctness gate before any P99 claim.
* No implementation, download, scheduler change, GPU benchmark, or performance
  claim was made in this session.

2026-06-24 Gate 1 correctness/synchronization runner update:

* Read `results/hb_gate1_after_xqueue_fix_20260624_170107`.
* Confirmed the XQueue association fix reached real HB_FIXED split launch on
  GPU, including transformed module load, transformed child submissions, child
  completions, and parent completions.
* Confirmed the remaining Gate 1 gap is correctness and synchronization
  evidence, not stream-to-XQueue association.
* Extended `tools/hb_xqueue_probe.cpp` with sync modes:
  `event`, `stream`, `context`, `same-stream`, and `parent`.
* Updated `tools/run_hb_gate1_smoke.sh` to generate full Gate 1 artifacts and
  `gate1_summary.env`.
* Correctness workload is `hb_open_resnet_like_runtime_eval` in
  `--correctness-mode lp-only` with deterministic input/weight initialization,
  fixed dimensions, one correctness iteration, and
  `lp-correctness-sync-boundary=iteration`.
* Workload correctness split blocks are set to `4096`; UXSched HB_FIXED split
  blocks remain `512`, so UXSched backend stats are the split evidence.
* Passed local non-GPU checks:
  * `tools/build_hb_xqueue_probe.sh build-hb/hb_xqueue_probe`
  * `cmake --build build-hb --target halcuda shimcuda -j2`
  * `cmake --build build-native --target halcuda shimcuda -j2`
  * `bash -n tools/run_hb_gate1_smoke.sh tools/build_hb_xqueue_probe.sh`
  * `git diff --check`
* GPU runtime validation was not run in Codex.

## Session completion checklist

2026-06-25 CUTLASS repeat=5 final result plotting:

* Selected final result directory:
  `results/cutlass_realtime_compare_split52_repeat5_20260625_141255`.
* The selected directory contains `summary.csv`, `comparison.csv`,
  `metadata.env`, repeats 0-4, and all three systems:
  `standalone_hp`, `uxsched_native_hp_lp`, and `uxsched_hb_fixed_hp_lp`.
* All 15 case rows are `COMPLETE`, all correctness checks passed, and each row
  has `hp_count=200`.
* HB_FIXED Gate passed for all five repeats:
  * `runtime_hb_metadata_bridge_pass=1`;
  * warmup transform count was 1 and measurement transform delta was 0;
  * every LP parent had exactly 6 child launches;
  * fallback count and `NO_XQUEUE` count were 0;
  * HP transform count was 0;
  * Global HPF log checks passed and local fallback count was 0.
* Added `tools/plot_cutlass_realtime_results.py`.
  * Reads `summary.csv`, `comparison.csv`, and HB backend stats.
  * Performs Gate checks before plotting.
  * Uses matplotlib without pandas or seaborn.
  * Generates PNG/PDF/SVG figures, `final_metrics.csv`,
    `final_derived_metrics.csv`, `final_metrics.md`, and `final_report.md`
    under the result directory.
* Generated final artifacts in the result directory, but did not add the
  `results/` tree to Git.
* Core paired-repeat metrics from CSV:
  * HP P99 ratio: 0.5024226243, reduction: 49.7577375653%;
  * HP P95 ratio: 0.4999064297, reduction: 50.0093570332%;
  * HP mean ratio: 0.5764918472, reduction: 42.3508152788%;
  * LP throughput ratio: 0.5728833846, retention: 57.2883384624%, loss:
    42.7116615376%.
* No GPU benchmark was run in Codex.

2026-06-25 CUTLASS fixed split and summary formatting update:

* User GPU testing compared `UXSCHED_HB_SPLIT_BLOCKS=52` and `64` with
  repeat=3 on RTX 5060 Laptop GPU for the SM120 FP32 SIMT CUTLASS GEMM kernel.
* Resource model:
  * 26 SMs;
  * 256 CUTLASS threads/block;
  * 128 registers/thread;
  * register limit gives 2 active CUTLASS blocks per SM;
  * Hummingbird hardware formula gives `26 * 2 = 52` blocks.
* Measured split=52 versus split=64:
  * HB HP P99 lowered by 7.76%;
  * HB LP throughput improved by 10.54%;
  * split=52 HB HP P99 range and LP throughput range did not overlap with
    split=64 in the reported repeat=3 runs.
* Updated the CUTLASS realtime benchmark default split blocks to `52`.
* This is a fixed configuration for the current hardware/kernel, not automatic
  split selection, not runtime profiling, and not a global optimum.
* Updated `tools/summarize_cutlass_realtime_compare.py` so `comparison.csv`
  includes per-repeat paired ratio rows with repeat IDs and aggregate rows over
  paired repeat ratios.
* Added `hp_p99_reduction_pct`, `lp_throughput_retention_pct`, and
  `lp_throughput_loss_pct` ratio metrics.
* Updated `tools/run_cutlass_realtime_compare.sh` with deterministic repeat
  case-order rotation and configurable cooldown, default 5 seconds.
* Codex did not run the real GPU benchmark.

2026-06-25 CUTLASS realtime HP/LP benchmark implementation:

* Added `benchmarks/cutlass/cutlass_realtime_worker.cu`.
  * Uses the same conservative FP32 SIMT SM120 CUTLASS GEMM configuration as
    the compatibility probe.
  * Creates one explicit CUDA stream, allocates/copies buffers once, initializes
    the CUTLASS GEMM object once, warms up before measurement, and emits JSONL.
  * HP mode schedules requests against absolute monotonic release times.
  * LP mode submits the same GEMM until duration or fixed request count.
  * Correctness is checked outside the formal measurement window and records
    checksum, output hash, element count, max abs/rel error, mismatch, NaN, and
    Inf counts.
  * File-based ready/start barriers prevent fixed-sleep benchmark start.
* Added `tools/run_cutlass_realtime_compare.sh`.
  * Supports `standalone_hp`, `uxsched_native_hp_lp`, and
    `uxsched_hb_fixed_hp_lp`.
  * Uses one worker binary for Native and HB_FIXED systems.
  * Starts UXSched Global HPF xserver for HP/LP systems.
  * Uses HP priority `10`, LP priority `-10`, `XSCHED_SCHEDULER=GLB`,
    `XSCHED_AUTO_XQUEUE=ON`, and the single UXSched CUDA shim.
  * Requires an exact HB verified-kernel allowlist for HB_FIXED; it does not use
    wildcard verification for realtime measurement.
  * Records counter snapshots around `UXSCHED_CUTLASS_PHASE=MEASUREMENT_START`
    markers and computes measurement deltas.
* Added `tools/summarize_cutlass_realtime_compare.py`.
  * Generates `summary.csv` and `comparison.csv` without pandas.
  * Uses sorted linear interpolation for P50/P95/P99 to match the original
    realtime benchmark semantics.
* Added `benchmarks/cutlass/verified_kernel_sm120_fp32_simt.txt` as a strict
  allowlist template. The user must fill the exact discovered CUTLASS kernel
  name before running the HB_FIXED realtime smoke.
* Static validation passed:
  * `bash -n tools/run_cutlass_realtime_compare.sh`
  * `python3 -m py_compile tools/summarize_cutlass_realtime_compare.py`
  * `python3 tools/summarize_cutlass_realtime_compare.py --self-test`
  * `cmake --build build-hb-cu128 --target halcuda shimcuda xserver -j2`
  * `tools/build_cutlass_launch_probe.sh --build-dir build-cutlass-cu128 ...`
* Codex did not run the real GPU realtime benchmark and made no P99 performance
  claim.

2026-06-25 project rule update:

* Updated `AGENTS.md` with the final CUTLASS realtime benchmark objective and
  mandatory proof requirements before any HP P99 improvement claim.
* The rule requires fair comparison between UXSched Global HPF + CUTLASS +
  `NATIVE` and UXSched Global HPF + CUTLASS + Hummingbird `HB_FIXED` under the
  same GPU, kernel, input scale, request model, scheduler, and measurement
  boundary.
* P99 claims require CUTLASS correctness, equal work, real HB transform/parent/
  child counts, no Native fallback, no `NO_XQUEUE`, correct split-group and
  Runtime synchronization semantics, parent completion after all children, and
  at least `repeat=3`.

Before ending a session:

1. Run `git status`.
2. Run build checks.
3. Run available tests.
4. Update this file.
5. Update `hb_integration_status.md`.
6. Commit relevant source and documentation changes separately.
7. Print the next manual command for the user.

2026-06-25 CUTLASS standalone HP stability diagnostics:

* Added `tools/analyze_cutlass_standalone_stability.py`.
  * Reads existing `standalone_hp/repeat_*/hp/output.jsonl` files.
  * Computes repeat-level mean/P50/P95/P99/max, release-lateness percentiles,
    and GPU event percentiles with the same sorted linear interpolation method
    used by the realtime summarizer.
  * Lists the slowest requests per repeat without dropping, smoothing, or
    modifying any outlier.
  * Classifies slow requests as `HOST_RELEASE_JITTER`,
    `HOST_COMPLETION_JITTER`, `GPU_EXECUTION_JITTER`, or
    `INSUFFICIENT_STEADY_STATE_WARMUP` using host latency, release lateness,
    request index, and CUDA event time.
* Generated analysis for
  `results/cutlass_realtime_compare_split52_repeat5_20260625_141255`:
  * `standalone_stability_analysis.md`
  * `standalone_stability_summary.csv`
  * `standalone_outliers.csv`
* Current diagnosis of the existing repeat=5 result:
  * abnormal Standalone P99 repeat is repeat `0`;
  * slowest request is repeat `0`, request `196`;
  * latency is `4907.229 us`, CUDA event time is `4621.792 us`, and release
    lateness is `102 us`;
  * initial classification is `GPU_EXECUTION_JITTER`, not HP release drift.
* Updated `tools/run_cutlass_realtime_compare.sh` with optional stability
  controls:
  * `--cpu-affinity <list>` records requested/effective affinity and applies it
    to the runner, workers, and telemetry helper when requested;
  * `--pre-run-idle-sec <seconds>` adds explicit idle time before each case;
  * `--enable-gpu-telemetry` and `--telemetry-interval-sec <seconds>` record
    `nvidia-smi` P-state, clocks, temperature, power, and utilization to
    per-case `gpu_telemetry.csv` files when supported by the driver;
  * metadata now includes an environment snapshot with WSL/Linux, CUDA, GPU,
    CPU, load, and governor information where available.
* Codex did not run any real GPU benchmark and did not modify the original
  result data.
