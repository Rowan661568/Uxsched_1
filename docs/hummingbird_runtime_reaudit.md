# Hummingbird Runtime Re-audit of 4146c1e

Baseline commit audited:

```text
4146c1e47b21f2a2f6f38a147894c0d979ec85c0 add CUDA Hummingbird split backend
```

This audit is based on source inspection, not on the previous design documents.
Hummingbird remains read-only.

## 1. Real Implementations

| Area | Status | Source Evidence | Notes |
| --- | --- | --- | --- |
| CUDA hook remains UXSched-only | IMPLEMENTED | `platforms/cuda/shim/src/intercept.cpp` routes module load/get/unload and `cuLaunchKernel` through UXSched shim symbols. | No second Hummingbird LD_PRELOAD was added. |
| Runtime switch | COMPILE VERIFIED | `platforms/cuda/hal/src/hb_split/backend.cpp`, `BackendModeFromEnv()` reads `UXSCHED_CUDA_PREEMPT_BACKEND`. | Supports old names `NATIVE`, `HB_SPLIT`, `AUTO`; not the new `UXSCHED_CUDA_RUNTIME_STRATEGY` names yet. |
| PTX detection | COMPILE VERIFIED | `LooksLikePtxText()` in `backend.cpp`. | Heuristic scan only; runtime not yet validated. |
| PTX transformation | COMPILE VERIFIED | `TransformKernelPtx()`, `InjectOffsetParams()`, `RewriteAxisMov()` in `backend.cpp`. | Rewrites recognized `mov.u32 ..., %ctaid.x/y/z` only. It can fail for other PTX forms. |
| Hidden transformed module | COMPILE VERIFIED | `TransformModulePtx()` loads transformed PTX via `Driver::ModuleLoadDataEx`; `ModuleInfo::transformed_module`. | Original module/function remains visible to the application for Native fallback. |
| Original to transformed function cache | COMPILE VERIFIED | `XModuleGetFunction()` stores `g_functions[*function]`. | Function mapping is process-local. |
| LP-only split gate | COMPILE VERIFIED | `CurrentPriority()`, `IsLowPriority()`, `TryLaunchKernel()`. | Priority is read from env, mainly `XSCHED_AUTO_XQUEUE_PRIORITY` or `UXSCHED_HB_PRIORITY`; it is not read from Global Scheduler state. |
| HP passthrough | COMPILE VERIFIED | `TryLaunchKernel()` returns false and logs `HIGH_PRIORITY_PASSTHROUGH` when priority is non-negative. | Runtime not yet GPU verified. |
| Fixed split size | COMPILE VERIFIED | `SplitBlocks()` defaults to `UXSCHED_HB_SPLIT_BLOCKS=512`; `DecomposeGrid()`. | No automatic split-size logic. |
| Split command creation | COMPILE VERIFIED | `SubmitSplitCommands()` creates `CudaKernelLaunchCommand` children with transformed function and appended offsets. | Relies on `CudaKernelCommand` deep-copy for parameter ownership. |
| LP threshold=1 | COMPILE VERIFIED | `SetLpSplitThresholdOnce()` calls `xqueue->SetLaunchConfig(1, 1)`. | This is real code. `AsyncXQueue::SetLaunchConfig()` waits all before changing config and `LaunchWorker` uses the threshold. |
| Module unload wait | COMPILE VERIFIED | `XModuleUnload()` calls `XQueueManager::ForEachWaitAll()` before unloading hidden transformed module. | Broad multi-thread unload stress is not tested. |

## 2. Compile Skeletons or Partial Implementations

| Area | Status | Source Evidence | Gap |
| --- | --- | --- | --- |
| RuntimeStrategy abstraction | BLOCKED | No `CudaRuntimeStrategy`, `NativeRuntimeStrategy`, or `HummingbirdRuntimeStrategy` exists in 4146c1e. | Existing logic is still called directly from `shim.cpp` into `hb_split::TryLaunchKernel()`. |
| HummingbirdRuntime queue | BLOCKED | No runtime-owned LP pending queue or tick thread exists. | Split children are submitted immediately to XQueue. |
| Device shared HB state | BLOCKED | No `DeviceHbRuntimeState` or shared memory/IPC state exists. | Global Scheduler does not expose HP pending state to LP runtime. |
| State machine | BLOCKED | No IDLE/HP_ACTIVE/SMALL_BUBBLE/LARGE_BUBBLE/etc. state machine exists. | Only environment-based backend gating exists. |
| Kernel profiler | BLOCKED | No SM count, occupancy, event timing, or split-plan cache code exists. | Split size is fixed. |
| Kernel-tick launcher | BLOCKED | No local asynchronous tick thread exists. | `threshold=1` provides coarse queue limiting, not kernel-tick launch control. |
| Bubble detection | BLOCKED | No small/large bubble hints or API pattern detection exists. | No bubble state in code. |
| Consolidation | BLOCKED | No code merges unlaunched split commands. | All split children are generated at original fixed size. |
| CUTLASS workload | BLOCKED | No new CUTLASS benchmark files were added. | Not started. |

## 3. Log-only Behavior

| Log | Source | Reality |
| --- | --- | --- |
| `unique_hook=UXSched-CUDA-shim` | `LogConfigOnce()` | Informational only, though source routing confirms UXSched is the only hook added by this patch. |
| `capability=splittable` | `SubmitSplitCommands()` | Printed only after checks pass; still not a proof of semantic correctness. |
| `split_group_completed` | State listener in `SubmitSplitCommands()` | Confirms child command state transitions reached completed; not a separate application-visible completion primitive. |
| fallback reason logs | `LogFallback()` | Useful audit trail, but fallback correctness still depends on native path and runtime tests. |

## 4. Synchronization Semantics Gaps

The current implementation relies on XQueue ordering:

```text
cuLaunchKernel -> submit all split children -> return
next cuEventRecord / stream command -> enqueued after children
cuStreamSynchronize -> XQueue WaitAll
cuCtxSynchronize -> XQueueManager::ForEachWaitAll
```

This is promising for single-stream ordering because the children enter the same
XQueue before subsequent commands. However:

- `SplitCommandGroup` is not itself an XCommand and does not block subsequent
  commands.
- No explicit parent command completion object is exposed.
- Child failure handling is incomplete because Lv1 launch uses `CUDA_ASSERT`.
- Multi-stream behavior is not validated.
- Device/context sync uses existing queue-wide wait, not group-specific wait.
- `cuLaunchKernelEx` remains native.
- `extra` launch format falls back native because `CudaKernelCommand` keeps
  `extra_` as a raw pointer.

Status: COMPILE VERIFIED, NOT TESTED for runtime correctness.

## 5. Runtime Suitability for Refactor

Good candidates for RuntimeStrategy extraction:

- launch metadata currently passed to `hb_split::TryLaunchKernel()`;
- native launch construction currently in `shim.cpp`;
- backend mode selection currently in `BackendModeFromEnv()`;
- LP/HP gating currently in `TryLaunchKernel()`;
- split submission currently in `SubmitSplitCommands()`;
- module/function capability cache currently in `backend.cpp`.

Less suitable until redesigned:

- module load wrappers are hook-level capability population and should remain
  near the CUDA platform support code;
- `SplitCommandGroup` needs a real runtime queue or command-group object before
  it can become a synchronization authority;
- threshold changes currently call `SetLaunchConfig()` directly and should move
  into HummingbirdRuntimeStrategy policy.

## 6. PTX Transformation Reality

The PTX transformation can run when:

- process is LP by environment priority;
- backend is not NATIVE;
- Level is 1;
- module image is PTX text;
- kernel name is in `UXSCHED_HB_VERIFIED_KERNELS` or `HB_SPLIT_KERNELS`;
- transformed hidden module successfully loads;
- `cuModuleGetFunction` is called for the original function.

It is not fully general:

- no proof of cooperative/persistent kernel safety;
- no robust cross-block synchronization proof beyond a few token checks;
- no support for all possible PTX `ctaid` patterns;
- no CUDA Graph support;
- no `cuModuleLoadFatBinary` transformation.

Status: COMPILE VERIFIED, NOT TESTED on GPU in this branch.

## 7. Threshold=1 Reality

`SetLpSplitThresholdOnce()` calls:

```cpp
xqueue->SetLaunchConfig(1, 1);
```

`AsyncXQueue::SetLaunchConfig()` waits all current commands, updates the
LaunchWorker threshold/batch size, and sends a config update event. In
`LaunchWorker::LaunchHwCommand()`, `cmd_log_.size() >= threshold_` forces sync
before launching more commands. Therefore threshold=1 is real and should bound
the number of in-flight commands for that XQueue.

Runtime caveat: the threshold is set at the first split submission and affects
the whole XQueue afterwards. This has not been verified under Global HPF with
actual HP/LP processes.

## 8. Summary Verdict

4146c1e is a compile-verified fixed-splitting prototype inside UXSched's CUDA
hook path. It is not yet a Hummingbird Runtime Strategy implementation. The next
safe step is to refactor launch handling into `CudaRuntimeStrategy` with native
and Hummingbird fixed modes while preserving the existing native behavior.

