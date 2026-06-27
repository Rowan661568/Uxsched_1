# Hummingbird Runtime Architecture

## Target Control Plane

Status: COMPILE VERIFIED for the RuntimeStrategy interface, NOT TESTED for GPU
runtime behavior.

```text
Application
-> UXSched CUDA XShim
-> RuntimeStrategy Selector
-> NativeRuntimeStrategy / HummingbirdRuntimeStrategy
-> XQueue / Command Buffer
-> XPreempt LaunchWorker
-> CUDA HwQueue
-> GPU
```

UXSched remains the only CUDA hook and the only global scheduler. Global
Scheduler / HPF still owns cross-process priority decisions, XQueue selection,
and suspend/resume. HummingbirdRuntimeStrategy is a local CUDA runtime strategy
inside the UXSched CUDA platform.

## Implemented Structure

| Component | Status | Source |
| --- | --- | --- |
| `CudaRuntimeStrategy` interface | COMPILE VERIFIED | `platforms/cuda/hal/include/xsched/cuda/hal/runtime/runtime_strategy.h` |
| `NativeRuntimeStrategy` | COMPILE VERIFIED | `platforms/cuda/hal/src/runtime/runtime_strategy.cpp` |
| `HummingbirdRuntimeStrategy` | COMPILE VERIFIED | `platforms/cuda/hal/src/runtime/runtime_strategy.cpp` |
| `HB_FIXED` fixed splitting mode | COMPILE VERIFIED | calls `hb_split::TryLaunchKernelFixed()` |
| `HB_RUNTIME` mode | IMPLEMENTED as fallback only | logs `HB_RUNTIME_NOT_IMPLEMENTED_YET` |
| `AUTO` mode | IMPLEMENTED as fallback only | logs `AUTO_RUNTIME_COORDINATOR_UNAVAILABLE` |
| Device shared HB coordinator | BLOCKED | not implemented |

## Runtime Modes

`UXSCHED_CUDA_RUNTIME_STRATEGY` supports:

- `NATIVE`: NativeRuntimeStrategy.
- `HB_FIXED`: fixed LP splitting, using the existing compile-verified split path.
- `HB_RUNTIME`: currently falls back to Native until coordinator/state machine,
  profiler, tick, and bubble code are implemented.
- `AUTO`: currently falls back to Native because cross-process HP/LP runtime
  state is not available yet.

Legacy `UXSCHED_CUDA_PREEMPT_BACKEND` remains accepted for compatibility.
PTX transformation is currently enabled only for `HB_FIXED` or the legacy
fixed-split backend path, so unimplemented `HB_RUNTIME` and `AUTO` modes do not
mutate module-load behavior before their coordinator is implemented.

## Current Gate

Current gate status:

- Gate 1 HB_FIXED GPU correctness: NOT TESTED.
- Gate 2 Global HP/LP smoke: NOT TESTED.
- Gate 3 synchronization semantics: NOT TESTED.
