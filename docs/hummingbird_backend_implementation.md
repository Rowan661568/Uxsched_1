# Hummingbird Split Backend Implementation

## 1. Modified Files

| File | Role |
| --- | --- |
| `CMakeLists.txt` | Adds `UXSCHED_ENABLE_HB_SPLIT`, default `OFF`. |
| `platforms/cuda/CMakeLists.txt` | Enables compile definition for CUDA HAL/shim when requested. |
| `platforms/RTX4060/CMakeLists.txt` | Enables the same compile definition for RTX4060, which reuses CUDA HAL/shim. |
| `platforms/cuda/shim/include/xsched/cuda/shim/shim.h` | Exposes the HB backend declarations to the CUDA shim. |
| `platforms/cuda/shim/src/intercept.cpp` | Routes `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleUnload`, and `cuModuleGetFunction` through UXSched HB-aware wrappers. |
| `platforms/cuda/shim/src/shim.cpp` | Invokes HB backend selection before creating the native `CudaKernelLaunchCommand`. |
| `platforms/cuda/hal/include/xsched/cuda/hal/hb_split/backend.h` | Defines backend mode, capability model, module wrappers, and launch selection API. |
| `platforms/cuda/hal/src/hb_split/backend.cpp` | Implements PTX transform, hidden transformed module cache, grid splitting, split command group, fallback, and logs. |

## 2. Call Chain

Native path remains:

```text
cuLaunchKernel
-> UXSched CUDA shim XLaunchKernel
-> HwQueueManager::GetXQueue
-> CudaKernelLaunchCommand
-> XQueue::Submit
-> LaunchWorker::LaunchHwCommand
-> CudaQueueLv1::Launch
-> CudaCommand::LaunchWrapper
-> Driver::LaunchKernel
```

HB split path is:

```text
cuModuleLoad/Data/DataEx
-> hb_split::XModuleLoad*
-> Driver::ModuleLoad* for original module
-> PTX transform for verified LP kernels
-> Driver::ModuleLoadDataEx for hidden transformed module

cuModuleGetFunction
-> hb_split::XModuleGetFunction
-> Driver::ModuleGetFunction for original function
-> Driver::ModuleGetFunction for hidden transformed function
-> original CUfunction -> transformed CUfunction cache

cuLaunchKernel
-> XLaunchKernel
-> hb_split::TryLaunchKernel
-> capability checks
-> DecomposeGrid
-> CudaKernelLaunchCommand child commands
-> XQueue::Submit for each child
-> LaunchWorker / CudaQueueLv1 / Driver::LaunchKernel
```

## 3. PTX Transformation

The PTX transform is adapted from the local Hummingbird prototype at:

- `/home/zm/project/hummingbird/kernel_splitter/src/ptx_transform.cpp`
- `/home/zm/project/hummingbird/kernel_splitter/src/grid_decompose.cpp`

Observed local Hummingbird source had no `LICENSE`, `NOTICE`, or `COPYING`
file during this audit. The UXSched file header records this source note.

The transform:

- finds `.visible .entry <kernel>`;
- appends `.param .u32 __hb_off_x/y/z`;
- injects offset register loads at kernel entry;
- rewrites recognized `mov.u32 ..., %ctaid.x/y/z;` instructions with an
  `add.s32` using the matching offset register;
- rejects PTX bodies containing recognized grid-level synchronization tokens:
  `grid.sync`, `griddepcontrol`, `barrier.cluster`.

If a grid dimension is greater than 1, the launch requires the matching ctaid
axis to have been rewritten. Otherwise the backend falls back to native.

## 4. Fallback Rules

Native fallback is used when:

- backend mode is `NATIVE`;
- the build was not configured with `UXSCHED_ENABLE_HB_SPLIT=ON`;
- the process priority is not negative;
- `XSCHED_AUTO_XQUEUE_LEVEL` is greater than 1;
- no XQueue exists for the stream;
- PTX is unavailable;
- kernel name is not in `UXSCHED_HB_VERIFIED_KERNELS`;
- transform failed;
- transformed function was not found;
- `extra` launch format is used;
- `kernelParams` is null;
- the grid is not larger than the split size;
- offset support does not cover the active grid dimensions.

With `UXSCHED_HB_STRICT=1`, several unsupported HB cases return
`CUDA_ERROR_NOT_SUPPORTED` or `CUDA_ERROR_INVALID_VALUE` for debugging.

## 5. Parameter Lifetime

Existing UXSched `CudaKernelCommand` deep-copies `kernelParams` when a command is
submitted to an XQueue. The HB backend builds each split command with
`deep_copy=true`. Offset values are provided from a temporary local array only
during command construction; the command immediately copies them via cuxtra
metadata for the transformed function.

The first stage intentionally does not split `extra` launch format because
UXSched's current command object stores `extra_` as a raw pointer. That format
falls back to native.

## 6. Completion and Synchronization

The backend submits all split children to the same XQueue before returning from
the intercepted `cuLaunchKernel`. Therefore:

- the next command on the same stream is enqueued after all children;
- `cuEventRecord` is submitted after all children;
- `cuStreamSynchronize`, `cuCtxSynchronize`, and XQueue wait logic continue to
  use existing UXSched command ordering.

`SplitCommandGroup` tracks child completion with command state listeners and
releases child ownership after the last child completes. The current UXSched Lv1
path still asserts on CUDA launch failures inside `CudaQueueLv1`, so first-error
recovery is not fully implemented beyond group tracking.

## 7. Logs

Important logs use the `[UXSCHED-HB]` prefix, including:

- backend configuration and unique hook path;
- HP passthrough;
- transform success/failure;
- selected split count;
- native fallback reason;
- LP threshold change to `1,1`;
- split group completion.

