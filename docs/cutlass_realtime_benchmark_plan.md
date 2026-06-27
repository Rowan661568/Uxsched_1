# CUTLASS Realtime Benchmark Plan

## Goal

Build a CUTLASS-based HP/LP realtime benchmark that preserves the measurement
shape of `benchmarks/realtime_inference_latency.py` while replacing the current
PyTorch/TorchVision workloads with open-source CUTLASS GEMM kernels.

Primary target:

```text
UXSched + Hummingbird HB_FIXED + CUTLASS HP P99
  < UXSched NATIVE + same CUTLASS HP P99
```

The first optimization objective is HP P99 latency. LP throughput must be
reported and cannot be reduced to zero to claim success.

Current Gate 1 interpretation for this phase:

```text
HB_FIXED runtime verified, correctness deferred to CUTLASS workload validation.
```

The older `open_resnet_like_runtime_eval` correctness path is not a blocker for
starting the CUTLASS workload, but no P99 claim is valid until CUTLASS
correctness and synchronization pass.

## CUTLASS Runtime Bridge Status

The first GPU run of `tools/run_cutlass_launch_probe.sh` with commit
`157662a` proved:

- CUTLASS Native SM120 builds with external CUTLASS revision `ad7b2f5`.
- CUTLASS Native Runtime correctness passes.
- CUTLASS UXSched NATIVE Runtime correctness passes.
- CUTLASS UXSched HB_FIXED Runtime did not exercise the HB backend:
  Runtime launch was not intercepted and all HB transform/parent/child counts
  were zero.

Root cause:

- The probe uses shared `libcudart.so.12`, not static cudart.
- The probe dynamically imports `cudaLaunchKernel` and Runtime registration
  symbols.
- The probe contains `sm_120` SASS and CUTLASS PTX.
- The prior `libshimcuda.so` only intercepted Driver API entry points, so CUDA
  Runtime registration and `cudaLaunchKernel` bypassed the UXSched HB_FIXED
  backend.

Bridge design now implemented in the UXSched CUDA shim:

```text
__cudaRegisterFatBinary
  -> extract uncompressed PTX from Runtime fatbin
__cudaRegisterFunction
  -> map host stub to device kernel name and fatbin
cudaLaunchKernel
  -> resolve host stub to registered CUTLASS kernel
  -> load PTX through existing hb_split module path
  -> resolve CUfunction through existing hb_split function path
  -> resolve/create stream XQueue
  -> call TryLaunchKernelFixed for HB_FIXED
  -> otherwise call true cudaLaunchKernel fallback
cudaStreamSynchronize / cudaEventRecord / cudaEventSynchronize / cudaDeviceSynchronize
  -> route to existing UXSched XStream/XEvent/XCtx synchronization wrappers
  -> wait for queued split children before correctness/timing boundaries complete
```

Safe fallback is required for unavailable PTX, missing registration, unverified
kernel, unsupported parameters, dynamic launch-ex path, missing XQueue,
NATIVE strategy, HP passthrough, or transform failure. Fallback must call the
real CUDA Runtime launch and preserve correctness.

`UXSCHED_CUDART_TRACE=1` records:

- `runtime_fatbin_registered`
- `runtime_function_registered`
- `runtime_launch_intercepted`
- `runtime_launch_function_resolved`
- `runtime_backend_selected`
- `runtime_launch_fallback`
- `runtime_sync_intercepted`

The CUTLASS probe build is now pinned to:

```text
CMAKE_CUDA_ARCHITECTURES=120-real;120-virtual
CUDA_RUNTIME_LIBRARY=Shared
--compress-mode=none
```

This preserves native SM120 cubin and extractable PTX for the Runtime bridge.

The updated runner saves:

- `probe_binary/ldd.txt`
- `probe_binary/dynamic_symbols.txt`
- `probe_binary/cuobjdump_elf.txt`
- `probe_binary/cuobjdump_ptx.txt`
- per-case `runtime_registration.log`
- per-case Runtime/HB backend counters
- `discovered_cutlass_kernel_name` in `cutlass_probe_summary.env`
- `runtime_sync_intercepted_count` in `cutlass_probe_summary.env`

The Runtime bridge is currently compile/static verified only. A normal WSL GPU
rerun must observe `runtime_launch_intercepted_count > 0`,
`runtime_function_resolved_count > 0`, `runtime_sync_intercepted_count > 0`,
transformed child launches, no fallback, no `NO_XQUEUE`, and CUTLASS
correctness before this phase can pass.

## Runtime Metadata Bridge

The first GPU rerun after Runtime launch/sync interception showed that Runtime
fatbin/PTX extraction and `CUfunction` resolution worked, but HB backend launch
selection still fell back with:

```text
function=<unknown>
reason=PTX_UNAVAILABLE
```

The source-level cause was a metadata handoff gap between CUDA Runtime
registration and the HB split backend registry. The HB backend only splits a
`CUfunction` after it can find that function in its `g_functions` registry,
which is normally populated by the Driver API `cuModuleGetFunction` hook. The
Runtime bridge had resolved a real `CUfunction` but had not guaranteed that the
same `CUmodule -> PTX` and `CUfunction -> kernel name` metadata was present in
the HB registry before dispatching to `TryLaunchKernelFixed`.

The bridge now explicitly performs:

```text
Runtime fatbin/PTX
  -> Driver::ModuleLoadDataEx
  -> hb_split::RegisterModuleMetadata
  -> Driver::ModuleGetFunction
  -> hb_split::RegisterFunctionMetadata
  -> hb_split::TryLaunchKernelFixed
```

The HB registry owns a PTX copy for the module lifetime, so Runtime fatbin
buffers or local extraction buffers are not used after registration. Runtime
unregister unloads the Runtime-created module through `XModuleUnload`, removing
module metadata, function metadata, and hidden transformed modules.

The runner records:

- `runtime_hb_module_registered_count`
- `runtime_hb_function_registered_count`
- `runtime_hb_registration_failed_count`
- `runtime_hb_module_registration_pass`
- `runtime_hb_function_registration_pass`
- `runtime_hb_metadata_bridge_pass`

The next GPU rerun should first prove:

```text
runtime_hb_module_registered_count > 0
runtime_hb_function_registered_count > 0
runtime_hb_metadata_bridge_pass=1
```

and should no longer show `function=<unknown>` with `PTX_UNAVAILABLE`. Any later
fallback such as `KERNEL_NOT_VERIFIED`, `ENTRY_NOT_FOUND`,
`OFFSET_Y_UNSUPPORTED`, `PARAM_COUNT_MISMATCH`, or `TRANSFORM_FAILED` is the
next layer of CUTLASS transform compatibility, not the Runtime metadata bridge.

## `realtime_inference_latency.py` Call Graph

Entry:

```text
main()
  parse_args()
  make_result_dir()
  for scenario:
    run_one()
      optionally start_xserver()
      optionally launch_worker(background)
      sleep(hp_delay)
      launch_worker(foreground)
      foreground.proc.wait()
      terminate background/xserver
      write_latency_csv()
      summarize_scenario()
  add_slowdowns()
  write comparison.json
```

Source references:

- Scenario list and top-level output are in
  `benchmarks/realtime_inference_latency.py:482-519`.
- `run_one()` starts xserver, background, foreground, and writes per-scenario
  outputs in `benchmarks/realtime_inference_latency.py:351-397`.
- Workers are spawned by re-running the same Python file with `--worker` in
  `worker_cmd()` at `benchmarks/realtime_inference_latency.py:237-258` and
  `launch_worker()` at `benchmarks/realtime_inference_latency.py:261-280`.
- Worker stdout is parsed by regex `latency_ms:` in
  `benchmarks/realtime_inference_latency.py:37` and consumed in
  `stream_worker()` at `benchmarks/realtime_inference_latency.py:283-309`.

## Current Workload Structure

HP workload:

- Foreground worker uses PyTorch/TorchVision ResNet50 inference.
- Code: `foreground_worker()` at
  `benchmarks/realtime_inference_latency.py:424-449`.
- Model: `torchvision.models.resnet50(weights=None).eval().to(device)` at
  `benchmarks/realtime_inference_latency.py:432`.
- Input: `torch.ones(batch_size, 3, image_size, image_size, device=device)` at
  `benchmarks/realtime_inference_latency.py:433`.
- Per request: one full model forward pass in `step()` at
  `benchmarks/realtime_inference_latency.py:435-437`.

LP workload:

- Background worker uses PyTorch/TorchVision MobileNetV2 training in an infinite
  loop.
- Code: `background_worker()` at
  `benchmarks/realtime_inference_latency.py:452-479`.
- Model: `torchvision.models.mobilenet_v2(...).train().to(device)` at
  `benchmarks/realtime_inference_latency.py:460`.
- Per iteration: zero grad, forward, cross entropy, backward, optimizer step in
  `step()` at `benchmarks/realtime_inference_latency.py:466-472`.

Process model:

- HP and LP are independent child processes. `run_one()` launches background
  first for every non-`alone` scenario at
  `benchmarks/realtime_inference_latency.py:365-372`, then launches foreground
  at `benchmarks/realtime_inference_latency.py:374-378`.
- The foreground process exits after `requests`; the background process loops
  forever and is terminated by the parent in
  `benchmarks/realtime_inference_latency.py:379-387`.

CUDA context/stream behavior:

- Each worker imports `torch`, checks CUDA visibility, and uses GPU `args.gpu`
  through `check_cuda()` at `benchmarks/realtime_inference_latency.py:414-421`.
- HP creates one explicit `torch.cuda.Stream` and sets it current at
  `benchmarks/realtime_inference_latency.py:430-431`.
- LP creates one explicit `torch.cuda.Stream` and sets it current at
  `benchmarks/realtime_inference_latency.py:458-459`.
- Each HP request calls `stream.synchronize()` before measuring completion at
  `benchmarks/realtime_inference_latency.py:443-448`.
- Each LP training iteration also calls `stream.synchronize()` at
  `benchmarks/realtime_inference_latency.py:477-479`.

XSched/XQueue setup:

- Non-XSched scenarios remove `XSCHED_*`, `CUXTRA_CUDA_LIB`, `LD_PRELOAD`, and
  strip `output/lib` from `LD_LIBRARY_PATH` in
  `benchmarks/realtime_inference_latency.py:94-104`.
- XSched scenarios set:
  - `XSCHED_CUDA_LIB`
  - `CUXTRA_CUDA_LIB`
  - `LD_LIBRARY_PATH=<output/lib>:$LD_LIBRARY_PATH`
  - `XSCHED_SCHEDULER=GLB`
  - `XSCHED_AUTO_XQUEUE=ON`
  - `XSCHED_AUTO_XQUEUE_PRIORITY=<priority>`
  - `XSCHED_AUTO_XQUEUE_LEVEL=<level>`
  - `XSCHED_AUTO_XQUEUE_THRESHOLD=<threshold>`
  - `XSCHED_AUTO_XQUEUE_BATCH_SIZE=<batch_commands>`
- Code: `child_env()` at `benchmarks/realtime_inference_latency.py:94-118`.
- The script currently does not set `LD_PRELOAD`; it relies on dynamic library
  path behavior in the install layout. For the CUTLASS version, use explicit
  `LD_PRELOAD=<build-hb>/platforms/cuda/libshimcuda.so` to preserve the
  UXSched-only hook invariant.

Priority mapping:

- Existing script uses background priority `0` and foreground priority `1`.
- Code: background launch at `benchmarks/realtime_inference_latency.py:367-368`;
  foreground launch at `benchmarks/realtime_inference_latency.py:374-375`.
- CUTLASS benchmark must change this to the integration convention:
  - HP: `XSCHED_AUTO_XQUEUE_PRIORITY=10`
  - LP: `XSCHED_AUTO_XQUEUE_PRIORITY=-10`

Scheduler:

- XSched scenarios use `XSCHED_SCHEDULER=GLB` in
  `benchmarks/realtime_inference_latency.py:111`.
- `xserver` is started with policy `args.policy`, default `HPF`, at
  `benchmarks/realtime_inference_latency.py:185-187`.
- Service default is also HPF/50000 in `service/server/src/main.cpp:20-43`.

xserver/client startup:

- Paths are hard-coded to `output/bin/xserver`, `output/bin/xcli`, and
  `output/lib` in `benchmarks/realtime_inference_latency.py:31-35`.
- Readiness is checked with `xcli --port <port> policy -q` in
  `benchmarks/realtime_inference_latency.py:121-132`.
- `xserver` is started as `[xserver, policy, port]` at
  `benchmarks/realtime_inference_latency.py:185-187`.
- `xcli policy -q` is implemented at `service/cli/src/main.cpp:48-88` and calls
  HTTP `/policy` through `Client::QueryPolicy()` at
  `service/common/src/client.cpp:84-90`.
- `service/README.md:134-144` documents `./output/bin/xserver [policy] [port]`.

Request arrival model:

- Background starts first.
- Parent sleeps a fixed `hp_delay` seconds before HP starts:
  `benchmarks/realtime_inference_latency.py:371-372`.
- HP requests are not period-paced; they run back-to-back after warmup in
  `benchmarks/realtime_inference_latency.py:443-448`.
- There is no trace, burst model, or fixed inter-arrival sleep in the current
  HP worker.

Warmup and measurement:

- Defaults: `--warmup=10`, `--requests=200`,
  `--batch-size=32`, `--train-batch-size=16`, `--image-size=224`,
  `--hp-delay=10.0`, `--threshold=16`, `--batch-commands=8`.
- Code: `parse_args()` at `benchmarks/realtime_inference_latency.py:51-78`.
- HP warmup runs `warmup` forward passes with stream synchronization at
  `benchmarks/realtime_inference_latency.py:439-441`.
- LP warmup runs `warmup` training iterations with stream synchronization at
  `benchmarks/realtime_inference_latency.py:474-476`.

Latency timing:

- HP latency starts immediately before `step()` and ends after
  `stream.synchronize()`.
- Code: `time.perf_counter()` around `step()` plus stream sync at
  `benchmarks/realtime_inference_latency.py:443-448`.
- This includes CPU launch overhead, framework overhead, and GPU wait time.
- It does not use CUDA events for HP latency.
- LP throughput is not explicitly summarized; LP stdout is not parsed unless it
  emits `latency_ms:`, which the current background worker does not.

Percentile method:

- `percentile()` sorts all foreground samples and linearly interpolates between
  adjacent ranks at `benchmarks/realtime_inference_latency.py:312-320`.
- `summarize_scenario()` reports avg, P50, P95, P99, min, max at
  `benchmarks/realtime_inference_latency.py:323-338`.
- `add_slowdowns()` divides scenario P50/P95/P99 by the `alone` scenario at
  `benchmarks/realtime_inference_latency.py:400-410`.
- `latency.csv` fields are written at
  `benchmarks/realtime_inference_latency.py:341-348`.
- Per-scenario `summary.json`, top-level `config.json`, and `comparison.json`
  are written at `benchmarks/realtime_inference_latency.py:391-397` and
  `benchmarks/realtime_inference_latency.py:493-508`.

Closed or non-reusable dependencies:

- PyTorch and TorchVision are required for the current HP/LP workload.
- cuDNN/cuBLAS or Torch-generated kernels may be used internally by those
  frameworks and are not controlled by UXSched-Hummingbird.
- The current workload does not expose a stable kernel name, fixed PTX module,
  or CUTLASS-level correctness outputs.

Parts to preserve:

- Parent/child process orchestration.
- Result directory layout.
- xserver readiness/start/stop pattern.
- CSV plus JSON summaries.
- Foreground P50/P90/P95/P99/Pmax from per-request samples.
- Scenario comparison discipline.

Parts to replace:

- `foreground_worker()` and `background_worker()` should become launchers for a
  CUTLASS C++ benchmark binary or a new Python runner should spawn that binary
  directly.
- PyTorch/TorchVision dependencies should be removed from the core benchmark.
- HP arrival model should become explicit fixed-period or trace-driven; first
  version should use fixed period for reproducibility.
- LP throughput and correctness must be emitted by the workload, not inferred
  from side effects.

## Local CUTLASS/CUDA Audit

Original audit result:

- No CUTLASS source tree or CUTLASS submodule was found under UXSched or
  Hummingbird by local filesystem search.
- UXSched submodules do not include CUTLASS; current submodules are CLI11,
  cpp-httplib, cuxtra, ftxui, ipc, jsoncpp, and nested ipc dependencies.
- CUDA toolkit found during the original audit was CUDA 12.0.

Current CUTLASS launch-compatibility probe environment:

- External CUTLASS source is available at `/home/zm/project/cutlass`.
- CUTLASS revision audited by the probe build is `ad7b2f5`.
- CUDA toolkit is `/usr/local/cuda-12.8`.
- `nvcc` is `/usr/local/cuda-12.8/bin/nvcc`, release `12.8, V12.8.93`.
- The probe uses native SM120 only:
  - `CMAKE_CUDA_ARCHITECTURES=120`
  - `CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`
  - `CUTLASS_MODE=NATIVE_SM120`
- The older Forward PTX compatibility idea is canceled for this phase.
- Hummingbird already has optional CUTLASS detection in
  `/home/zm/project/hummingbird/benchmarks/CMakeLists.txt:24-40`.
  It searches:
  - `third_party/cutlass/include`
  - `external/cutlass/include`
  - `$CUTLASS_ROOT/include`
  - `/usr/local/include`
  - `/usr/include`
- Hummingbird's existing `hb_open_gemm_eval` is not a CUTLASS workload. It uses
  a hand-written tiled CUDA GEMM kernel at
  `/home/zm/project/hummingbird/benchmarks/open_gemm_eval.cu:14-39`.

Design implication:

- First implementation should add CUTLASS as a UXSched-side external dependency
  or submodule, but this audit phase must not download it.
- Preferred future location:

```text
3rdparty/cutlass
```

or, if kept outside the repo:

```text
CUTLASS_ROOT=/path/to/cutlass
```

The build should fail with a clear message when CUTLASS headers are absent and
the CUTLASS benchmark target is requested.

## CUTLASS Replacement Boundary

Add a new CUTLASS benchmark and runner rather than modifying
`realtime_inference_latency.py` in place.

Initial launch-compatibility probe files:

```text
benchmarks/cutlass/CMakeLists.txt
benchmarks/cutlass/cutlass_launch_probe.cu
benchmarks/cutlass/cutlass_probe_common.h
tools/build_cutlass_launch_probe.sh
tools/run_cutlass_launch_probe.sh
```

Future realtime benchmark files:

```text
benchmarks/cutlass_realtime_latency.py
benchmarks/cutlass/cutlass_realtime_gemm.cu
benchmarks/cutlass/cutlass_realtime_common.h
tools/run_cutlass_realtime_gate.sh
```

The Python runner should preserve the existing orchestration contract:

- start/reuse xserver;
- launch independent HP and LP processes;
- assign HP priority `10` and LP priority `-10`;
- collect per-request HP samples;
- collect LP progress/throughput;
- write `latency.csv`, `summary.json`, `comparison.json`, and per-process logs.

The C++ CUTLASS binary should own workload behavior:

- `--role hp|lp`
- `--mode correctness|latency|throughput`
- `--m`, `--n`, `--k`
- `--dtype fp16_tc|fp32_simt`
- `--requests`
- `--warmup`
- `--period-us`
- `--duration-ms`
- `--stream default|explicit`
- `--sync event|stream|context`
- `--output-jsonl`
- `--dump-output`

## Minimal CUTLASS Workload Design

Use CUTLASS C++ template API, not PyTorch or cuBLAS.

Core operation:

```text
C = alpha * A * B + beta * C
```

Correctness requirements:

- deterministic input initialization;
- deterministic `C` initialization;
- CPU reference for small validation shapes;
- optional CUDA reference only after CPU path passes;
- output matrix copy back;
- max absolute error;
- max relative error;
- NaN/Inf count;
- FNV-1a or xxHash-style output hash over raw output bytes;
- output element count;
- JSONL fields for all correctness metrics.

Timing requirements:

- CUDA events around the GEMM launch for GPU time;
- CPU `steady_clock` around request submission plus synchronization for
  end-to-end HP latency;
- HP P99 should use CPU-observed request latency because the realtime goal is
  request response time;
- GPU event time should be reported as a diagnostic, not as the P99 source of
  truth.

Launch requirements:

- The benchmark must use CUDA Runtime or Driver API in a way intercepted by the
  UXSched CUDA shim.
- For transformability, the first version should prefer CUTLASS paths that
  produce visible kernel launches through `cuLaunchKernel` with `kernelParams`,
  not CUDA Graphs or `cuLaunchKernelEx`.
- Compile options should preserve PTX in the binary or provide a PTX-loading
  path that current UXSched module hooks can inspect. If CUTLASS only arrives as
  cubin/SASS in a fatbin path that UXSched cannot transform, first version must
  record that as `PTX_UNAVAILABLE` and fall back.

## Initial Shape Candidates

These are starting points for RTX 5060 Laptop GPU validation, not claimed
optimal settings.

Tensor Core FP16 input, FP32 accumulation:

| Role | Candidate | Rationale |
| --- | --- | --- |
| HP | `M=N=K=256` | Small request, high frequency, should expose tail latency under LP pressure. |
| HP | `M=N=K=384` | Slightly larger HP if 256 is too short to measure robustly. |
| LP | `M=N=K=4096` | Long background GEMM, likely enough CTAs for split experiments. |
| LP | `M=N=K=8192` | Larger stress case if 4096 is too short, memory permitting. |

SIMT FP32 fallback:

| Role | Candidate | Rationale |
| --- | --- | --- |
| HP | `M=N=K=256` FP32 SIMT | Conservative kernel structure if Tensor Core PTX is not transformable. |
| LP | `M=N=K=2048` FP32 SIMT | Long enough to split, but less likely to use unsupported Tensor Core/TMA forms. |
| LP | `M=N=K=4096` FP32 SIMT | Stress option if memory and runtime are acceptable. |

Initial request model:

- HP fixed period: start with `period_us=1000` or `2000`.
- HP warmup: `50`.
- HP measured requests: `1000` for smoke, `5000` for formal P99.
- LP duration: run until HP completes plus shutdown grace.
- Repeat count: `1` until Phase 9; `3` or `5` only after correctness and split
  evidence pass.

## CUTLASS/HB PTX Compatibility Risk

Current transformer behavior:

- Runtime mode and split size are read from env at
  `platforms/cuda/hal/src/hb_split/backend.cpp:130-170`.
- LP gate uses negative priority at
  `platforms/cuda/hal/src/hb_split/backend.cpp:184-198`.
- PTX text detection is heuristic at
  `platforms/cuda/hal/src/hb_split/backend.cpp:240-255`.
- Transform only injects three offset params at
  `platforms/cuda/hal/src/hb_split/backend.cpp:311-321`.
- It rejects a small set of cross-block sync tokens:
  `grid.sync`, `griddepcontrol`, `barrier.cluster` at
  `platforms/cuda/hal/src/hb_split/backend.cpp:324-329`.
- It rewrites only `mov.u32 <reg>, %ctaid.x/y/z;` forms via regex at
  `platforms/cuda/hal/src/hb_split/backend.cpp:331-365`.
- It requires the offset axis to be observed in PTX before splitting grids on
  that axis at `platforms/cuda/hal/src/hb_split/backend.cpp:660-674`.
- It does not support `extra` launch format or null `kernelParams`, as shown at
  `platforms/cuda/hal/src/hb_split/backend.cpp:986-1000`.
- It falls back when the grid has no more blocks than `UXSCHED_HB_SPLIT_BLOCKS`
  at `platforms/cuda/hal/src/hb_split/backend.cpp:1003-1008`.
- `cuLaunchKernelEx` remains native in
  `platforms/cuda/shim/src/shim.cpp:221-225`.

Likely directly supportable:

- Non-persistent single-kernel GEMM with independent CTAs.
- Regular 1D/2D/3D CTA grids if `%ctaid.*` appears in the recognized `mov.u32`
  form.
- Static or dynamic shared memory at launch, as `shared_mem_bytes` is forwarded
  to child launches at `platforms/cuda/hal/src/hb_split/backend.cpp:692-748`.
- Intra-CTA barriers such as `bar.sync` should be safe conceptually because the
  split preserves whole CTAs. They are not rejected by current token scan.

May work but requires validation:

- Tensor Core MMA instructions.
- `cp.async` pipelines.
- Complex epilogues with multiple stores.
- Dynamic shared-memory kernels.
- Kernels whose PTX uses `%ctaid` through forms other than the current
  `mov.u32` regex.
- CUTLASS kernels launched through Runtime API wrappers if the resulting driver
  launch still reaches `cuLaunchKernel` and has transformable PTX metadata.

Currently unsupported or high risk:

- TMA / Hopper-style kernels.
- Cluster launch and `barrier.cluster`.
- Cooperative grid sync.
- Persistent kernels.
- Grouped GEMM.
- Split-K with inter-block reductions or workspace reductions.
- CUDA Graph launch path.
- `cuLaunchKernelEx`.
- Kernels only visible as cubin/SASS with no PTX text to transform.
- Any kernel using `extra` launch format rather than `kernelParams`.

Features to actively disable in first version:

- grouped GEMM;
- split-K;
- persistent kernels;
- TMA;
- cluster launch;
- cooperative groups requiring grid-level sync;
- CUDA Graphs;
- runtime-selected kernels that hide the concrete kernel name;
- very new SM90/SM100 collectives until transformer compatibility is proven.

First version should use the most conservative CUTLASS GEMM that still launches
one visible GEMM kernel and exposes a stable kernel name.

## Launch Compatibility Probe

The first implemented CUTLASS artifact is a single-process launch-path probe,
not the HP/LP realtime benchmark.

Implemented probe:

```text
benchmarks/cutlass/cutlass_launch_probe.cu
```

Build helper:

```text
tools/build_cutlass_launch_probe.sh
```

Runner:

```text
tools/run_cutlass_launch_probe.sh
```

Runtime mode:

- Uses `cutlass::gemm::device::Gemm` with FP32 input, FP32 output, FP32
  accumulation, SIMT, `cutlass::arch::Sm120`, and identity threadblock swizzle.
- The CUTLASS call chain is:

```text
ProbeGemm::run(stream)
  -> cutlass::gemm::device::Gemm::run(stream)
  -> cutlass::Kernel<GemmKernel><<<grid, block, smem, stream>>>(params_)
  -> CUDA Runtime launch
  -> CUDA Driver launch path, if libcudart resolves through intercepted Driver API
```

- For CUTLASS 3.x universal kernels, `cutlass::kernel_launch` uses
  `device_kernel<GemmKernel><<<...>>>(kernel_params)` when PDL is disabled; PDL
  would use `cudaLaunchKernelEx`, which the current UXSched HB path does not
  split.
- The current UXSched shim can only split this mode if the Runtime launch reaches
  intercepted `cuLaunchKernel` and module/function metadata is visible to
  `cuModuleLoad*` / `cuModuleGetFunction`.
- If the Runtime launch runs correctly but HB backend counts remain zero, the
  runner records `runtime_launch_not_intercepted=1` rather than treating it as a
  CUDA correctness failure.

Driver / `CudaHostAdapter` mode:

- CUTLASS `CudaHostAdapter` is an abstract interface intended to let CUTLASS
  call a host-provided launcher.
- It does not by itself provide a generic official path to obtain `CUmodule`,
  `CUfunction`, the exact CUTLASS kernel parameter layout, dynamic shared-memory
  size, and stream for a `cuLaunchKernel` call for this GEMM.
- The probe therefore implements driver mode as an explicit blocked case that
  emits:

```text
cutlass_driver_launch_integration_blocked
```

- This avoids replacing CUTLASS with a custom CUDA kernel or bypassing UXSched.

Correctness model:

- A and B are deterministic FP32 matrices with values chosen so the CPU
  reference can be computed in O(M*N) rather than O(M*N*K).
- The probe outputs checksum, raw output hash, output element count, max
  absolute error, max relative error, mismatch count, NaN count, Inf count,
  CPU request time, and CUDA event time.
- FP32 tolerances are emitted by the binary:
  - absolute tolerance `1e-2`
  - relative tolerance `1e-4`

Current non-GPU build status:

- `tools/build_cutlass_launch_probe.sh` configured and built
  `build-cutlass-cu128/cutlass_launch_probe`.
- `build-hb-cu128` was configured separately and built `halcuda`, `shimcuda`,
  `xserver`, and `xcli`.
- Codex did not run Runtime GPU cases or claim HB_FIXED CUTLASS compatibility.

## Fair Core Comparison

Baseline A: UXSched Native

```text
same CUTLASS HP workload
same CUTLASS LP workload
XSCHED_SCHEDULER=GLB
xserver policy=HPF
HP priority=10
LP priority=-10
UXSCHED_CUDA_RUNTIME_STRATEGY=NATIVE
no Hummingbird split
```

System B: UXSched + Hummingbird

```text
same CUTLASS HP workload
same CUTLASS LP workload
same arrival rate
same request count
same warmup
same GPU
same xserver policy=HPF
same priorities
HP priority=10, must log HIGH_PRIORITY_PASSTHROUGH
LP priority=-10
UXSCHED_CUDA_RUNTIME_STRATEGY=HB_FIXED
UXSCHED_HB_SPLIT_BLOCKS=<selected split size>
UXSCHED_HB_VERIFIED_KERNELS=<CUTLASS LP kernel name after verification>
```

Only LP runtime strategy and required HB parameters may differ.

Do not change between A and B:

- GEMM shape;
- dtype;
- input initialization;
- request count;
- warmup;
- HP period;
- scheduler policy;
- priority;
- measurement window;
- GPU clocks or power mode;
- sync boundary;
- output correctness thresholds.

## Metrics

Core metrics:

- HP P50;
- HP P90;
- HP P95;
- HP P99;
- HP max;
- HP deadline miss ratio;
- LP throughput;
- LP completion count;
- GPU error count;
- fallback count;
- transformed launch count;
- child launch count;
- parent completion count;
- no-XQueue count.

Primary success:

```text
HB_FIXED HP P99 < UXSched NATIVE HP P99
```

Required guards:

- CUTLASS correctness pass;
- no CUDA illegal memory access;
- no CUDA invalid argument;
- no segmentation fault;
- no local scheduler fallback;
- HP never split;
- LP transformed launch count > 0;
- LP child launch count > 1;
- `NO_XQUEUE=0`.

Suggested derived metrics:

- P99 improvement percentage:
  `(native_p99 - hb_p99) / native_p99 * 100`;
- LP throughput retention:
  `hb_lp_throughput / native_lp_throughput`;
- repeat variance for P99 and LP throughput in Phase 9.

## Result Directory Structure

Proposed:

```text
results/cutlass_realtime_<timestamp>/
  config.json
  xserver/
    command.txt
    env.txt
    stdout.log
    stderr.log
    status.txt
  native/
    hp/
      command.txt
      env.txt
      stdout.log
      stderr.log
      output.jsonl
      correctness.json
    lp/
      command.txt
      env.txt
      stdout.log
      stderr.log
      output.jsonl
      correctness.json
    latency.csv
    summary.json
    backend_stats.env
  hb_fixed/
    hp/
    lp/
    latency.csv
    summary.json
    backend_stats.env
    split_trace.log
    child_completion.log
    parent_completion.log
  comparison.json
  gate_summary.env
```

`gate_summary.env` should include at least:

```text
cutlass_correctness_pass
native_hp_p99_us
hb_hp_p99_us
p99_improvement_pct
native_lp_throughput
hb_lp_throughput
lp_throughput_retention
hp_passthrough_pass
lp_transformed_launch_count
lp_child_launch_count
lp_fallback_count
lp_no_xqueue_count
no_cuda_error
no_local_scheduler_fallback
phase_pass
```

## Phase Gates

Phase 1: single-process CUTLASS correctness

- Run HP shape and LP shape without UXSched.
- CPU reference, max abs/rel error, NaN/Inf, output hash must pass.

Phase 2: single-process CUTLASS Native latency

- Run the binary with UXSched shim and `UXSCHED_CUDA_RUNTIME_STRATEGY=NATIVE`.
- Record CPU request latency and CUDA event latency.

Phase 3: single-process LP HB_FIXED split correctness

- Run LP shape with `HB_FIXED`.
- Must observe transformed module load, parent launch, child launches,
  transformed launch evidence, child completion, parent completion, and
  correctness pass.

Phase 4: synchronization probes

- Default stream, explicit stream, event sync, stream sync, context sync, and
  same-stream dependent correctness must pass for the CUTLASS workload.

Phase 5: Global HP-only baseline

- Start xserver HPF.
- Run HP only with UXSched NATIVE.
- Establish HP-only P99 and deadline target.

Phase 6: Global HP+LP UXSched Native

- Same HP/LP workloads.
- `UXSCHED_CUDA_RUNTIME_STRATEGY=NATIVE`.
- Collect baseline HP P99 and LP throughput.

Phase 7: Global HP+LP HB_FIXED

- Same HP/LP workloads.
- HP priority `10` passthrough.
- LP priority `-10` split.
- Compare P99 and throughput to Phase 6.

Phase 8: split-size sweep

- Only after Phase 7 passes.
- The current fixed experiment setting is `UXSCHED_HB_SPLIT_BLOCKS=52` for the
  RTX 5060 Laptop GPU and the SM120 FP32 SIMT CUTLASS GEMM kernel.
- This value comes from the Hummingbird hardware formula: 26 SMs times 2 active
  CUTLASS blocks per SM, where the resident block count is register-limited.
- repeat=3 testing showed split=52 reduced HB HP P99 by 7.76% relative to
  split=64 and improved LP throughput by 10.54% relative to split=64.
- This is a fixed benchmark configuration, not automatic split selection and
  not runtime profiling.
- The selected value is not a global optimum for other GPUs, other CUTLASS
  kernels, or other shapes.
- Additional sweeps may be run after correctness and fairness checks pass.
- Do not change workload or arrival model.

Phase 9: formal repeat

- Repeat `3` or `5` times after selecting a split size.
- Report variance and do not claim performance from a single run.

## PASS/FAIL Conditions

Phase 1 PASS:

- correctness pass for both HP and LP shapes;
- output hash stable across repeated runs;
- no CUDA error.

Phase 3 PASS:

- correctness pass;
- LP transformed launch count > 0;
- child launch count > 1;
- no fallback for verified LP GEMM kernel;
- `NO_XQUEUE=0`;
- HP passthrough still verified separately.

Phase 7 PASS:

- all Phase 1-4 checks still pass;
- HP P99 in HB_FIXED is lower than UXSched Native;
- LP throughput retention reported;
- no local scheduler fallback;
- HP never split;
- LP actually split.

FAIL:

- correctness mismatch;
- NaN/Inf output;
- illegal memory access;
- invalid argument;
- segmentation fault;
- local scheduler fallback;
- missing transformed launch evidence for LP;
- HP transformed or child launch count nonzero;
- changing workload or arrival parameters between baseline and HB_FIXED.

## Next Implementation Steps

1. CUTLASS dependency plumbing is implemented through external `CUTLASS_ROOT`;
   no CUTLASS source is copied into UXSched.
2. `benchmarks/cutlass/cutlass_realtime_worker.cu` implements the first
   dual-process realtime worker with deterministic FP32 SIMT GEMM inputs,
   correctness metrics, JSONL output, explicit CUDA stream, HP fixed-period
   pacing, LP steady submission, and file-based ready/start barriers.
3. `tools/run_cutlass_realtime_compare.sh` implements repeat=1 smoke entry
   points for:
   - `standalone_hp`;
   - `uxsched_native_hp_lp`;
   - `uxsched_hb_fixed_hp_lp`.
4. `tools/summarize_cutlass_realtime_compare.py` generates `summary.csv` and
   `comparison.csv` without pandas and uses the same sorted linear-interpolation
   percentile definition as `benchmarks/realtime_inference_latency.py`.
5. The HB_FIXED realtime path requires an exact kernel allowlist file. The
   template is `benchmarks/cutlass/verified_kernel_sm120_fp32_simt.txt`; formal
   measurement must fill it from compatibility-probe
   `discovered_cutlass_kernel_name` and must not use
   `UXSCHED_HB_VERIFIED_KERNELS=*`.
6. Repeat=1 smoke is only an integration check. Formal P99 claims still require
   repeat=3 or repeat=5, correctness pass, identical workload settings, no
   fallback, no `NO_XQUEUE`, HP passthrough, and real LP parent/child split
   deltas during measurement.

## Current Implementation Status

- CUTLASS compatibility/correctness gate: PASS in user GPU testing.
- Runtime metadata bridge to HB backend: PASS in user GPU testing.
- CUTLASS HP/LP realtime benchmark: IMPLEMENTED and COMPILE VERIFIED in Codex.
- CUTLASS fixed HB split size for the current formal experiment: 52 blocks.
  Source: Hummingbird hardware formula plus repeat=3 endpoint validation.
- repeat=1 GPU smoke: waiting for user manual WSL GPU execution.
- repeat=5 final P99 benchmark: user GPU run completed for the current RTX 5060
  Laptop GPU, CUDA 12.8, SM120 FP32 SIMT CUTLASS GEMM, M=N=K=2048 setup.
- Final plotting/report script: `tools/plot_cutlass_realtime_results.py`.
- Final artifacts were generated under
  `results/cutlass_realtime_compare_split52_repeat5_20260625_141255/figures/`
  and `final_report.md`. These artifacts are not committed to Git.
- The main conclusion is limited to the current GPU, current CUTLASS kernel, and
  current benchmark configuration.
