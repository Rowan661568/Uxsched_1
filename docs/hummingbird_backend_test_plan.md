# Hummingbird Split Backend Test Plan

## 1. Build Tests

Already run on this machine:

```bash
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

Both builds must pass. The default build keeps HB runtime logic disabled because
`UXSCHED_ENABLE_HB_SPLIT` defaults to `OFF`.

## 2. Runtime Smoke: Native Compatibility

Run an existing UXSched CUDA workload with:

```bash
export LD_PRELOAD=/path/to/libshimcuda.so
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_LEVEL=1
export UXSCHED_CUDA_PREEMPT_BACKEND=NATIVE
```

Expected:

- no PTX transformation log;
- original UXSched Global Lv1 behavior;
- HP/LP correctness equal to current UXSched baseline.

## 3. Runtime Smoke: HP Passthrough

Use a high-priority process:

```bash
export XSCHED_AUTO_XQUEUE_PRIORITY=10
export UXSCHED_CUDA_PREEMPT_BACKEND=AUTO
export UXSCHED_HB_VERIFIED_KERNELS=hb_open_resnet_conv2d_kernel,hb_open_resnet_relu_kernel,hb_open_resnet_residual_add_kernel,hb_open_resnet_checksum_kernel
```

Expected log:

```text
[UXSCHED-HB] backend_selected=NATIVE reason=HIGH_PRIORITY_PASSTHROUGH
```

Expected correctness:

- no split-count log for HP kernels;
- HP checksum equals native;
- stream and event ordering unchanged.

## 4. Runtime Smoke: LP Fixed Splitting

Use the Hummingbird open-resnet-like Driver API PTX workload or an equivalent
PTX-visible custom kernel:

```bash
export XSCHED_AUTO_XQUEUE_PRIORITY=-10
export UXSCHED_CUDA_PREEMPT_BACKEND=AUTO
export UXSCHED_HB_SPLIT_BLOCKS=512
export UXSCHED_HB_STRICT=0
export UXSCHED_HB_VERIFIED_KERNELS=hb_open_resnet_conv2d_kernel,hb_open_resnet_relu_kernel,hb_open_resnet_residual_add_kernel,hb_open_resnet_checksum_kernel
```

Expected log:

```text
[UXSCHED-HB] transform_succeeded function=<name>
[UXSCHED-HB] capability=splittable
[UXSCHED-HB] split_blocks=512
[UXSCHED-HB] split_count=<N>
[UXSCHED-HB] xqueue=<...> lp_in_flight_threshold=1 batch_size=1
```

Expected correctness:

- LP checksum equals native or Hummingbird validated baseline;
- each split child uses the same original stream;
- subsequent event record fires only after all split children.

## 5. Fallback Tests

At least one unsupported kernel should be tested with `AUTO`:

- no PTX module available;
- kernel name not listed in `UXSCHED_HB_VERIFIED_KERNELS`;
- `extra` launch format;
- `XSCHED_AUTO_XQUEUE_LEVEL=2` or `3`;
- `kernelParams=nullptr`.

Expected:

- application still runs in `STRICT=0`;
- log contains `backend_selected=NATIVE reason=<reason>`;
- output equals native.

## 6. Synchronization Tests

Add or run workloads that explicitly exercise:

- `cuEventRecord` after a split launch;
- `cuEventSynchronize`;
- `cuStreamSynchronize`;
- `cuCtxSynchronize`;
- module unload after split completion;
- multiple streams in the same process.

Expected:

- no event completes before all split children complete;
- no stream/device sync returns early;
- module unload waits for outstanding split work.

## 7. Workload A: open_resnet_like

Use the current Hummingbird open-resnet-like workload because its PTX-visible
custom kernels have already been used with fixed split blocks in local
experiments. Required checks:

- HP process: priority `10`, passthrough;
- LP process: priority `-10`, split blocks `512`;
- Global Scheduler and HPF in xserver;
- checksum match;
- fallback switch by setting `UXSCHED_CUDA_PREEMPT_BACKEND=NATIVE`;
- compare HP latency and LP throughput.

## 8. Workload B: CUTLASS ResNet-like

Not implemented in this patch. Stage 1 test requirement remains:

- first compile and run CUTLASS workload in native mode;
- confirm whether CUTLASS kernels expose PTX and can pass the current transform;
- keep HB splitting disabled until transform correctness is verified;
- use native fallback for unsupported CUTLASS kernels.

## 9. Performance Metrics

For final repeat runs, collect:

- HP mean/P50/P95/P99 latency;
- HP relative slowdown versus standalone HP;
- LP throughput and normalized throughput;
- split count;
- estimated Lv1 preemption wait bound;
- transformation overhead;
- native fallback count;
- runtime error count.

