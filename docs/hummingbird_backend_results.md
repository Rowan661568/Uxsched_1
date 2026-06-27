# Hummingbird Split Backend Results

## 1. Build Results

Baseline commit:

```text
73039c69e67a0e66bd28741e8784e7dda65749ed
```

Branch:

```text
feature/hummingbird-split-backend
```

Submodules were initially uninitialized. After running:

```bash
git submodule update --init --recursive
```

the required third-party sources were available.

### HB-enabled CUDA build

Command:

```bash
cmake -S . -B build-hb \
  -DPLATFORM_CUDA=ON \
  -DUXSCHED_ENABLE_HB_SPLIT=ON \
  -DBUILD_TEST=OFF \
  -DCMAKE_INSTALL_INCLUDEDIR=include
cmake --build build-hb --target halcuda shimcuda -j2
```

Result:

```text
Built target halcuda
Built target shimcuda
```

### Default native CUDA build

Command:

```bash
cmake -S . -B build-native \
  -DPLATFORM_CUDA=ON \
  -DBUILD_TEST=OFF \
  -DCMAKE_INSTALL_INCLUDEDIR=include
cmake --build build-native --target halcuda shimcuda -j2
```

Result:

```text
Built target halcuda
Built target shimcuda
```

### Service targets for Global Lv1 smoke

Command:

```bash
cmake --build build-hb --target xserver xcli -j2
cmake --build build-native --target xserver xcli -j2
```

Result:

```text
Built target xserver
Built target xcli
```

Confirmed paths:

```text
build-hb/platforms/cuda/libhalcuda.so
build-hb/platforms/cuda/libshimcuda.so
build-native/platforms/cuda/libhalcuda.so
build-native/platforms/cuda/libshimcuda.so
build-hb/service/xserver
build-hb/service/xcli
build-native/service/xserver
build-native/service/xcli
/home/zm/project/hummingbird/build-lite/benchmarks/hb_open_resnet_like_eval
/home/zm/project/hummingbird/build-lite/benchmarks/hb_open_resnet_like_runtime_eval
```

## 2. Runtime Results

2026-06-24 manual Gate 1 smoke result directory:

```text
results/hb_gate1_manual_20260624_163059
```

Artifact status:

| Case | Result |
| --- | --- |
| Native open_resnet_like LP | RAN on RTX 5060 |
| UXSched `NATIVE` LP | RAN with UXSched shim loaded |
| UXSched `HB_FIXED` LP | FAILED/PARTIAL: PTX transformed, launches fell back `NO_XQUEUE` |
| UXSched `HB_FIXED` HP passthrough probe | RAN: `HIGH_PRIORITY_PASSTHROUGH` |
| UXSched `HB_FIXED` unverified-kernel fallback probe | RAN, but old code reported `<unknown>/PTX_UNAVAILABLE`; fixed code must rerun |
| Event-boundary sync probe | RAN workload-internal split counters only; no UXSched split trace |
| xserver HPF | Started, accepted clients, stopped |

No benchmark numbers are claimed in this document.

Observed evidence:

- `transform_succeeded` appeared for open_resnet_like kernels;
- `backend_selected=NATIVE reason=NO_XQUEUE` appeared on LP launches;
- transformed CUfunction child launch evidence: not observed;
- child completion evidence: not observed;
- workload fields `lp_split_launched` and `fixed_split_blocks` are workload
  internal counters and are not UXSched backend split evidence;
- Global Lv1 HPF smoke: not run because single-process HB_FIXED split execution
  did not pass.

2026-06-24 NO_XQUEUE fix:

```text
RUNTIME VERIFIED for stream-to-XQueue association and fixed split launch.
```

Implemented after the manual result:

- launch-time auto-association from CUDA stream to stable XQueue when
  `XSCHED_AUTO_XQUEUE=ON`;
- default stream support through a per-context synthetic HwQueue handle;
- `UXSCHED_XQUEUE_TRACE=1` logs for API, pid/tid, CUDA context, stream handle,
  default-stream flag, auto-create attempt/result, HwQueue/XQueue pointers,
  lookup result, `KernelLaunch.xqueue`, runtime strategy, and fallback path;
- `KERNEL_NOT_VERIFIED` fallback metadata for PTX entries that are present but
  excluded by the verified kernel list;
- explicit `transformed_module_loaded`, `parent_launch_submitted`,
  `child_launch_submitted`, `child_launch_completed`, and
  `parent_launch_completed` logs;
- minimal Driver API probe for default and explicit streams:
  `tools/hb_xqueue_probe.cpp`.

Manual GPU rerun after the XQueue fix:

```text
results/hb_gate1_after_xqueue_fix_20260624_170107
```

Observed UXSched backend evidence:

| Case | Runtime evidence |
| --- | --- |
| `probe_default_stream_hb_fixed_lp` | `transform_count=1`, `parent_launch_count=1`, `child_launch_count=2`, `transformed_launch_count=2`, `NO_XQUEUE=0`, `mismatches=0` |
| `probe_explicit_stream_hb_fixed_lp` | `transform_count=1`, `parent_launch_count=1`, `child_launch_count=2`, `transformed_launch_count=2`, `NO_XQUEUE=0`, `mismatches=0` |
| `uxsched_hb_fixed_lp` | `transform_count=4`, `parent_launch_count=78`, `child_launch_count=312`, `transformed_launch_count=312`, `fallback_count=0`, `NO_XQUEUE=0` |
| `probe_fallback_ptx_unavailable` | `reason=PTX_UNAVAILABLE`, normal exit |
| `probe_fallback_kernel_not_verified` | `reason=KERNEL_NOT_VERIFIED`, `function=hb_xqueue_probe_kernel`, normal exit |

Important limitations of that artifact set:

- `sync_event_boundary_probe` did not trigger UXSched HB_FIXED split; its
  UXSched transform/parent/child/transformed counts were all zero. Its
  `fixed_split_blocks=16` and `lp_split_launched=1402` were workload-internal
  counters and are not UXSched backend evidence.
- The ordinary `hb_open_resnet_like_eval` UXSched cases did not emit checksum,
  output hash, or output element count. Several ordinary UXSched open_resnet_like
  cases returned 139, so they are not Gate 1 correctness evidence.
- Gate 1 correctness and synchronization remained unverified before the current
  runner update.

Build checks after this fix:

```bash
tools/build_hb_xqueue_probe.sh build-hb/hb_xqueue_probe
cmake --build build-hb --target halcuda shimcuda -j2
cmake --build build-native --target halcuda shimcuda -j2
bash -n tools/run_hb_gate1_smoke.sh tools/build_hb_xqueue_probe.sh
```

Current Gate 1 runner update:

- `tools/hb_xqueue_probe.cpp` now supports
  `--sync event|stream|context|same-stream|parent` and emits checksum,
  output hash, output element count, pass flags, child/parent completion logs,
  and same-stream dependency validation.
- `tools/run_hb_gate1_smoke.sh` now creates these correctness cases:
  `native_correctness`, `uxsched_native_correctness`,
  `uxsched_hb_fixed_correctness`.
- All correctness cases use `hb_open_resnet_like_runtime_eval` with identical
  batch size, channels, height, width, num blocks, deterministic memset input
  and weight initialization, one correctness iteration, and
  `lp-correctness-sync-boundary=iteration`.
- The workload correctness split size is set to `4096`, while UXSched HB_FIXED
  uses `UXSCHED_HB_SPLIT_BLOCKS=512`, so UXSched backend stats remain the split
  evidence.
- The runner now writes `gate1_summary.env` with pass/fail fields for
  correctness, checksum/hash/count comparison, HB backend counts, sync probes,
  HP passthrough, fallback cases, CUDA errors, local scheduler fallback, and
  `gate1_pass`.

Pending runtime checks after this update:

- manual rerun of the updated Gate 1 runner in normal WSL GPU environment;
- Native, UXSched NATIVE, and UXSched HB_FIXED correctness checksum/hash/count
  comparison;
- event, stream, context, same-stream ordering, and parent-completion sync
  probes with real transformed child launches;
- HP passthrough with transformed and child launch counts equal to zero;
- separate `PTX_UNAVAILABLE` and `KERNEL_NOT_VERIFIED` fallback cases;
- no CUDA illegal memory access, invalid argument, segmentation fault, or local
  scheduler fallback.

## 3. Known Missing Results

- HP P50/P95/P99 latency for UXSched-HB AUTO;
- LP normalized throughput for UXSched-HB AUTO;
- estimated preemption delay with split size 512;
- transformation overhead;
- fallback count across workload runs;
- CUTLASS ResNet-like native run;
- CUTLASS transformability check.

## 4. Observed Build Notes

- The top-level CMake currently expects `CMAKE_INSTALL_INCLUDEDIR`; the build
  commands above pass `-DCMAKE_INSTALL_INCLUDEDIR=include`.
- `UXSCHED_ENABLE_HB_SPLIT` defaults to `OFF`.
- The backend compiles in both enabled and disabled builds.
- No Hummingbird source files were modified.
