# Hummingbird Runtime Results

## Build Results

| Build | Command Summary | Status |
| --- | --- | --- |
| HB enabled CUDA | `cmake -S . -B build-hb -DPLATFORM_CUDA=ON -DUXSCHED_ENABLE_HB_SPLIT=ON -DBUILD_TEST=OFF -DCMAKE_INSTALL_INCLUDEDIR=include` then `cmake --build build-hb --target halcuda shimcuda -j2` | COMPILE VERIFIED |
| Native default CUDA | `cmake -S . -B build-native -DPLATFORM_CUDA=ON -DBUILD_TEST=OFF -DCMAKE_INSTALL_INCLUDEDIR=include` then `cmake --build build-native --target halcuda shimcuda -j2` | COMPILE VERIFIED |

The default build produced a clock-skew warning once, but `halcuda` and
`shimcuda` were built successfully.

## Runtime Results

| Result | Status |
| --- | --- |
| GPU visibility | BLOCKED |
| HB_FIXED GPU run | BLOCKED |
| HP passthrough GPU run | NOT TESTED |
| LP split checksum | NOT TESTED |
| synchronization correctness | NOT TESTED |
| kernel profiler | BLOCKED |
| kernel tick | BLOCKED |
| small bubble | BLOCKED |
| large bubble / consolidation | BLOCKED |
| CUTLASS workload | BLOCKED |

No HP latency, LP throughput, or preemption-delay performance claims are made
from this branch yet.

GPU access check:

```text
nvidia-smi
Failed to initialize NVML: GPU access blocked by the operating system
Failed to properly shut down NVML: GPU access blocked by the operating system
```

Because Gate 1 requires real HB_FIXED GPU execution and checksum correctness,
later runtime stages remain blocked in this environment.

## Required Metrics for Future Runs

- HP P50/P95/P99;
- HP relative slowdown;
- LP throughput;
- LP normalized throughput;
- preemption delay;
- split size;
- split count;
- tick count;
- tick miss;
- small bubble count;
- small bubble harvested time;
- large bubble count;
- consolidation count;
- fallback count;
- transform overhead;
- profiling overhead;
- checksum/error.
