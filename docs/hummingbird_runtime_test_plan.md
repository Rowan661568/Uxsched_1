# Hummingbird Runtime Test Plan

## Status Legend

Allowed status values:

- IMPLEMENTED
- COMPILE VERIFIED
- RUNTIME VERIFIED
- CORRECTNESS VERIFIED
- PERFORMANCE VERIFIED
- NOT TESTED
- BLOCKED
- FAILED

## Stage A: Fixed Split Correctness

| Test | Status | Required Before Passing |
| --- | --- | --- |
| Native CUDA baseline | NOT TESTED | Run workload without UXSched. |
| UXSched NATIVE | NOT TESTED | `UXSCHED_CUDA_RUNTIME_STRATEGY=NATIVE`. |
| HB_FIXED checksum | NOT TESTED | LP split output equals baseline. |
| Event ordering | NOT TESTED | event after split group is not early. |
| Stream sync | NOT TESTED | stream sync waits all children. |
| Device/context sync | NOT TESTED | sync waits all children. |
| Native fallback | NOT TESTED | unsupported kernel runs correctly. |

Gate 1 requires HB_FIXED GPU correctness and checksum success.

Current environment status: BLOCKED. `nvidia-smi` reports GPU access blocked by
the operating system, so Gate 1 cannot be executed in this session.

## Stage B: HP/LP Runtime Smoke

| Test | Status |
| --- | --- |
| HP passthrough | NOT TESTED |
| HP arrival stops new LP split | BLOCKED |
| LP at most one split in flight | NOT TESTED |
| Global HPF active | NOT TESTED |
| No Local fallback | NOT TESTED |

Gate 2 requires Global HP/LP smoke to pass.

## Stage C: Kernel Tick

Status: BLOCKED until profiler produces real split timing.

Required metrics:

- tick count;
- tick miss;
- late launch;
- actual interval;
- preemption delay;
- LP throughput.

## Stage D: Small Bubble

Status: BLOCKED until explicit bubble hints exist.

Required microbenchmarks:

- CPU sleep/processing gap;
- `cudaMemcpyAsync`;
- stream synchronization;
- explicit bubble hint begin/end.

## Stage E: Large Bubble

Status: BLOCKED until large-bubble threshold and consolidation exist.

Required checks:

- threshold trigger;
- consolidation count;
- HP unexpected arrival delay;
- LP throughput impact.

## Stage F: End-to-End

| Workload | Status |
| --- | --- |
| open_resnet_like | NOT TESTED |
| CUTLASS ResNet-like | BLOCKED |

Comparisons:

- Standalone HP;
- Native HP+LP;
- UXSched Native;
- UXSched HB_FIXED;
- UXSched HB_RUNTIME;
- original Hummingbird.
