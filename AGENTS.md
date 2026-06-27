# UXSched Agent Rules

## UXSched-Hummingbird Integration Rules

### Project Paths

- UXSched: `/home/zm/project/UXSched`
- Hummingbird: `/home/zm/project/hummingbird`
- Python environment: `/home/zm/project/hummingbird/.venv`

### Architecture Invariants

1. UXSched is the only global scheduler.
2. UXSched CUDA shim is the only CUDA hook entry.
3. Do not use two CUDA `LD_PRELOAD` hook libraries.
4. Hummingbird is integrated as a CUDA runtime strategy inside UXSched.
5. Do not start an independent Hummingbird scheduler.
6. UXSched decides which XQueue may run.
7. Hummingbird Runtime only controls fine-grained execution of eligible LP kernels.
8. HP kernels always use passthrough and are never split.
9. Unsupported LP kernels must safely fall back to UXSched Native.
10. `/home/zm/project/hummingbird` is read-only unless the user explicitly changes this rule.

### Runtime Strategies

- `NATIVE`: original UXSched behavior.
- `HB_FIXED`: fixed-size LP splitting.
- `HB_RUNTIME`: future full Hummingbird runtime.
- `AUTO`: future capability-aware selection.

Currently, only `NATIVE` and `HB_FIXED` have implementation paths. `HB_RUNTIME`
and `AUTO` must not be described as complete until runtime validation proves
otherwise.

### Validation Rules

Use the following statuses precisely:

- IMPLEMENTED
- COMPILE VERIFIED
- RUNTIME VERIFIED
- CORRECTNESS VERIFIED
- GLOBAL SCHEDULING VERIFIED
- PERFORMANCE VERIFIED
- NOT TESTED
- BLOCKED
- FAILED

Compilation success is not runtime verification.

A feature cannot be marked correctness verified without real GPU execution and
output validation.

Do not fabricate performance data.

### Development Gates

Gate 1 must pass before implementing the complete runtime:

- HB_FIXED executes on a real GPU.
- LP produces more than one real split launch.
- The transformed CUfunction is actually submitted.
- Native and HB_FIXED checksums match.
- HP passthrough is verified.
- Native fallback is verified.
- Event, stream, context/device synchronization semantics are correct.
- Global Lv1 HPF smoke test passes without local fallback.

### CUTLASS Realtime Benchmark Goal

Reference `benchmarks/realtime_inference_latency.py` and replace the original
PyTorch/TorchVision workload with an open-source CUTLASS workload.

Under the same GPU, same CUTLASS kernel, same input sizes, same request model,
same scheduling policy, and same measurement boundaries, compare:

1. UXSched Global HPF + CUTLASS + `NATIVE`
2. UXSched Global HPF + CUTLASS + Hummingbird `HB_FIXED`

The final performance goal is to show, under fair, correct, and repeatable
conditions, that UXSched + Hummingbird + CUTLASS improves HP P99 latency over
the original UXSched baseline.

Before making any P99 improvement claim, prove all of the following:

- CUTLASS output is correct.
- Native and `HB_FIXED` execute the same amount of work.
- HB transform, parent launch, and child launch statistics are real and greater
  than zero.
- There is no Native fallback.
- There is no `NO_XQUEUE`.
- Split group and CUDA Runtime synchronization semantics are correct.
- Parent completion occurs only after all children complete.
- The benchmark uses at least `repeat=3`.
- No advantage is obtained by changing workload, reducing computation, skipping
  kernels, or changing measurement boundaries.
- `repeat=1` runs are smoke tests only and must not be used for a final P99
  improvement claim.

For the current RTX 5060 Laptop GPU and SM120 FP32 SIMT CUTLASS GEMM kernel,
the recommended fixed HB split size is `52` blocks. This comes from the
Hummingbird hardware formula using 26 SMs and 2 resident CUTLASS blocks per SM,
then repeat=3 end-to-end validation against split size 64. This is not automatic
split selection, not runtime profiling, and not a global optimum for other GPUs
or kernels.

### Git Rules

- Work on `feature/hummingbird-split-backend`.
- Inspect `git status`, current branch, and recent commits before changes.
- Keep commits small and independently buildable.
- Do not commit build directories or generated benchmark outputs unless
  explicitly requested.
- Update `hb_integration_status.md` and `docs/codex_handoff.md` before ending a
  work session.
