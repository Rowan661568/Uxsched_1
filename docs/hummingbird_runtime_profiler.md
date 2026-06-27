# Hummingbird Runtime Profiler

## Current Status

Status: BLOCKED.

No kernel profiler or automatic split-plan cache exists in the current branch.
`HB_FIXED` uses `UXSCHED_HB_SPLIT_BLOCKS`, default `512`.

## Target SplitPlan

```cpp
struct SplitPlan {
    uint32_t split_blocks;
    double estimated_exec_us;
    double launch_overhead_us;
    bool profiled;
    std::string reason;
};
```

## Target Inputs

| Input | Status |
| --- | --- |
| GPU UUID / compute capability | BLOCKED |
| SM count | BLOCKED |
| block threads | BLOCKED |
| occupancy | BLOCKED |
| register count | BLOCKED |
| static shared memory | BLOCKED |
| dynamic shared memory | BLOCKED |
| CUDA event timing | BLOCKED |
| launch overhead | BLOCKED |
| candidate split sizes | BLOCKED |

## Rules

- HP kernels must never be profiled or split.
- LP profiling failure must fall back to fixed 512 or Native according to the
  selected policy.
- Split plans must be persisted as auditable logs before being used for
  performance claims.
- No automatic split-size result is currently claimed.

