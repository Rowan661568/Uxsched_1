# Hummingbird Runtime Bubble Detection

## Current Status

Status: BLOCKED.

No small-bubble hint API, automatic CUDA API pattern detection, large-bubble
timeout, or consolidation implementation exists in the current branch.

## Target Small Bubble Path

First implementation should use explicit hints:

```cpp
uxschedHbBubbleBegin(device, SMALL_BUBBLE);
uxschedHbBubbleEnd(device);
```

or the equivalent UXSched hint/XCLI mechanism. The marker must update
per-device shared state and be visible to LP runtimes without per-split xserver
round trips.

## Target Large Bubble Path

Large bubble begins when:

- `hp_pending=false`;
- the conservative threshold expires;
- no new HP command appears.

Consolidation must:

- merge only unlaunched splits of the same original kernel;
- not cross stream;
- not cross command group;
- preserve grid/offset coverage;
- cap estimated runtime;
- stop launching new consolidated work when HP arrives.

## Current Implementation Gaps

| Feature | Status |
| --- | --- |
| explicit small-bubble hint | BLOCKED |
| CUDA API pattern detection | BLOCKED |
| NCCL pattern detection | BLOCKED |
| large bubble threshold | BLOCKED |
| consolidation queue | BLOCKED |
| consolidation metrics | BLOCKED |

