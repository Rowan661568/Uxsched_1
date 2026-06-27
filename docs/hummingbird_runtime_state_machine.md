# Hummingbird Runtime State Machine

## Current Status

Status: BLOCKED for runtime state machine implementation.

This document records the intended state machine. The current code only defines
the `DeviceRuntimeState` data carrier in the RuntimeStrategy interface and stores
the last state inside `HummingbirdRuntimeStrategy`. No cross-process coordinator
or event-driven state transition engine exists yet.

## Target States

| State | Status | Intended Meaning |
| --- | --- | --- |
| `IDLE` | BLOCKED | No HP work and no LP runtime work selected. |
| `HP_ACTIVE` | BLOCKED | HP queue is running or ready. LP launches must stop. |
| `WAIT_HP_DRAIN` | BLOCKED | HP just became idle; runtime waits for safe drain/transition. |
| `SMALL_BUBBLE` | BLOCKED | Short bubble where only short split/tick launches are allowed. |
| `LARGE_BUBBLE` | BLOCKED | Longer bubble where consolidation can be considered. |
| `LP_TICKING` | BLOCKED | LP split launcher controlled by kernel tick. |
| `PREEMPTING` | BLOCKED | HP arrived; stop issuing LP split commands. |
| `SUSPENDED` | BLOCKED | Global Scheduler suspended this LP XQueue. |

## Target Events

| Event | Status |
| --- | --- |
| `HP_READY` | BLOCKED |
| `HP_IDLE` | BLOCKED |
| `BUBBLE_BEGIN` | BLOCKED |
| `BUBBLE_END` | BLOCKED |
| `LARGE_BUBBLE_TIMEOUT` | BLOCKED |
| `GLOBAL_SUSPEND` | BLOCKED |
| `GLOBAL_RESUME` | BLOCKED |
| `SPLIT_COMPLETED` | BLOCKED |
| `STATE_EPOCH_CHANGED` | BLOCKED |

## Implementation Boundary

The current refactor prepares the interface:

```cpp
virtual void Suspend() = 0;
virtual void Resume() = 0;
virtual void Wait(CommandId command) = 0;
virtual void OnDeviceRuntimeState(const DeviceRuntimeState &state) = 0;
```

The next required implementation step is a per-device shared coordinator that
can publish HP readiness and selected LP queue state without per-split xserver
round trips.

