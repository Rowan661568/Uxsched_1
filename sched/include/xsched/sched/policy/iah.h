#pragma once

#include <unordered_map>

#include "xsched/types.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

/// @brief Interference-Aware Heterogeneous scheduling policy.
///
/// Extends HHPF with cross-device interference modelling.
/// When a high-priority task runs on one device (e.g. GPU),
/// lower-priority tasks on *different* devices (e.g. NPU) may
/// still execute if the interference risk (memory bandwidth
/// contention on the SoC) is low.
///
/// Each device maintains a memory-bandwidth intensity factor
/// in [0.0, 1.0] that can be set via MemoryIntensityHint:
///   - 0.0 = compute-bound (no bandwidth pressure)
///   - 1.0 = memory-bound (maximum bandwidth pressure)
///
/// Decision logic per scheduler tick:
///   1. Find the globally highest-priority task and its device (G).
///   2. For each device D, select its local highest-priority task:
///      a. If D == G → unconditionally resume (HHPF logic).
///      b. If the task on D has equal priority to G → resume.
///      c. Otherwise → check interference between D and G:
///         - If intensity(D) + intensity(G) > HIGH_THRESHOLD → suspend.
///         - Otherwise → resume (co-schedule when safe).
class InterferenceAwareHeterogeneousPolicy : public Policy
{
public:
    InterferenceAwareHeterogeneousPolicy()
        : Policy(kPolicyInterferenceAwareHeterogeneous) {}
    virtual ~InterferenceAwareHeterogeneousPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    Priority GetPriority(XQueueHandle handle);
    double GetDeviceIntensity(XDevice device) const;
    bool IsHighInterference(XDevice dev_a, XDevice dev_b) const;

    static constexpr double kIntensityDefault = 0.5;
    static constexpr double kIntensityThreshold = 1.2;  // sum threshold for high interference

    std::unordered_map<XQueueHandle, Priority> priorities_;
    std::unordered_map<XDevice, double> device_intensities_;
};

} // namespace xsched::sched
