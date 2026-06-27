#include <map>

#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/iah.h"

using namespace xsched::sched;

void InterferenceAwareHeterogeneousPolicy::Sched(const Status &status)
{
    // Phase 1: Find the globally highest-priority ready task and its device.
    Priority global_max_prio = PRIORITY_MIN;
    XDevice global_max_device = 0;
    for (auto &st : status.xqueue_status) {
        if (!st.second->ready) continue;
        Priority prio = GetPriority(st.second->handle);
        if (prio > global_max_prio) {
            global_max_prio = prio;
            global_max_device = st.second->device;
        }
    }

    // Phase 2: Find the highest-priority ready task per device.
    std::map<XDevice, Priority> device_max_prio;
    std::map<XDevice, XQueueHandle> device_best_handle;
    for (auto &st : status.xqueue_status) {
        if (!st.second->ready) continue;
        XDevice dev = st.second->device;
        XQueueHandle handle = st.second->handle;
        Priority prio = GetPriority(handle);

        auto it = device_max_prio.find(dev);
        if (it == device_max_prio.end() || prio > it->second) {
            device_max_prio[dev] = prio;
            device_best_handle[dev] = handle;
        }
    }

    // Phase 3: Per-XQueue decision.
    for (auto &st : status.xqueue_status) {
        XQueueHandle handle = st.second->handle;
        if (!st.second->ready) continue;

        XDevice dev = st.second->device;
        Priority prio = GetPriority(handle);

        bool should_run = false;

        // Is this the local-best XQueue on its device?
        auto best_it = device_best_handle.find(dev);
        if (best_it != device_best_handle.end() && best_it->second == handle) {
            if (prio >= global_max_prio) {
                // a) This task has the global highest priority → run.
                should_run = true;
            } else {
                // b) This device's best is lower than the global best.
                if (global_max_device == dev) {
                    // Same device → classic HPF: the higher-priority one wins.
                    should_run = false;
                } else {
                    // Cross-device: check interference risk.
                    should_run = !IsHighInterference(dev, global_max_device);
                }
            }
        }

        if (should_run) {
            this->Resume(handle);
        } else {
            this->Suspend(handle);
        }
    }
}

void InterferenceAwareHeterogeneousPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    switch (hint->Type())
    {
    case kHintTypePriority:
    {
        auto h = std::dynamic_pointer_cast<const PriorityHint>(hint);
        XASSERT(h != nullptr, "hint type not match");
        Priority priority = h->Prio();
        if (priority < PRIORITY_MIN) priority = PRIORITY_MIN;
        if (priority > PRIORITY_MAX) priority = PRIORITY_MAX;
        if (priority != h->Prio()) {
            XWARN("IAH: priority %d not in range [%d, %d], override for XQueue 0x" FMT_64X " to %d",
                  h->Prio(), PRIORITY_MIN, PRIORITY_MAX, h->Handle(), priority);
        }
        XINFO("IAH: set priority %d for XQueue 0x" FMT_64X, priority, h->Handle());
        priorities_[h->Handle()] = priority;
        break;
    }
    case kHintTypeMemoryIntensity:
    {
        auto h = std::dynamic_pointer_cast<const MemoryIntensityHint>(hint);
        XASSERT(h != nullptr, "hint type not match");
        double intensity = h->Intensity();
        if (intensity < 0.0) intensity = 0.0;
        if (intensity > 1.0) intensity = 1.0;
        XINFO("IAH: set memory intensity %.2f for device 0x" FMT_64X,
              intensity, h->Device());
        device_intensities_[h->Device()] = intensity;
        break;
    }
    default:
        XWARN("IAH: unsupported hint type: %d", hint->Type());
        break;
    }
}

Priority InterferenceAwareHeterogeneousPolicy::GetPriority(XQueueHandle handle)
{
    auto it = priorities_.find(handle);
    if (it != priorities_.end()) return it->second;
    return PRIORITY_DEFAULT;
}

double InterferenceAwareHeterogeneousPolicy::GetDeviceIntensity(XDevice device) const
{
    auto it = device_intensities_.find(device);
    if (it != device_intensities_.end()) return it->second;
    return kIntensityDefault;
}

bool InterferenceAwareHeterogeneousPolicy::IsHighInterference(
    XDevice dev_a, XDevice dev_b) const
{
    if (dev_a == dev_b) return true;  // same device → always interferes
    double intensity_a = GetDeviceIntensity(dev_a);
    double intensity_b = GetDeviceIntensity(dev_b);
    return (intensity_a + intensity_b) > kIntensityThreshold;
}
