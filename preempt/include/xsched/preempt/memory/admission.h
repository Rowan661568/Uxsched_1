#pragma once

#include <cstddef>
#include <cstdint>

#include "xsched/types.h"
#include "xsched/utils/common.h"

namespace xsched::preempt
{

struct MemoryBackendCaps
{
    bool query_mem_info = false;
    bool unified_memory = false;
    bool async_prefetch = false;
    bool evict_to_host = false;
};

class XpuMemoryBackend
{
public:
    virtual ~XpuMemoryBackend() = default;

    virtual const char *Name() const = 0;
    virtual bool Enabled() const = 0;
    virtual MemoryBackendCaps Caps() const = 0;

    virtual bool QueryMemory(uintptr_t context, int64_t device,
                             size_t &free, size_t &total) = 0;
    virtual bool EvictToHost(uintptr_t context, int64_t device,
                             uintptr_t ptr, size_t size) = 0;
    virtual bool PrefetchToDevice(uintptr_t context, int64_t device,
                                  uintptr_t ptr, size_t size) = 0;
    virtual bool Synchronize(uintptr_t context, int64_t device) = 0;
};

class MemoryAdmissionManager
{
public:
    STATIC_CLASS(MemoryAdmissionManager);

    static void RegisterRegion(XpuMemoryBackend *backend, uintptr_t ptr, size_t size,
                               uintptr_t context, int64_t device, XQueueHandle owner = 0);
    static void UnregisterRegion(XpuMemoryBackend *backend, uintptr_t ptr);
    static void TouchRegion(XpuMemoryBackend *backend, uintptr_t context, int64_t device,
                            uintptr_t ptr, XQueueHandle owner);

    static void SetQueuePriority(XQueueHandle queue, Priority priority);
    static void RemoveQueue(XQueueHandle queue);

    static void OnQueueSuspend(XpuMemoryBackend *backend, uintptr_t context,
                               int64_t device, XQueueHandle queue);
    static void BeforeQueueResume(XpuMemoryBackend *backend, uintptr_t context,
                                  int64_t device, XQueueHandle queue);
};

} // namespace xsched::preempt
