#include <cstdlib>
#include <cstring>
#include <strings.h>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/preempt/memory/admission.h"
#include "xsched/ascend/hal/driver.h"
#include "xsched/ascend/hal/memory_manager.h"

using namespace xsched::ascend;
using namespace xsched::preempt;

namespace
{

constexpr const char *kEnableEnv = "XSCHED_ASCEND_MEM_OVERSUB";

bool EnvEnabled()
{
    static bool enabled = []() {
        const char *env = std::getenv(kEnableEnv);
        return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 &&
               strcasecmp(env, "off") != 0 && strcasecmp(env, "false") != 0;
    }();
    return enabled;
}

class AscendMemoryBackend : public XpuMemoryBackend
{
public:
    virtual const char *Name() const override { return "ascend"; }
    virtual bool Enabled() const override { return EnvEnabled(); }

    virtual MemoryBackendCaps Caps() const override
    {
        return MemoryBackendCaps {
            .query_mem_info = true,
            .unified_memory = false,
            .async_prefetch = false,
            .evict_to_host = false,
        };
    }

    virtual bool QueryMemory(uintptr_t context, int64_t device,
                             size_t &free, size_t &total) override
    {
        UNUSED(context);
        UNUSED(device);
        aclError ret = Driver::rtGetMemInfo(ACL_HBM_MEM, &free, &total);
        if (ret != ACL_SUCCESS) {
            XWARN("SchedUM[ascend]: aclrtGetMemInfo failed, error=%d", ret);
            return false;
        }
        return true;
    }

    virtual bool EvictToHost(uintptr_t context, int64_t device,
                             uintptr_t ptr, size_t size) override
    {
        UNUSED(context);
        UNUSED(device);
        UNUSED(ptr);
        UNUSED(size);
        return false;
    }

    virtual bool PrefetchToDevice(uintptr_t context, int64_t device,
                                  uintptr_t ptr, size_t size) override
    {
        UNUSED(context);
        UNUSED(device);
        UNUSED(ptr);
        UNUSED(size);
        return false;
    }

    virtual bool Synchronize(uintptr_t context, int64_t device) override
    {
        UNUSED(context);
        UNUSED(device);
        return true;
    }
};

AscendMemoryBackend g_ascend_backend;

} // namespace

void AscendMemoryManager::RegisterDeviceAllocation(void *ptr, size_t size,
                                                   aclrtContext ctx, int32_t dev)
{
    MemoryAdmissionManager::RegisterRegion(&g_ascend_backend, (uintptr_t)ptr, size,
                                           (uintptr_t)ctx, (int64_t)dev);
}

void AscendMemoryManager::UnregisterAllocation(void *ptr)
{
    MemoryAdmissionManager::UnregisterRegion(&g_ascend_backend, (uintptr_t)ptr);
}

void AscendMemoryManager::OnQueueSuspend(aclrtContext ctx, int32_t dev, XQueueHandle queue)
{
    MemoryAdmissionManager::OnQueueSuspend(&g_ascend_backend, (uintptr_t)ctx,
                                           (int64_t)dev, queue);
}

void AscendMemoryManager::BeforeQueueResume(aclrtContext ctx, int32_t dev, XQueueHandle queue)
{
    MemoryAdmissionManager::BeforeQueueResume(&g_ascend_backend, (uintptr_t)ctx,
                                              (int64_t)dev, queue);
}
