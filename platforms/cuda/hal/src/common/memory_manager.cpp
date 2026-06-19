#include <cstdlib>
#include <cstring>
#include <mutex>
#include <strings.h>
#include <unordered_map>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/preempt/memory/admission.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/memory_manager.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

namespace
{

constexpr const char *kEnableEnv = "XSCHED_CUDA_MEM_OVERSUB";

bool EnvEnabled()
{
    static bool enabled = []() {
        const char *env = std::getenv(kEnableEnv);
        return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 &&
               strcasecmp(env, "off") != 0 && strcasecmp(env, "false") != 0;
    }();
    return enabled;
}

class CudaMemoryBackend : public XpuMemoryBackend
{
public:
    virtual const char *Name() const override { return "cuda"; }
    virtual bool Enabled() const override { return EnvEnabled(); }

    virtual MemoryBackendCaps Caps() const override
    {
        return MemoryBackendCaps {
            .query_mem_info = true,
            .unified_memory = true,
            .async_prefetch = true,
            .evict_to_host = true,
        };
    }

    virtual bool QueryMemory(uintptr_t context, int64_t device,
                             size_t &free, size_t &total) override
    {
        UNUSED(context);
        UNUSED(device);
        CUresult ret = Driver::MemGetInfo_v2(&free, &total);
        if (ret != CUDA_SUCCESS) {
            XWARN("SchedUM[cuda]: cuMemGetInfo_v2 failed, error=%d", ret);
            return false;
        }
        return true;
    }

    virtual bool EvictToHost(uintptr_t context, int64_t device,
                             uintptr_t ptr, size_t size) override
    {
        UNUSED(device);
        CUstream stream = MigrationStream((CUcontext)context);
        if (stream == nullptr || ptr == 0 || size == 0) return false;

        CUmemLocation location {
            .type = CU_MEM_LOCATION_TYPE_HOST,
            .id = 0,
        };
        CUresult ret = Driver::MemPrefetchAsync_v2_ptsz((CUdeviceptr)ptr, size, location,
                                                        0, stream);
        if (ret == CUDA_SUCCESS) return true;
        if (ret == CUDA_ERROR_INVALID_DEVICE) {
            XWARN("SchedUM[cuda]: host prefetch unsupported for region 0x" FMT_64X
                  ", size=%zu; using logical eviction fallback",
                  (uint64_t)ptr, size);
            return true;
        }
        XWARN("SchedUM[cuda]: failed to evict region 0x" FMT_64X ", size=%zu, error=%d",
              (uint64_t)ptr, size, ret);
        return false;
    }

    virtual bool PrefetchToDevice(uintptr_t context, int64_t device,
                                  uintptr_t ptr, size_t size) override
    {
        CUstream stream = MigrationStream((CUcontext)context);
        if (stream == nullptr || ptr == 0 || size == 0) return false;

        CUmemLocation location {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = (int)device,
        };
        CUresult ret = Driver::MemPrefetchAsync_v2_ptsz((CUdeviceptr)ptr, size, location,
                                                        0, stream);
        if (ret == CUDA_SUCCESS) return true;
        XWARN("SchedUM[cuda]: failed to prefetch region 0x" FMT_64X
              ", size=%zu, dst=%d, error=%d",
              (uint64_t)ptr, size, (int)device, ret);
        return false;
    }

    virtual bool Synchronize(uintptr_t context, int64_t device) override
    {
        UNUSED(device);
        CUstream stream = MigrationStream((CUcontext)context);
        if (stream == nullptr) return false;
        CUresult ret = Driver::StreamSynchronize(stream);
        if (ret == CUDA_SUCCESS) return true;
        XWARN("SchedUM[cuda]: migration stream synchronization failed, error=%d", ret);
        return false;
    }

private:
    CUstream MigrationStream(CUcontext ctx)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = streams_.find(ctx);
        if (it != streams_.end()) return it->second;

        CUstream stream = nullptr;
        CUresult ret = Driver::StreamCreate(&stream, CU_STREAM_NON_BLOCKING);
        if (ret != CUDA_SUCCESS) {
            XWARN("SchedUM[cuda]: failed to create migration stream, error=%d", ret);
            return nullptr;
        }
        streams_[ctx] = stream;
        return stream;
    }

    std::mutex mtx_;
    std::unordered_map<CUcontext, CUstream> streams_;
};

CudaMemoryBackend g_cuda_backend;

} // namespace

void CudaMemoryManager::RegisterManagedAllocation(CUdeviceptr ptr, size_t size,
                                                  CUcontext ctx, CUdevice dev)
{
    MemoryAdmissionManager::RegisterRegion(&g_cuda_backend, (uintptr_t)ptr, size,
                                           (uintptr_t)ctx, (int64_t)dev);
}

void CudaMemoryManager::UnregisterAllocation(CUdeviceptr ptr)
{
    MemoryAdmissionManager::UnregisterRegion(&g_cuda_backend, (uintptr_t)ptr);
}

void CudaMemoryManager::TouchAllocation(CUdeviceptr ptr, CUcontext ctx,
                                        CUdevice dev, XQueueHandle queue)
{
    MemoryAdmissionManager::TouchRegion(&g_cuda_backend, (uintptr_t)ctx,
                                        (int64_t)dev, (uintptr_t)ptr, queue);
}

void CudaMemoryManager::OnQueueSuspend(CUcontext ctx, CUdevice dev, XQueueHandle queue)
{
    MemoryAdmissionManager::OnQueueSuspend(&g_cuda_backend, (uintptr_t)ctx,
                                           (int64_t)dev, queue);
}

void CudaMemoryManager::BeforeQueueResume(CUcontext ctx, CUdevice dev, XQueueHandle queue)
{
    MemoryAdmissionManager::BeforeQueueResume(&g_cuda_backend, (uintptr_t)ctx,
                                              (int64_t)dev, queue);
}
