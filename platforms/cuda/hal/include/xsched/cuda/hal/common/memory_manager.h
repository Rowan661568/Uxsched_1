#pragma once

#include <cstddef>

#include "xsched/types.h"
#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

class CudaMemoryManager
{
public:
    static void RegisterManagedAllocation(CUdeviceptr ptr, size_t size,
                                          CUcontext ctx, CUdevice dev);
    static void UnregisterAllocation(CUdeviceptr ptr);

    static void TouchAllocation(CUdeviceptr ptr, CUcontext ctx, CUdevice dev, XQueueHandle queue);
    static void OnQueueSuspend(CUcontext ctx, CUdevice dev, XQueueHandle queue);
    static void BeforeQueueResume(CUcontext ctx, CUdevice dev, XQueueHandle queue);
};

} // namespace xsched::cuda
