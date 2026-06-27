#pragma once

#include "xsched/types.h"
#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

constexpr HwQueueHandle kDefaultStreamHwQueueTag = 0xd5f7000000000000ULL;
constexpr HwQueueHandle kDefaultStreamHwQueueMask = 0x0000FFFFFFFFFFFFULL;

inline HwQueueHandle GetHwQueueHandle(CUstream stream)
{
    return (HwQueueHandle)stream;
}

inline HwQueueHandle GetDefaultStreamHwQueueHandle(CUcontext context)
{
    return kDefaultStreamHwQueueTag |
           (((HwQueueHandle)(uintptr_t)context) & kDefaultStreamHwQueueMask);
}

inline HwQueueHandle GetHwQueueHandle(CUstream stream, CUcontext context)
{
    if (stream != nullptr) return GetHwQueueHandle(stream);
    return GetDefaultStreamHwQueueHandle(context);
}

} // namespace xsched::cuda
