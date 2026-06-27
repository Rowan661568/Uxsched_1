#include <list>
#include <cstdlib>
#include <cstring>
#include <string>
#include <strings.h>
#include <unordered_map>

#include "xsched/xqueue.h"
#include "xsched/utils/map.h"
#include "xsched/protocol/def.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cuda/hal.h"
#include "xsched/cuda/shim/shim.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/level1/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_command.h"
#include "xsched/cuda/hal/common/memory_manager.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/hint.h"

using namespace xsched::preempt;

namespace xsched::cuda
{

static utils::ObjectMap<CUevent, std::shared_ptr<CudaEventRecordCommand>> g_events;

// ---------------------------------------------------------------------------
// IAH kernel intensity auto-detection
// ---------------------------------------------------------------------------
static std::unordered_map<CUfunction, std::string> g_func_names;
static std::unordered_map<XDevice, double> g_device_intensity;

static double EstimateKernelIntensity(const std::string &name)
{
    // compute-bound patterns
    if (name.find("conv") != std::string::npos ||
        name.find("Conv") != std::string::npos ||
        name.find("gemm") != std::string::npos ||
        name.find("Gemm") != std::string::npos ||
        name.find("fft") != std::string::npos ||
        name.find("FFT") != std::string::npos ||
        name.find("mma") != std::string::npos ||
        name.find("warp") != std::string::npos ||
        name.find("wmma") != std::string::npos)
        return 0.3;

    // memory-bound patterns
    if (name.find("copy") != std::string::npos ||
        name.find("Copy") != std::string::npos ||
        name.find("memcpy") != std::string::npos ||
        name.find("Memcpy") != std::string::npos ||
        name.find("broadcast") != std::string::npos ||
        name.find("Broadcast") != std::string::npos ||
        name.find("relu") != std::string::npos ||
        name.find("Relu") != std::string::npos ||
        name.find("elementwise") != std::string::npos ||
        name.find("add_kernel") != std::string::npos ||
        name.find("mul_kernel") != std::string::npos)
        return 0.8;

    // embedding / gather / scatter — bandwidth-heavy
    if (name.find("embed") != std::string::npos ||
        name.find("Embed") != std::string::npos ||
        name.find("gather") != std::string::npos ||
        name.find("Gather") != std::string::npos ||
        name.find("scatter") != std::string::npos)
        return 0.9;

    // softmax, layer_norm — moderate bandwidth
    if (name.find("softmax") != std::string::npos ||
        name.find("Softmax") != std::string::npos ||
        name.find("norm") != std::string::npos ||
        name.find("Norm") != std::string::npos)
        return 0.6;

    // default: mixed
    return 0.5;
}

static void UpdateDeviceIntensity(CUfunction f, XDevice device)
{
    auto it = g_func_names.find(f);
    if (it == g_func_names.end()) {
        const char *name = nullptr;
        if (Driver::FuncGetName(&name, f) == CUDA_SUCCESS && name != nullptr) {
            it = g_func_names.emplace(f, std::string(name)).first;
        } else {
            return;
        }
    }

    double intensity = EstimateKernelIntensity(it->second);
    auto dev_it = g_device_intensity.find(device);
    if (dev_it == g_device_intensity.end() || std::abs(dev_it->second - intensity) > 0.05) {
        g_device_intensity[device] = intensity;
        XHintMemoryIntensity(device, intensity);
    }
}

#define XQUEUE_TRACE(format, ...) \
    do { \
        if (XQueueTraceEnabled()) { \
            XINFO("[UXSCHED-XQUEUE] " format __VA_OPT__(,) __VA_ARGS__); \
        } \
    } while (0)

namespace
{

bool XQueueTraceEnabled()
{
    static const bool enabled = []() {
        const char *env = std::getenv("UXSCHED_XQUEUE_TRACE");
        if (env == nullptr || env[0] == '\0') return false;
        return std::strcmp(env, "0") != 0 && strcasecmp(env, "off") != 0 &&
               strcasecmp(env, "false") != 0 && strcasecmp(env, "no") != 0;
    }();
    return enabled;
}

bool AutoXQueueEnabled()
{
    const char *env = std::getenv(XSCHED_AUTO_XQUEUE_ENV_NAME);
    if (env == nullptr || env[0] == '\0') return false;
    return std::strcmp(env, "0") != 0 && strcasecmp(env, "off") != 0 &&
           strcasecmp(env, "false") != 0 && strcasecmp(env, "no") != 0;
}

HwQueueHandle LookupHwQueueHandle(CUstream stream, CUcontext *ctx_out, CUdevice *dev_out,
                                  CUresult *ctx_ret_out, CUresult *dev_ret_out)
{
    CUcontext ctx = nullptr;
    CUdevice dev = CU_DEVICE_INVALID;
    CUresult ctx_ret = Driver::CtxGetCurrent(&ctx);
    CUresult dev_ret = CUDA_ERROR_INVALID_CONTEXT;
    if (ctx_ret == CUDA_SUCCESS && ctx != nullptr) {
        dev_ret = Driver::CtxGetDevice(&dev);
    }

    if (ctx_out != nullptr) *ctx_out = ctx;
    if (dev_out != nullptr) *dev_out = dev;
    if (ctx_ret_out != nullptr) *ctx_ret_out = ctx_ret;
    if (dev_ret_out != nullptr) *dev_ret_out = dev_ret;

    if (stream == nullptr) {
        if (ctx_ret != CUDA_SUCCESS || ctx == nullptr) return 0;
        return GetHwQueueHandle(stream, ctx);
    }
    return GetHwQueueHandle(stream);
}

} // namespace

std::shared_ptr<XQueue> ResolveXQueueForStream(const char *api, CUstream stream, bool auto_create)
{
    CUcontext ctx = nullptr;
    CUdevice dev = CU_DEVICE_INVALID;
    CUresult ctx_ret = CUDA_SUCCESS;
    CUresult dev_ret = CUDA_SUCCESS;
    HwQueueHandle lookup_hwq = LookupHwQueueHandle(stream, &ctx, &dev, &ctx_ret, &dev_ret);
    auto hwq = HwQueueManager::Get(lookup_hwq);
    auto xq = hwq == nullptr ? nullptr : hwq->GetXQueue();
    const bool is_default_stream = stream == nullptr;
    const bool auto_enabled = AutoXQueueEnabled();
    const XQueueHandle xq_handle = xq == nullptr ? 0 : xq->GetHandle();

    XQUEUE_TRACE("api=%s pid=" FMT_PID " tid=" FMT_TID
                 " context=%p ctx_ret=%d device=%d dev_ret=%d stream=%p default_stream=%d "
                 "auto_xqueue=%d auto_create_allowed=%d lookup_hwq=0x" FMT_64X
                 " hwqueue=%p stream_to_xqueue=%p xqueue_id=0x" FMT_64X,
                 api, GetProcessId(), GetThreadId(), ctx, (int)ctx_ret, (int)dev, (int)dev_ret,
                 stream, is_default_stream ? 1 : 0, auto_enabled ? 1 : 0,
                 auto_create ? 1 : 0, lookup_hwq, hwq.get(), xq.get(), xq_handle);

    if (xq != nullptr) {
        XQUEUE_TRACE("api=%s auto_create_attempted=0 reason=lookup_hit KernelLaunch.xqueue=%p",
                     api, xq.get());
        return xq;
    }

    if (!auto_create) {
        XQUEUE_TRACE("api=%s auto_create_attempted=0 reason=auto_create_not_allowed "
                     "KernelLaunch.xqueue=(nil)", api);
        return nullptr;
    }

    if (!auto_enabled) {
        XQUEUE_TRACE("api=%s auto_create_attempted=0 reason=auto_xqueue_disabled "
                     "KernelLaunch.xqueue=(nil)", api);
        return nullptr;
    }

    HwQueueHandle created_hwq = 0;
    XResult create_res = XQueueManager::AutoCreate([&](HwQueueHandle *hwq_out) {
        XResult res = CudaQueueCreate(hwq_out, stream);
        if (hwq_out != nullptr) created_hwq = *hwq_out;
        return res;
    });

    auto hwq_after = HwQueueManager::Get(lookup_hwq);
    auto xq_after = hwq_after == nullptr ? nullptr : hwq_after->GetXQueue();
    auto xq_by_created = created_hwq == 0 ? nullptr : HwQueueManager::GetXQueue(created_hwq);
    const XQueueHandle xq_after_handle = xq_after == nullptr ? 0 : xq_after->GetHandle();
    const bool key_match = created_hwq == lookup_hwq;

    XQUEUE_TRACE("api=%s auto_create_attempted=1 create_result=%d created_hwq=0x" FMT_64X
                 " lookup_hwq=0x" FMT_64X " created_key_match=%d hwqueue=%p "
                 "stream_to_xqueue=%p xqueue_id=0x" FMT_64X " created_key_xqueue=%p",
                 api, (int)create_res, created_hwq, lookup_hwq, key_match ? 1 : 0,
                 hwq_after.get(), xq_after.get(), xq_after_handle, xq_by_created.get());

    if (create_res == kXSchedSuccess && xq_after == nullptr) {
        XQUEUE_TRACE("api=%s auto_create_diagnosis=%s", api,
                     (!key_match && xq_by_created != nullptr)
                         ? "created_success_lookup_key_mismatch"
                         : "created_success_no_stream_mapping");
    } else if (create_res != kXSchedSuccess) {
        XQUEUE_TRACE("api=%s auto_create_diagnosis=auto_create_failed create_result=%d",
                     api, (int)create_res);
    }

    return xq_after;
}

void WaitBlockingXQueues()
{
    std::list<std::shared_ptr<XQueueWaitAllCommand>> wait_cmds;
    XResult res = XQueueManager::ForEach([&](std::shared_ptr<XQueue> xq)->XResult {
        auto hwq = xq->GetHwQueue();
        auto cuda_q = std::dynamic_pointer_cast<CudaQueueLv1>(hwq);
        if (cuda_q == nullptr) return kXSchedErrorUnknown;
        // does not need to wait a non-blocking stream
        if (cuda_q->GetStreamFlags() & CU_STREAM_NON_BLOCKING) return kXSchedSuccess;
        auto wait_cmd = xq->SubmitWaitAll();
        if (wait_cmd == nullptr) return kXSchedErrorUnknown;
        wait_cmds.push_back(wait_cmd);
        return kXSchedSuccess;
    });
    XASSERT(res == kXSchedSuccess, "Fail to submit wait all commands");
    for (auto &cmd : wait_cmds) cmd->Wait();
}

CUresult XMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
{
    CUresult ret = Driver::MemAllocManaged(dptr, bytesize, flags);
    if (ret != CUDA_SUCCESS || dptr == nullptr || *dptr == 0) return ret;

    CUcontext ctx = nullptr;
    CUdevice dev = CU_DEVICE_INVALID;
    if (Driver::CtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) return ret;
    if (Driver::CtxGetDevice(&dev) != CUDA_SUCCESS) return ret;

    CudaMemoryManager::RegisterManagedAllocation(*dptr, bytesize, ctx, dev);
    return ret;
}

CUresult XMemFree_v2(CUdeviceptr dptr)
{
    CUresult ret = Driver::MemFree_v2(dptr);
    if (ret == CUDA_SUCCESS) CudaMemoryManager::UnregisterAllocation(dptr);
    return ret;
}

CUresult XMemFree(CUdeviceptr_v1 dptr)
{
    CUresult ret = Driver::MemFree(dptr);
    if (ret == CUDA_SUCCESS) CudaMemoryManager::UnregisterAllocation((CUdeviceptr)dptr);
    return ret;
}

CUresult XLaunchKernel(CUfunction f,
                       unsigned int gdx, unsigned int gdy, unsigned int gdz,
                       unsigned int bdx, unsigned int bdy, unsigned int bdz,
                       unsigned int shmem, CUstream stream, void **params, void **extra)
{
    XDEBG("XLaunchKernel(func: %p, stream: %p, grid: [%u, %u, %u], block: [%u, %u, %u], "
          "shm: %u, params: %p, extra: %p)", f, stream, gdx, gdy, gdz, bdx, bdy, bdz,
          shmem, params, extra);

    if (stream == nullptr) {
        WaitBlockingXQueues();
    }

    auto xq = ResolveXQueueForStream("cuLaunchKernel", stream);
    if (xq) {
        UpdateDeviceIntensity(f, xq->GetDevice());
    }
    const runtime::RuntimeStrategyMode mode = runtime::CurrentRuntimeStrategyMode();
    XQUEUE_TRACE("api=cuLaunchKernel KernelLaunch.xqueue=%p xqueue_id=0x" FMT_64X
                 " runtime_strategy=%s",
                 xq.get(), xq == nullptr ? 0 : xq->GetHandle(),
                 runtime::RuntimeStrategyModeName(mode));
    runtime::KernelLaunch launch{
        f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, stream, params, extra, xq
    };
    return runtime::SubmitKernelWithRuntimeStrategy(launch).result;
}

CUresult XLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **params, void **extra)
{
    XDEBG("XLaunchKernelEx(cfg: %p, func: %p, params: %p, extra: %p)", config, f, params, extra);
    if (config == nullptr) return Driver::LaunchKernelEx(config, f, params, extra);

    CUstream stream = config->hStream;

    if (stream == nullptr) {
        WaitBlockingXQueues();
    }

    auto xq = ResolveXQueueForStream("cuLaunchKernelEx", stream);
    if (xq) {
        UpdateDeviceIntensity(f, xq->GetDevice());
    }
    auto kn = std::make_shared<CudaKernelLaunchExCommand>(config, f, params, extra, xq != nullptr);

    if (xq == nullptr) return DirectLaunch(kn, stream);
    xq->Submit(kn);
    return CUDA_SUCCESS;
}

CUresult XLaunchHostFunc(CUstream stream, CUhostFn fn, void *data)
{
    if (stream == 0) {
        WaitBlockingXQueues();
    }
    auto xq = ResolveXQueueForStream("cuLaunchHostFunc", stream);
    if (xq == nullptr) return Driver::LaunchHostFunc(stream, fn, data);
    auto hw_cmd = std::make_shared<CudaHostFuncCommand>(fn, data);
    xq->Submit(hw_cmd);
    return CUDA_SUCCESS;
}

CUresult XEventQuery(CUevent event)
{
    XDEBG("XEventQuery(event: %p)", event);
    if (event == nullptr) return Driver::EventQuery(event);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::EventQuery(event);

    auto state = xevent->GetState();
    if (state >= kCommandStateCompleted) return CUDA_SUCCESS;
    return CUDA_ERROR_NOT_READY;
}

CUresult XEventRecord(CUevent event, CUstream stream)
{
    XDEBG("XEventRecord(event: %p, stream: %p)", event, stream);
    if (event == nullptr) return Driver::EventRecord(event, stream);

    CUresult result;
    auto xevent = std::make_shared<CudaEventRecordCommand>(event);

    if (stream == nullptr) {
        WaitBlockingXQueues();
    }

    auto xq = ResolveXQueueForStream("cuEventRecord", stream);
    if (xq == nullptr) {
        result = Driver::EventRecord(event, stream);
    } else {
        xq->Submit(xevent);
        result = CUDA_SUCCESS;
    }

    g_events.Add(event, xevent);
    return result;
}

CUresult XEventRecordWithFlags(CUevent event, CUstream stream, unsigned int flags)
{
    XDEBG("XEventRecordWithFlags(event: %p, stream: %p, flags: %u)", event, stream, flags);
    if (event == nullptr) return Driver::EventRecordWithFlags(event, stream, flags);

    CUresult result;
    auto xevent = std::make_shared<CudaEventRecordWithFlagsCommand>(event, flags);

    if (stream == nullptr) {
        WaitBlockingXQueues();
    }

    auto xq = ResolveXQueueForStream("cuEventRecordWithFlags", stream);
    if (xq == nullptr) {
        result = Driver::EventRecordWithFlags(event, stream, flags);
    } else {
        xq->Submit(xevent);
        result = CUDA_SUCCESS;
    }

    g_events.Add(event, xevent);
    return result;
}

CUresult XEventSynchronize(CUevent event)
{
    XDEBG("XEventSynchronize(event: %p)", event);
    if (event == nullptr) return Driver::EventSynchronize(event);

    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::EventSynchronize(event);

    xevent->Wait();
    return CUDA_SUCCESS;
}

CUresult XStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags)
{
    XDEBG("XStreamWaitEvent(stream: %p, event: %p, flags: %u)", stream, event, flags);
    if (event == nullptr)return Driver::StreamWaitEvent(stream, event, flags);

    auto xevent = g_events.Get(event, nullptr);
    // the event is not recorded yet
    if (xevent == nullptr) return Driver::StreamWaitEvent(stream, event, flags);

    if (stream == nullptr) {
        // sync a event on default stream
        WaitBlockingXQueues();
    }

    auto xq = ResolveXQueueForStream("cuStreamWaitEvent", stream);
    if (xq == nullptr) {
        // waiting stream is not an xqueue
        if (xevent->GetXQueueHandle() == 0) {
            // the event is not recorded on an xqueue
            return Driver::StreamWaitEvent(stream, event, flags);
        }
        xevent->Wait();
        return CUDA_SUCCESS;
    }

    auto cmd = std::make_shared<CudaEventWaitCommand>(xevent, flags);
    xq->Submit(cmd);
    return CUDA_SUCCESS;
}

CUresult XEventDestroy(CUevent event)
{
    XDEBG("XEventDestroy(event: %p)", event);
    if (event == nullptr) return Driver::EventDestroy(event);

    auto xevent = g_events.DoThenDel(event, nullptr, [](auto xevent) {
        xevent->DestroyEvent();
    });
    if (xevent == nullptr) return Driver::EventDestroy(event);
    return CUDA_SUCCESS;
}

CUresult XEventDestroy_v2(CUevent event)
{
    XDEBG("XEventDestroy_v2(event: %p)", event);
    if (event == nullptr) return Driver::EventDestroy_v2(event);

    auto xevent = g_events.DoThenDel(event, nullptr, [](auto xevent) {
        xevent->DestroyEvent();
    });
    if (xevent == nullptr) return Driver::EventDestroy_v2(event);
    return CUDA_SUCCESS;
}

CUresult XStreamSynchronize(CUstream stream)
{
    XDEBG("XStreamSynchronize(stream: %p)", stream);
    auto xq = ResolveXQueueForStream("cuStreamSynchronize", stream);
    if (xq == nullptr) return Driver::StreamSynchronize(stream);
    xq->WaitAll();
    return CUDA_SUCCESS;
}

CUresult XStreamQuery(CUstream stream)
{
    XDEBG("XStreamQuery(stream: %p)", stream);
    auto xq = ResolveXQueueForStream("cuStreamQuery", stream, false);
    if (xq == nullptr) return Driver::StreamQuery(stream);

    switch (xq->Query())
    {
    case kQueueStateIdle:
        return CUDA_SUCCESS;
    case kQueueStateReady:
        return CUDA_ERROR_NOT_READY;
    default:
        return Driver::StreamQuery(stream);
    }
}
CUresult XCtxSynchronize()
{
    XDEBG("XCtxSynchronize()");
    XQueueManager::ForEachWaitAll();
    return Driver::CtxSynchronize();
}

CUresult XStreamCreate(CUstream *stream, unsigned int flags)
{
    CUresult res = Driver::StreamCreate(stream, flags);
    if (res != CUDA_SUCCESS) return res;
    ResolveXQueueForStream("cuStreamCreate", *stream);
    XDEBG("XStreamCreate(stream: %p, flags: 0x%x)", *stream, flags);
    return res;
}

CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority)
{
    CUresult res = Driver::StreamCreateWithPriority(stream, flags, priority);
    if (res != CUDA_SUCCESS) return res;
    ResolveXQueueForStream("cuStreamCreateWithPriority", *stream);
    XDEBG("XStreamCreateWithPriority(stream: %p, flags: 0x%x, priority: %d)",
          *stream, flags, priority);
    return res;
}

CUresult XStreamDestroy(CUstream stream)
{
    XDEBG("XStreamDestroy(stream: %p)", stream);
    XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
    return Driver::StreamDestroy(stream);
}

CUresult XStreamDestroy_v2(CUstream stream)
{
    XDEBG("XStreamDestroy_v2(stream: %p)", stream);
    XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
    return Driver::StreamDestroy_v2(stream);
}

} // namespace xsched::cuda
