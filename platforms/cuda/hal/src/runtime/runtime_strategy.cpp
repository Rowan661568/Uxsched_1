#include "xsched/cuda/hal/runtime/runtime_strategy.h"

#include <cstdlib>
#include <cstring>
#include <strings.h>

#include "xsched/utils/common.h"
#include "xsched/utils/log.h"
#include "xsched/protocol/def.h"
#include "xsched/cuda/hal/common/cuda_command.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/hb_split/backend.h"
#include "xsched/preempt/xqueue/xqueue.h"

namespace xsched::cuda::runtime
{

namespace
{

bool XQueueTraceEnabled()
{
    const char *env = std::getenv("UXSCHED_XQUEUE_TRACE");
    if (env == nullptr || env[0] == '\0') return false;
    return std::strcmp(env, "0") != 0 && strcasecmp(env, "off") != 0 &&
           strcasecmp(env, "false") != 0 && strcasecmp(env, "no") != 0;
}

bool BuildHasHbSplit()
{
#ifdef UXSCHED_ENABLE_HB_SPLIT
    return true;
#else
    return false;
#endif
}

RuntimeStrategyMode ParseModeName(const char *mode, RuntimeStrategyMode fallback)
{
    if (mode == nullptr || mode[0] == '\0') return fallback;
    if (strcasecmp(mode, "NATIVE") == 0) return RuntimeStrategyMode::kNative;
    if (strcasecmp(mode, "HB_FIXED") == 0) return RuntimeStrategyMode::kHbFixed;
    if (strcasecmp(mode, "HB_SPLIT") == 0) return RuntimeStrategyMode::kHbFixed;
    if (strcasecmp(mode, "HB_RUNTIME") == 0) return RuntimeStrategyMode::kHbRuntime;
    if (strcasecmp(mode, "AUTO") == 0) return RuntimeStrategyMode::kAuto;
    XWARN("[UXSCHED-HB] invalid CUDA runtime strategy=%s; using NATIVE", mode);
    return RuntimeStrategyMode::kNative;
}

RuntimeStrategyMode LegacyBackendMode()
{
    const char *legacy = std::getenv("UXSCHED_CUDA_PREEMPT_BACKEND");
    if (legacy == nullptr || legacy[0] == '\0') return RuntimeStrategyMode::kNative;
    return ParseModeName(legacy, RuntimeStrategyMode::kNative);
}

} // namespace

const char *RuntimeStrategyModeName(RuntimeStrategyMode mode)
{
    switch (mode) {
    case RuntimeStrategyMode::kNative: return "NATIVE";
    case RuntimeStrategyMode::kHbFixed: return "HB_FIXED";
    case RuntimeStrategyMode::kHbRuntime: return "HB_RUNTIME";
    case RuntimeStrategyMode::kAuto: return "AUTO";
    }
    return "NATIVE";
}

RuntimeStrategyMode CurrentRuntimeStrategyMode()
{
    const char *strategy = std::getenv("UXSCHED_CUDA_RUNTIME_STRATEGY");
    if (strategy != nullptr && strategy[0] != '\0') {
        return ParseModeName(strategy, RuntimeStrategyMode::kNative);
    }
    return LegacyBackendMode();
}

SubmitResult NativeRuntimeStrategy::SubmitKernel(const KernelLaunch &launch)
{
    auto kernel = std::make_shared<CudaKernelLaunchCommand>(
        launch.function,
        launch.grid_dim_x, launch.grid_dim_y, launch.grid_dim_z,
        launch.block_dim_x, launch.block_dim_y, launch.block_dim_z,
        launch.shared_mem_bytes, launch.kernel_params, launch.extra,
        launch.xqueue != nullptr);

    if (launch.xqueue == nullptr) {
        return SubmitResult::Submitted(DirectLaunch(kernel, launch.stream));
    }

    launch.xqueue->Submit(kernel);
    return SubmitResult::Submitted(CUDA_SUCCESS);
}

void NativeRuntimeStrategy::Wait(CommandId command)
{
    UNUSED(command);
}

void NativeRuntimeStrategy::OnDeviceRuntimeState(const DeviceRuntimeState &state)
{
    UNUSED(state);
}

SubmitResult HummingbirdRuntimeStrategy::SubmitKernel(const KernelLaunch &launch)
{
    if (!BuildHasHbSplit()) return SubmitResult::Fallback("HB_SPLIT_BUILD_DISABLED");

    if (mode_ == RuntimeStrategyMode::kHbFixed) {
        CUresult ret = CUDA_SUCCESS;
        if (hb_split::TryLaunchKernelFixed(
                launch.function,
                launch.grid_dim_x, launch.grid_dim_y, launch.grid_dim_z,
                launch.block_dim_x, launch.block_dim_y, launch.block_dim_z,
                launch.shared_mem_bytes, launch.stream, launch.kernel_params,
                launch.extra, launch.xqueue, &ret)) {
            return SubmitResult::Submitted(ret);
        }
        return SubmitResult::Fallback("HB_FIXED_FALLBACK_NATIVE");
    }

    if (mode_ == RuntimeStrategyMode::kHbRuntime) {
        XINFO("[UXSCHED-HB] backend_selected=NATIVE reason=HB_RUNTIME_NOT_IMPLEMENTED_YET");
        return SubmitResult::Fallback("HB_RUNTIME_NOT_IMPLEMENTED_YET");
    }

    if (mode_ == RuntimeStrategyMode::kAuto) {
        XINFO("[UXSCHED-HB] backend_selected=NATIVE reason=AUTO_RUNTIME_COORDINATOR_UNAVAILABLE");
        return SubmitResult::Fallback("AUTO_RUNTIME_COORDINATOR_UNAVAILABLE");
    }

    return SubmitResult::Fallback("NOT_HB_MODE");
}

void HummingbirdRuntimeStrategy::Suspend()
{
    XINFO("[UXSCHED-HB] runtime_strategy_suspend mode=%s", RuntimeStrategyModeName(mode_));
}

void HummingbirdRuntimeStrategy::Resume()
{
    XINFO("[UXSCHED-HB] runtime_strategy_resume mode=%s", RuntimeStrategyModeName(mode_));
}

void HummingbirdRuntimeStrategy::Wait(CommandId command)
{
    UNUSED(command);
}

void HummingbirdRuntimeStrategy::OnDeviceRuntimeState(const DeviceRuntimeState &state)
{
    last_state_ = state;
}

SubmitResult SubmitKernelWithRuntimeStrategy(const KernelLaunch &launch)
{
    const RuntimeStrategyMode mode = CurrentRuntimeStrategyMode();
    if (mode == RuntimeStrategyMode::kNative) {
        NativeRuntimeStrategy native;
        return native.SubmitKernel(launch);
    }

    HummingbirdRuntimeStrategy hb(mode);
    if (XQueueTraceEnabled()) {
        XINFO("[UXSCHED-XQUEUE] runtime_strategy_enter mode=%s stream=%p "
              "KernelLaunch.xqueue=%p",
              RuntimeStrategyModeName(mode), launch.stream, launch.xqueue.get());
    }
    SubmitResult hb_result = hb.SubmitKernel(launch);
    if (hb_result.status == SubmitStatus::kSubmitted) return hb_result;

    if (XQueueTraceEnabled()) {
        XINFO("[UXSCHED-XQUEUE] runtime_strategy_fallback mode=%s reason=%s "
              "KernelLaunch.xqueue=%p",
              RuntimeStrategyModeName(mode), hb_result.fallback_reason.c_str(),
              launch.xqueue.get());
    }

    NativeRuntimeStrategy native;
    return native.SubmitKernel(launch);
}

} // namespace xsched::cuda::runtime
