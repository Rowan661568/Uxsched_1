#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::preempt
{
class XQueue;
}

namespace xsched::cuda::runtime
{

using CommandId = uint64_t;

enum class RuntimeStrategyMode
{
    kNative,
    kHbFixed,
    kHbRuntime,
    kAuto,
};

enum class SubmitStatus
{
    kSubmitted,
    kFallbackNative,
};

struct KernelLaunch
{
    CUfunction function = nullptr;
    unsigned int grid_dim_x = 1;
    unsigned int grid_dim_y = 1;
    unsigned int grid_dim_z = 1;
    unsigned int block_dim_x = 1;
    unsigned int block_dim_y = 1;
    unsigned int block_dim_z = 1;
    unsigned int shared_mem_bytes = 0;
    CUstream stream = nullptr;
    void **kernel_params = nullptr;
    void **extra = nullptr;
    std::shared_ptr<preempt::XQueue> xqueue = nullptr;
};

struct DeviceRuntimeState
{
    uint64_t state_epoch = 0;
    uint32_t hp_ready_count = 0;
    bool hp_pending = false;
    uint32_t bubble_type = 0;
    uint64_t bubble_deadline_ns = 0;
    uint64_t selected_lp_queue = 0;
};

struct SubmitResult
{
    SubmitStatus status = SubmitStatus::kFallbackNative;
    CUresult result = CUDA_SUCCESS;
    std::string fallback_reason;

    static SubmitResult Submitted(CUresult ret = CUDA_SUCCESS)
    {
        return SubmitResult{SubmitStatus::kSubmitted, ret, ""};
    }

    static SubmitResult Fallback(std::string reason)
    {
        return SubmitResult{SubmitStatus::kFallbackNative, CUDA_SUCCESS, std::move(reason)};
    }
};

class CudaRuntimeStrategy
{
public:
    virtual ~CudaRuntimeStrategy() = default;
    virtual SubmitResult SubmitKernel(const KernelLaunch &launch) = 0;
    virtual void Suspend() = 0;
    virtual void Resume() = 0;
    virtual void Wait(CommandId command) = 0;
    virtual void OnDeviceRuntimeState(const DeviceRuntimeState &state) = 0;
};

class NativeRuntimeStrategy final : public CudaRuntimeStrategy
{
public:
    SubmitResult SubmitKernel(const KernelLaunch &launch) override;
    void Suspend() override {}
    void Resume() override {}
    void Wait(CommandId command) override;
    void OnDeviceRuntimeState(const DeviceRuntimeState &state) override;
};

class HummingbirdRuntimeStrategy final : public CudaRuntimeStrategy
{
public:
    explicit HummingbirdRuntimeStrategy(RuntimeStrategyMode mode): mode_(mode) {}

    SubmitResult SubmitKernel(const KernelLaunch &launch) override;
    void Suspend() override;
    void Resume() override;
    void Wait(CommandId command) override;
    void OnDeviceRuntimeState(const DeviceRuntimeState &state) override;

private:
    RuntimeStrategyMode mode_;
    DeviceRuntimeState last_state_{};
};

RuntimeStrategyMode CurrentRuntimeStrategyMode();
const char *RuntimeStrategyModeName(RuntimeStrategyMode mode);
SubmitResult SubmitKernelWithRuntimeStrategy(const KernelLaunch &launch);

} // namespace xsched::cuda::runtime
