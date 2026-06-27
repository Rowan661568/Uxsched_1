#pragma once

#include <memory>
#include <string>

#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::preempt
{
class XQueue;
}

namespace xsched::cuda::hb_split
{

enum class BackendMode
{
    kNative,
    kHbSplit,
    kAuto,
};

struct KernelCapability
{
    bool ptx_available = false;
    bool transform_attempted = false;
    bool transform_succeeded = false;
    bool supports_offset_x = false;
    bool supports_offset_y = false;
    bool supports_offset_z = false;
    bool cooperative = false;
    bool persistent = false;
    bool cross_block_sync = false;
    bool supports_kernel_params = false;
    bool supports_extra = false;
    bool splittable = false;
    std::string fallback_reason;
};

struct MetadataRegistrationResult
{
    bool ok = false;
    std::string reason;
    size_t ptx_bytes = 0;
    size_t record_count = 0;
};

CUresult XModuleLoad(CUmodule *module, const char *fname);
CUresult XModuleLoadData(CUmodule *module, const void *image);
CUresult XModuleLoadDataEx(CUmodule *module, const void *image, unsigned int num_options,
                           CUjit_option *options, void **option_values);
CUresult XModuleUnload(CUmodule module);
CUresult XModuleGetFunction(CUfunction *function, CUmodule module, const char *name);

MetadataRegistrationResult RegisterModuleMetadata(CUmodule module, const void *ptx,
                                                  size_t ptx_size);
MetadataRegistrationResult RegisterFunctionMetadata(CUfunction function, CUmodule module,
                                                    const char *name);
void UnregisterModuleMetadata(CUmodule module);
bool LookupFunctionMetadata(CUfunction function, std::string *kernel_name,
                            std::string *fallback_reason);

bool TryLaunchKernel(CUfunction function,
                     unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
                     unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
                     unsigned int shared_mem_bytes, CUstream stream, void **kernel_params,
                     void **extra, std::shared_ptr<preempt::XQueue> xqueue, CUresult *result);

bool TryLaunchKernelFixed(CUfunction function,
                          unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
                          unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
                          unsigned int shared_mem_bytes, CUstream stream, void **kernel_params,
                          void **extra, std::shared_ptr<preempt::XQueue> xqueue, CUresult *result);

} // namespace xsched::cuda::hb_split
