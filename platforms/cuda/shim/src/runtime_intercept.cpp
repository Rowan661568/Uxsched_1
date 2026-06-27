#ifdef UXSCHED_ENABLE_HB_SPLIT
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <strings.h>
#include <unordered_map>
#include <vector>

#include "xsched/utils/common.h"
#include "xsched/utils/log.h"
#include "xsched/cuda/shim/shim.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/hb_split/backend.h"
#include "xsched/cuda/hal/runtime/runtime_strategy.h"

#ifdef cudaLaunchKernel
#undef cudaLaunchKernel
#endif

#ifdef cudaLaunchKernelExC
#undef cudaLaunchKernelExC
#endif

#ifdef cudaDeviceSynchronize
#undef cudaDeviceSynchronize
#endif

#ifdef cudaStreamSynchronize
#undef cudaStreamSynchronize
#endif

#ifdef cudaStreamQuery
#undef cudaStreamQuery
#endif

#ifdef cudaStreamWaitEvent
#undef cudaStreamWaitEvent
#endif

#ifdef cudaStreamDestroy
#undef cudaStreamDestroy
#endif

#ifdef cudaEventRecord
#undef cudaEventRecord
#endif

#ifdef cudaEventRecordWithFlags
#undef cudaEventRecordWithFlags
#endif

#ifdef cudaEventQuery
#undef cudaEventQuery
#endif

#ifdef cudaEventSynchronize
#undef cudaEventSynchronize
#endif

#ifdef cudaEventDestroy
#undef cudaEventDestroy
#endif

namespace
{

using RegisterFatBinaryFn = void **(*)(void *);
using RegisterFatBinaryEndFn = void (*)(void **);
using UnregisterFatBinaryFn = void (*)(void **);
using RegisterFunctionFn = void (*)(void **, const char *, char *, const char *, int,
                                    uint3 *, uint3 *, dim3 *, dim3 *, int *);
using CudaLaunchKernelFn = cudaError_t (*)(const void *, dim3, dim3, void **, size_t,
                                           cudaStream_t);
using CudaLaunchKernelExCFn = cudaError_t (*)(const cudaLaunchConfig_t *, const void *,
                                             void **);

constexpr uint32_t kFatCubinWrapperMagic = 0x466243b1;
constexpr uint32_t kFatbinMagic = 0xba55ed50;
constexpr size_t kFatbinScanLimit = 128 * 1024 * 1024;
constexpr size_t kMinimumPtxBytes = 128;

#define CUDART_TRACE(format, ...) \
    do { \
        if (CudartTraceEnabled()) { \
            XINFO("[UXSCHED-CUDART] " format __VA_OPT__(,) __VA_ARGS__); \
        } \
    } while (0)

struct FatbinWrapper
{
    uint32_t magic;
    uint32_t version;
    const void *data;
    void *filename;
};

struct FatbinRecord
{
    void *fat_cubin = nullptr;
    void **handle = nullptr;
    std::string ptx;
    std::vector<CUmodule> modules;
};

struct RuntimeFunctionRecord
{
    const void *host_stub = nullptr;
    void **fatbin_handle = nullptr;
    std::string device_fun;
    std::string device_name;
};

struct ResolvedRuntimeFunction
{
    bool attempted = false;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    std::string kernel_name;
    std::string reason;
};

std::mutex g_mu;
std::unordered_map<void **, FatbinRecord> g_fatbins;
std::unordered_map<const void *, RuntimeFunctionRecord> g_functions;
std::unordered_map<const void *, ResolvedRuntimeFunction> g_resolved_functions;

bool IsTruthyEnv(const char *name)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return false;
    return std::strcmp(env, "0") != 0 && strcasecmp(env, "off") != 0 &&
           strcasecmp(env, "false") != 0 && strcasecmp(env, "no") != 0;
}

bool CudartTraceEnabled()
{
    static const bool enabled = IsTruthyEnv("UXSCHED_CUDART_TRACE");
    return enabled;
}

template <typename Fn>
Fn ResolveNext(const char *symbol)
{
    void *ptr = dlsym(RTLD_NEXT, symbol);
    if (ptr == nullptr) {
        XWARN("[UXSCHED-CUDART] dlsym_failed symbol=%s error=%s", symbol, dlerror());
    }
    return reinterpret_cast<Fn>(ptr);
}

RegisterFatBinaryFn RealRegisterFatBinary()
{
    static RegisterFatBinaryFn fn = ResolveNext<RegisterFatBinaryFn>("__cudaRegisterFatBinary");
    return fn;
}

RegisterFatBinaryEndFn RealRegisterFatBinaryEnd()
{
    static RegisterFatBinaryEndFn fn =
        ResolveNext<RegisterFatBinaryEndFn>("__cudaRegisterFatBinaryEnd");
    return fn;
}

UnregisterFatBinaryFn RealUnregisterFatBinary()
{
    static UnregisterFatBinaryFn fn =
        ResolveNext<UnregisterFatBinaryFn>("__cudaUnregisterFatBinary");
    return fn;
}

RegisterFunctionFn RealRegisterFunction()
{
    static RegisterFunctionFn fn = ResolveNext<RegisterFunctionFn>("__cudaRegisterFunction");
    return fn;
}

CudaLaunchKernelFn RealCudaLaunchKernel()
{
    static CudaLaunchKernelFn fn = ResolveNext<CudaLaunchKernelFn>("cudaLaunchKernel");
    return fn;
}

CudaLaunchKernelExCFn RealCudaLaunchKernelExC()
{
    static CudaLaunchKernelExCFn fn = ResolveNext<CudaLaunchKernelExCFn>("cudaLaunchKernelExC");
    return fn;
}

cudaError_t RuntimeErrorFromCuResult(CUresult ret)
{
    if (ret == CUDA_SUCCESS) return cudaSuccess;
    if (ret == CUDA_ERROR_NOT_READY) return cudaErrorNotReady;
    if (ret == CUDA_ERROR_INVALID_HANDLE) return cudaErrorInvalidResourceHandle;
    if (ret == CUDA_ERROR_INVALID_VALUE) return cudaErrorInvalidValue;
    if (ret == CUDA_ERROR_INVALID_CONTEXT) return cudaErrorInvalidDevice;
    if (ret == CUDA_ERROR_OUT_OF_MEMORY) return cudaErrorMemoryAllocation;
    if (ret == CUDA_ERROR_NOT_SUPPORTED) return cudaErrorNotSupported;
    return cudaErrorUnknown;
}

bool IsAsciiPtxByte(unsigned char c)
{
    return c == '\n' || c == '\r' || c == '\t' || c == '\f' ||
           (c >= 0x20 && c <= 0x7e);
}

std::string ExtractPtxText(const void *fat_cubin, std::string *reason)
{
    if (fat_cubin == nullptr) {
        if (reason != nullptr) *reason = "RUNTIME_FATBIN_NULL";
        return {};
    }

    const auto *wrapper = static_cast<const FatbinWrapper *>(fat_cubin);
    if (wrapper->magic != kFatCubinWrapperMagic || wrapper->data == nullptr) {
        if (reason != nullptr) *reason = "RUNTIME_FATBIN_WRAPPER_UNSUPPORTED";
        return {};
    }

    const auto *bytes = static_cast<const unsigned char *>(wrapper->data);
    uint32_t magic = 0;
    uint64_t fatbin_size = 0;
    std::memcpy(&magic, bytes, sizeof(magic));
    std::memcpy(&fatbin_size, bytes + 8, sizeof(fatbin_size));
    if (magic != kFatbinMagic || fatbin_size < kMinimumPtxBytes ||
        fatbin_size > kFatbinScanLimit) {
        if (reason != nullptr) *reason = "RUNTIME_FATBIN_HEADER_UNSUPPORTED";
        return {};
    }

    const char *data = reinterpret_cast<const char *>(bytes);
    const char *end = data + fatbin_size;
    const char *version = std::search(data, end, ".version", ".version" + 8);
    if (version == end) {
        if (reason != nullptr) *reason = "RUNTIME_PTX_UNAVAILABLE";
        return {};
    }

    const char *cursor = version;
    const char *last_close_brace = nullptr;
    size_t bad_run = 0;
    while (cursor < end) {
        const unsigned char c = static_cast<unsigned char>(*cursor);
        if (IsAsciiPtxByte(c)) {
            bad_run = 0;
            if (c == '}') last_close_brace = cursor + 1;
            ++cursor;
            continue;
        }
        if (c == '\0' && last_close_brace != nullptr) break;
        ++bad_run;
        if (bad_run >= 16 && last_close_brace != nullptr) break;
        ++cursor;
    }

    const char *ptx_end = last_close_brace != nullptr ? last_close_brace : cursor;
    while (ptx_end > version &&
           std::isspace(static_cast<unsigned char>(*(ptx_end - 1))) != 0) {
        --ptx_end;
    }
    if (ptx_end <= version || (size_t)(ptx_end - version) < kMinimumPtxBytes) {
        if (reason != nullptr) *reason = "RUNTIME_PTX_EXTRACT_FAILED";
        return {};
    }

    std::string ptx(version, ptx_end);
    if (ptx.find(".entry") == std::string::npos ||
        ptx.find(".address_size") == std::string::npos) {
        if (reason != nullptr) *reason = "RUNTIME_PTX_EXTRACT_FAILED";
        return {};
    }

    if (reason != nullptr) *reason = "";
    return ptx;
}

ResolvedRuntimeFunction ResolveRuntimeFunction(const void *host_stub,
                                               RuntimeFunctionRecord *record_out)
{
    RuntimeFunctionRecord record;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto fn_it = g_functions.find(host_stub);
        if (fn_it == g_functions.end()) {
            return ResolvedRuntimeFunction{true, nullptr, nullptr, "", "RUNTIME_KERNEL_NOT_REGISTERED"};
        }
        record = fn_it->second;
        if (record_out != nullptr) *record_out = record;

        auto resolved_it = g_resolved_functions.find(host_stub);
        if (resolved_it != g_resolved_functions.end() &&
            resolved_it->second.attempted) {
            return resolved_it->second;
        }
    }

    FatbinRecord fatbin;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto fat_it = g_fatbins.find(record.fatbin_handle);
        if (fat_it == g_fatbins.end()) {
            ResolvedRuntimeFunction out{true, nullptr, nullptr, "", "RUNTIME_FATBIN_NOT_REGISTERED"};
            g_resolved_functions[host_stub] = out;
            return out;
        }
        fatbin = fat_it->second;
    }

    if (fatbin.ptx.empty()) {
        ResolvedRuntimeFunction out{true, nullptr, nullptr, "", "RUNTIME_PTX_UNAVAILABLE"};
        std::lock_guard<std::mutex> lock(g_mu);
        g_resolved_functions[host_stub] = out;
        return out;
    }

    std::vector<char> ptx(fatbin.ptx.begin(), fatbin.ptx.end());
    ptx.push_back('\0');

    CUmodule module = nullptr;
    CUresult load_ret = xsched::cuda::Driver::ModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr);
    if (load_ret != CUDA_SUCCESS || module == nullptr) {
        ResolvedRuntimeFunction out{true, module, nullptr, "", "RUNTIME_PTX_LOAD_FAILED"};
        std::lock_guard<std::mutex> lock(g_mu);
        g_resolved_functions[host_stub] = out;
        return out;
    }

    auto module_reg = xsched::cuda::hb_split::RegisterModuleMetadata(
        module, fatbin.ptx.data(), fatbin.ptx.size());
    CUDART_TRACE("runtime_hb_module_registered module=%p ptx_bytes=%zu record_count=%zu "
                 "result=%s",
                 module, module_reg.ptx_bytes, module_reg.record_count,
                 module_reg.ok ? "OK" : module_reg.reason.c_str());
    if (!module_reg.ok) {
        (void)xsched::cuda::Driver::ModuleUnload(module);
        ResolvedRuntimeFunction out{true, nullptr, nullptr, "", module_reg.reason};
        std::lock_guard<std::mutex> lock(g_mu);
        g_resolved_functions[host_stub] = out;
        return out;
    }

    CUfunction function = nullptr;
    std::vector<std::string> candidates;
    if (!record.device_name.empty()) candidates.push_back(record.device_name);
    if (!record.device_fun.empty() &&
        (candidates.empty() || candidates.front() != record.device_fun)) {
        candidates.push_back(record.device_fun);
    }

    std::string resolved_name;
    CUresult func_ret = CUDA_ERROR_NOT_FOUND;
    for (const std::string &candidate : candidates) {
        func_ret = xsched::cuda::Driver::ModuleGetFunction(&function, module, candidate.c_str());
        if (func_ret == CUDA_SUCCESS && function != nullptr) {
            resolved_name = candidate;
            break;
        }
    }
    if (func_ret != CUDA_SUCCESS || function == nullptr) {
        xsched::cuda::hb_split::XModuleUnload(module);
        ResolvedRuntimeFunction out{true, nullptr, nullptr, "", "RUNTIME_FUNCTION_RESOLVE_FAILED"};
        std::lock_guard<std::mutex> lock(g_mu);
        g_resolved_functions[host_stub] = out;
        return out;
    }

    auto function_reg = xsched::cuda::hb_split::RegisterFunctionMetadata(
        function, module, resolved_name.c_str());
    CUDART_TRACE("runtime_hb_function_registered function=%p module=%p kernel_name=%s "
                 "record_count=%zu result=%s",
                 function, module, resolved_name.c_str(), function_reg.record_count,
                 function_reg.ok ? "OK" : function_reg.reason.c_str());
    if (!function_reg.ok) {
        xsched::cuda::hb_split::XModuleUnload(module);
        ResolvedRuntimeFunction out{true, nullptr, nullptr, "", function_reg.reason};
        std::lock_guard<std::mutex> lock(g_mu);
        g_resolved_functions[host_stub] = out;
        return out;
    }

    ResolvedRuntimeFunction out{true, module, function, resolved_name, ""};
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_resolved_functions[host_stub] = out;
        g_fatbins[record.fatbin_handle].modules.push_back(module);
    }
    return out;
}

void RemoveFatbin(void **fatbin_handle)
{
    std::vector<CUmodule> modules;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto fat_it = g_fatbins.find(fatbin_handle);
        if (fat_it != g_fatbins.end()) {
            modules = std::move(fat_it->second.modules);
            g_fatbins.erase(fat_it);
        }
        for (auto it = g_functions.begin(); it != g_functions.end();) {
            if (it->second.fatbin_handle == fatbin_handle) {
                g_resolved_functions.erase(it->first);
                it = g_functions.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (CUmodule module : modules) {
        if (module != nullptr) xsched::cuda::hb_split::XModuleUnload(module);
    }
}

cudaError_t LaunchNativeRuntime(const void *func, dim3 grid_dim, dim3 block_dim, void **args,
                                size_t shared_mem, cudaStream_t stream)
{
    CudaLaunchKernelFn real = RealCudaLaunchKernel();
    if (real == nullptr) return cudaErrorUnknown;
    return real(func, grid_dim, block_dim, args, shared_mem, stream);
}

int64_t CurrentRuntimePriority()
{
    const char *own = std::getenv("UXSCHED_HB_PRIORITY");
    const char *auto_priority = std::getenv(XSCHED_AUTO_XQUEUE_PRIORITY_ENV_NAME);
    const char *env = (own != nullptr && own[0] != '\0') ? own : auto_priority;
    if (env == nullptr || env[0] == '\0') return 0;
    char *end = nullptr;
    long long value = std::strtoll(env, &end, 10);
    if (end == env) return 0;
    return (int64_t)value;
}

} // namespace

EXPORT_C_FUNC void **__cudaRegisterFatBinary(void *fat_cubin)
{
    RegisterFatBinaryFn real = RealRegisterFatBinary();
    void **handle = real == nullptr ? nullptr : real(fat_cubin);

    std::string reason;
    std::string ptx = ExtractPtxText(fat_cubin, &reason);
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_fatbins[handle] = FatbinRecord{fat_cubin, handle, ptx, {}};
    }

    CUDART_TRACE("runtime_fatbin_registered pid=" FMT_PID " tid=" FMT_TID
                 " fatbin=%p handle=%p ptx_available=%d ptx_bytes=%zu reason=%s",
                 GetProcessId(), GetThreadId(), fat_cubin, (void *)handle, ptx.empty() ? 0 : 1,
                 ptx.size(), reason.empty() ? "OK" : reason.c_str());
    return handle;
}

EXPORT_C_FUNC void __cudaRegisterFatBinaryEnd(void **fatbin_handle)
{
    RegisterFatBinaryEndFn real = RealRegisterFatBinaryEnd();
    if (real != nullptr) real(fatbin_handle);
    CUDART_TRACE("runtime_fatbin_register_end pid=" FMT_PID " tid=" FMT_TID " handle=%p",
                 GetProcessId(), GetThreadId(), (void *)fatbin_handle);
}

EXPORT_C_FUNC void __cudaUnregisterFatBinary(void **fatbin_handle)
{
    CUDART_TRACE("runtime_fatbin_unregister pid=" FMT_PID " tid=" FMT_TID " handle=%p",
                 GetProcessId(), GetThreadId(), (void *)fatbin_handle);
    RemoveFatbin(fatbin_handle);
    UnregisterFatBinaryFn real = RealUnregisterFatBinary();
    if (real != nullptr) real(fatbin_handle);
}

EXPORT_C_FUNC void __cudaRegisterFunction(void **fatbin_handle, const char *host_fun,
                                          char *device_fun, const char *device_name,
                                          int thread_limit, uint3 *tid, uint3 *bid,
                                          dim3 *b_dim, dim3 *g_dim, int *w_size)
{
    RegisterFunctionFn real = RealRegisterFunction();
    if (real != nullptr) {
        real(fatbin_handle, host_fun, device_fun, device_name, thread_limit,
             tid, bid, b_dim, g_dim, w_size);
    }

    const void *host_stub = reinterpret_cast<const void *>(host_fun);
    RuntimeFunctionRecord record;
    record.host_stub = host_stub;
    record.fatbin_handle = fatbin_handle;
    if (device_fun != nullptr) record.device_fun = device_fun;
    if (device_name != nullptr) record.device_name = device_name;

    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_functions[host_stub] = record;
        g_resolved_functions.erase(host_stub);
    }

    CUDART_TRACE("runtime_function_registered pid=" FMT_PID " tid=" FMT_TID
                 " host_stub=%p fatbin=%p device_fun=%s kernel_name=%s",
                 GetProcessId(), GetThreadId(), host_stub, (void *)fatbin_handle,
                 record.device_fun.empty() ? "<unknown>" : record.device_fun.c_str(),
                 record.device_name.empty() ? "<unknown>" : record.device_name.c_str());
}

EXPORT_C_FUNC cudaError_t cudaLaunchKernel(const void *func, dim3 grid_dim, dim3 block_dim,
                                           void **args, size_t shared_mem,
                                           cudaStream_t stream)
{
    const auto mode = xsched::cuda::runtime::CurrentRuntimeStrategyMode();
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);
    CUDART_TRACE("runtime_launch_intercepted pid=" FMT_PID " tid=" FMT_TID
                 " host_stub=%p grid=(%u,%u,%u) block=(%u,%u,%u) shared_mem=%zu "
                 "stream=%p priority=" FMT_64D " strategy=%s",
                 GetProcessId(), GetThreadId(), func, grid_dim.x, grid_dim.y, grid_dim.z,
                 block_dim.x, block_dim.y, block_dim.z, shared_mem, cu_stream,
                 CurrentRuntimePriority(),
                 xsched::cuda::runtime::RuntimeStrategyModeName(mode));

    if (mode != xsched::cuda::runtime::RuntimeStrategyMode::kHbFixed) {
        CUDART_TRACE("runtime_backend_selected backend=NATIVE reason=RUNTIME_STRATEGY_NATIVE "
                     "host_stub=%p strategy=%s",
                     func, xsched::cuda::runtime::RuntimeStrategyModeName(mode));
        return LaunchNativeRuntime(func, grid_dim, block_dim, args, shared_mem, stream);
    }

    RuntimeFunctionRecord record;
    ResolvedRuntimeFunction resolved = ResolveRuntimeFunction(func, &record);
    if (resolved.function == nullptr) {
        CUDART_TRACE("runtime_launch_fallback host_stub=%p reason=%s strategy=%s",
                     func, resolved.reason.empty() ? "RUNTIME_FUNCTION_RESOLVE_FAILED"
                                                   : resolved.reason.c_str(),
                     xsched::cuda::runtime::RuntimeStrategyModeName(mode));
        return LaunchNativeRuntime(func, grid_dim, block_dim, args, shared_mem, stream);
    }

    CUDART_TRACE("runtime_launch_function_resolved host_stub=%p kernel_name=%s module=%p "
                 "function=%p",
                 func, resolved.kernel_name.c_str(), resolved.module, resolved.function);

    if (cu_stream == nullptr) xsched::cuda::WaitBlockingXQueues();
    auto xq = xsched::cuda::ResolveXQueueForStream("cudaLaunchKernel", cu_stream);
    if (xq == nullptr) {
        CUDART_TRACE("runtime_launch_fallback host_stub=%p kernel_name=%s "
                     "reason=RUNTIME_NO_XQUEUE stream=%p",
                     func, resolved.kernel_name.c_str(), cu_stream);
        return LaunchNativeRuntime(func, grid_dim, block_dim, args, shared_mem, stream);
    }

    CUresult cu_result = CUDA_SUCCESS;
    bool submitted = xsched::cuda::hb_split::TryLaunchKernelFixed(
        resolved.function, grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y,
        block_dim.z, (unsigned int)shared_mem, cu_stream, args, nullptr, xq, &cu_result);

    if (submitted) {
        CUDART_TRACE("runtime_backend_selected backend=HB_FIXED host_stub=%p kernel_name=%s "
                     "xqueue=0x" FMT_64X " stream=%p cu_result=%d",
                     func, resolved.kernel_name.c_str(), xq->GetHandle(), cu_stream,
                     (int)cu_result);
        return cu_result == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown;
    }

    std::string hb_reason;
    (void)xsched::cuda::hb_split::LookupFunctionMetadata(resolved.function, nullptr, &hb_reason);
    CUDART_TRACE("runtime_launch_fallback host_stub=%p kernel_name=%s reason=%s "
                 "xqueue=0x" FMT_64X " stream=%p",
                 func, resolved.kernel_name.c_str(),
                 hb_reason.empty() ? "RUNTIME_HB_FIXED_FALLBACK_NATIVE" : hb_reason.c_str(),
                 xq->GetHandle(), cu_stream);
    return LaunchNativeRuntime(func, grid_dim, block_dim, args, shared_mem, stream);
}

EXPORT_C_FUNC cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                              const void *func, void **args)
{
    CUDART_TRACE("runtime_launch_intercepted api=cudaLaunchKernelExC host_stub=%p "
                 "reason=RUNTIME_LAUNCH_EX_UNSUPPORTED",
                 func);
    CudaLaunchKernelExCFn real = RealCudaLaunchKernelExC();
    if (real == nullptr) return cudaErrorUnknown;
    return real(config, func, args);
}

EXPORT_C_FUNC cudaError_t cudaDeviceSynchronize()
{
    CUDART_TRACE("runtime_sync_intercepted api=cudaDeviceSynchronize");
    return RuntimeErrorFromCuResult(xsched::cuda::XCtxSynchronize());
}

EXPORT_C_FUNC cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);
    CUDART_TRACE("runtime_sync_intercepted api=cudaStreamSynchronize stream=%p", cu_stream);
    return RuntimeErrorFromCuResult(xsched::cuda::XStreamSynchronize(cu_stream));
}

EXPORT_C_FUNC cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);
    CUDART_TRACE("runtime_sync_intercepted api=cudaStreamQuery stream=%p", cu_stream);
    return RuntimeErrorFromCuResult(xsched::cuda::XStreamQuery(cu_stream));
}

EXPORT_C_FUNC cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                              unsigned int flags)
{
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);
    CUevent cu_event = reinterpret_cast<CUevent>(event);
    CUDART_TRACE("runtime_sync_intercepted api=cudaStreamWaitEvent stream=%p event=%p "
                 "flags=%u",
                 cu_stream, cu_event, flags);
    return RuntimeErrorFromCuResult(xsched::cuda::XStreamWaitEvent(cu_stream, cu_event, flags));
}

EXPORT_C_FUNC cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);
    CUDART_TRACE("runtime_sync_intercepted api=cudaStreamDestroy stream=%p", cu_stream);
    return RuntimeErrorFromCuResult(xsched::cuda::XStreamDestroy(cu_stream));
}

EXPORT_C_FUNC cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    CUevent cu_event = reinterpret_cast<CUevent>(event);
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);
    CUDART_TRACE("runtime_sync_intercepted api=cudaEventRecord event=%p stream=%p",
                 cu_event, cu_stream);
    return RuntimeErrorFromCuResult(xsched::cuda::XEventRecord(cu_event, cu_stream));
}

EXPORT_C_FUNC cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                                   unsigned int flags)
{
    CUevent cu_event = reinterpret_cast<CUevent>(event);
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);
    CUDART_TRACE("runtime_sync_intercepted api=cudaEventRecordWithFlags event=%p "
                 "stream=%p flags=%u",
                 cu_event, cu_stream, flags);
    return RuntimeErrorFromCuResult(
        xsched::cuda::XEventRecordWithFlags(cu_event, cu_stream, flags));
}

EXPORT_C_FUNC cudaError_t cudaEventQuery(cudaEvent_t event)
{
    CUevent cu_event = reinterpret_cast<CUevent>(event);
    CUDART_TRACE("runtime_sync_intercepted api=cudaEventQuery event=%p", cu_event);
    return RuntimeErrorFromCuResult(xsched::cuda::XEventQuery(cu_event));
}

EXPORT_C_FUNC cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    CUevent cu_event = reinterpret_cast<CUevent>(event);
    CUDART_TRACE("runtime_sync_intercepted api=cudaEventSynchronize event=%p", cu_event);
    return RuntimeErrorFromCuResult(xsched::cuda::XEventSynchronize(cu_event));
}

EXPORT_C_FUNC cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    CUevent cu_event = reinterpret_cast<CUevent>(event);
    CUDART_TRACE("runtime_sync_intercepted api=cudaEventDestroy event=%p", cu_event);
    return RuntimeErrorFromCuResult(xsched::cuda::XEventDestroy(cu_event));
}
#endif // UXSCHED_ENABLE_HB_SPLIT
