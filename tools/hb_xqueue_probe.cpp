#include <dlfcn.h>

#include <cstdint>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "xsched/cuda/hal/common/cuda.h"

namespace
{

constexpr const char *kProbePtx = R"PTX(
.version 7.0
.target sm_50
.address_size 64

.visible .entry hb_xqueue_probe_kernel(
    .param .u64 hb_param_out,
    .param .u32 hb_param_n
)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd1, [hb_param_out];
    ld.param.u32 %r1, [hb_param_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE;
    mul.wide.u32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r5;
DONE:
    ret;
}

.visible .entry hb_xqueue_check_kernel(
    .param .u64 hb_param_out,
    .param .u32 hb_param_n,
    .param .u64 hb_param_error
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [hb_param_out];
    ld.param.u32 %r1, [hb_param_n];
    ld.param.u64 %rd4, [hb_param_error];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra CHECK_DONE;
    mul.wide.u32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.u32 %r6, [%rd3];
    setp.eq.u32 %p2, %r6, %r5;
    @%p2 bra CHECK_DONE;
    mov.u32 %r7, 1;
    st.global.u32 [%rd4], %r7;
CHECK_DONE:
    ret;
}
)PTX";

template <typename Fn>
Fn Load(void *handle, const char *name)
{
    dlerror();
    void *sym = dlsym(RTLD_DEFAULT, name);
    if (sym == nullptr) sym = dlsym(handle, name);
    const char *err = dlerror();
    if (sym == nullptr || err != nullptr) {
        std::fprintf(stderr, "missing CUDA symbol %s: %s\n", name, err == nullptr ? "not found" : err);
        std::exit(2);
    }
    return reinterpret_cast<Fn>(sym);
}

void Check(CUresult ret, const char *op)
{
    if (ret == CUDA_SUCCESS) return;
    std::fprintf(stderr, "%s failed: CUDA error %d\n", op, static_cast<int>(ret));
    std::exit(1);
}

bool ArgEquals(const char *arg, const char *name)
{
    return std::strcmp(arg, name) == 0;
}

} // namespace

int main(int argc, char **argv)
{
    std::string stream_mode = "default";
    std::string sync_mode = "basic";
    unsigned int blocks = 1024;
    unsigned int threads = 1;

    for (int i = 1; i < argc; ++i) {
        if (ArgEquals(argv[i], "--stream") && i + 1 < argc) {
            stream_mode = argv[++i];
        } else if (ArgEquals(argv[i], "--sync") && i + 1 < argc) {
            sync_mode = argv[++i];
        } else if (ArgEquals(argv[i], "--blocks") && i + 1 < argc) {
            blocks = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
        } else if (ArgEquals(argv[i], "--threads") && i + 1 < argc) {
            threads = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
        } else {
            std::fprintf(stderr,
                         "usage: %s --stream default|explicit "
                         "[--sync basic|event|stream|context|same-stream|parent] "
                         "[--blocks N] [--threads N]\n",
                         argv[0]);
            return 2;
        }
    }

    if (stream_mode != "default" && stream_mode != "explicit") {
        std::fprintf(stderr, "--stream must be default or explicit\n");
        return 2;
    }
    if (sync_mode != "basic" && sync_mode != "event" && sync_mode != "stream" &&
        sync_mode != "context" && sync_mode != "same-stream" && sync_mode != "parent") {
        std::fprintf(stderr, "--sync must be basic, event, stream, context, same-stream, or parent\n");
        return 2;
    }
    if (blocks == 0 || threads == 0) {
        std::fprintf(stderr, "--blocks and --threads must be positive\n");
        return 2;
    }

    const char *cuda_lib = std::getenv("XSCHED_CUDA_LIB");
    if (cuda_lib == nullptr || cuda_lib[0] == '\0') cuda_lib = "libcuda.so.1";
    void *libcuda = dlopen(cuda_lib, RTLD_NOW | RTLD_GLOBAL);
    if (libcuda == nullptr) {
        std::fprintf(stderr, "dlopen(%s) failed: %s\n", cuda_lib, dlerror());
        return 2;
    }

    auto cuInit = Load<CUresult (*)(unsigned int)>(libcuda, "cuInit");
    auto cuDeviceGet = Load<CUresult (*)(CUdevice *, int)>(libcuda, "cuDeviceGet");
    auto cuDeviceGetName = Load<CUresult (*)(char *, int, CUdevice)>(libcuda, "cuDeviceGetName");
    auto cuDevicePrimaryCtxRetain =
        Load<CUresult (*)(CUcontext *, CUdevice)>(libcuda, "cuDevicePrimaryCtxRetain");
    auto cuDevicePrimaryCtxRelease =
        Load<CUresult (*)(CUdevice)>(libcuda, "cuDevicePrimaryCtxRelease");
    auto cuCtxSetCurrent = Load<CUresult (*)(CUcontext)>(libcuda, "cuCtxSetCurrent");
    auto cuCtxSynchronize = Load<CUresult (*)()>(libcuda, "cuCtxSynchronize");
    auto cuEventCreate = Load<CUresult (*)(CUevent *, unsigned int)>(libcuda, "cuEventCreate");
    auto cuEventRecord = Load<CUresult (*)(CUevent, CUstream)>(libcuda, "cuEventRecord");
    auto cuEventSynchronize = Load<CUresult (*)(CUevent)>(libcuda, "cuEventSynchronize");
    auto cuEventDestroy = Load<CUresult (*)(CUevent)>(libcuda, "cuEventDestroy_v2");
    auto cuModuleLoadDataEx =
        Load<CUresult (*)(CUmodule *, const void *, unsigned int, CUjit_option *, void **)>(
            libcuda, "cuModuleLoadDataEx");
    auto cuModuleGetFunction =
        Load<CUresult (*)(CUfunction *, CUmodule, const char *)>(libcuda, "cuModuleGetFunction");
    auto cuModuleUnload = Load<CUresult (*)(CUmodule)>(libcuda, "cuModuleUnload");
    auto cuMemAlloc = Load<CUresult (*)(CUdeviceptr *, size_t)>(libcuda, "cuMemAlloc_v2");
    auto cuMemFree = Load<CUresult (*)(CUdeviceptr)>(libcuda, "cuMemFree_v2");
    auto cuMemcpyDtoH = Load<CUresult (*)(void *, CUdeviceptr, size_t)>(libcuda, "cuMemcpyDtoH_v2");
    auto cuMemsetD32 = Load<CUresult (*)(CUdeviceptr, unsigned int, size_t)>(libcuda, "cuMemsetD32_v2");
    auto cuStreamCreate = Load<CUresult (*)(CUstream *, unsigned int)>(libcuda, "cuStreamCreate");
    auto cuStreamSynchronize = Load<CUresult (*)(CUstream)>(libcuda, "cuStreamSynchronize");
    auto cuStreamDestroy = Load<CUresult (*)(CUstream)>(libcuda, "cuStreamDestroy_v2");
    auto cuLaunchKernel =
        Load<CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                          unsigned int, unsigned int, unsigned int, unsigned int,
                          CUstream, void **, void **)>(libcuda, "cuLaunchKernel");

    Check(cuInit(0), "cuInit");
    CUdevice dev = 0;
    Check(cuDeviceGet(&dev, 0), "cuDeviceGet");
    char device_name[128] = {};
    Check(cuDeviceGetName(device_name, sizeof(device_name), dev), "cuDeviceGetName");
    CUcontext ctx = nullptr;
    Check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    Check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

    CUmodule module = nullptr;
    Check(cuModuleLoadDataEx(&module, kProbePtx, 0, nullptr, nullptr), "cuModuleLoadDataEx");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "hb_xqueue_probe_kernel"), "cuModuleGetFunction");
    CUfunction check_function = nullptr;
    Check(cuModuleGetFunction(&check_function, module, "hb_xqueue_check_kernel"), "cuModuleGetFunction");

    CUstream stream = nullptr;
    if (stream_mode == "explicit") {
        Check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    }

    const uint32_t n = blocks * threads;
    CUdeviceptr out = 0;
    CUdeviceptr error_flag = 0;
    Check(cuMemAlloc(&out, n * sizeof(uint32_t)), "cuMemAlloc");
    Check(cuMemAlloc(&error_flag, sizeof(uint32_t)), "cuMemAlloc(error_flag)");
    Check(cuMemsetD32(out, 0xffffffffu, n), "cuMemsetD32(out)");
    Check(cuMemsetD32(error_flag, 0, 1), "cuMemsetD32(error_flag)");

    void *params[] = {&out, const_cast<uint32_t *>(&n)};
    std::printf("cuda_device=%s\n", device_name);
    std::printf("stream_mode=%s\n", stream_mode.c_str());
    std::printf("sync_mode=%s\n", sync_mode.c_str());
    std::printf("stream_handle=%p\n", stream);
    std::printf("kernel=hb_xqueue_probe_kernel\n");
    std::printf("blocks=%u\n", blocks);
    std::printf("threads=%u\n", threads);

    Check(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, stream, params, nullptr),
          "cuLaunchKernel");

    bool event_sync_pass = false;
    bool stream_sync_pass = false;
    bool context_sync_pass = false;
    bool same_stream_ordering_pass = false;
    bool parent_completion_probe = false;

    if (sync_mode == "same-stream") {
        void *check_params[] = {&out, const_cast<uint32_t *>(&n), &error_flag};
        std::printf("dependent_kernel=hb_xqueue_check_kernel\n");
        Check(cuLaunchKernel(check_function, blocks, 1, 1, threads, 1, 1, 0, stream,
                             check_params, nullptr),
              "cuLaunchKernel(check)");
    }

    if (sync_mode == "event") {
        CUevent event = nullptr;
        Check(cuEventCreate(&event, CU_EVENT_DEFAULT), "cuEventCreate");
        Check(cuEventRecord(event, stream), "cuEventRecord");
        Check(cuEventSynchronize(event), "cuEventSynchronize");
        Check(cuEventDestroy(event), "cuEventDestroy");
        event_sync_pass = true;
    } else if (sync_mode == "stream" || sync_mode == "same-stream" || sync_mode == "parent") {
        Check(cuStreamSynchronize(stream), "cuStreamSynchronize");
        stream_sync_pass = true;
        parent_completion_probe = sync_mode == "parent";
    } else if (sync_mode == "context") {
        Check(cuCtxSynchronize(), "cuCtxSynchronize");
        context_sync_pass = true;
    } else {
        if (stream == nullptr) {
            Check(cuCtxSynchronize(), "cuCtxSynchronize");
            context_sync_pass = true;
        } else {
            Check(cuStreamSynchronize(stream), "cuStreamSynchronize");
            stream_sync_pass = true;
        }
    }

    std::vector<uint32_t> host(n);
    Check(cuMemcpyDtoH(host.data(), out, host.size() * sizeof(uint32_t)), "cuMemcpyDtoH");
    uint32_t device_error = 0;
    Check(cuMemcpyDtoH(&device_error, error_flag, sizeof(device_error)), "cuMemcpyDtoH(error_flag)");
    uint64_t checksum = 0;
    uint64_t hash = 1469598103934665603ull;
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < n; ++i) {
        checksum += host[i];
        const uint32_t v = host[i];
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<unsigned char>((v >> (byte * 8)) & 0xffu);
            hash *= 1099511628211ull;
        }
        if (host[i] != i) ++mismatches;
    }
    same_stream_ordering_pass = (sync_mode == "same-stream" && device_error == 0 && mismatches == 0);
    std::printf("checksum=%" PRIu64 "\n", checksum);
    std::printf("output_hash=%016" PRIx64 "\n", hash);
    std::printf("output_element_count=%u\n", n);
    std::printf("mismatches=%u\n", mismatches);
    std::printf("device_error=%u\n", device_error);
    if (sync_mode == "event") {
        std::printf("event_sync_pass=%u\n", event_sync_pass && mismatches == 0 ? 1u : 0u);
    } else if (sync_mode == "stream") {
        std::printf("stream_sync_pass=%u\n", stream_sync_pass && mismatches == 0 ? 1u : 0u);
    } else if (sync_mode == "context") {
        std::printf("context_sync_pass=%u\n", context_sync_pass && mismatches == 0 ? 1u : 0u);
    } else if (sync_mode == "same-stream") {
        std::printf("same_stream_ordering_pass=%u\n", same_stream_ordering_pass ? 1u : 0u);
    } else if (sync_mode == "parent") {
        std::printf("parent_completion_probe=%u\n", parent_completion_probe && mismatches == 0 ? 1u : 0u);
    }

    Check(cuMemFree(error_flag), "cuMemFree(error_flag)");
    Check(cuMemFree(out), "cuMemFree");
    if (stream != nullptr) Check(cuStreamDestroy(stream), "cuStreamDestroy");
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(dev), "cuDevicePrimaryCtxRelease");
    return mismatches == 0 && device_error == 0 ? 0 : 1;
}
