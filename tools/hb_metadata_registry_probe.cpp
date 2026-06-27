#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>

#include "xsched/cuda/hal/hb_split/backend.h"

namespace
{

constexpr const char *kPtx = R"ptx(
.version 8.7
.target sm_120
.address_size 64

.entry __hb_registry_probe_kernel(
    .param .u64 __hb_registry_probe_kernel_param_0
)
{
    ret;
}
)ptx";

} // namespace

int main()
{
    setenv("UXSCHED_HB_VERIFIED_KERNELS", "__not_the_probe_kernel", 1);

    CUmodule module = reinterpret_cast<CUmodule>(static_cast<uintptr_t>(0x12345000));
    CUfunction function = reinterpret_cast<CUfunction>(static_cast<uintptr_t>(0x12346000));
    const char *kernel_name = "__hb_registry_probe_kernel";

    auto module_res = xsched::cuda::hb_split::RegisterModuleMetadata(
        module, kPtx, std::char_traits<char>::length(kPtx));
    if (!module_res.ok) {
        std::cerr << "module_register_failed reason=" << module_res.reason << "\n";
        return 1;
    }

    auto function_res = xsched::cuda::hb_split::RegisterFunctionMetadata(
        function, module, kernel_name);
    if (!function_res.ok) {
        std::cerr << "function_register_failed reason=" << function_res.reason << "\n";
        return 2;
    }

    std::string found_name;
    std::string fallback_reason;
    if (!xsched::cuda::hb_split::LookupFunctionMetadata(function, &found_name, &fallback_reason)) {
        std::cerr << "lookup_failed_before_unregister\n";
        return 3;
    }
    if (found_name != kernel_name) {
        std::cerr << "lookup_name_mismatch found=" << found_name << "\n";
        return 4;
    }
    if (fallback_reason != "KERNEL_NOT_VERIFIED") {
        std::cerr << "fallback_reason_mismatch reason=" << fallback_reason << "\n";
        return 5;
    }

    xsched::cuda::hb_split::UnregisterModuleMetadata(module);
    if (xsched::cuda::hb_split::LookupFunctionMetadata(function, nullptr, nullptr)) {
        std::cerr << "lookup_succeeded_after_unregister\n";
        return 6;
    }

    std::cout << "hb_metadata_registry_probe_pass=1\n";
    std::cout << "module_ptx_bytes=" << module_res.ptx_bytes << "\n";
    std::cout << "module_record_count=" << module_res.record_count << "\n";
    std::cout << "function_record_count=" << function_res.record_count << "\n";
    std::cout << "fallback_reason=" << fallback_reason << "\n";
    return 0;
}
