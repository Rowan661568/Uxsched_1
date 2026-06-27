#include "xsched/cuda/hal/hb_split/backend.h"

// PTX offset injection and grid decomposition are adapted from the local
// Hummingbird prototype in /home/zm/project/hummingbird/kernel_splitter.
// No LICENSE/NOTICE file was found in that local source tree during this audit.
// UXSched remains the only CUDA hook and scheduler; this file only provides an
// optional split backend behind the UXSched CUDA shim.

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <strings.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuxtra/cuxtra.h>

#include "xsched/protocol/def.h"
#include "xsched/utils/common.h"
#include "xsched/utils/log.h"
#include "xsched/cuda/hal/common/cuda_command.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/preempt/xqueue/xqueue.h"

namespace xsched::cuda::hb_split
{

namespace
{

constexpr size_t kImageScanLimit = 64 * 1024 * 1024;
constexpr size_t kOffsetParamCount = 3;

struct Grid3D
{
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;
};

struct SplitSpec
{
    Grid3D grid;
    std::array<uint32_t, kOffsetParamCount> offset{};
};

struct PtxTransformResult
{
    std::string ptx_text;
    bool modified = false;
    bool supports_offset_x = false;
    bool supports_offset_y = false;
    bool supports_offset_z = false;
    bool cross_block_sync = false;
    std::string diagnostic;
};

struct KernelTransformRecord
{
    KernelCapability capability;
    size_t original_param_count = 0;
};

struct ModuleInfo
{
    bool ptx_available = false;
    std::shared_ptr<const std::string> original_ptx;
    size_t ptx_bytes = 0;
    CUcontext context = nullptr;
    CUdevice device = CU_DEVICE_INVALID;
    CUmodule transformed_module = nullptr;
    std::unordered_map<std::string, KernelTransformRecord> records;
};

struct FunctionInfo
{
    KernelCapability capability;
    size_t original_param_count = 0;
    std::string kernel_name;
    CUmodule original_module = nullptr;
    CUfunction original_function = nullptr;
    CUfunction transformed_function = nullptr;
};

struct SplitCommandGroup
{
    std::string kernel_name;
    size_t split_count = 0;
    std::atomic<size_t> completed{0};
    CUresult first_error = CUDA_SUCCESS;
    std::vector<std::shared_ptr<CudaKernelLaunchCommand>> children;
};

std::mutex g_mu;
std::unordered_map<CUmodule, ModuleInfo> g_modules;
std::unordered_map<CUfunction, FunctionInfo> g_functions;
std::unordered_set<XQueueHandle> g_threshold_adjusted_xqueues;

bool IsTruthyEnv(const char *name, bool fallback)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return fallback;
    if (std::strcmp(env, "0") == 0 || strcasecmp(env, "off") == 0 ||
        strcasecmp(env, "false") == 0 || strcasecmp(env, "no") == 0) {
        return false;
    }
    return true;
}

bool BuildEnabled()
{
#ifdef UXSCHED_ENABLE_HB_SPLIT
    return true;
#else
    return false;
#endif
}

BackendMode BackendModeFromEnv()
{
    const char *env = std::getenv("UXSCHED_CUDA_RUNTIME_STRATEGY");
    if (env == nullptr || env[0] == '\0') env = std::getenv("UXSCHED_CUDA_PREEMPT_BACKEND");
    if (env == nullptr || env[0] == '\0' || strcasecmp(env, "NATIVE") == 0) {
        return BackendMode::kNative;
    }
    if (strcasecmp(env, "HB_SPLIT") == 0) return BackendMode::kHbSplit;
    if (strcasecmp(env, "HB_FIXED") == 0) return BackendMode::kHbSplit;
    if (strcasecmp(env, "HB_RUNTIME") == 0) return BackendMode::kHbSplit;
    if (strcasecmp(env, "AUTO") == 0) return BackendMode::kAuto;
    XWARN("[UXSCHED-HB] invalid CUDA runtime strategy=%s; using NATIVE", env);
    return BackendMode::kNative;
}

const char *BackendModeName(BackendMode mode)
{
    switch (mode) {
    case BackendMode::kNative: return "NATIVE";
    case BackendMode::kHbSplit: return "HB_SPLIT";
    case BackendMode::kAuto: return "AUTO";
    }
    return "NATIVE";
}

int64_t EnvInt64(const char *name, int64_t fallback)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return fallback;
    char *end = nullptr;
    long long value = std::strtoll(env, &end, 10);
    if (end == env) return fallback;
    return (int64_t)value;
}

int SplitBlocks()
{
    int64_t value = EnvInt64("UXSCHED_HB_SPLIT_BLOCKS", 512);
    if (value <= 0 || value > std::numeric_limits<int>::max()) return 512;
    return (int)value;
}

bool StrictMode()
{
    return IsTruthyEnv("UXSCHED_HB_STRICT", false);
}

bool HbLogEnabled()
{
    const char *env = std::getenv("UXSCHED_HB_LOG_LEVEL");
    if (env == nullptr || env[0] == '\0') return true;
    return strcasecmp(env, "OFF") != 0 && strcasecmp(env, "NONE") != 0;
}

int64_t CurrentPriority()
{
    const char *own = std::getenv("UXSCHED_HB_PRIORITY");
    if (own != nullptr && own[0] != '\0') return EnvInt64("UXSCHED_HB_PRIORITY", 0);
    return EnvInt64(XSCHED_AUTO_XQUEUE_PRIORITY_ENV_NAME, 0);
}

bool IsLowPriority()
{
    return CurrentPriority() < 0;
}

bool Lv1Compatible()
{
    return EnvInt64(XSCHED_AUTO_XQUEUE_LEVEL_ENV_NAME, 1) <= 1;
}

bool RuntimeEnabled()
{
    if (!BuildEnabled()) return false;

    const char *strategy = std::getenv("UXSCHED_CUDA_RUNTIME_STRATEGY");
    if (strategy != nullptr && strategy[0] != '\0') {
        return strcasecmp(strategy, "HB_FIXED") == 0 || strcasecmp(strategy, "HB_SPLIT") == 0;
    }

    return BackendModeFromEnv() != BackendMode::kNative;
}

void LogConfigOnce()
{
    static std::once_flag flag;
    std::call_once(flag, []() {
        if (!HbLogEnabled()) return;
        const char *sched = std::getenv(XSCHED_SCHEDULER_ENV_NAME);
        const char *policy = std::getenv(XSCHED_POLICY_ENV_NAME);
        XINFO("[UXSCHED-HB] backend=%s build_enabled=%d split_blocks=%d strict=%d "
              "scheduler=%s policy=%s unique_hook=UXSched-CUDA-shim",
              BackendModeName(BackendModeFromEnv()), BuildEnabled() ? 1 : 0, SplitBlocks(),
              StrictMode() ? 1 : 0, sched == nullptr ? "<unset>" : sched,
              policy == nullptr ? "<unset>" : policy);
    });
}

void LogFallback(const char *reason, const FunctionInfo *info = nullptr)
{
    if (!HbLogEnabled()) return;
    XINFO("[UXSCHED-HB] backend_selected=NATIVE reason=%s function=%s priority=" FMT_64D,
          reason, info == nullptr ? "<unknown>" : info->kernel_name.c_str(), CurrentPriority());
}

bool IsSpace(char c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

bool LooksLikePtxText(const void *image, size_t nbytes)
{
    if (image == nullptr || nbytes < 8) return false;
    const char *s = static_cast<const char *>(image);
    const size_t scan = std::min(nbytes, (size_t)4096);
    size_t i = 0;
    while (i < scan && IsSpace(s[i])) ++i;
    if (i + 8 <= scan && std::strncmp(s + i, ".version", 8) == 0) return true;
    if (i + 2 <= scan && s[i] == '/' && s[i + 1] == '/') return true;
    if (nbytes >= 32) {
        std::string head(s, s + std::min(nbytes, (size_t)2048));
        return head.find(".version") != std::string::npos &&
               head.find(".address_size") != std::string::npos;
    }
    return false;
}

std::optional<size_t> FindEntryName(const std::string &s, const std::string &kernel_name)
{
    const std::array<std::string, 2> needles{
        ".visible .entry " + kernel_name,
        ".entry " + kernel_name,
    };
    for (const std::string &needle : needles) {
        size_t pos = s.find(needle);
        if (pos != std::string::npos) return pos + needle.size();
    }
    return std::nullopt;
}

bool FindEntryParamClose(const std::string &s, const std::string &kernel_name,
                         size_t &open_paren, size_t &close_paren)
{
    auto entry_end = FindEntryName(s, kernel_name);
    if (!entry_end) return false;
    size_t pos = s.find('(', *entry_end);
    if (pos == std::string::npos) return false;
    int depth = 0;
    for (size_t i = pos; i < s.size(); ++i) {
        if (s[i] == '(') {
            ++depth;
        } else if (s[i] == ')') {
            --depth;
            if (depth == 0) {
                open_paren = pos;
                close_paren = i;
                return true;
            }
        }
    }
    return false;
}

bool ParamListNonEmpty(const std::string &s, size_t open_paren, size_t close_paren)
{
    for (size_t i = open_paren + 1; i < close_paren; ++i) {
        if (!IsSpace(s[i])) return true;
    }
    return false;
}

bool ParamListHasOffsets(const std::string &s, size_t open_paren, size_t close_paren)
{
    const std::string params = s.substr(open_paren + 1, close_paren - open_paren - 1);
    return params.find("__hb_off_x") != std::string::npos &&
           params.find("__hb_off_y") != std::string::npos &&
           params.find("__hb_off_z") != std::string::npos;
}

std::optional<size_t> FindClosingBrace(const std::string &s, size_t open_brace)
{
    int depth = 0;
    for (size_t i = open_brace; i < s.size(); ++i) {
        if (s[i] == '{') {
            ++depth;
        } else if (s[i] == '}') {
            --depth;
            if (depth == 0) return i;
        }
    }
    return std::nullopt;
}

std::string InjectOffsetParams(const std::string &s, size_t open_paren, size_t close_paren)
{
    static constexpr const char *kExtraParams =
        ".param .u32 __hb_off_x, .param .u32 __hb_off_y, .param .u32 __hb_off_z";
    std::string out;
    out.reserve(s.size() + 128);
    out.append(s, 0, close_paren);
    out.append(ParamListNonEmpty(s, open_paren, close_paren) ? ", " : " ");
    out.append(kExtraParams);
    out.append(s, close_paren, std::string::npos);
    return out;
}

bool ContainsCrossBlockSync(const std::string &body)
{
    return body.find("grid.sync") != std::string::npos ||
           body.find("griddepcontrol") != std::string::npos ||
           body.find("barrier.cluster") != std::string::npos;
}

std::string RewriteAxisMov(const std::string &input, char axis, const char *offset_reg, bool *modified)
{
    std::string pattern = std::string(R"((^|\n)([\t ]*)mov\.u32\s+(%[A-Za-z0-9_]+)\s*,\s*%ctaid\.)")
                        + axis + R"(\s*;)";
    const std::regex rx(pattern);
    std::string out;
    size_t last = 0;
    bool any = false;
    for (auto it = std::sregex_iterator(input.begin(), input.end(), rx);
         it != std::sregex_iterator(); ++it) {
        const std::smatch &m = *it;
        out.append(input, last, (size_t)m.position() - last);
        out.append(m.str(1));
        out.append(m.str(2));
        out.append("mov.u32 ");
        out.append(m.str(3));
        out.append(", %ctaid.");
        out.push_back(axis);
        out.append(";\n");
        out.append(m.str(2));
        out.append("add.s32 ");
        out.append(m.str(3));
        out.append(", ");
        out.append(m.str(3));
        out.append(", ");
        out.append(offset_reg);
        out.append(";");
        last = (size_t)m.position() + (size_t)m.length();
        any = true;
    }
    if (!any) return input;
    out.append(input, last, std::string::npos);
    if (modified != nullptr) *modified = true;
    return out;
}

PtxTransformResult TransformKernelPtx(const std::string &ptx, const std::string &kernel_name)
{
    PtxTransformResult out;
    out.ptx_text = ptx;

    size_t open = 0;
    size_t close = 0;
    if (!FindEntryParamClose(ptx, kernel_name, open, close)) {
        out.diagnostic = "ENTRY_NOT_FOUND";
        return out;
    }
    if (ParamListHasOffsets(ptx, open, close)) {
        out.diagnostic = "ALREADY_TRANSFORMED";
        return out;
    }

    const std::string with_params = InjectOffsetParams(ptx, open, close);
    size_t open2 = 0;
    size_t close2 = 0;
    if (!FindEntryParamClose(with_params, kernel_name, open2, close2)) {
        out.diagnostic = "INTERNAL_ENTRY_LOST";
        return out;
    }
    const size_t body_open = with_params.find('{', close2);
    if (body_open == std::string::npos) {
        out.diagnostic = "BODY_NOT_FOUND";
        return out;
    }
    auto body_close = FindClosingBrace(with_params, body_open);
    if (!body_close) {
        out.diagnostic = "UNBALANCED_BRACES";
        return out;
    }

    std::string body = with_params.substr(body_open + 1, *body_close - body_open - 1);
    if (ContainsCrossBlockSync(body)) {
        out.cross_block_sync = true;
        out.diagnostic = "CROSS_BLOCK_SYNC";
        return out;
    }

    bool x = false;
    bool y = false;
    bool z = false;
    try {
        body = RewriteAxisMov(body, 'x', "%__hb_rx", &x);
        body = RewriteAxisMov(body, 'y', "%__hb_ry", &y);
        body = RewriteAxisMov(body, 'z', "%__hb_rz", &z);
    } catch (const std::regex_error &) {
        out.diagnostic = "REGEX_REWRITE_FAILED";
        return out;
    }

    static constexpr const char *kOffsetHeader =
        "\n\t.reg .b32 %__hb_rx;\n\t.reg .b32 %__hb_ry;\n\t.reg .b32 %__hb_rz;\n\t"
        "ld.param.u32 %__hb_rx, [__hb_off_x];\n\t"
        "ld.param.u32 %__hb_ry, [__hb_off_y];\n\t"
        "ld.param.u32 %__hb_rz, [__hb_off_z];\n";

    out.supports_offset_x = x;
    out.supports_offset_y = y;
    out.supports_offset_z = z;
    out.ptx_text = with_params.substr(0, body_open + 1) + kOffsetHeader + body +
                   with_params.substr(*body_close);
    out.modified = true;
    out.diagnostic = "OK";
    return out;
}

std::vector<std::string> CollectEntryNames(const std::string &ptx)
{
    std::vector<std::string> names;
    static constexpr const char *kNeedle = ".entry ";
    size_t pos = 0;
    while ((pos = ptx.find(kNeedle, pos)) != std::string::npos) {
        if (pos > 0 && (std::isalnum((unsigned char)ptx[pos - 1]) ||
                        ptx[pos - 1] == '_' || ptx[pos - 1] == '.')) {
            pos += std::strlen(kNeedle);
            continue;
        }
        pos += std::strlen(kNeedle);
        while (pos < ptx.size() && IsSpace(ptx[pos])) ++pos;
        size_t end = pos;
        while (end < ptx.size() && !IsSpace(ptx[end]) && ptx[end] != '(') ++end;
        if (end > pos) names.emplace_back(ptx.substr(pos, end - pos));
        pos = end;
    }
    return names;
}

std::set<std::string> VerifiedKernelNames(bool *wildcard)
{
    if (wildcard != nullptr) *wildcard = false;
    const char *env = std::getenv("UXSCHED_HB_VERIFIED_KERNELS");
    if (env == nullptr || env[0] == '\0') env = std::getenv("HB_SPLIT_KERNELS");
    std::set<std::string> out;
    if (env == nullptr || env[0] == '\0') return out;

    std::string current;
    for (const char *p = env; *p != '\0'; ++p) {
        if (*p == ',') {
            if (!current.empty()) out.insert(current);
            current.clear();
        } else if (!std::isspace((unsigned char)*p)) {
            current.push_back(*p);
        }
    }
    if (!current.empty()) out.insert(current);
    if (out.find("*") != out.end()) {
        if (wildcard != nullptr) *wildcard = true;
        out.clear();
    }
    return out;
}

std::optional<size_t> CountParamsForKernel(const std::string &ptx, const std::string &kernel_name)
{
    size_t open = 0;
    size_t close = 0;
    if (!FindEntryParamClose(ptx, kernel_name, open, close)) return std::nullopt;
    const std::string param_list = ptx.substr(open + 1, close - open - 1);
    size_t count = 0;
    size_t pos = 0;
    while ((pos = param_list.find(".param", pos)) != std::string::npos) {
        ++count;
        pos += 6;
    }
    return count;
}

ModuleInfo TransformModulePtx(const std::shared_ptr<const std::string> &ptx_owner)
{
    ModuleInfo info;
    info.ptx_available = true;
    info.original_ptx = ptx_owner;
    info.ptx_bytes = ptx_owner == nullptr ? 0 : ptx_owner->size();
    if (Driver::CtxGetCurrent(&info.context) != CUDA_SUCCESS) info.context = nullptr;
    if (Driver::CtxGetDevice(&info.device) != CUDA_SUCCESS) info.device = CU_DEVICE_INVALID;

    const std::string &ptx = *ptx_owner;

    bool wildcard = false;
    const std::set<std::string> verified = VerifiedKernelNames(&wildcard);
    if (!wildcard && verified.empty()) {
        if (HbLogEnabled()) {
            XINFO("[UXSCHED-HB] no verified kernels supplied; PTX module will use Native fallback");
        }
        return info;
    }

    std::string transformed = ptx;
    for (const std::string &name : CollectEntryNames(ptx)) {
        KernelTransformRecord rec;
        rec.capability.ptx_available = true;
        rec.capability.supports_kernel_params = true;
        rec.capability.supports_extra = false;
        rec.capability.fallback_reason = "KERNEL_NOT_VERIFIED";

        const auto param_count = CountParamsForKernel(ptx, name);
        if (!param_count) {
            rec.capability.fallback_reason = "PARAM_COUNT_UNAVAILABLE";
            info.records[name] = rec;
            continue;
        }
        rec.original_param_count = *param_count;

        if (!wildcard && verified.find(name) == verified.end()) {
            info.records[name] = rec;
            continue;
        }

        rec.capability.transform_attempted = true;
        rec.capability.fallback_reason = "TRANSFORM_NOT_ATTEMPTED";

        PtxTransformResult tr = TransformKernelPtx(transformed, name);
        rec.capability.transform_succeeded = tr.modified;
        rec.capability.supports_offset_x = tr.supports_offset_x;
        rec.capability.supports_offset_y = tr.supports_offset_y;
        rec.capability.supports_offset_z = tr.supports_offset_z;
        rec.capability.cross_block_sync = tr.cross_block_sync;

        if (tr.modified) {
            transformed = std::move(tr.ptx_text);
            rec.capability.splittable = true;
            rec.capability.fallback_reason.clear();
            if (HbLogEnabled()) {
                XINFO("[UXSCHED-HB] transform_succeeded function=%s orig_params=%zu "
                      "offset_axes=(%d,%d,%d)",
                      name.c_str(), rec.original_param_count, rec.capability.supports_offset_x ? 1 : 0,
                      rec.capability.supports_offset_y ? 1 : 0,
                      rec.capability.supports_offset_z ? 1 : 0);
            }
        } else {
            rec.capability.fallback_reason = tr.diagnostic.empty() ? "TRANSFORM_FAILED" : tr.diagnostic;
            if (HbLogEnabled()) {
                XINFO("[UXSCHED-HB] transform_failed function=%s reason=%s",
                      name.c_str(), rec.capability.fallback_reason.c_str());
            }
        }
        info.records[name] = rec;
    }

    if (info.records.empty()) return info;
    bool any_modified = false;
    for (const auto &kv : info.records) {
        any_modified = any_modified || kv.second.capability.transform_succeeded;
    }
    if (!any_modified) return info;

    std::vector<char> transformed_buf(transformed.begin(), transformed.end());
    transformed_buf.push_back('\0');
    CUmodule transformed_module = nullptr;
    CUresult ret = Driver::ModuleLoadDataEx(&transformed_module, transformed_buf.data(), 0, nullptr, nullptr);
    if (ret != CUDA_SUCCESS || transformed_module == nullptr) {
        for (auto &kv : info.records) {
            kv.second.capability.splittable = false;
            kv.second.capability.fallback_reason = "TRANSFORMED_MODULE_LOAD_FAILED";
        }
        return info;
    }
    info.transformed_module = transformed_module;
    if (HbLogEnabled()) {
        XINFO("[UXSCHED-HB] transformed_module_loaded transformed_module=%p records=%zu",
              transformed_module, info.records.size());
    }
    return info;
}

MetadataRegistrationResult RegisterModuleInfo(CUmodule module, ModuleInfo info)
{
    MetadataRegistrationResult result;
    if (module == nullptr) {
        result.reason = "RUNTIME_HB_MODULE_REGISTER_FAILED";
        return result;
    }
    result.ok = true;
    result.reason = "OK";
    result.ptx_bytes = info.ptx_bytes;
    result.record_count = info.records.size();
    std::lock_guard<std::mutex> lock(g_mu);
    g_modules[module] = std::move(info);
    return result;
}

std::optional<ModuleInfo> FindModuleInfo(CUmodule module)
{
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_modules.find(module);
    if (it == g_modules.end()) return std::nullopt;
    return it->second;
}

std::optional<FunctionInfo> FindFunctionInfo(CUfunction function)
{
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_functions.find(function);
    if (it == g_functions.end()) return std::nullopt;
    return it->second;
}

MetadataRegistrationResult RegisterFunctionInfo(CUfunction function, CUmodule module,
                                                const char *name)
{
    MetadataRegistrationResult result;
    if (function == nullptr || module == nullptr || name == nullptr || name[0] == '\0') {
        result.reason = "RUNTIME_HB_FUNCTION_REGISTER_FAILED";
        return result;
    }

    auto module_info = FindModuleInfo(module);
    if (!module_info) {
        result.reason = "RUNTIME_HB_MODULE_NOT_REGISTERED";
        return result;
    }

    FunctionInfo info;
    auto rec_it = module_info->records.find(name);
    if (rec_it != module_info->records.end()) {
        info.capability = rec_it->second.capability;
        info.original_param_count = rec_it->second.original_param_count;
    } else {
        info.capability.ptx_available = module_info->ptx_available;
        info.capability.supports_kernel_params = true;
        info.capability.supports_extra = false;
        info.capability.fallback_reason = "ENTRY_NOT_FOUND";
        if (module_info->original_ptx != nullptr) {
            auto param_count = CountParamsForKernel(*module_info->original_ptx, name);
            if (param_count) info.original_param_count = *param_count;
        }
    }

    info.kernel_name = name;
    info.original_module = module;
    info.original_function = function;

    if (info.capability.transform_succeeded && module_info->transformed_module != nullptr) {
        CUfunction transformed = nullptr;
        CUresult hidden_ret = Driver::ModuleGetFunction(&transformed, module_info->transformed_module, name);
        if (hidden_ret == CUDA_SUCCESS && transformed != nullptr) {
            info.transformed_function = transformed;
        } else {
            info.capability.splittable = false;
            info.capability.fallback_reason = "TRANSFORMED_FUNCTION_NOT_FOUND";
        }
    } else if (info.capability.transform_succeeded) {
        info.capability.splittable = false;
        info.capability.fallback_reason = "TRANSFORMED_MODULE_UNAVAILABLE";
    }

    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_functions[function] = std::move(info);
    }
    result.ok = true;
    result.reason = "OK";
    result.ptx_bytes = module_info->ptx_bytes;
    result.record_count = module_info->records.size();
    return result;
}

CUmodule RemoveModuleInfo(CUmodule module)
{
    CUmodule transformed_module = nullptr;
    std::lock_guard<std::mutex> lock(g_mu);
    auto mod_it = g_modules.find(module);
    if (mod_it != g_modules.end()) {
        transformed_module = mod_it->second.transformed_module;
        g_modules.erase(mod_it);
    }
    for (auto it = g_functions.begin(); it != g_functions.end();) {
        if (it->second.original_module == module) {
            it = g_functions.erase(it);
        } else {
            ++it;
        }
    }
    return transformed_module;
}

uint64_t Volume(Grid3D grid)
{
    return (uint64_t)grid.x * (uint64_t)grid.y * (uint64_t)grid.z;
}

void DecomposeBox(uint32_t x0, uint32_t y0, uint32_t z0,
                  uint32_t x1, uint32_t y1, uint32_t z1,
                  uint64_t max_blocks, std::vector<SplitSpec> *out)
{
    const uint32_t dx = x1 - x0;
    const uint32_t dy = y1 - y0;
    const uint32_t dz = z1 - z0;
    const uint64_t vol = (uint64_t)dx * (uint64_t)dy * (uint64_t)dz;
    if (vol == 0) return;
    if (vol <= max_blocks) {
        out->push_back(SplitSpec{Grid3D{dx, dy, dz}, {x0, y0, z0}});
        return;
    }

    std::array<uint32_t, 3> lens{dx, dy, dz};
    int axis = 0;
    if (lens[1] >= lens[0] && lens[1] >= lens[2]) axis = 1;
    else if (lens[2] >= lens[0] && lens[2] >= lens[1]) axis = 2;

    const uint32_t len = lens[(size_t)axis];
    uint64_t denom = 1;
    if (axis == 0) denom = (uint64_t)dy * (uint64_t)dz;
    else if (axis == 1) denom = (uint64_t)dx * (uint64_t)dz;
    else denom = (uint64_t)dx * (uint64_t)dy;

    uint32_t left_len = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(len, max_blocks / std::max<uint64_t>(1, denom)));
    if (left_len >= len) left_len = std::max<uint32_t>(1, len / 2);

    if (axis == 0) {
        DecomposeBox(x0, y0, z0, x0 + left_len, y1, z1, max_blocks, out);
        DecomposeBox(x0 + left_len, y0, z0, x1, y1, z1, max_blocks, out);
    } else if (axis == 1) {
        DecomposeBox(x0, y0, z0, x1, y0 + left_len, z1, max_blocks, out);
        DecomposeBox(x0, y0 + left_len, z0, x1, y1, z1, max_blocks, out);
    } else {
        DecomposeBox(x0, y0, z0, x1, y1, z0 + left_len, max_blocks, out);
        DecomposeBox(x0, y0, z0 + left_len, x1, y1, z1, max_blocks, out);
    }
}

std::vector<SplitSpec> DecomposeGrid(Grid3D grid, int split_blocks)
{
    std::vector<SplitSpec> splits;
    if (grid.x == 0 || grid.y == 0 || grid.z == 0 || split_blocks <= 0) return splits;
    DecomposeBox(0, 0, 0, grid.x, grid.y, grid.z, (uint64_t)split_blocks, &splits);
    return splits;
}

bool CapabilitySupportsGrid(const FunctionInfo &info, Grid3D grid, std::string *reason)
{
    if (grid.x > 1 && !info.capability.supports_offset_x) {
        if (reason != nullptr) *reason = "OFFSET_X_UNSUPPORTED";
        return false;
    }
    if (grid.y > 1 && !info.capability.supports_offset_y) {
        if (reason != nullptr) *reason = "OFFSET_Y_UNSUPPORTED";
        return false;
    }
    if (grid.z > 1 && !info.capability.supports_offset_z) {
        if (reason != nullptr) *reason = "OFFSET_Z_UNSUPPORTED";
        return false;
    }
    return true;
}

void SetLpSplitThresholdOnce(std::shared_ptr<preempt::XQueue> xqueue)
{
    if (xqueue == nullptr) return;
    const XQueueHandle handle = xqueue->GetHandle();
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_threshold_adjusted_xqueues.find(handle) != g_threshold_adjusted_xqueues.end()) return;
        g_threshold_adjusted_xqueues.insert(handle);
    }
    xqueue->SetLaunchConfig(1, 1);
    if (HbLogEnabled()) {
        XINFO("[UXSCHED-HB] xqueue=0x" FMT_64X " lp_in_flight_threshold=1 batch_size=1", handle);
    }
}

bool SubmitSplitCommands(const FunctionInfo &info, Grid3D grid, Grid3D block,
                         unsigned int shared_mem_bytes, CUstream stream, void **kernel_params,
                         std::shared_ptr<preempt::XQueue> xqueue, CUresult *result)
{
    std::string grid_reason;
    if (!CapabilitySupportsGrid(info, grid, &grid_reason)) {
        if (StrictMode()) {
            if (result != nullptr) *result = CUDA_ERROR_NOT_SUPPORTED;
            LogFallback(grid_reason.c_str(), &info);
            return true;
        }
        LogFallback(grid_reason.c_str(), &info);
        return false;
    }

    std::vector<SplitSpec> splits = DecomposeGrid(grid, SplitBlocks());
    if (splits.empty()) {
        if (result != nullptr) *result = CUDA_ERROR_INVALID_VALUE;
        return true;
    }
    if (splits.size() <= 1) {
        LogFallback("GRID_SMALLER_THAN_SPLIT_BLOCKS", &info);
        return false;
    }

    const size_t transformed_param_count = cuXtraGetParamCount(info.transformed_function);
    if (transformed_param_count != info.original_param_count + kOffsetParamCount) {
        if (result != nullptr) *result = CUDA_ERROR_INVALID_VALUE;
        if (HbLogEnabled()) {
            XWARN("[UXSCHED-HB] fallback=NATIVE reason=PARAM_COUNT_MISMATCH function=%s "
                  "expected=%zu actual=%zu",
                  info.kernel_name.c_str(), info.original_param_count + kOffsetParamCount,
                  transformed_param_count);
        }
        return StrictMode();
    }

    auto group = std::make_shared<SplitCommandGroup>();
    group->kernel_name = info.kernel_name;
    group->split_count = splits.size();
    group->children.reserve(splits.size());

    SetLpSplitThresholdOnce(xqueue);

    for (size_t child_idx = 0; child_idx < splits.size(); ++child_idx) {
        const SplitSpec &split = splits[child_idx];
        std::vector<void *> params;
        params.reserve(info.original_param_count + kOffsetParamCount);
        for (size_t i = 0; i < info.original_param_count; ++i) params.push_back(kernel_params[i]);
        std::array<uint32_t, kOffsetParamCount> offsets = split.offset;
        params.push_back(&offsets[0]);
        params.push_back(&offsets[1]);
        params.push_back(&offsets[2]);

        auto cmd = std::make_shared<CudaKernelLaunchCommand>(
            info.transformed_function, split.grid.x, split.grid.y, split.grid.z,
            block.x, block.y, block.z, shared_mem_bytes, params.data(), nullptr, true);
        cmd->AddStateListener([group, child_idx](preempt::XCommandState state) {
            if (state < preempt::kCommandStateCompleted) return;
            if (HbLogEnabled()) {
                XINFO("[UXSCHED-HB] child_launch_completed function=%s child_index=%zu "
                      "split_count=%zu",
                      group->kernel_name.c_str(), child_idx, group->split_count);
            }
            const size_t done = group->completed.fetch_add(1) + 1;
            if (done == group->split_count) {
                if (HbLogEnabled()) {
                    XINFO("[UXSCHED-HB] split_group_completed function=%s split_count=%zu",
                          group->kernel_name.c_str(), group->split_count);
                    XINFO("[UXSCHED-HB] parent_launch_completed function=%s split_count=%zu",
                          group->kernel_name.c_str(), group->split_count);
                }
                group->children.clear();
            }
        });
        group->children.push_back(cmd);
    }

    if (HbLogEnabled()) {
        XINFO("[UXSCHED-HB] function=%s priority=" FMT_64D " capability=splittable "
              "original_grid=(%u,%u,%u) split_blocks=%d split_count=%zu "
              "backend_selected=HB_SPLIT stream=%p",
              info.kernel_name.c_str(), CurrentPriority(), grid.x, grid.y, grid.z,
              SplitBlocks(), splits.size(), stream);
        XINFO("[UXSCHED-HB] parent_launch_submitted function=%s split_count=%zu "
              "transformed_function=%p xqueue=0x" FMT_64X " stream=%p",
              info.kernel_name.c_str(), splits.size(), info.transformed_function,
              xqueue == nullptr ? 0 : xqueue->GetHandle(), stream);
    }

    for (size_t child_idx = 0; child_idx < group->children.size(); ++child_idx) {
        const SplitSpec &split = splits[child_idx];
        if (HbLogEnabled()) {
            XINFO("[UXSCHED-HB] child_launch_submitted function=%s child_index=%zu "
                  "split_count=%zu transformed_function=%p grid=(%u,%u,%u) "
                  "offset=(%u,%u,%u) xqueue=0x" FMT_64X " stream=%p",
                  info.kernel_name.c_str(), child_idx, group->split_count,
                  info.transformed_function, split.grid.x, split.grid.y, split.grid.z,
                  split.offset[0], split.offset[1], split.offset[2],
                  xqueue == nullptr ? 0 : xqueue->GetHandle(), stream);
        }
        xqueue->Submit(group->children[child_idx]);
    }
    if (result != nullptr) *result = CUDA_SUCCESS;
    return true;
}

} // namespace

CUresult XModuleLoad(CUmodule *module, const char *fname)
{
    LogConfigOnce();
    if (!RuntimeEnabled() || !IsLowPriority() || !Lv1Compatible() || fname == nullptr) {
        return Driver::ModuleLoad(module, fname);
    }

    std::ifstream in(fname, std::ios::binary);
    if (!in) return Driver::ModuleLoad(module, fname);
    in.seekg(0, std::ios::end);
    const std::streamoff end = in.tellg();
    if (end <= 0 || end > (std::streamoff)kImageScanLimit) return Driver::ModuleLoad(module, fname);
    in.seekg(0, std::ios::beg);
    std::vector<char> raw((size_t)end);
    if (!in.read(raw.data(), (std::streamsize)raw.size())) return Driver::ModuleLoad(module, fname);
    if (!LooksLikePtxText(raw.data(), raw.size())) return Driver::ModuleLoad(module, fname);

    const CUresult ret = Driver::ModuleLoad(module, fname);
    if (ret != CUDA_SUCCESS || module == nullptr || *module == nullptr) return ret;

    auto ptx = std::make_shared<const std::string>(raw.data(), raw.data() + raw.size());
    RegisterModuleInfo(*module, TransformModulePtx(ptx));
    return ret;
}

CUresult XModuleLoadData(CUmodule *module, const void *image)
{
    LogConfigOnce();
    if (!RuntimeEnabled() || !IsLowPriority() || !Lv1Compatible() || image == nullptr ||
        !LooksLikePtxText(image, kImageScanLimit)) {
        return Driver::ModuleLoadData(module, image);
    }

    const size_t nbytes = strnlen(static_cast<const char *>(image), kImageScanLimit);
    const CUresult ret = Driver::ModuleLoadData(module, image);
    if (ret != CUDA_SUCCESS || module == nullptr || *module == nullptr || nbytes == 0) return ret;

    auto ptx = std::make_shared<const std::string>(static_cast<const char *>(image), nbytes);
    RegisterModuleInfo(*module, TransformModulePtx(ptx));
    return ret;
}

CUresult XModuleLoadDataEx(CUmodule *module, const void *image, unsigned int num_options,
                           CUjit_option *options, void **option_values)
{
    LogConfigOnce();
    if (!RuntimeEnabled() || !IsLowPriority() || !Lv1Compatible() || image == nullptr ||
        !LooksLikePtxText(image, kImageScanLimit)) {
        return Driver::ModuleLoadDataEx(module, image, num_options, options, option_values);
    }

    const size_t nbytes = strnlen(static_cast<const char *>(image), kImageScanLimit);
    const CUresult ret = Driver::ModuleLoadDataEx(module, image, num_options, options, option_values);
    if (ret != CUDA_SUCCESS || module == nullptr || *module == nullptr || nbytes == 0) return ret;

    auto ptx = std::make_shared<const std::string>(static_cast<const char *>(image), nbytes);
    RegisterModuleInfo(*module, TransformModulePtx(ptx));
    return ret;
}

CUresult XModuleUnload(CUmodule module)
{
    CUmodule transformed_module = RemoveModuleInfo(module);

    if (transformed_module != nullptr) {
        preempt::XQueueManager::ForEachWaitAll();
        CUresult hidden_ret = Driver::ModuleUnload(transformed_module);
        if (hidden_ret != CUDA_SUCCESS && StrictMode()) return hidden_ret;
    }
    return Driver::ModuleUnload(module);
}

CUresult XModuleGetFunction(CUfunction *function, CUmodule module, const char *name)
{
    LogConfigOnce();
    CUresult ret = Driver::ModuleGetFunction(function, module, name);
    if (ret != CUDA_SUCCESS || function == nullptr || *function == nullptr || name == nullptr) return ret;

    auto module_info = FindModuleInfo(module);
    if (!module_info) return ret;

    RegisterFunctionInfo(*function, module, name);
    return ret;
}

MetadataRegistrationResult RegisterModuleMetadata(CUmodule module, const void *ptx,
                                                  size_t ptx_size)
{
    LogConfigOnce();
    MetadataRegistrationResult result;
    if (module == nullptr || ptx == nullptr || ptx_size == 0) {
        result.reason = "RUNTIME_HB_MODULE_REGISTER_FAILED";
        return result;
    }
    if (!LooksLikePtxText(ptx, ptx_size)) {
        result.reason = "RUNTIME_HB_PTX_LIFETIME_INVALID";
        return result;
    }

    auto ptx_owner = std::make_shared<const std::string>(static_cast<const char *>(ptx), ptx_size);
    result = RegisterModuleInfo(module, TransformModulePtx(ptx_owner));
    return result;
}

MetadataRegistrationResult RegisterFunctionMetadata(CUfunction function, CUmodule module,
                                                    const char *name)
{
    LogConfigOnce();
    return RegisterFunctionInfo(function, module, name);
}

void UnregisterModuleMetadata(CUmodule module)
{
    CUmodule transformed_module = RemoveModuleInfo(module);
    if (transformed_module != nullptr) {
        preempt::XQueueManager::ForEachWaitAll();
        (void)Driver::ModuleUnload(transformed_module);
    }
}

bool LookupFunctionMetadata(CUfunction function, std::string *kernel_name,
                            std::string *fallback_reason)
{
    auto info = FindFunctionInfo(function);
    if (!info) return false;
    if (kernel_name != nullptr) *kernel_name = info->kernel_name;
    if (fallback_reason != nullptr) *fallback_reason = info->capability.fallback_reason;
    return true;
}

bool TryLaunchKernelFixed(CUfunction function,
                          unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
                          unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
                          unsigned int shared_mem_bytes, CUstream stream, void **kernel_params,
                          void **extra, std::shared_ptr<preempt::XQueue> xqueue, CUresult *result)
{
    LogConfigOnce();
    if (result != nullptr) *result = CUDA_SUCCESS;
    if (!BuildEnabled()) return false;

    if (!Lv1Compatible()) {
        LogFallback("LV2_LV3_UNSUPPORTED_WITH_HB_SPLIT");
        if (StrictMode()) {
            if (result != nullptr) *result = CUDA_ERROR_NOT_SUPPORTED;
            return true;
        }
        return false;
    }

    if (!IsLowPriority()) {
        if (HbLogEnabled()) {
            XINFO("[UXSCHED-HB] backend_selected=NATIVE reason=HIGH_PRIORITY_PASSTHROUGH "
                  "priority=" FMT_64D " function=%p", CurrentPriority(), function);
        }
        return false;
    }

    const auto info_opt = FindFunctionInfo(function);
    if (!info_opt) {
        LogFallback("PTX_UNAVAILABLE");
        if (StrictMode()) {
            if (result != nullptr) *result = CUDA_ERROR_NOT_SUPPORTED;
            return true;
        }
        return false;
    }
    FunctionInfo info = *info_opt;

    if (xqueue == nullptr) {
        LogFallback("NO_XQUEUE", &info);
        if (StrictMode()) {
            if (result != nullptr) *result = CUDA_ERROR_NOT_SUPPORTED;
            return true;
        }
        return false;
    }
    if (!info.capability.splittable || info.transformed_function == nullptr) {
        LogFallback(info.capability.fallback_reason.empty() ? "NOT_SPLITTABLE"
                                                            : info.capability.fallback_reason.c_str(), &info);
        if (StrictMode()) {
            if (result != nullptr) *result = CUDA_ERROR_NOT_SUPPORTED;
            return true;
        }
        return false;
    }
    if (extra != nullptr) {
        LogFallback("EXTRA_LAUNCH_FORMAT_UNSUPPORTED", &info);
        if (StrictMode()) {
            if (result != nullptr) *result = CUDA_ERROR_NOT_SUPPORTED;
            return true;
        }
        return false;
    }
    if (kernel_params == nullptr) {
        LogFallback("KERNEL_PARAMS_NULL", &info);
        if (StrictMode()) {
            if (result != nullptr) *result = CUDA_ERROR_INVALID_VALUE;
            return true;
        }
        return false;
    }

    const Grid3D grid{grid_dim_x, grid_dim_y, grid_dim_z};
    const Grid3D block{block_dim_x, block_dim_y, block_dim_z};
    if (Volume(grid) <= (uint64_t)SplitBlocks()) {
        LogFallback("GRID_SMALLER_THAN_SPLIT_BLOCKS", &info);
        return false;
    }

    return SubmitSplitCommands(info, grid, block, shared_mem_bytes, stream, kernel_params, xqueue, result);
}

bool TryLaunchKernel(CUfunction function,
                     unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
                     unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
                     unsigned int shared_mem_bytes, CUstream stream, void **kernel_params,
                     void **extra, std::shared_ptr<preempt::XQueue> xqueue, CUresult *result)
{
    const BackendMode mode = BackendModeFromEnv();
    if (mode == BackendMode::kNative) return false;
    return TryLaunchKernelFixed(function,
                                grid_dim_x, grid_dim_y, grid_dim_z,
                                block_dim_x, block_dim_y, block_dim_z,
                                shared_mem_bytes, stream, kernel_params, extra,
                                xqueue, result);
}

} // namespace xsched::cuda::hb_split
