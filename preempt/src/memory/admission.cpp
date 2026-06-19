#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <strings.h>
#include <unordered_map>
#include <vector>

#include "xsched/utils/log.h"
#include "xsched/preempt/memory/admission.h"

using namespace xsched::preempt;

namespace
{

constexpr const char *kEnableEnv = "XSCHED_XPU_MEM_OVERSUB";
constexpr const char *kCudaCompatEnableEnv = "XSCHED_CUDA_MEM_OVERSUB";
constexpr const char *kHighWatermarkEnv = "XSCHED_XPU_MEM_HIGH_WATERMARK";
constexpr const char *kLowWatermarkEnv = "XSCHED_XPU_MEM_LOW_WATERMARK";
constexpr const char *kCudaHighWatermarkEnv = "XSCHED_CUDA_MEM_HIGH_WATERMARK";
constexpr const char *kCudaLowWatermarkEnv = "XSCHED_CUDA_MEM_LOW_WATERMARK";
constexpr const char *kPrefetchLimitMbEnv = "XSCHED_XPU_MEM_PREFETCH_LIMIT_MB";
constexpr const char *kEvictLimitMbEnv = "XSCHED_XPU_MEM_EVICT_LIMIT_MB";

struct MemoryRegion
{
    XpuMemoryBackend *backend = nullptr;
    uintptr_t ptr = 0;
    size_t size = 0;
    uintptr_t context = 0;
    int64_t device = 0;
    XQueueHandle owner = 0;
    bool resident = true;
    uint64_t last_touch = 0;
};

struct QueueState
{
    XpuMemoryBackend *backend = nullptr;
    uintptr_t context = 0;
    int64_t device = 0;
    Priority priority = PRIORITY_DEFAULT;
    bool active = false;
    uint64_t last_active = 0;
};

std::mutex g_mtx;
std::unordered_map<uintptr_t, MemoryRegion> g_regions;
std::unordered_map<XQueueHandle, QueueState> g_queues;
uint64_t g_clock = 0;

bool EnvTruthy(const char *name)
{
    const char *env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 &&
           strcasecmp(env, "off") != 0 && strcasecmp(env, "false") != 0;
}

bool GloballyEnabled()
{
    return EnvTruthy(kEnableEnv) || EnvTruthy(kCudaCompatEnableEnv);
}

int64_t EnvInt(const char *name, int64_t fallback)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return fallback;
    char *end = nullptr;
    long val = std::strtol(env, &end, 10);
    if (end == env || *end != '\0') return fallback;
    return val;
}

int64_t EnvPercent(const char *name, const char *compat_name, int64_t fallback)
{
    int64_t val = EnvInt(name, EnvInt(compat_name, fallback));
    if (val <= 0 || val >= 100) return fallback;
    return val;
}

size_t HighWatermark(size_t total)
{
    return total * (size_t)EnvPercent(kHighWatermarkEnv, kCudaHighWatermarkEnv, 85) / 100;
}

size_t LowWatermark(size_t total)
{
    int64_t high = EnvPercent(kHighWatermarkEnv, kCudaHighWatermarkEnv, 85);
    int64_t low = EnvPercent(kLowWatermarkEnv, kCudaLowWatermarkEnv, 75);
    if (low >= high) low = std::max<int64_t>(1, high - 10);
    return total * (size_t)low / 100;
}

size_t MbLimit(const char *name, size_t fallback_mb)
{
    int64_t mb = EnvInt(name, (int64_t)fallback_mb);
    if (mb <= 0) return 0;
    return (size_t)mb << 20;
}

bool UsableBackend(XpuMemoryBackend *backend)
{
    return backend != nullptr && GloballyEnabled() && backend->Enabled();
}

void AdoptContextRegions(XpuMemoryBackend *backend, uintptr_t context,
                         int64_t device, XQueueHandle queue)
{
    if (queue == 0) return;
    for (auto &entry : g_regions) {
        MemoryRegion &region = entry.second;
        if (region.backend == backend && region.context == context &&
            region.device == device && region.owner == 0) {
            region.owner = queue;
            region.last_touch = ++g_clock;
        }
    }
}

size_t TrackedResidentBytes(XpuMemoryBackend *backend, uintptr_t context, int64_t device)
{
    size_t bytes = 0;
    for (const auto &entry : g_regions) {
        const MemoryRegion &region = entry.second;
        if (region.backend == backend && region.context == context &&
            region.device == device && region.resident) {
            bytes += region.size;
        }
    }
    return bytes;
}

size_t QueueResidentBytes(XpuMemoryBackend *backend, uintptr_t context,
                          int64_t device, XQueueHandle queue)
{
    size_t bytes = 0;
    for (const auto &entry : g_regions) {
        const MemoryRegion &region = entry.second;
        if (region.backend == backend && region.context == context &&
            region.device == device && region.owner == queue && region.resident) {
            bytes += region.size;
        }
    }
    return bytes;
}

size_t QueueHotBytes(XpuMemoryBackend *backend, uintptr_t context,
                     int64_t device, XQueueHandle queue)
{
    size_t bytes = 0;
    for (const auto &entry : g_regions) {
        const MemoryRegion &region = entry.second;
        if (region.backend == backend && region.context == context &&
            region.device == device && region.owner == queue) {
            bytes += region.size;
        }
    }
    return bytes;
}

uint64_t QueueWeight(const QueueState &queue)
{
    int64_t normalized = (int64_t)queue.priority - PRIORITY_MIN + 1;
    return (uint64_t)std::max<int64_t>(1, normalized);
}

size_t QueueBudget(XpuMemoryBackend *backend, uintptr_t context, int64_t device,
                   XQueueHandle queue, size_t total)
{
    size_t usable = LowWatermark(total);
    uint64_t sum_weight = 0;
    uint64_t this_weight = 1;
    size_t active_count = 0;

    for (const auto &entry : g_queues) {
        const QueueState &state = entry.second;
        if (!state.active || state.backend != backend || state.context != context ||
            state.device != device) {
            continue;
        }
        uint64_t weight = QueueWeight(state);
        sum_weight += weight;
        active_count++;
        if (entry.first == queue) this_weight = weight;
    }

    if (active_count == 0 || sum_weight == 0) return usable;

    size_t budget = usable * this_weight / sum_weight;
    size_t hot = QueueHotBytes(backend, context, device, queue);
    if (hot != 0) budget = std::min(budget, hot);
    return budget;
}

} // namespace

void MemoryAdmissionManager::RegisterRegion(XpuMemoryBackend *backend, uintptr_t ptr, size_t size,
                                             uintptr_t context, int64_t device, XQueueHandle owner)
{
    if (!UsableBackend(backend) || ptr == 0 || size == 0 || context == 0) return;

    std::lock_guard<std::mutex> lock(g_mtx);
    g_regions[ptr] = MemoryRegion {
        .backend = backend,
        .ptr = ptr,
        .size = size,
        .context = context,
        .device = device,
        .owner = owner,
        .resident = true,
        .last_touch = ++g_clock,
    };
    XINFO("SchedUM[%s]: track region 0x" FMT_64X ", size=%zu, queue=0x" FMT_64X,
          backend->Name(), (uint64_t)ptr, size, owner);
}

void MemoryAdmissionManager::UnregisterRegion(XpuMemoryBackend *backend, uintptr_t ptr)
{
    if (!UsableBackend(backend) || ptr == 0) return;

    std::lock_guard<std::mutex> lock(g_mtx);
    auto it = g_regions.find(ptr);
    if (it == g_regions.end() || it->second.backend != backend) return;
    XINFO("SchedUM[%s]: untrack region 0x" FMT_64X ", size=%zu",
          backend->Name(), (uint64_t)ptr, it->second.size);
    g_regions.erase(it);
}

void MemoryAdmissionManager::TouchRegion(XpuMemoryBackend *backend, uintptr_t context,
                                          int64_t device, uintptr_t ptr, XQueueHandle owner)
{
    if (!UsableBackend(backend) || ptr == 0 || owner == 0) return;

    std::lock_guard<std::mutex> lock(g_mtx);
    for (auto &entry : g_regions) {
        MemoryRegion &region = entry.second;
        if (region.backend != backend || region.context != context || region.device != device) {
            continue;
        }
        if (ptr < region.ptr || ptr >= region.ptr + region.size) continue;
        region.owner = owner;
        region.last_touch = ++g_clock;
        return;
    }
}

void MemoryAdmissionManager::SetQueuePriority(XQueueHandle queue, Priority priority)
{
    if (queue == 0) return;

    std::lock_guard<std::mutex> lock(g_mtx);
    QueueState &state = g_queues[queue];
    state.priority = std::min(std::max(priority, PRIORITY_MIN), PRIORITY_MAX);
}

void MemoryAdmissionManager::RemoveQueue(XQueueHandle queue)
{
    if (queue == 0) return;

    std::lock_guard<std::mutex> lock(g_mtx);
    g_queues.erase(queue);
    for (auto &entry : g_regions) {
        MemoryRegion &region = entry.second;
        if (region.owner == queue) region.owner = 0;
    }
}

void MemoryAdmissionManager::OnQueueSuspend(XpuMemoryBackend *backend, uintptr_t context,
                                            int64_t device, XQueueHandle queue)
{
    if (!UsableBackend(backend) || context == 0) return;
    MemoryBackendCaps caps = backend->Caps();
    if (!caps.query_mem_info || !caps.evict_to_host) return;

    std::lock_guard<std::mutex> lock(g_mtx);
    QueueState &state = g_queues[queue];
    state.backend = backend;
    state.context = context;
    state.device = device;
    state.active = false;

    AdoptContextRegions(backend, context, device, queue);

    size_t free = 0;
    size_t total = 0;
    if (!backend->QueryMemory(context, device, free, total) || total == 0) return;

    size_t driver_used = total - free;
    size_t tracked_used = TrackedResidentBytes(backend, context, device);
    size_t used = std::max(driver_used, tracked_used);
    size_t high = HighWatermark(total);
    if (used <= high) return;

    size_t low = LowWatermark(total);
    size_t need = used > low ? used - low : 0;
    size_t evict_limit = MbLimit(kEvictLimitMbEnv, 4096);
    if (evict_limit != 0) need = std::min(need, evict_limit);
    if (need == 0) return;

    std::vector<MemoryRegion *> victims;
    victims.reserve(g_regions.size());
    for (auto &entry : g_regions) {
        MemoryRegion &region = entry.second;
        if (region.backend == backend && region.context == context &&
            region.device == device && region.resident) {
            victims.push_back(&region);
        }
    }

    std::sort(victims.begin(), victims.end(), [](const MemoryRegion *a, const MemoryRegion *b) {
        auto aq = g_queues.find(a->owner);
        auto bq = g_queues.find(b->owner);
        bool a_active = aq != g_queues.end() && aq->second.active;
        bool b_active = bq != g_queues.end() && bq->second.active;
        if (a_active != b_active) return !a_active;
        Priority ap = aq == g_queues.end() ? PRIORITY_DEFAULT : aq->second.priority;
        Priority bp = bq == g_queues.end() ? PRIORITY_DEFAULT : bq->second.priority;
        if (ap != bp) return ap < bp;
        if (a->last_touch != b->last_touch) return a->last_touch < b->last_touch;
        return a->size > b->size;
    });

    size_t evicted = 0;
    for (MemoryRegion *region : victims) {
        if (!backend->EvictToHost(context, device, region->ptr, region->size)) continue;
        region->resident = false;
        region->last_touch = ++g_clock;
        evicted += region->size;
        if (evicted >= need) break;
    }

    if (evicted == 0) return;
    if (!backend->Synchronize(context, device)) return;
    XINFO("SchedUM[%s]: pressure eviction requested=%zu, evicted=%zu, "
          "driver_used=%zu, tracked_used=%zu, total=%zu",
          backend->Name(), need, evicted, driver_used, tracked_used, total);
}

void MemoryAdmissionManager::BeforeQueueResume(XpuMemoryBackend *backend, uintptr_t context,
                                               int64_t device, XQueueHandle queue)
{
    if (!UsableBackend(backend) || context == 0 || queue == 0) return;
    MemoryBackendCaps caps = backend->Caps();
    if (!caps.query_mem_info || !caps.async_prefetch) return;

    std::lock_guard<std::mutex> lock(g_mtx);
    QueueState &state = g_queues[queue];
    state.backend = backend;
    state.context = context;
    state.device = device;
    state.active = true;
    state.last_active = ++g_clock;

    AdoptContextRegions(backend, context, device, queue);

    size_t free = 0;
    size_t total = 0;
    if (!backend->QueryMemory(context, device, free, total) || total == 0) return;

    size_t budget = QueueBudget(backend, context, device, queue, total);
    size_t resident = QueueResidentBytes(backend, context, device, queue);
    if (resident >= budget) return;

    size_t need = budget - resident;
    size_t prefetch_limit = MbLimit(kPrefetchLimitMbEnv, 1024);
    if (prefetch_limit != 0) need = std::min(need, prefetch_limit);
    if (need == 0) return;

    std::vector<MemoryRegion *> candidates;
    candidates.reserve(g_regions.size());
    for (auto &entry : g_regions) {
        MemoryRegion &region = entry.second;
        if (region.backend == backend && region.context == context &&
            region.device == device && region.owner == queue && !region.resident) {
            candidates.push_back(&region);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const MemoryRegion *a,
                                                       const MemoryRegion *b) {
        if (a->last_touch != b->last_touch) return a->last_touch > b->last_touch;
        return a->size > b->size;
    });

    size_t prefetched = 0;
    for (MemoryRegion *region : candidates) {
        if (!backend->PrefetchToDevice(context, device, region->ptr, region->size)) continue;
        region->resident = true;
        region->last_touch = ++g_clock;
        prefetched += region->size;
        if (prefetched >= need) break;
    }

    if (prefetched == 0) return;
    if (!backend->Synchronize(context, device)) return;
    XINFO("SchedUM[%s]: queue 0x" FMT_64X " prefetched=%zu, budget=%zu, resident_before=%zu",
          backend->Name(), queue, prefetched, budget, resident);
}
