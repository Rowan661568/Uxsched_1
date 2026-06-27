# SchedUM: Scheduling-aware Unified Memory Admission

## 1. 为什么 XSched 需要补上内存管理

XSched 原本解决的是一个很核心的问题：当多个 AI 任务共享同一块 GPU、NPU 或其他 XPU 时，谁应该先运行，谁应该被暂停，谁应该被恢复。

这对计算调度是有效的。例如一个低优先级训练任务 B 正在跑，一个高优先级实时推理任务 A 到达，XSched 可以通过用户态 API 拦截和 XQueue 控制，让 B 暂停，让 A 尽快恢复执行。

但是这里有一个隐藏问题：**计算资源被让出来了，不代表显存也让出来了。**

如果 B 在被暂停前已经占用了大量显存，而 A 恰好需要申请或访问新的模型参数、KV cache、输入输出 tensor，那么 A 仍然可能被 B 的内存占用拖住。表现出来就是：

```text
低优先级任务 B 被暂停了
高优先级任务 A 被恢复了
但 A 访问数据时仍然遇到显存压力、page fault、迁移抖动甚至 OOM
```

这说明只做计算调度还不够。调度器知道“未来谁要运行”，但内存系统仍然不知道这个未来。SchedUM 要补上的就是这部分能力：

```text
XSched 决定谁运行
SchedUM 根据这个调度结果提前做显存准入
```

所以 SchedUM 不是要替代 CUDA Unified Memory、Ascend ACL 内存接口或底层驱动。它做的是一个用户态、调度感知的内存管理层：当任务被暂停或恢复时，根据显存压力、任务优先级和当前活跃队列集合，决定哪些数据可以让出 GPU，哪些数据应该提前迁入 GPU。

## 2. 内存超额分配到底是什么

内存超额分配指的是：程序需要访问的数据总量超过了设备物理显存容量。

例如 RTX4060 Laptop GPU 只有约 8GB 显存，但程序用 CUDA Unified Memory 申请了 10GB managed memory：

```cpp
cudaMallocManaged(&p, 10GB);
```

这不一定立刻失败。Unified Memory 会把这 10GB 数据切成很多 managed page。每个 page 在某一时刻可能驻留在 GPU 显存，也可能驻留在 CPU 内存。GPU kernel 访问某个不在显存里的 page 时，会触发 GPU page fault，驱动再把这个 page 迁回 GPU。

这个机制让程序可以“看起来”使用超过显存容量的数据。但如果多个任务混跑，问题会变得严重：

```text
B: 低优先级训练，占用 6GB 热数据
A: 高优先级推理，需要 2GB 热数据
GPU: 只有 8GB
```

如果不做调度感知的内存管理，A 恢复运行后可能一边执行，一边被动触发 page fault，同时驱动还要临时决定把谁的 page 挤出去。这样高优先级任务虽然被调度器放行了，但仍然会被低优先级任务残留的显存工作集影响。

SchedUM 的目标不是让 10GB 数据同时塞进 8GB 显存，这是不可能的。它的目标是：

```text
让即将运行的高优先级任务拥有足够的启动工作集；
让暂停或低优先级任务主动让出一部分冷数据；
把不可避免的迁移成本放在更合理的调度时机。
```

## 3. 为什么不完整复现 DeepUM

DeepUM 的核心思路是通过神经网络预测 page 未来是否会被访问，从而提前决定哪些 page 应该迁入 GPU，哪些 page 可以迁出。这个方向很有启发，但完整复现并不适合当前 XSched 的赛题目标。

原因有三点。

第一，DeepUM 的重点是 page-level 访问预测，而 XSched 的优势是调度器已经知道“谁未来会运行”。我们不一定需要训练一个预测器来猜未来，因为调度器本身已经提供了一个强信号：

```text
即将 Resume 的 XQueue 大概率马上会访问它的热数据
刚被 Suspend 的 XQueue 短期内不应该继续占据大量显存
```

第二，神经网络预测器实现成本高，训练数据、特征采集、模型部署都会引入额外复杂度。赛题强调用户态 API 拦截、调度策略和低干扰实现，过重的 predictor 反而会模糊主线。

第三，调度场景中最需要解决的是跨任务竞争，而不是单个任务内部的最优 page 预测。对于多个 AI 进程共享 GPU，最关键的问题通常是：

```text
高优先级任务来了，低优先级任务要不要让显存？
多个高优先级任务同时来了，每个任务应该拿多少显存预算？
```

因此我们参考 DeepUM 的“按需迁移、冷热数据、超额分配处理”思想，但实现成更贴合 XSched 的方法：

```text
SchedUM = Scheduling-aware Memory Admission
```

它不是预测所有 page 的未来访问，而是利用调度器已知的执行未来，对显存做准入、预算和部分迁移。

## 4. SchedUM 的整体故事

可以把 SchedUM 理解成给 XSched 增加了一个“运行前的显存门禁”。

原本的执行链是：

```text
任务提交 kernel
XSched 拦截 API
调度器决定 XQueue Suspend / Resume
硬件队列继续发射计算
```

加入 SchedUM 后，执行链变成：

```text
任务提交 kernel
XSched 拦截 API
调度器决定 XQueue Suspend / Resume
        |
        v
SchedUM 在 Suspend 后观察显存压力，必要时迁出冷数据
SchedUM 在 Resume 前给即将运行的队列做显存预算和预迁入
        |
        v
硬件队列继续发射计算
```

这里有两个关键时机。

第一个时机是 `Suspend` 之后：

```text
低优先级 XQueue 被暂停
SchedUM 判断当前显存压力是否超过 high watermark
如果压力过高，就从低优先级、非活跃、较冷的 region 中迁出一部分
迁出目标是降到 low watermark 附近
```

第二个时机是 `Resume` 之前：

```text
高优先级 XQueue 即将恢复运行
SchedUM 把它加入 active set
根据当前 active set 计算每个队列的 memory budget
只把该队列 budget 内的热 region prefetch 回 GPU
然后 XSched 真正 Resume 队列
```

这样做的好处是：任务不是恢复后才被动碰到缺页和显存竞争，而是在恢复之前就获得一个合理的启动工作集。

## 5. 高优先级并发发射时怎么处理

这个问题是 SchedUM 里最重要的部分。

如果多个高优先级任务同时到达，不能让每个任务都无限制 prefetch 自己的全部数据。否则会出现显存抖动：

```text
A1 Resume -> A1 把自己的数据大量迁入 GPU
A2 Resume -> A2 又把 A1 刚迁入的数据挤出去
A3 Resume -> A3 再次制造迁移压力
```

SchedUM 采用的是全局预算思路。

它维护一个 active queue set。每个即将恢复或正在运行的 XQueue 都会进入这个集合。然后 SchedUM 根据所有 active queue 的权重统一分配显存预算：

```text
usable_gpu_mem = total_mem * low_watermark
queue_weight = f(priority)
budget_i = usable_gpu_mem * weight_i / sum(weight)
```

当前实现中，权重主要来自 XQueue priority。priority 越高，预算权重越大。这样多个高优先级任务并发时，不会出现第一个恢复的任务把显存吃满的情况。

举一个具体例子。假设设备是 RTX4060 8GB：

```text
低优先级训练 B: 已经占用 6GB managed working set
高优先级推理 A1/A2/A3: 同时到达
每个推理任务总数据 2GB
但马上要用的热数据约 1GB
```

不好的做法是：

```text
把 B 迁出很多
A1/A2/A3 各自尝试全量 prefetch
显存反复挤出和迁入
高优先级任务之间互相干扰
```

SchedUM 的做法是：

```text
保留系统余量
把 A1/A2/A3 放入 active set
根据优先级统一计算预算
每个任务只迁入 budget 内的热 region
B 的冷 region 只按实际缺口迁出
```

这并不意味着 A1/A2/A3 “只迁入数据但不能运行”。真实流程是：

```text
XSched 决定 A1/A2/A3 可以运行
SchedUM 在运行前迁入一部分热数据
XSched Resume A1/A2/A3
kernel 正常发射执行
如果后续访问到未提前迁入的 managed page，再由 CUDA UVM 按需迁移
```

也就是说，budget 限制的是“提前迁入多少”，不是限制任务能不能运行。

## 6. 为了低开销做了哪些设计

SchedUM 不能把普通内存充足场景变慢。因此实现时重点做了快速路径和懒触发。

第一，默认关闭。

CUDA 路径需要显式设置：

```bash
XSCHED_CUDA_MEM_OVERSUB=1
```

通用层也支持：

```bash
XSCHED_XPU_MEM_OVERSUB=1
```

未开启时，hook 函数会快速返回，不做内存管理。

第二，没有可管理 region 时不做迁移。

只有拦截到 managed allocation 或平台 allocation 后，SchedUM 才会跟踪 region。没有 region 时，即使 Suspend/Resume 触发 hook，也不会进入重逻辑。

第三，使用 watermark 控制迁移时机。

默认高低水位线为：

```text
high watermark = 85%
low watermark  = 75%
```

只有显存使用超过 high watermark，Suspend 时才会触发 eviction。迁出目标不是越多越好，而是尽量回落到 low watermark 附近。

第四，迁移是部分迁移。

SchedUM 不会无脑迁出整个任务，也不会无脑迁入整个任务。它会按缺口决定迁多少：

```text
need = current_used - low_watermark
```

然后按 victim 排序迁出足够的 region。

第五，迁移有限额。

通用层提供环境变量限制单次迁移规模：

```bash
XSCHED_XPU_MEM_EVICT_LIMIT_MB=4096
XSCHED_XPU_MEM_PREFETCH_LIMIT_MB=1024
```

这样可以避免一次抢占路径里做太多迁移动作。

第六，平台能力可查询。

不是所有 XPU 都有 CUDA Unified Memory 那样的 page migration 能力。因此抽象层里有 `MemoryBackendCaps`：

```cpp
struct MemoryBackendCaps {
    bool query_mem_info;
    bool unified_memory;
    bool async_prefetch;
    bool evict_to_host;
};
```

策略层只表达“应该迁入/迁出”，具体能不能做由 backend 决定。

## 7. 代码结构

### 7.1 通用层

通用层位于：

```text
preempt/include/xsched/preempt/memory/admission.h
preempt/src/memory/admission.cpp
```

它定义了两个核心对象。

第一个是平台后端接口：

```cpp
class XpuMemoryBackend {
public:
    virtual const char *Name() const = 0;
    virtual bool Enabled() const = 0;
    virtual MemoryBackendCaps Caps() const = 0;

    virtual bool QueryMemory(...) = 0;
    virtual bool EvictToHost(...) = 0;
    virtual bool PrefetchToDevice(...) = 0;
    virtual bool Synchronize(...) = 0;
};
```

第二个是通用准入管理器：

```cpp
class MemoryAdmissionManager {
public:
    static void RegisterRegion(...);
    static void UnregisterRegion(...);
    static void TouchRegion(...);

    static void SetQueuePriority(...);
    static void RemoveQueue(...);

    static void OnQueueSuspend(...);
    static void BeforeQueueResume(...);
};
```

通用层内部维护：

```text
MemoryRegion:
  ptr
  size
  context
  device
  owner XQueue
  resident
  last_touch

QueueState:
  XQueueHandle
  priority
  active
  context
  device
  last_active
```

这样它可以在 XPU 无关的层面完成：

```text
谁是 active queue
每个 queue 有多少热数据
每个 queue 当前 resident bytes
哪个 region 更适合被迁出
哪个 region 更适合被迁入
```

### 7.2 XQueue hook

为了让内存管理知道调度事件，在 `HwQueue` 中增加了两个 hook：

```cpp
virtual void OnXQueueSuspend() {}
virtual void BeforeXQueueResume() {}
```

位置：

```text
preempt/include/xsched/preempt/hal/hw_queue.h
```

调用位置在：

```text
preempt/src/xqueue/async_xqueue.cpp
```

Suspend 路径：

```cpp
launch_worker_->Pause();
kHwQueue->OnXQueueSuspend();
```

Resume 路径：

```cpp
kHwQueue->BeforeXQueueResume();
```

这两个位置很关键。`OnXQueueSuspend()` 发生在队列暂停后，适合迁出被暂停任务的冷数据；`BeforeXQueueResume()` 发生在队列恢复前，适合给即将运行的任务做显存准入。

### 7.3 Priority 接入

为了让内存预算和调度优先级一致，在：

```text
preempt/src/sched/hint.cpp
```

中把 `XHintPriority()` 同步给 SchedUM：

```cpp
MemoryAdmissionManager::SetQueuePriority(xq, prio);
```

XQueue 删除时，在：

```text
preempt/src/xqueue/xqueue.cpp
```

清理内存管理层状态：

```cpp
MemoryAdmissionManager::RemoveQueue(xq_h);
```

### 7.4 CUDA backend

CUDA backend 位于：

```text
platforms/cuda/hal/include/xsched/cuda/hal/common/memory_manager.h
platforms/cuda/hal/src/common/memory_manager.cpp
```

它实现了真实的 managed memory 支持：

```text
cuMemAllocManaged -> RegisterManagedAllocation
cuMemFree/cuMemFree_v2 -> UnregisterAllocation
Suspend -> OnQueueSuspend
Resume -> BeforeQueueResume
```

CUDA backend 的能力声明为：

```cpp
query_mem_info = true
unified_memory = true
async_prefetch = true
evict_to_host = true
```

显存查询使用：

```cpp
cuMemGetInfo_v2
```

迁出到 CPU 使用：

```cpp
cuMemPrefetchAsync_v2_ptsz(..., CU_MEM_LOCATION_TYPE_HOST, ...)
```

迁入 GPU 使用：

```cpp
cuMemPrefetchAsync_v2_ptsz(..., CU_MEM_LOCATION_TYPE_DEVICE, ...)
```

对应的队列 hook 在：

```text
platforms/cuda/hal/src/level1/cuda_queue.cpp
```

### 7.5 CUDA shim

CUDA shim 拦截了 managed memory 的分配和释放：

```text
platforms/cuda/shim/include/xsched/cuda/shim/shim.h
platforms/cuda/shim/src/shim.cpp
platforms/cuda/shim/src/intercept.cpp
```

具体拦截：

```text
cuMemAllocManaged
cuMemFree_v2
cuMemFree
```

这样 SchedUM 能知道进程中有哪些 managed region、大小是多少、属于哪个 CUDA context 和 device。

### 7.6 Ascend backend

Ascend backend 位于：

```text
platforms/ascend/hal/include/xsched/ascend/hal/memory_manager.h
platforms/ascend/hal/src/memory_manager.cpp
```

Ascend 接入了同一套抽象，但当前能力边界和 CUDA 不一样。

Ascend backend 的能力声明为：

```cpp
query_mem_info = true
unified_memory = false
async_prefetch = false
evict_to_host = false
```

它当前支持：

```text
aclrtGetMemInfo(ACL_HBM_MEM, ...)
```

来查询 HBM 空间。

Ascend shim 目前拦截：

```text
aclrtMalloc
aclrtMallocAlign32
aclrtMallocCached
aclrtFree
```

位置：

```text
platforms/ascend/shim/include/xsched/ascend/shim/shim.h
platforms/ascend/shim/src/shim.cpp
platforms/ascend/shim/src/intercept.cpp
```

这使得 SchedUM 可以在 Ascend 上跟踪 device allocation 和显存压力。但由于当前 ACL 接口没有 CUDA Unified Memory 那种通用 page prefetch/evict 能力，所以 Ascend 这版是“机制接入 + allocation tracking + pressure query”，不是完整的页级交换。

这也正好体现了抽象层的意义：

```text
通用策略层可以复用
平台 backend 根据硬件和 runtime 能力选择实现深度
```

## 8. 当前实现的执行流程

### 8.1 CUDA managed allocation

当应用调用：

```cpp
cuMemAllocManaged(&ptr, size, flags);
```

shim 会转到：

```cpp
XMemAllocManaged(...)
```

成功后查询当前 CUDA context 和 device：

```cpp
cuCtxGetCurrent(&ctx)
cuCtxGetDevice(&dev)
```

然后注册到 SchedUM：

```cpp
CudaMemoryManager::RegisterManagedAllocation(ptr, size, ctx, dev)
```

最终进入通用层：

```cpp
MemoryAdmissionManager::RegisterRegion(...)
```

### 8.2 XQueue Suspend

当 XSched 决定暂停某个队列时：

```text
AsyncXQueue::Suspend
  -> launch_worker_->Pause()
  -> HwQueue::OnXQueueSuspend()
  -> CudaMemoryManager::OnQueueSuspend(...)
  -> MemoryAdmissionManager::OnQueueSuspend(...)
```

通用层做这些事：

```text
1. 把该 queue 标记为 inactive
2. 查询当前显存 free/total
3. 计算 driver_used 和 tracked_used
4. used = max(driver_used, tracked_used)
5. 如果 used <= high watermark，直接返回
6. 如果 used > high watermark，计算 need = used - low watermark
7. 从候选 region 中选择 victim
8. 调用 backend->EvictToHost(...)
9. backend->Synchronize(...)
```

victim 排序优先级大致是：

```text
非 active queue 的 region 优先
低 priority queue 的 region 优先
last_touch 更老的 region 优先
size 更大的 region 优先
```

### 8.3 XQueue Resume

当 XSched 决定恢复某个队列时：

```text
AsyncXQueue::Resume
  -> HwQueue::BeforeXQueueResume()
  -> CudaMemoryManager::BeforeQueueResume(...)
  -> MemoryAdmissionManager::BeforeQueueResume(...)
```

通用层做这些事：

```text
1. 把该 queue 标记为 active
2. 查询当前显存 free/total
3. 根据 active set 计算该 queue 的 budget
4. 如果该 queue resident bytes 已经超过 budget，直接返回
5. 否则从该 queue 的 nonresident region 里选 hot region
6. 调用 backend->PrefetchToDevice(...)
7. backend->Synchronize(...)
```

这就是“高优先级任务运行前的显存准入”。

## 9. RTX4060 实测数据

测试脚本：

```text
tools/schedum_oversub.py
```

运行环境：

```text
GPU: RTX4060 Laptop GPU
显存: 约 8GB
workload: CUDA Driver API + managed memory + 手写 PTX kernel
```

运行命令：

```bash
LD_LIBRARY_PATH=/home/cyk/miniconda3/lib:$LD_LIBRARY_PATH \
  /usr/bin/python3 tools/schedum_oversub.py
```

测试构造了一个超过显存容量的 managed memory workload：

```text
B working set: 7 x 1GiB managed regions
A pressure set: 3 x 1GiB managed regions
总 managed allocation: 10GiB
GPU physical memory: 约 8GiB
```

脚本会用 GPU kernel 触碰每个 4KB page，确保这些 allocation 不是只分配地址空间，而是真的被 GPU 访问过。

### 9.1 第一次 Suspend

B 触碰 7GiB managed working set 后，触发第一次 Suspend。

日志：

```text
SchedUM[cuda]: pressure eviction requested=1077280768,
               evicted=2147483648,
               driver_used=1141374976,
               tracked_used=7516192768,
               total=8585216000
```

解释：

```text
total ≈ 8.00GiB
tracked_used ≈ 7.00GiB
high watermark = 85%
low watermark = 75%
```

根据 low watermark，SchedUM 计算大约需要释放 1GiB。但 region 粒度是 1GiB，并且 victim 排序后一次迁出完整 region，所以实际迁出 2GiB。

这体现了当前策略的特点：

```text
不是全量迁出 B
而是根据压力只迁出足够的部分
```

### 9.2 第二次 Suspend

随后又分配并触碰 A 的 3GiB pressure set，总 managed working set 达到 10GiB。再次 Suspend 时日志为：

```text
SchedUM[cuda]: pressure eviction requested=2151022592,
               evicted=3221225472,
               driver_used=1141374976,
               tracked_used=8589934592,
               total=8585216000
```

解释：

```text
tracked_used ≈ 8.00GiB
需要释放 ≈ 2GiB
实际迁出 3GiB
```

同样，因为 region 粒度是 1GiB，最终迁出量略大于 requested。这是可接受的：它避免了过细粒度管理带来的高开销，也符合当前实现“region-level partial eviction”的定位。

### 9.3 WSL 环境限制

在当前 RTX4060/WSL 环境中，CUDA managed prefetch 返回：

```text
CUDA_ERROR_INVALID_DEVICE = 101
```

日志中可以看到：

```text
host prefetch unsupported ... using logical eviction fallback
failed to prefetch region ..., dst=0, error=101
```

这说明这套环境下 CUDA UVM 的显式 prefetch 到 host/device 受限。SchedUM 对此做了 fallback：当 host prefetch 不支持时，仍然进行 logical eviction 标记，以便验证调度感知的 tracking、pressure detection、budget decision 和 victim selection。

因此当前 RTX4060 实测能证明：

```text
managed allocation tracking 生效
显存压力判断生效
partial eviction 决策生效
全局抽象层没有破坏 CUDA 路径
```

但真实物理 page migration 的性能收益，需要在支持 managed prefetch 的 native Linux CUDA 环境上进一步验证。

## 10. 编译验证

CUDA/RTX4060 路径：

```bash
cmake -S . -B build
cmake --build build --target preempt halrtx4060 shimrtx4060
```

结果：

```text
Built target preempt
Built target halrtx4060
Built target shimrtx4060
```

Ascend 路径单独配置：

```bash
cmake -S . -B build_ascend -DPLATFORM_ASCEND=ON
cmake --build build_ascend --target preempt halascend shimascend
```

结果：

```text
Built target preempt
Built target halascend
Built target shimascend
```

## 11. 当前能力和边界

已经实现的能力：

```text
1. 通用 XPU memory admission 抽象层
2. XQueue Suspend/Resume hook
3. priority hint 接入 memory budget
4. CUDA managed memory allocation tracking
5. CUDA pressure-driven partial eviction
6. CUDA budget-aware prefetch 入口
7. Ascend allocation tracking
8. Ascend HBM memory pressure query
9. 平台能力 caps，支持不同 XPU backend 逐步扩展
10. RTX4060 oversubscription workload 验证
```

当前边界：

```text
1. CUDA 真实 prefetch 在当前 WSL/RTX4060 环境受限，返回 error=101
2. Ascend 当前没有实现真实 page migration，只做 tracking 和 pressure query
3. managed region owner 当前主要通过 context/queue adopt，kernel 参数级 hotness 还没有完全细化
4. active set 是在 Resume/Suspend hook 中维护，还没有和全局调度器的批量决策深度融合
5. 当前迁移粒度是 allocation region，不是 4KB page
```

这些边界并不影响机制故事。相反，它们说明 SchedUM 采用了一个可扩展架构：

```text
先把调度感知内存准入机制立起来
CUDA backend 先验证 managed memory 路径
Ascend backend 先验证 XPU 抽象接入
后续根据平台能力逐步增强真实迁移能力
```

## 12. 后续可以继续增强的点

第一，kernel 参数扫描。

CUDA kernel launch 时可以扫描 kernel 参数，如果参数指针落在某个 managed region 内，就把这个 region 绑定到对应 XQueue，并更新 `last_touch`。

这样 hotness 会更准确：

```text
不是“这个 context 下的 region 可能属于这个 queue”
而是“这个 queue 最近发射的 kernel 确实用到了这个 region”
```

第二，接入调度器批量 active set。

当前 active set 是通过单个 queue Resume/Suspend 维护的。后续可以让 policy 在做出调度决策后，把“一批即将运行的队列”直接告诉 SchedUM。

这样高优先级并发预算会更准确：

```text
SchedPolicy 先决定 A1/A2/A3 一起运行
SchedUM 一次性给 A1/A2/A3 分预算
然后再 Resume
```

第三，环境自适应能力检测。

CUDA backend 可以在初始化时探测：

```text
cudaDevAttrConcurrentManagedAccess
cuMemPrefetchAsync host/device 是否可用
```

如果发现平台不支持 prefetch，就自动降级到 logical tracking 或 admission-only 模式。

第四，Ascend coarse-grained swap。

如果 Ascend 没有统一内存 page migration，可以考虑用更粗粒度方式做 swap：

```text
拦截 aclrtMalloc
必要时为低优先级 buffer 分配 host shadow
Suspend 时 D2H copy
Resume 时 H2D copy
```

这不是 page-level UVM，但可以作为 Ascend backend 的显式交换能力。

## 13. 一句话总结

SchedUM 的核心不是“预测每个 page 的未来”，而是利用 XSched 已经知道的调度未来：

```text
谁即将运行，谁应该拿显存预算；
谁已经暂停，谁可以让出冷数据。
```

因此它和 XSched 的关系非常自然：

```text
XSched 管计算准入
SchedUM 管内存准入
```

在内存充足时，SchedUM 通过快速路径尽量不干扰抢占延迟；在内存超额分配时，SchedUM 通过 priority-aware、pressure-driven、budget-aware 的方式，把内存迁移成本放到更合理的调度时机，并降低低优先级任务对高优先级任务的显存干扰。
