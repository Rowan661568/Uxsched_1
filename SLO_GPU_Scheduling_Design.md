# SLO-Constrained Preemptive Scheduling for AI Accelerator Resource Sharing

---

## 1. 背景与问题定义

### 1.1 多进程共享加速器的工程困境

AI 大模型推理、深度学习训练等任务的普及，使 GPU、NPU 等加速器卡成为计算系统的核心资源。实际部署中，多个 AI 进程共享单块或多块加速器卡是常态：一个在线推理服务与一个离线训练任务共用同一张 GPU，或一个实时渲染进程与一个模型微调进程共享同一个 NPU。

这种共享面临一个根本矛盾。加速器硬件的设计目标是大规模并行计算吞吐，而非多进程隔离。传统加速器驱动提供的调度机制非常原始——要么进程独占整张卡，要么所有进程在驱动层面以 FIFO 或简单的时间片轮转争夺硬件资源。灵活的优先级支持、可配置的抢占策略、量化服务质量约束，这些在 CPU 调度器中早已成熟的机制，在加速器调度栈中几乎是空白。

这意味着，一个高优先级的实时推理进程和一个低优先级的批量训练进程共享同一张 GPU 时，系统缺乏有效的机制来确保推理进程的延迟不会因为训练进程的 kernel 发射而被拉长。进程间的资源竞争干扰是全方位的——计算单元争抢、显存带宽争抢、缓存污染——而操作系统的标准进程调度对这些硬件资源的竞争几乎没有可见性。

### 1.2 用户态拦截作为基础架构

本系统的核心设计选择是：**不修改操作系统内核，不修改加速器驱动**。通过在用户态拦截加速器 API 调用，实现对计算启动、资源申请、同步等待、流操作等关键接口的透明接管。

拦截层通过 `LD_PRELOAD` 机制注入，在运行时拦截 CUDA Runtime API、Ascend ACL API 等加速器调用。每一条 `cudaMemcpy`、`cudaLaunchKernel`、`aclrtMemcpy`、`aclrtLaunchKernel` 都经过拦截层处理：调用被截获、上下文被映射到调度器的内部队列（XQueue）、提交被调度器决策调度后才真正下发到驱动。

这套架构的核心优势在于兼容性。不依赖特定驱动版本，不绑定特定操作系统，不需要内核模块。同一套调度逻辑可以同时作用于 CUDA GPU 和昇腾 NPU ——只要各自的用户态 API 可被拦截。

### 1.3 基线策略 HPF 的局限性

系统内置的基线策略是 Highest Priority First（HPF）：每个调度 tick 扫描所有 XQueue，在每个设备上选出优先级最高的任务执行，其余挂起。

HPF 的直觉足够直观——高优先级任务优先使用 GPU。但在实际运行中，这条直觉的成立前提——加速器可以被干净地、即时地抢占——并不成立。

**优先级失效窗口。** GPU（以及当前主流 NPU）的 kernel 一旦发射到硬件，就不能在中间被掐断。一个正在执行的 kernel 运行时，调度器对其所在队列调用 Suspend，但这个 Suspend 信号要等到当前 kernel 执行到某个可中断点（通常是 kernel 边界或特定的同步点）才能生效。这意味着高优先级任务即使被调度器选中，也必须等待正在执行的低优先级 kernel 自然完成。这个窗口的长度取决于低优先级 kernel 的粒度——数百微秒到数毫秒不等。

**Tail Latency 的不稳定性。** 在混合负载下，HPF 的高优先级任务延迟呈现出不规律的模式。由于训练任务的 kernel 长度在不同 iteration 间不同，推理任务每次等待低优先级 kernel 完成的时间也不一样。这种随机性集中体现在 tail latency 上：p50 可能只比独占运行多几毫秒，但 p99 可能跳升数倍。

**低优先级任务的 Backlog 积压。** HPF 的另一面是低优先级任务的命运。只要高优先级任务在提交，低优先级队列就一直被 Suspend。系统将加速器的利用率限制在高优先级任务的空闲窗口内——高优先级任务在 CPU 上做预处理时、或同步等待下一次输入时，低优先级任务才能跑几下。在实验数据中，HPF 策略下 background 训练吞吐量仅为裸跑的 25%，而高优先级任务的延迟并没有得到理想中的完全隔离。

**缺乏可量化的调度目标。** 更深层的问题是 HPF 没有可量化的目标。优先级是一个相对量——两个优先级相差 1 的任务，其调度行为差异没有量化定义。对于有明确服务质量要求的场景（如推理服务 p99 < 100ms），调度器需要的不只是 "谁更重要"，还有 "还能拖多久"。

这些局限并非 HPF 独有，而是传统优先级驱动调度在不可抢占硬件上的系统性问题。解决方案不是寻找更优的优先级排序方式，而是建立一种新的调度范式。

---

## 2. 设计动机

### 2.1 优先级不等于延迟保障

上述实验揭示了一个反直觉的事实：最高优先级的任务在 HPF 下仍然经历了显著的延迟增长。

在多任务共享加速器的场景中，一个请求的端到端延迟由三部分构成：

```
latency = queuing_delay + preemption_wait + execution_time
```

HPF 只影响第一项——它在队列中选择下一个执行的任务。对第二项几乎无能为力：当前正在执行的 kernel 必须跑完，无论调度器怎么喊 Suspend。而恰恰是第二项构成了高优先级任务延迟抖动的主要来源。

这意味着，单纯靠提高优先级来保障延迟是不够的。调度器需要感知正在执行任务的进度，并以此估算高优先级任务需要等多长时间。如果这个等待时间超过可容忍的范围，调度器需要提前干预，而不是等到高优任务到达后再排队。

### 2.2 Kernel Duration 的不确定性

深度学习训练任务的 kernel 长度是不确定的。backward pass 可能比 forward pass 长数倍；混合精度训练中某些算子的 kernel 长度随输入尺寸变化；数据加载延迟导致 GPU 空闲窗口忽长忽短。

高优先级推理任务每次被调度时，遇到的低优先级 kernel 处于不同的执行阶段。这种不确定性在尾延迟上集中体现——我们的实验中，不同配置下 p99 的差异跨幅超过 30ms，直观地说明了这种不可预测性。

解决这个问题需要调度器对加速器上的执行有可见性——需要知道当前在跑什么、还需要多久。不可见的调度策略，无论优先级多精细，都无法消除尾延迟的不确定性。

### 2.3 优先级反转与 Backlog 系统性抖动

HPF 的严格优先级策略在存在依赖链的系统中会引发更严重的连锁问题。考虑一条典型的 AI 推理流水线：推理服务进程依赖一个数据预处理进程，后者又依赖一个存储读取进程。如果将低优先级的存储读取完全挂起，数据预处理进程会因为没有输入而阻塞，推理服务进程最终空转等待——虽然它是最高优先级，却因为依赖链底层被挂起而无法推进。

这是操作系统中经典的优先级反转问题，在 CPU 调度领域已有成熟解法（优先级继承、优先级置顶），但在加速器调度上下文中尚未被系统性地处理。一个健康的调度系统需要保底机制，允许低优先级任务在合理范围内取得进展，以避免依赖链上的系统性死锁。

### 2.4 需要一个可量化的目标

这三个问题指向同一个根源：调度器执行的是相对决策（"A 比 B 优先级高"），而应用需要的是绝对保障（"A 的延迟不超过 70ms"）。

相对优先级的好处是简单。但缺乏量化目标意味着调度器无法回答以下问题：
- 当前这个高优先级任务还能等多久？
- 如果让低优先级任务再跑一会来换取更多吞吐，高优先级任务还来得及跑完吗？
- 系统是否接近 SLO 边界？

需要一个对标量来回答这些问题。这个标量就是 Service Level Objective（SLO）。

不是把优先级扔掉，而是在优先级之上建立一层 SLO 约束，用 deadline 和 slack time 来量化每个任务的紧急程度。调度决策的出发点从 "谁优先级更高" 转变为 "谁更接近违反 SLO"。

---

## 3. 核心思想：SLO-Constrained Scheduling

### 3.1 从 Priority 到 Violation Risk

SLO-Constrained Scheduling 的核心转变是：将调度决策的依据，从优先级替换为 SLO 违规风险（violation risk）。

优先级是一个静态量——两个任务各有一个数字，大的优先。违规风险是一个动态量，它在每个时刻取决于：

- 任务剩余的可用时间（deadline - 已等待时间）
- 系统当前的负载状态（队列深度、就绪任务数）

一个任务即使优先级不高，但如果它的 deadline 即将耗尽，它的违规风险升高，调度器理应让它先跑。反之，一个高优先级任务如果刚刚提交、距 deadline 还远，调度器可以等一小会，让低优先级任务继续推进。

### 3.2 SLO 的表达方式

SLO 在系统中通过 deadline 来表达。一个任务的 deadline 定义为：

```
deadline = ready_time + latency_budget
```

latency_budget 的来源可以是：

1. **显式指定**：用户为实时推理任务设置一个绝对延迟上限（如 50ms）。调度器保证每个 batch 的端到端延迟不超过这个值。
2. **自动推导**：先离线 profiling 测得任务在独占加速器时的 p99 latency 作为 baseline，再乘以一个容忍因子（如 1.25），作为 deadline。
3. **周期性启发**：对于周期提交的任务，可用任务的 inter-arrival time 作为 deadline 的近似值。

一个任务如果未指定 deadline，系统将其归类为 best-effort：不对延迟负责，但也不被高优任务完全饿死。

### 3.3 Slack Time 与三层调度模型

有了 deadline，调度器在每个 tick 计算每个就绪任务的 slack time：

```
slack = deadline - now
slack_remaining = slack - estimated_execution_time
```

slack time 反映了任务的紧急程度。当 slack 很大时，任务还很宽松；当 slack 趋近于零时，这个任务随时可能违反 SLO，需要立即干预。

基于 slack 和优先级，任务被分为三个层级：

1. **Urgent**（deadline 即将耗尽，`slack <= tick_threshold`）：必须立即执行，无视其他所有任务。这是系统对 SLO 违规的最后防线。
2. **Latency-Sensitive**（有 deadline 或 priority > 默认值，但 slack 充足）：按优先级排序，优先级高的先跑。这个层级是常规的延迟保障层。
3. **Batch / Best-Effort**（无 SLO 参数）：使用 CFS 风格的公平调度，按 vruntime 分配加速器时间。这个层级在系统没有 SLO 压力时运行，最大化资源利用率。

三个层级的执行顺序严格嵌套：Urgent 优先于一切，Latency 次之，Batch 在前两者都不活跃时运行。

### 3.4 防止 SLO 饿死的 Aging 机制

在 SLO 调度中，低优先级的 Batch 任务可能面临饥饿——如果 Latency 任务持续提交，Batch 任务得不到加速器时间。

解决方法是引入累积等待时间作为调度的隐式因子。Batch 层使用 vruntime（虚拟运行时间）进行公平调度：

```
vruntime += elapsed_time × (default_weight / task_weight)
```

task_weight 由 utilization hint 决定。利用率越高的任务，权重越大，vruntime 增长越慢，被调度的机会越多。所有 Batch 任务未指定 utilization 时权重一致，退化为轮转。

这个机制确保：即使 Batch 任务在 Latency 活跃时被压制，一旦 Latency 趋于空闲，Batch 任务的 vruntime 更小，会被优先调度。系统不会出现永久性饥饿。

---

## 4. 系统设计与实现

### 4.1 系统架构

整个系统由四个层次构成：

```
Application (CUDA / Ascend workloads)
       │
       ▼
API Interception Layer (LD_PRELOAD / hook)
  - 拦截计算启动、资源申请、同步等待、流操作等关键接口
  - 支持 CUDA Runtime API, Ascend ACL API 等
  - 自动创建 XQueue 映射进程 GPU/NPU context
       │
       ▼
Scheduler Core (xserver + pluggable policy)
  - 接收 Hint 更新任务元数据
  - 每个调度 tick 执行选定策略
  - 输出 Suspend / Resume / AddTimer 指令
       │
       ▼
Preemption & Resource Isolation Layer
  - XQueue 级别的上下文切换
  - 可中断点管理（cooperation-based preemption）
  - 多设备负载均衡
```

每个层次有清晰的职责边界。Interception Layer 只负责截获调用和路由，不参与调度决策。Scheduler Core 运行在独立的服务进程中，与业务进程解耦。Preemption Layer 屏蔽底层加速器的硬件差异，向上提供统一的队列操作接口。

### 4.2 用户态 API 拦截的实现

拦截层需要覆盖加速器编程模型中的四类关键操作：

**计算启动。** `cudaLaunchKernel`、`aclrtLaunchKernel` 等接口是 GPU/NPU 计算任务进入硬件的入口。拦截层在每个发射指令前插入调度检查——如果当前队列未被调度器选中，该发射被悬停或重定向到调度等待队列。

**资源申请。** `cudaMalloc`、`aclrtMalloc` 等内存分配操作触发显存资源的管理。拦截层记录每个 XQueue 的内存使用量，为资源隔离提供依据。

**同步等待。** `cudaStreamSynchronize`、`aclrtSynchronizeStream` 等同步操作是任务完成的关键信号。拦截层通过这些接口观察到任务的实际完成时间，用于更新调度器的状态。

**流操作。** `cudaStreamCreate`、`aclrtCreateStream` 等流管理操作决定了任务的执行序和并发度。拦截层识别流结构，将其映射到 XQueue 层级结构中。

实际实现中，CUDA 平台通过 `LD_PRELOAD` 注入动态库拦截，Ascend ACL API 通过类似的符号劫持机制实现。拦截层本身不做调度决策——它收集信息、转发指令。

### 4.3 SLO 参数的绑定机制

SLO 参数通过 Hint 机制注入到调度器。系统定义了一组 hint 类型：

- `PriorityHint`：设置静态优先级
- `DeadlineHint`：设置 SLO deadline（微秒），这是三层调度模型的核心输入
- `UtilizationHint`：设置利用率权重，影响 Batch 层的 vruntime 分配
- `TimesliceHint`：设置调度 tick 间隔，控制抢占粒度

Hint 的注入有两种途径：
1. **程序化 API**：应用在初始化时调用 `XHintDeadline(xq, ddl_us)`
2. **环境变量**：通过 `XSCHED_AUTO_XQUEUE_DEADLINE` 自动注入，不修改应用代码

### 4.4 运行时状态收集

每个调度 tick 触发时，调度器收到当前系统的 Status 快照，包含：

- 每个 XQueue 的运行状态（ready / blocked，suspended / running，所属设备）
- 每个进程持有的 XQueue 列表
- 各队列的 ready_time（可用于计算等待时间和 slack）

调度器不直接测量 kernel duration——这需要硬件层面的性能计数器支持，不是所有加速器都提供。而是通过 ready_time 隐含地追踪任务的等待历史：一个任务如果 ready_time 很早但还没运行，说明它已经等了很久。这个信息在计算 slack 时已有体现。

### 4.5 调度决策过程

在每个 tick 中，调度器对每个加速器设备独立执行决策：

```
1. Update vruntime for all running tasks
2. For each task, compute:
   - slack = deadline - (now - ready_time)
   - latency_sensitive = (has deadline or prio > default)
   - urgent = (has deadline and slack <= tick)
3. Select task per device:
   a. Urgent exists → pick the one with minimal slack
   b. Latency-sensitive exists → pick highest prio, then minimal slack
   c. Otherwise → pick batch task with minimal vruntime
4. Resume selected task, Suspend all others on the same device
5. Set timer for next tick
```

决策过程的时间复杂度是 O(n)，其中 n 是当前设备上的 XQueue 数量。每个 tick 扫描一次所有队列，做常数时间的计算。调度器本身不会成为系统的瓶颈。

### 4.6 GPU / NPU 软抢占

当前的 GPU 和 NPU 硬件不支持细粒度的指令级抢占。系统通过以下三种机制组合实现可用的软抢占：

**XQueue 抽象与切换。** 每个进程的加速器 context 被映射到独立的 XQueue。调度器在 XQueue 粒度上执行 Suspend/Resume。Suspend 不会立即停止硬件执行，而是标记该队列为"需暂停"。当前运行的 kernel 完成（或到达可中断点）后，系统暂停该队列的后继提交。Resume 恢复队列的提交通路。

**三级抢占粒度控制。** 系统通过 xqueue_level 参数控制抢占的响应速度：Level 1 在 kernel 间切换，适用于延迟敏感场景；Level 2 在命令缓冲（command buffer）边界切换，响应更快但 Suspend/Resume 频率更高；Level 3 在更细粒度上操作。抢占粒度的选择需要在切换开销和延迟响应之间做权衡。

**协同式抢占。** 对于长时间运行的 kernel，可以通过 kernel slicing 或 micro-batching 的方式，在 kernel 内部插入可中断点，将长 kernel 拆分为可抢占的小段。这需要应用层的配合，但可以显著缩小优先级失效窗口。

### 4.7 资源隔离与负载均衡

资源隔离方面，系统记录每个 XQueue 的显存使用量，通过 xqueue threshold 参数控制单次提交的命令数量，防止单个进程过度占用硬件资源。在更细粒度上，系统可以通过限制 XQueue 发射到硬件的并发命令数量，间接控制带宽竞争。

负载均衡方面，当系统管理多块加速器卡时，调度器将进程的 XQueue 分配到负载较轻的设备上。分配依据包括设备当前的队列深度、运行中任务数、以及历史占用率。当前实现中负载均衡是静态的（在进程启动时分配），但在策略层面预留了动态迁移的接口。

空闲资源复用方面，Batch 层的设计直接服务于这个目标：当系统中没有 SLO 压力时（无 Urgent 或 Latency 任务），Batch 任务自然接管整个加速器，填满空闲的计算资源。

### 4.8 冷启动与 SLO 设定流程

对于需要自动推导 SLO 的场景，系统的工作流程是：

1. **Profiling run**：在独占模式下运行 benchmark，采集任务在无干扰时的延迟分布，计算 p99。
2. **SLO 计算**：根据用户指定的容忍因子计算 deadline（如 `deadline = p99 × 1.25 × 1000`，微秒）。
3. **参数注入**：deadline 通过环境变量 `XSCHED_AUTO_XQUEUE_DEADLINE` 注入，在运行时生效。
4. **调度运行**：SLO 策略使用 deadline 驱动三层调度。

---

## 5. 实验设计与效果

### 5.1 Benchmark 设计

实验使用 ResNet50 batch inference 作为高优先级 foreground workload，MobileNetV2 batch training 作为低优先级 background workload。两个进程共享同一台 RTX4060 GPU。Foreground 在延迟敏感场景下承载推理服务负载，Background 在吞吐优先场景下承载训练负载。

四种实验场景构成完整的对比基线：

1. **alone**：foreground 独占 GPU，测量 baseline 延迟。
2. **native**：无调度干预，foreground 和 background 自由竞争 GPU。
3. **xsched**：调度介入，选择 HPF 或 SLO 策略，xqueue level 1。
4. **xsched_lv2**：同策略，更细粒度的抢占（xqueue level 2）。

每个场景收集 foreground 每个 batch 的推理延迟，计算 p50/p95/p99。Background 进程记录退出前的总 iteration 数，折算为吞吐量（iters/s）。

### 5.2 HPF 策略的表现

| 场景 | Foreground p99 | Background 吞吐 |
|------|:-------------:|:--------------:|
| alone | 57.0 ms | — |
| native | 145.8 ms | 11.75 it/s |
| xsched | 69.3 ms | 3.58 it/s |
| xsched_lv2 | 95.3 ms | 3.35 it/s |

HPF 将 foreground 延迟从 145.8ms（native）压低到 69.3ms，代价是 background 吞吐从 11.75 it/s 降至 3.58 it/s——70% 的吞吐被牺牲。

xsched_lv2 的回归值得注意：更细粒度的抢占没有改善延迟，反而从 69.3ms 恶化到 95.3ms。原因在于 Level 2 增加了 Suspend/Resume 的频率，切换开销积累后拖慢了 foreground。HPF 配合细粒度抢占出现"过度调度"的问题——调度器跑得越快，系统性能反而越差。

### 5.3 SLO 策略的表现

SLO 策略的 deadline 设置采用自动 profiling 方式：先运行 alone 场景测得 baseline p99 ≈ 56ms，设容忍因子为 1.25，计算 deadline 约 70ms。

| 场景 | Foreground p99 | Background 吞吐 |
|------|:-------------:|:--------------:|
| alone | 56.9 ms | — |
| native | 145.5 ms | 14.37 it/s |
| xsched | 68.2 ms | 7.21 it/s |
| xsched_lv2 | 70.5 ms | 6.92 it/s |

SLO 在两个指标上都优于 HPF。Foreground p99 为 68.2ms（与 HPF 的 69.3ms 相当），但 background 吞吐达到 7.21 it/s——HPF 的两倍。

更关键的是 SLO 在细粒度抢占下的表现：Foreground p99 从 68.2ms 仅微升至 70.5ms，background 吞吐稳定在 6.92 it/s。与 HPF 在 Level 2 下 95.3ms 的延迟退化形成鲜明对比。SLO 因为有 deadline 兜底，在更高频率的调度 tick 下可以选择不触发切换——知道 deadline 还来得及，就不必打断低优先级任务的执行。

### 5.4 双指标权衡分析

将两种策略在相同场景下对比：

| 策略 | p99 退化（vs alone） | 吞吐保留率（vs native） |
|------|:------------------:|:-------------------:|
| HPF | +12.3 ms（+22%） | 30%（3.58 / 11.75）|
| SLO | +11.3 ms（+20%） | 50%（7.21 / 14.37）|

HPF 用 70% 的吞吐损失换来 22% 的延迟退化控制。SLO 用 50% 的吞吐损失换来 20% 的延迟退化控制。在相同的延迟保护水平下，SLO 保留了更多系统吞吐。

SLO 调度器在 foreground 延迟压力不大时（slack time 充足），不会像 HPF 那样激进取缔低优先级任务。HPF 在每个 tick 评估"是否有更高优先级的任务在运行"，而 SLO 检查"当前的高优先级任务还有多少 slack"。如果 slack 充足，调度器允许当前运行的低优先级任务继续，从而降低了 Suspend/Resume 的频率和切换开销。

### 5.5 干扰控制与利用率

- **Alone 场景**：foreground 独占 GPU，p99 56ms，无进程间干扰，为 SLO 设定提供 baseline。
- **Native 场景**：无调度场景下 foreground 延迟膨胀到 145ms（2.56× baseline），background 虽然吞吐最高（14.37 it/s）但以严重牺牲高优任务为代价，不满足服务等级要求。
- **HPF 场景**：延迟压回到 69ms，但 background 被近乎饿死（3.58 it/s），加速器在高优任务空闲之前几乎完全被独占，资源利用率不高。
- **SLO 场景**：延迟 68ms 与 HPF 持平，吞吐 7.21 it/s 为 HPF 的二倍。SLO 在 SLO 约束允许的范围内主动放松对低优任务的压制，让 batch 任务填补高优任务留下的空闲窗口。

---

## 6. 工程总结

HPF 策略的局限本质上是相对优先级模型的局限：调度器知道哪个任务更重要，但不知道哪个任务更紧急。将高优先级任务排在前面执行并不总能转化为低延迟——尤其在 GPU/NPU 的不可抢占特性使得优先级在 kernel 执行窗口内失效时。HPF 在每个调度 tick 反复 Suspend/Resume 低优先级任务，消耗了可观的上下文切换开销，却未能从系统中提取更多的延迟保障。

SLO-Constrained Scheduling 将调度决策从比较优先级转向量化违规风险。Deadline 和 slack time 提供了调度的参考指标：任务距 deadline 越近，调度器越需要优先处理。三个调度层级（Urgent / Latency-Sensitive / Batch）按 SLO 风险嵌套排列，保证 deadline 濒临超限的任务得到及时处理，同时也允许非延迟敏感的任务在空闲窗口中取得进展。这套机制落地于一个纯用户态的加速器调度系统——通过 API 拦截技术实现无侵入的调度接入，通过 XQueue 抽象屏蔽底层加速器的硬件差异，使同一套策略可以作用于 CUDA GPU 和 Ascend NPU 等不同平台。

实验表明，在 ResNet50 推理 + MobileNetV2 训练的混合负载下，SLO 策略在保护高优先级尾延迟的同时，能保留两倍于 HPF 的低优先级吞吐量。更重要的是，SLO 在面对细粒度抢占配置时表现出更好的稳定性——deadline 兜底使调度器不需要频繁决策，从而避免了 HPF 因过度调度导致的性能退化。

这套系统并不试图在 GPU/NPU 上实现完美的实时可调度性——现有的硬件约束决定了这是一个不切实际的目标。它的设计出发点是：在不可抢占的加速器硬件上，通过 SLO 约束和软抢占机制，为混合负载的调度提供一个可量化的、可调参的权衡框架。
