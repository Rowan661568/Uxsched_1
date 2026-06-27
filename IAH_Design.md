# Interference-Aware Co-Scheduling for Heterogeneous Accelerator Resource Sharing

---

## 1. 背景与问题定义

### 1.1 异构加速器共享中的新维度

SLO-Constrained Scheduling 解决了单加速器场景下优先级到 deadline 的映射问题：当一个高优先级任务和低优先级任务共享同一张 GPU 时，deadline 给了调度器一个量化约束，使其能在 SLO 允许的范围内更聪明地分配时间，不再像 HPF 那样一刀切地压制低优任务。

但单卡场景只是一个维度。**当系统中存在多张异构加速器——GPU 和 NPU 共享同一块 SoC、或两张 GPU 共享同一根 PCIe 总线——时，调度问题出现了新的复杂性。**

在独立的多 GPU 场景中（每张卡有各自的显存和内存控制器），经典 HPF 或 SLO 的 per-device 独立调度已经足够：每张卡各自选优先级最高或最紧急的任务运行，互不干扰，因为两张卡之间的资源是完全独立的。

但在**异构 SoC** 场景中（GPU + NPU 集成在同一芯片上，如 NVIDIA Jetson AGX Orin、部分昇腾模组），情况不同了。GPU 和 NPU 拥有各自独立的计算单元——GPU 跑它的 kernel，NPU 跑它的 kernel，计算上完全不重叠。但它们共享 SoC 级的资源：内存控制器、DRAM 带宽、片上互联总线、以及总功耗预算。当一个高带宽需求的 kernel 在 GPU 上运行时，它可能拉满内存控制器的带宽，导致 NPU 的访存请求被阻塞。反之亦然。

这意味着：跨设备的任务是否应该并行，不只取决于它们的优先级关系，还取决于它们**对共享资源的需求是否冲突**。

### 1.2 HHPF 的跨设备调度

系统现有的 Heterogeneous Highest Priority First（HHPF）策略是专门为这个场景设计的。它的决策逻辑很直接：扫描系统中所有设备上的所有就绪队列，找出**全局**优先级最高的那个任务，让它运行，将其余所有设备上的所有任务全部挂起。

```
HHPF 调度逻辑（简化）：

global_max_prio = max(all_queues, key=priority)
for each queue:
    if queue.priority >= global_max_prio:
        Resume(queue)
    else:
        Suspend(queue)
```

这个策略在形式上是对的——它确保了全局最高优先级的任务无论在哪里都不会被其他设备上的低优任务干扰。在论文的原始实验场景中（Jetson TX2 上的 GPU + NPU 异构），HHPF 将 foreground NPU 推理任务的 p99 延迟从 native 的 1.67× baseline 降至 1.18× baseline，验证了跨设备统一调度的必要性。

### 1.3 HHPF 的隐性代价

但 HHPF 的决策有一个隐含假设：**只要存在跨设备优先级差异，就应该挂起低优先级的设备。** 这个假设在 GPU 和 NPU 的计算单元完全独立的情况下是不准确的。

考虑以下场景：

```
时间轴（HHPF）：

GPU ── 高优推理（卷积，计算密集，内存带宽需求低）███████████
NPU ── 低优训练（矩阵乘，计算密集，内存带宽需求低）░░░░░░░░░  ← Suspend！

实际情况：
  - GPU 运行的卷积 kernel 以 FMA 计算为主，几乎不产生 DRAM 访问
  - NPU 运行的矩阵乘同样以计算为主
  - 两者共享的内存控制器几乎没有争抢
  - NPU 的安全域完全可以继续训练，不会影响 GPU 的推理延迟
```

HHPF 无法区分这种"不冲突"的场景。它不具备对跨设备干扰水平的可见性——不知道当前运行的 kernel 是计算密集还是访存密集，不知道两者的内存带宽需求是否叠加。因此它选择了一个在所有条件下都安全的策略：有高优任务时，低优设备的任务全部停掉。

这个策略的代价是异构系统的利用率损失。NPU 的计算单元是高优任务不需要的时候也不让用——即使它本可以安全地并行——因为调度器不知道它是否安全。

### 1.4 干扰模型的缺失

问题根源在于：**传统优先级调度只建模了"谁优先"的关系，没有建模"谁跟谁争抢什么资源"的关系。**

对于同一张 GPU 上的多个进程，这个缺失不是问题——它们天然共享一切资源，竞争是全方位的，要么让 A 跑、要么让 B 跑。

但对于异构 SoC 上的 GPU 和 NPU，计算单元是不重叠的。竞争只发生在一部分共享资源上（内存带宽、功耗预算），而这些资源的争抢程度取决于当前运行的 kernel 的特征。一个 compute-bound 的 GEMM kernel 和另一个 compute-bound 的矩阵乘 kernel 可以安全并行；两个 memory-bound 的 copy kernel 则不能。

HHPF 没有这个模型。它假设"跨设备竞争总是存在"，因此采用最保守的调度策略。但这个假设在异构场景下过于悲观——多数 AI kernel 是计算密集的，共享总线的压力远小于其理论峰值，跨设备并行在大部分时间内是安全的。

---

## 2. 设计动机

### 2.1 计算与访存的不对称

深度学习 kernel 可以大致分为两类：

**计算密集型（Compute-bound）。** 卷积、矩阵乘法、Transformer 注意力等 kernel，其执行时间由 FMA 单元的算力决定，内存带宽利用率通常不高。这类 kernel 在执行时，SoC 的内存总线往往有大量空闲带宽。一个计算密集的 GPU kernel 和一个计算密集的 NPU kernel 可以在同一时间并行执行，而不会显著影响彼此的完成时间。

**访存密集型（Memory-bound）。** embedding lookup、broadcast、某些逐元素操作（ReLU、add）等 kernel，其执行时间由内存带宽决定，计算单元大部分时间处于等待状态。两个访存密集的 kernel 同时在 GPU 和 NPU 上运行时，会竞争内存控制器的服务能力，导致双方的延迟都显著增加。

HHPF 不区分这两种情况。一个 compute-bound 的高优推理任务和一个 compute-bound 的低优训练任务在不同设备上时，HHPF 仍然选择 Suspend 训练任务——即使没有资源争抢。对系统的吞吐而言，这相当于在计算单元完全独立的情况下浪费了一半的算力。

### 2.2 跨设备干扰的可判断性

跨设备干扰的程度不是随机的，而是由当前正在执行的 kernel 特征决定。这意味着**调度器可以做出判断**——不需要硬件性能计数器，只需要知道每个设备上当前 kernel 的访存特征。

具体来说，一个简单的足够判断准则是：

```
interference_risk = intensity(device_A) + intensity(device_B)
```

其中 `intensity` 是一个 [0, 1] 之间的值，表示设备当前运行的 kernel 的访存密集程度。0 表示纯计算（无 DRAM 访问），1 表示纯访存（计算单元空闲）。两个设备的 intensity 之和如果超过一个阈值，说明它们对内存总线的竞争足够激烈，需要 Suspend 其中之一。否则，它们可以安全并行。

这个表达方式足够简单，可以在每个调度 tick 内完成计算——不需要复杂的模型，不需要硬件计数器，只需要应用在初始化时或运行时提供一个 hint。

### 2.3 从"一刀切"到"按需切换"

基于上述观察，一个自然的改进方向是：在 HHPF 的跨设备优先级框架上，增加一个**干扰判断层**。

当跨设备优先级冲突发生时（高优在设备 A，低优在设备 B），调度器不再直接 Suspend 设备 B，而是先判断两个设备当前 kernel 的干扰风险：

- **高干扰**（两个 intensity 之和 > 阈值）→ 执行 HHPF 逻辑：Suspend 设备 B 的低优任务。
- **低干扰**（两个 intensity 之和 ≤ 阈值）→ 允许并行：让设备 B 继续运行。

这个判断是**对称的**——即使设备 A 上的任务优先级更高，如果它与设备 B 之间的干扰低，设备 B 上的低优任务可以继续执行。唯一的例外是同一设备上的优先级冲突：同一张 GPU 上的高优和低优任务，计算单元共享，必须严格优先级调度。

### 2.4 可退化的设计

这种"先判断干扰，再决定是否 Suspend"的设计还有一个工程上的好处：**可退化性**。

如果应用未提供任何 intensity hint，所有设备的默认 intensity 为 0.5。两个默认设备之和为 1.0，低于默认阈值 1.2——这意味着在无 hint 的情况下，IAH 默认允许跨设备并行，比 HHPF 更激进。

如果应用希望恢复到 HHPF 的行为（绝对不允许跨设备并行），可以将任一设备的 intensity 设为 0.8 以上——0.8 + 0.5 = 1.3 > 1.2，阈值被突破，IAH 退化为 HHPF 行为。

这种设计避免了 HHPF 的另一个问题：**信息缺失时的保守行为。** HHPF 在信息缺失时选择最保守的策略（全部 Suspend），IAH 在信息缺失时选择一个合理的默认值（默认允许并行），只有在明确的信息提示冲突时才保守。

---

## 3. 核心思想：Interference-Aware Co-Scheduling

### 3.1 干扰模型的定义

IAH 的核心是一个量化跨设备干扰的模型。模型的两个输入是各设备当前 kernel 的访存密集度（memory intensity），输出是干扰风险的二元判断。

**访存密集度（Memory Intensity）** 定义为 [0.0, 1.0] 之间的值：

- **0.0**：纯计算密集。kernel 几乎不访问 DRAM，指令执行完全在计算单元内部完成。典型场景：大矩阵乘法、深度卷积、FFT 等。这类 kernel 在执行时，SoC 的内存总线处于相对空闲状态，其他设备可以安全地并发访问内存。
- **0.5**：混合型。kernel 同时包含计算和访存操作，两者都不是瓶颈。这是许多常规 AI kernel 的特征，也是 IAH 的默认值。
- **1.0**：纯访存密集。kernel 的执行时间完全由内存带宽决定，计算单元大部分时间空闲等待数据到达。典型场景：memory copy、embedding lookup、broadcast 操作、逐元素大向量操作。

**干扰风险**由两个设备的 intensity 之和决定：

```
interference_risk = intensity_A + intensity_B
high_interference = interference_risk > kIntensityThreshold
```

其中 `kIntensityThreshold = 1.2`。这个阈值的含义是两个设备的访存密集度都明显高于中等水平时（如 0.7 + 0.7 = 1.4），才禁止并行。双边都低（0.3 + 0.3 = 0.6）或一边低一边高（0.3 + 0.8 = 1.1）时，允许并行。

### 3.2 四类组合决策表

干扰公式对四种组合场景的判定结果如下：

| 场景 | GPU 当前 kernel | NPU 请求的 kernel | intensity 和 | 是否 > 阈值 (1.2) | 决策 |
|------|----------------|-------------------|:-----------:|:----------------:|------|
| ① 双方访存密集 | 访存密集 (0.8) | 访存密集 (0.8) | **1.6** | ✅ 是 | **暂停 NPU** — 争抢内存带宽，拖慢高优任务 |
| ② GPU 访存密集，NPU 计算密集 | 访存密集 (0.8) | 计算密集 (0.3) | **1.1** | ❌ 否 | **让 NPU 跑** — 总线压力小，高优不受影响 |
| ③ 双方计算密集 | 计算密集 (0.3) | 计算密集 (0.3) | **0.6** | ❌ 否 | **让 NPU 跑** — 完全无竞争 |
| ④ GPU 计算密集，NPU 未知 | 计算密集 (0.3) | 默认混合 (0.5) | **0.8** | ❌ 否 | **让 NPU 跑** — 高优任务不依赖带宽 |

这四种情况统一由 `intensity_A + intensity_B > 1.2` 一个公式表达。公式更加紧凑简洁，而表格在展示时更直观。IAH 的代码实现使用公式，但在概念上等价于这个四类决策表。

### 3.3 三层决策结构

在每个调度 tick 中，IAH 对每个设备上的 XQueue 执行以下决策过程：

```
1. 全局扫描 —— 找到全局最高优先级任务 G 及其设备 dev(G)

2. 对每个设备 D，选其本地最高优先级任务 L

3. 对 L 执行决策：
   a. 如果 D == dev(G):
      L 和 G 在同一设备 → 经典 HPF，G 跑（同设备计算单元冲突，必须严格优先级）
   
   b. 如果 priority(L) >= priority(G):
      优先级不低 → L 不受 G 的压制
   
   c. 如果 priority(L) < priority(G) 且 D != dev(G):
      跨设备优先级冲突：
      - intensity(D) + intensity(dev(G)) > 阈值 → 高干扰 → Suspend L
      - 否则 → 低干扰 → Resume L（co-schedule）
```

这个结构保证了：

- **同一设备上**的调度行为与 HPF 完全一致——最高优先级的任务在它所在的设备上享有绝对优先权
- **跨设备低干扰**时，低优任务不受全局最高优先级的压制——填充 SoC 的空闲算力
- **跨设备高干扰**时，行为与 HHPF 完全一致——全局最高优先级的任务优先，低优设备挂起

### 3.4 与 HHPF 和 HPF 的关系

三种策略构成了一个递进序列：

| 维度 | HPF | HHPF | IAH |
|------|-----|------|-----|
| 调度域 | 单设备 | 所有设备 | 所有设备 |
| 跨设备优先级感知 | 不感知 | 全局优先级统一 | 全局优先级统一 |
| 低优设备在冲突时 | — | Suspend | Suspend |
| 低优设备在**不冲突**时 | — | 仍然 Suspend | **允许并行** |
| 需要的信息 | 仅优先级 | 仅优先级 | 优先级 + 干扰模型 |
| 对算力独立但总线共享的系统利用率 | 低（单卡范围） | 更低（一刀切） | 高（智能并行） |

IAH 不是对 HHPF 的推翻，而是对其的**扩展**：在 HHPF 的跨设备统一优先级框架上，增加一个干扰判断层。当干扰高时，IAH = HHPF。当干扰低时，IAH ≈ 每设备独立 HPF（但保留全局优先级感知）。

### 3.5 干扰模型的局限性

这个模型是简化的。它存在以下已知局限：

**不支持多设备叠加分析。** 当前模型只考虑两两设备的干扰。当同时存在 GPU、NPU、DLA 三个设备时，三者的总干扰判断还本归为两两判断。

**没有细分的共享资源建模。** 所有干扰都抽象为 memory intensity 一个维度。实际上可能存在某些 kernel 高内存带宽但不参与共享总线冲突的情况（如通过专用路径访问片内 SRAM）。

**强度来自 hint 而非测量。** 系统不主动测量设备的瞬时带宽消耗。intensity 由应用或初始化时的 hint 指定，可能无法反映 kernel 在微秒级时间内的动态变化。

这些局限可以后续改进，但当前的简化版本已经能覆盖异构 SoC 场景中大部分有价值的并行机会——多数 AI kernel 的访存特征是稳定的，应用开发者知道自己的 kernel 是计算密集还是访存密集。

---

## 4. 系统设计与实现

### 4.1 策略注册与集成

IAH 作为一个独立的 Policy 子类，复用 XSched 现有的完整调度架构。新增的组件如下：

| 组件 | 路径 | 说明 |
|------|------|------|
| 策略头文件 | `sched/include/xsched/sched/policy/iah.h` | 类声明，继承 Policy |
| 策略实现 | `sched/src/policy/iah.cpp` | Sched + RecvHint 实现 |
| Hint 扩展 | `sched/include/xsched/sched/protocol/hint.h` | 新增 `kHintTypeMemoryIntensity = 8` + `MemoryIntensityHint` |
| 类型注册 | `include/xsched/types.h` | `kPolicyInterferenceAwareHeterogeneous = 12` |
| 名字注册 | `protocol/include/xsched/protocol/def.h` | `XSCHED_POLICY_NAME_IAH` |
| 策略名映射 | `protocol/src/names.cpp` | `IAH` → `kPolicyInterferenceAwareHeterogeneous` |
| 工厂注册 | `sched/src/policy/policy.cpp` | CreatePolicy 新增 case |
| HTTP Hint 分发 | `service/server/src/server.cpp` | MemoryIntensityHint 的 REST 接口 |

全部新增代码约 200 行，不修改任何现有策略的逻辑。

### 4.2 Memory Intensity 的设定

intensity 值通过 Hint 机制注入调度器，支持的途径包括：

**程序化 API（运行时设置）：**

```cpp
XHintMemoryIntensity(device, 0.3);  // 设备 0x... 设强度为 0.3（计算密集）
```

**HTTP Hint（运行时动态修改）：**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"hint_type": 8, "device": 0, "intensity": 0.7}' \
  http://localhost:50000/hint
```

intensity 值在策略运行时通过 `RecvHint` 接收并保存在 `device_intensities_` 表中。每个调度 tick 从该表查询当前强度值。如果还未收到任何 hint，默认返回 `kIntensityDefault = 0.5`。

### 4.3 访存密集度的自动检测（Kernel Name Heuristic）

IAH 支持两种 intensity 设定方式：手动 hint 注入和自动检测。

手动 hint 已在上一节描述。自动检测通过在 CUDA shim 层（`platforms/cuda/shim/src/shim.cpp`）拦截 `cuLaunchKernel` 和 `cuLaunchKernelEx` 实现。在每个 kernel launch 时自动执行以下流程：

```
1. 解析 CUfunction 句柄 → 通过 cuFuncGetName 获取 kernel 名称
2. 将 kernel 名称映射到 [0.0, 1.0] 的访存密集度
3. 检查该设备的 intensity 是否与上次发送的不同
4. 如果发生变化，调用 XHintMemoryIntensity() 自动注入 hint
```

kernel 名称到 intensity 的映射采用启发式规则表：

| kernel 名称特征 | 访存密集度 | 示例 |
|----------------|:---------:|------|
| 包含 `conv`, `Conv`, `gemm`, `Gemm`, `fft`, `mma`, `wmma`, `warp` | **0.3**（计算密集） | 卷积、矩阵乘法、FFT |
| 包含 `softmax`, `Softmax`, `norm`, `Norm` | **0.6**（中等） | softmax、layer norm |
| 包含 `relu`, `Relu`, `add_kernel`, `mul_kernel`, `elementwise` | **0.8**（访存密集） | ReLU、逐元素操作 |
| 包含 `copy`, `Copy`, `memcpy`, `broadcast` | **0.8**（访存密集） | 数据搬运、广播 |
| 包含 `embed`, `Embed`, `gather`, `Gather`, `scatter` | **0.9**（高访存密集） | embedding lookup |
| 其他（不匹配上述规则） | **0.5**（默认混合） | — |

这个启发式的依据是：kernel 名称通常反映了其计算模式。卷积和 GEMM 是典型的 compute-bound 操作，计算密度高（FLOPs/byte 比值大）；embedding 和 element-wise 是典型的 memory-bound 操作，计算几乎不消耗，执行时间完全由访存带宽决定。

为了减少不必要的 IPC 通信，每个设备只在上次发送的 intensity 与当前估算值的差异超过 5% 时才发送新的 hint。在典型推理/训练负载中，同一设备的 kernel 类型相对稳定（如 ResNet50 大多数 kernel 是卷积），intensity 更新频率远低于 kernel launch 频率。

这种自动检测不需要修改应用程序代码，用户不需要手动设置 `XHintMemoryIntensity`。IAH 策略在运行时通过 hint 的 IPC 通道自动接收这些强度更新，并用于跨设备干扰判断。

### 4.4 调度决策的代码结构

IAH 的 `Sched()` 实现与 HHPF 在结构上类似，区别在于跨设备分支处增加干扰判断。核心路径如下：

```
Phase 1: 全局扫描找到 global_max_prio 和对应的 global_max_device

Phase 2: 对每个设备，找到本地最高优先级的 handle

Phase 3: 对每个就绪的 XQueue：
  1. 跳过非本地最优队列（同一设备上只有本地最优者可能运行）
  2. 如果它是本地最优：
     a. priority >= global_max_prio → Resume（同设备或同优先级）
     b. 同设备但优先级低 → Suspend（经典 HPF 行为）
     c. 不同设备 → 检查干扰：
        - 高干扰 → Suspend
        - 低干扰 → Resume（co-schedule）
```

复杂度为 O(n)，其中 n 是系统中的 XQueue 总数。与 HPF 和 HHPF 在同一量级。

### 4.5 状态管理

IAH 状态包括两个 unordered_map：

- `priorities_`：`XQueueHandle → Priority`，与 HPF/HHPF 完全相同的机制，通过 `PriorityHint` 设置。
- `device_intensities_`：`XDevice → double`，存储每台设备当前的访存密集度，通过 `MemoryIntensityHint` 设置。

不维护队列运行历史，不跟踪 kernel 长度，不预设设备拓扑。决策完全基于当前 tick 的 `Status` 快照和 hint 提供的信息。

### 4.6 默认行为与退化路径

IAH 的设计强调了在信息缺失时的理性行为：

- **无 MemoryIntensityHint 时**：所有设备 intensity 默认为 0.5，两个设备之和 1.0 < 1.2 → 默认允许跨设备并行。此时代码的行为等价于「跨设备 HPF 但不限制低优设备」——比 HHPF 更激进，对所有设备上的所有就绪队列按优先级排序，但不同设备上的低优任务不会被挂起。

- **强度设为极端值时**：某设备设 intensity = 1.0，其余设备保持 0.5 → 1.0 + 0.5 = 1.5 > 1.2 → 禁止与它跨设备并行。此时 IAH 的行为退化为 HHPF。

- **kIntensityThreshold 可调**：如果将阈值设为 0（任何跨设备情况都被视为高干扰），IAH 退化为 HHPF。如果将阈值设为 2.0（从不认为有任何干扰），IAH 退化为 per-device HPF 但保留跨设备全局优先级排序。

这种可退化性意味着 IAH 不是一个注定比 HHPF 更好的策略——它是一个参数空间更大的策略，在 HHPF（阈值 = 0）和 per-device HPF（阈值 = +∞）之间提供了一个连续可调的权衡。`kIntensityThreshold = 1.2` 是这个权衡中的一个合理默认值。

---

## 5. 实验场景设计

### 5.1 验证目标

IAH 的验证需要在异构 SoC 平台（如 Jetson AGX Orin、昇腾模组）上进行。测试目标是验证以下两个命题：

1. **高干扰场景下 IAH 的保护能力不弱于 HHPF** — 当跨设备内存争抢严重时，IAH 应当 Suspend 低优设备任务，保证高优任务延迟不受到显著影响。
2. **低干扰场景下 IAH 的吞吐优于 HHPF** — 当跨设备内存争抢不严重时，IAH 应当允许低优设备任务继续执行，提升 SoC 的整体利用率。

### 5.2 Benchmark 设计建议

**Case 1：双方 compute-bound**

在 GPU 上运行高优先级的计算密集型推理任务（大 batch 卷积网络），在 NPU 上同时运行低优先级的计算密集型训练任务（大矩阵乘法）。两者的 intensity 都低（约 0.3 左右），IAH 应允许并行。

对比指标：
- GPU 高优任务 p99 延迟（vs alone baseline）
- NPU 低优任务吞吐量

预期：IAH 的延迟保护 ≥ HHPF，吞吐显著高于 HHPF。

**Case 2：双方 memory-bound**

在 GPU 上运行高优先级的内存密集型操作（大向量 broadcast + element-wise），在 NPU 上运行低优先级的内存密集型操作（embedding lookup）。两者的 intensity 都高（约 0.8 以上），IAH 应禁止并行。

对比指标：
- GPU 高优任务 p99 延迟
- NPU 低优任务吞吐

预期：IAH 的延迟保护 = HHPF，吞吐 = HHPF（因为两者行为相同）。

**Case 3：混合不对称**

GPU 运行高优先级的内存密集型操作（intensity 高），NPU 运行低优先级的计算密集型操作（intensity 低）。IAH 应允许 NPU 继续运行。

对比指标：
- GPU 高优任务 p99 延迟
- NPU 低优任务吞吐

预期：IAH 的延迟保护轻微劣于 HHPF（因为总线上的低负载操作不会显著争抢），但 NPU 吞吐显著提升。

### 5.3 硬件平台要求

IAH 的效果高度依赖于 SoC 的内存子系统架构。适用 IAH 的典型平台特征：

- GPU 和 NPU 集成在同一芯片或同一封装内
- 两者共享统一的内存控制器（如 LPDDR5 控制器）
- 两者各自拥有独立计算单元（非时分复用）
- 不存在 NVLink/片间互联的点对点高带宽连接

不适用 IAH 的平台包括：
- 两张独立 PCIe GPU（各自独立显存和内存控制器）
- GPU 和 CPU 之间的标准 PCIe 连接（单边带宽消耗，不构成对称争抢）

---

## 6. 工程总结

HHPF 策略首次将加速器调度的视野从单设备扩展到整个异构 SoC，验证了跨设备统一优先级调度对 tail latency 的价值。但它引入了一个新的代价：在系统吞吐上的保守行为——只要有高优先级任务在任意设备上运行，其他所有设备上的低优任务都被挂起，即使跨设备的资源竞争可以忽略。

Interference-Aware Co-Scheduling（IAH）在这个框架上增加了一个工程上的修正：不是简单地"存在高优就全停"，而是先判断跨设备干扰的程度，再决定是否允许并行。判断模型是轻量级的——每个设备维护一个 memory intensity 因子，通过两个因子的和是否超过阈值来估算干扰风险。这个模型不依赖硬件性能计数器，不需要修改驱动程序，所有信息通过 hint 机制注入，复用了 XSched 现有的架构。

IAH 的工程哲学是：**在不确定时，选择更积极的默认值（允许并行），只有明确的信息表明冲突时才保守。** 这与 HHPF 的"保守是安全"的哲学形成对比。两种哲学在各自的设计空间内都是合理的——但异构 SoC 的场景中，多数 AI kernel 是计算密集的，积极默认值更适合这种负载特征。

IAH 的代码量很小——约 200 行新增代码，不修改现有策略。它可以在 HHPF（阈值设为 0）和 per-device 独立调度（阈值设为 +∞）之间连续调节。与 SLO 策略（延时感知的串行抢占）互补：SLO 回答的是"同一设备上保延迟还是保吞吐"，IAH 回答的是"不同设备上的任务能不能一起跑"。
