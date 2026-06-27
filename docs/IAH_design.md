# Interference-Aware Heterogeneous Scheduling (IAH)

---

## 1. 动机

### 1.1 HHPF 的局限性

Heterogeneous Highest Priority First (HHPF) 策略在异构加速器场景中解决了跨设备优先级感知的问题：当一个设备上的高优先级任务就绪时，它会 Suspend 其他所有设备上的低优先级任务。这在 GPU 和 NPU 共享 SoC 内存带宽和功耗预算的场景下是必要的——NVLink 或共享总线上的争抢确实会影响高优先级任务的延迟。

但 HHPF 的问题是：它**总是** Suspend，即使当前没有资源争抢。

```
时间轴（HHPF）:

GPU ── 高优推理（计算密集，带宽需求低）███████████████
NPU ── 低优训练（计算密集，带宽需求低）░░░░░░░░░░░░░░░░░  ← 被 Suspend！

实际情况：
  - GPU 跑 kernel 是 compute-bound，几乎不访存
  - NPU 跑 kernel 也是 compute-bound
  - 两者共享总线上几乎没有争抢
  - NPU 本可以安全地继续训练
```

这种一刀切导致异构系统利用率大幅下降：HHPF 下低优设备的吞吐量被迫降到零，而竞争资源实际上并未被争抢。

### 1.2 核心观察

异构 SoC（如 Jetson AGX Orin、部分昇腾模组）上的 GPU 和 NPU 虽然计算单元完全独立，但它们共享：

- **内存控制器与带宽**：同时高带宽的 kernel 会争抢
- **功耗预算**：当前的总功耗可能让 task 频率受到限制
- **LLC/Last-Level Cache**：某些 SoC 架构上共享末级缓存

但并非所有 kernel 都具有高带宽需求。一个计算密集型 kernel（如矩阵乘法）的带宽压力远小于一个访存密集型 kernel（如 embedding lookup 或某些逐元素操作）。

因此，跨设备是否应该 co-schedule 取决于**当前运行的 kernel 的干扰风险**，而不是简单地"有高优任务就全停"。

---

## 2. 设计与决策模型

### 2.1 干扰模型

IAH 引入一个简单的干扰模型：每个设备维护一个**内存带宽强度**（Memory Intensity）因子，范围 [0.0, 1.0]：

- **0.0**：纯计算密集（如 GEMM、卷积），几乎不产生内存带宽压力
- **0.5**：混合型
- **1.0**：纯访存密集（如 embedding lookup、broadcast 操作），产生最大带宽压力

两个设备上的任务能否同时运行，取决于它们的强度之和是否超过阈值：

```
interference_risk = intensity(dev_a) + intensity(dev_b)
high_interference = interference_risk > kIntensityThreshold  // 默认 1.2
```

阈值 1.2 是一个保守值，表示只有在两个设备**都明显带宽敏感**时才禁止 co-schedule。一个设备 bandwidth-heavy（0.7）+ 另一个 bandwidth-heavy（0.7）= 1.4 > 1.2 → 禁止。一个 compute-bound（0.3）+ 另一个 compute-bound（0.3）= 0.6 < 1.2 → 允许。

### 2.2 调度决策流程

每个调度 tick，决策逻辑：

```
1. 扫描所有就绪队列，找到全局最高优先级任务 G 及其设备 dev(G)

2. 对每台设备 D：
   a. 找出 D 上本地优先级最高的任务 L
   b. 如果 D == dev(G) → G 和 L 在同一设备 → 经典优先级调度，G 跑
   c. 如果 priority(L) >= priority(G) → 同优先级，L 跑
   d. 否则：检查干扰风险
      - intensity(dev(G)) + intensity(D) > 阈值 → 高 → Suspend L
      - 否则 → 低 → Resume L（co-schedule！）

3. 每设备上非 L 的队列按优先级 Suspend
```

### 2.3 与 HHPF 的对比

| 维度 | HHPF | IAH |
|------|------|-----|
| 决策依据 | 全局最高优先级 | 全局最高优先级 + 跨设备干扰风险 |
| 低优设备在冲突时的行为 | 全部 Suspend | 全部 Suspend（与 HHPF 一致） |
| 低优设备在**不冲突**时的行为 | 仍然 Suspend | **允许继续运行（co-schedule）** |
| 对系统吞吐的影响 | 低优设备吞吐归零 | 不冲突时吞吐最高 |
| 对高优延迟的保护 | 严格 | 同样严格（冲突时才 Suspend）|
| 需要的外部信息 | 仅优先级 | 优先级 + 每设备内存强度 |

### 2.4 默认行为的安全性

在没有显式设置 `MemoryIntensityHint` 的情况下，所有设备的默认强度为 **0.5**，两个默认设备之和为 1.0 < 1.2，**默认允许 co-schedule**。这意味着 IAH 比 HHPF 更激进地允许跨设备并行，但一旦收到显式的强度 hint，行为会更加精确。

对于绝对不允许并行的高优场景，可以通过 hint 显式设置强度为 1.0（对端）+ 0.3（本端）= 1.3 > 1.2 → 禁止 co-schedule，退化为 HHPF 行为。

---

## 3. 实现

### 3.1 新增组件

| 文件 | 类型 | 说明 |
|------|------|------|
| `sched/include/xsched/sched/policy/iah.h` | 头文件 | IAH 策略类声明 |
| `sched/src/policy/iah.cpp` | 实现 | 调度决策 + Hint 处理 |
| `sched/include/xsched/sched/protocol/hint.h` | 扩展 | 新增 `kHintTypeMemoryIntensity` + `MemoryIntensityHint` |

### 3.2 注册改动

| 文件 | 改动 |
|------|------|
| `include/xsched/types.h` | 新增 `kPolicyInterferenceAwareHeterogeneous = 12` |
| `protocol/include/xsched/protocol/def.h` | 新增 `XSCHED_POLICY_NAME_IAH` |
| `protocol/src/names.cpp` | 注册策略名映射 |
| `sched/src/policy/policy.cpp` | 工厂 case + header include |
| `service/server/src/server.cpp` | 新增 `MemoryIntensityHint` 的 HTTP hint 分发 |

### 3.3 使用方式

**启动 xserver：**

```bash
/home/cyk/xsched/output/bin/xserver IAH 50000
```

**设置跨设备干扰参数（可选）：**

通过 HTTP hint 设置某设备的内存强度：

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"hint_type": 8, "device": <device_addr>, "intensity": 0.7}' \
  http://localhost:50000/hint
```

或者在 benchmark 中通过 `XSCHED_AUTO_XQUEUE_INTENSITY` 环境变量设置（如需扩展）。

**默认行为（无 hint）：**
- 所有设备的默认强度为 0.5
- 两台设备之和 0.5 + 0.5 = 1.0 < 1.2 → 默认允许 co-schedule
- 一台设备 0.5 + 另一台设到 0.8 = 1.3 > 1.2 → 禁止 co-schedule

---

## 4. 实验建议

要验证 IAH 相比 HHPF 的收益，建议覆盖以下场景：

- **Case 1：双方 compute-bound** — GPU 跑大卷积、NPU 跑矩阵乘。IAH 应该允许两者并行，HHPF 则 Suspend NPU → IAH 吞吐显著高于 HHPF，高优延迟不变。
- **Case 2：双方 memory-bound** — GPU 跑 broadcast-heavy 的 kernel，NPU 跑 embedding。IAH 应该 Suspend NPU，行为等价于 HHPF。
- **Case 3：混合** — GPU memory-bound、NPU compute-bound。IAH 允许 NPU 跑，延迟微增但吞吐明显提升。

---

## 5. 与集群级方案的关系（非本系统范围）

集群调度器（如 Kubernetes + YARN）负责将任务分配到不同的计算节点，解决的是"任务去哪儿跑"的问题。IAH 解决的是"同一台设备上的多个进程如何共享单张/多张加速器卡"的问题。

两者是互补的：集群调度器的决策粒度通常在秒级，无法感知 SoC 内的微秒级带宽争抢。IAH 填补的是这一层调度空白，以 XUSec 级别的 tick 频率动态调整跨设备的并行度。
