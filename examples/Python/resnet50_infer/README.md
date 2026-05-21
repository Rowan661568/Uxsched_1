# ResNet50 单卡推理 + XSched

在 **单张 NVIDIA GPU** 上跑 ResNet50 推理，并为每次 forward 打印任务耗时（与 `examples/Linux/1_transparent_sched` 中 `Task N completed in X ms` 风格一致）。

采用 **透明调度**：无需改模型代码，通过 `libshimcuda` 拦截 CUDA 调用并自动为每个 CUDA stream 创建 XQueue。

## 环境要求

- NVIDIA GPU + 驱动（WSL2 需安装 [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)）
- Python 3.8+
- 已编译的 XSched（`make cuda`）
- PyTorch（CUDA 版）+ torchvision

## 1. 编译 XSched

```bash
cd /path/to/xsched
git submodule update --init --recursive
# 需要 cmake、CUDA Toolkit (nvcc) 等构建依赖
make cuda INSTALL_PATH=/path/to/xsched/output
```

安装路径默认为仓库根目录下的 `output/`。

## 2. 安装 Python 依赖

```bash
cd examples/Python/resnet50_infer
pip install -r requirements.txt
# 若需指定 CUDA 版本，请参考 https://pytorch.org/get-started/locally/
```

## 3. 运行

### 基线（不启用 XSched）

```bash
chmod +x run_baseline.sh run_with_xsched.sh
./run_baseline.sh --num-tasks 50
```

### 启用 XSched（单进程、单卡，推荐先试）

使用进程内调度器 `LCL`，**不需要** 单独启动 `xserver`：

```bash
./run_with_xsched.sh --num-tasks 50 --gpu 0
```

示例输出：

```
Device: cuda:0 (NVIDIA GeForce RTX ...)
XSched: transparent mode (scheduler=LCL, priority=0)
Loading ResNet50...
Task 0 completed in 12.34 ms
Task 1 completed in 11.89 ms
...
```

### 多进程优先级（可选）

终端 1 启动全局调度器：

```bash
/path/to/xsched/output/bin/xserver HPF 50000
```

终端 2（高优先级）：

```bash
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE_PRIORITY=1
./run_with_xsched.sh --num-tasks 100
```

终端 3（低优先级）：

```bash
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE_PRIORITY=0
export XSCHED_AUTO_XQUEUE_THRESHOLD=4
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=2
./run_with_xsched.sh --num-tasks 100
```

## 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--gpu` | GPU 编号 | 0 |
| `--batch-size` | batch 大小 | 1 |
| `--image-size` | 输入边长 | 224 |
| `--num-tasks` | 计时的推理次数 | 100 |
| `--warmup` | 预热次数 | 10 |
| `--sleep-ms` | 任务间隔（毫秒） | 0 |
| `--no-dedicated-stream` | 使用默认 stream | 否 |

环境变量 `XSCHED_INSTALL` 可覆盖 XSched 安装目录（默认 `<xsched>/output`）。

## 说明

- 计时为 **单次 forward + GPU 同步** 的墙钟时间，与仓库内其他示例一致；XSched 不提供单独的推理延迟 API。
- 默认使用独立 `torch.cuda.Stream`，便于 XSched 为 stream 自动建 XQueue；可用 `--no-dedicated-stream` 对比行为。
- RTX 40 系列（如 sm89）在官方矩阵中可能标记为进行中；若 shim 异常，可先 `./run_baseline.sh` 确认 PyTorch/CUDA 正常，再查 [platforms/cuda](../../../platforms/cuda)。
