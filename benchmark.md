# XSched Benchmark Guide

本文档用于说明当前测试环境下的手工演示方式，以及两个基准脚本的标准运行方法。

当前赛题测试用例支持在终端中直接自定义：

- 高优先级与低优先级任务的优先级配置
- 高优先级与低优先级任务的并发数
- 不同 workload 的类型选择
- XQueue level、threshold、batch size 等调度相关参数

因此，本文中的命令示例是推荐配置和最小演示配置，不是固定写死的唯一运行方式。

环境假设：

- 当前工程根目录为 `/home/cyk/xsched`
- 已完成编译，且存在：
  - `/home/cyk/xsched/output/bin/xserver`
  - `/home/cyk/xsched/output/bin/xcli`
  - `/home/cyk/xsched/output/lib`
- 当前平台为 WSL + NVIDIA CUDA，真实 `libcuda.so.1` 路径为 `/usr/lib/wsl/lib/libcuda.so.1`

## 1. 手工演示

以下演示以 `1` 个高优先级任务和 `1` 个低优先级任务为例。

这只是最小演示配置，不是固定限制。实际使用时可以继续增加终端，分别启动更多高优先级或低优先级任务，从而调整并发数。

以本节为例：

- 一个终端运行 `xserver`
- 一个终端运行高优先级任务
- 一个终端运行低优先级任务

如果需要更高并发，可以继续增加终端，例如：

- 再开一个终端，额外运行一个低优先级任务，则低优先级并发数变为 `2`
- 再开一个终端，额外运行一个高优先级任务，则高优先级并发数变为 `2`

### 1.1 终端一：启动 `xserver`

```bash
cd /home/cyk/xsched
/home/cyk/xsched/output/bin/xserver HPF 50000
```

作用：

- `xserver` 是 XSched 的用户态调度服务端。
- 开启透明调度后，CUDA 任务创建出的 XQueue 会连接到 `xserver`。
- `xserver` 根据调度策略决定哪些 XQueue 继续运行，哪些 XQueue 需要被挂起或恢复。

命令含义：

- `HPF`：调度策略名，表示 `Highest Priority First`
- `50000`：服务监听端口

可更改项：

- 调度策略可以改成别的策略，例如：
  - `HPF`
  - 其他当前构建支持的策略
- 端口可以修改，例如：
  - `50000`
  - `50001`

如果修改了端口，后续所有需要连接 `xserver` 的脚本或 `xcli` 都要同步改成同一端口。

### 1.2 终端二：高优先级 ResNet50 任务

```bash
cd /home/cyk/xsched

export XSCHED_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1
export CUXTRA_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1

export LD_LIBRARY_PATH=/home/cyk/xsched/output/lib:${LD_LIBRARY_PATH:-}
unset LD_PRELOAD

export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_PRIORITY=1
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=16
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8

python3 platforms/RTX4060/test/resnet50_full.py -c 10
```

作用：

- 启动一个高优先级 ResNet50 推理进程。
- `resnet50_full.py` 默认使用 dedicated CUDA stream，便于 XSched 接管其 GPU kernel。

关键环境变量：

- `XSCHED_SCHEDULER=GLB`
  - 使用全局调度器，进程会连接到 `xserver`
- `XSCHED_AUTO_XQUEUE=ON`
  - 自动为 CUDA stream 创建 XQueue
- `XSCHED_AUTO_XQUEUE_PRIORITY=1`
  - 设置为高优先级
- `XSCHED_AUTO_XQUEUE_LEVEL=1`
  - 使用 Lv1 调度级别
- `XSCHED_AUTO_XQUEUE_THRESHOLD=16`
  - 自动提交阈值
- `XSCHED_AUTO_XQUEUE_BATCH_SIZE=8`
  - 自动批量提交命令数

### 1.3 终端三：低优先级 ResNet50 任务

```bash
cd /home/cyk/xsched

export XSCHED_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1
export CUXTRA_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1

export LD_LIBRARY_PATH=/home/cyk/xsched/output/lib:${LD_LIBRARY_PATH:-}
unset LD_PRELOAD

export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_PRIORITY=0
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=16
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8

python3 platforms/RTX4060/test/resnet50_full.py -c 10
```

作用：

- 启动一个低优先级 ResNet50 推理进程。
- 与终端二的区别主要是 `XSCHED_AUTO_XQUEUE_PRIORITY=0`。

### 1.4 如何增加并发数

如果要增加并发，不需要修改 `xserver` 命令，只需要新增终端，再运行更多高优先级或低优先级任务即可。

例如：

- 当前示例：
  - 高优先级并发数 = `1`
  - 低优先级并发数 = `1`
- 如果再开一个终端，重复执行低优先级命令：
  - 高优先级并发数 = `1`
  - 低优先级并发数 = `2`
- 如果再开一个终端，重复执行高优先级命令：
  - 高优先级并发数 = `2`
  - 低优先级并发数 = `1`
- 如果同时新增一个高优先级终端和一个低优先级终端：
  - 高优先级并发数 = `2`
  - 低优先级并发数 = `2`

### 1.5 `resnet50_full.py` 可修改参数

查看帮助：

```bash
python3 platforms/RTX4060/test/resnet50_full.py --help
```

常用参数：

- `-c, --run-cnt`
  - 每轮吞吐统计中执行多少次推理
- `--gpu`
  - 使用哪一块 CUDA 设备
- `--batch-size`
  - 推理 batch size
- `--image-size`
  - 输入分辨率，默认 `224`
- `--warmup`
  - 预热次数
- `--per-task`
  - 按单次任务输出延迟，而不是无限吞吐循环
- `--num-tasks`
  - 在 `--per-task` 模式下，统计多少个任务
- `--no-dedicated-stream`
  - 使用默认 stream；不建议在 XSched 演示中使用

单任务延迟示例：

```bash
python3 platforms/RTX4060/test/resnet50_full.py --batch-size 1 --per-task --num-tasks 100
```

### 1.6 `xcli` 的辅助使用

查看当前策略：

```bash
/home/cyk/xsched/output/bin/xcli --port 50000 policy -q
```

查看当前 XQueue：

```bash
/home/cyk/xsched/output/bin/xcli --port 50000 list
```

实时查看状态：

```bash
/home/cyk/xsched/output/bin/xcli --port 50000 top -f 1
```

说明：

- PyTorch + cuDNN 会创建较多内部 CUDA 队列，因此 `xcli top` 里可能看到很多 XQueue。
- 这不表示有很多 Python 进程，而是一个 PyTorch 进程内部产生了多个可被 XSched 管理的 XQueue。

## 2. `priority_preemption_throughput.py`

脚本路径：

```text
/home/cyk/xsched/benchmarks/priority_preemption_throughput.py
```

作用：

- 同时启动低优先级任务和高优先级任务。
- 记录每个任务的吞吐变化。
- 用于展示高优先级任务加入后，低优先级和高优先级任务的吞吐表现。

### 2.1 标准运行命令

```bash
cd /home/cyk/xsched

LD_PRELOAD=/home/cyk/miniconda3/lib/libstdc++.so.6 \
LD_LIBRARY_PATH=/home/cyk/miniconda3/lib:$LD_LIBRARY_PATH \
/home/cyk/miniconda3/bin/python3 benchmarks/priority_preemption_throughput.py \
  --mode xsched \
  --lp-workload train \
  --hp-workload cnn \
  --lp-count 1 \
  --hp-count 1 \
  --hp-delay 20 \
  --duration-after-hp 40 \
  -c 10 \
  --batch-size 32 \
  --transformer-batch-size 8 \
  --train-batch-size 16 \
  --warmup 1 \
  --gpu 0 \
  --xqueue-level 1 \
  --threshold 16 \
  --batch-commands 8 \
  --policy HPF \
  --port 50000 \
  --real-libcuda /usr/lib/wsl/lib/libcuda.so.1 \
  --restart-xserver
```

说明：

- `lp-workload=train`：低优先级任务为 MobileNetV2 训练
- `hp-workload=cnn`：高优先级任务为 ResNet50 推理
- `lp-count=1`：低优先级进程数
- `hp-count=1`：高优先级进程数
- `hp-delay=20`：先让低优先级任务运行 20 秒，再启动高优先级任务
- `duration-after-hp=40`：高优先级启动后继续采样 40 秒

### 2.2 可修改参数

查看帮助：

```bash
python3 benchmarks/priority_preemption_throughput.py --help
```

常用参数：

- `--mode`
  - `xsched` 或 `native`
- `--lp-workload`
  - `cnn`
  - `transformer`
  - `train`
- `--hp-workload`
  - `cnn`
  - `transformer`
  - `train`
- `--lp-count`
  - 低优先级并发进程数
- `--hp-count`
  - 高优先级并发进程数
- `--hp-delay`
  - 高优先级启动前，低优先级先运行多久
- `--duration-after-hp`
  - 高优先级启动后继续记录多久
- `-c, --run-cnt`
  - 每轮吞吐统计的内部迭代次数
- `--batch-size`
  - `cnn` 任务 batch size
- `--transformer-batch-size`
  - `transformer` 任务 batch size
- `--train-batch-size`
  - `train` 任务 batch size
- `--gpu`
  - CUDA 设备编号
- `--xqueue-level`
  - XSched 预抢占级别
- `--threshold`
  - XQueue 自动提交阈值
- `--batch-commands`
  - XQueue 自动批量提交命令数
- `--policy`
  - `xserver` 调度策略
- `--port`
  - `xserver` 端口
- `--real-libcuda`
  - 真实 `libcuda.so.1` 路径
- `--no-start-xserver`
  - 不自动启动 `xserver`
- `--restart-xserver`
  - 自动重启干净的 `xserver`

### 2.3 Native 对照命令

```bash
cd /home/cyk/xsched

LD_PRELOAD=/home/cyk/miniconda3/lib/libstdc++.so.6 \
LD_LIBRARY_PATH=/home/cyk/miniconda3/lib:$LD_LIBRARY_PATH \
/home/cyk/miniconda3/bin/python3 benchmarks/priority_preemption_throughput.py \
  --mode native \
  --lp-workload train \
  --hp-workload cnn \
  --lp-count 1 \
  --hp-count 1 \
  --hp-delay 20 \
  --duration-after-hp 40 \
  -c 10 \
  --batch-size 32 \
  --train-batch-size 16
```

## 3. `realtime_inference_latency.py`

脚本路径：

```text
/home/cyk/xsched/benchmarks/realtime_inference_latency.py
```

作用：

- 测试前台高优先级推理延迟。
- 后台运行低优先级 MobileNetV2 训练任务。
- 前台 workload 可在终端选择为 `ResNet50` 或 `Transformer` 推理。
- 输出 `avg / p50 / p95 / p99` 延迟，以及相对独占运行时的 slowdown。

### 3.1 标准运行命令

```bash
cd /home/cyk/xsched

LD_PRELOAD=/home/cyk/miniconda3/lib/libstdc++.so.6 \
LD_LIBRARY_PATH=/home/cyk/miniconda3/lib:$LD_LIBRARY_PATH \
/home/cyk/miniconda3/bin/python3 benchmarks/realtime_inference_latency.py \
  --scenario all \
  --foreground-workload resnet50 \
  --requests 500 \
  --warmup 50 \
  --batch-size 32 \
  --train-batch-size 16 \
  --gpu 0 \
  --hp-delay 10 \
  --background-count 1 \
  --xqueue-level 1 \
  --threshold 16 \
  --batch-commands 8 \
  --policy HPF \
  --port 50000 \
  --real-libcuda /usr/lib/wsl/lib/libcuda.so.1 \
  --restart-xserver
```

说明：

- `scenario=all` 会依次运行：
  - `alone`
  - `native`
  - `xsched`
  - `xsched_lv2`
- `foreground-workload=resnet50`：前台高优先级任务为 ResNet50 推理
- `requests=500`：记录 500 个前台推理请求
- `warmup=50`：前台推理预热 50 次
- `background-count=1`：后台低优先级训练进程数为 1

### 3.2 可修改参数

查看帮助：

```bash
python3 benchmarks/realtime_inference_latency.py --help
```

常用参数：

- `--scenario`
  - `alone`
  - `native`
  - `xsched`
  - `xsched_lv2`
  - `all`
- `--requests`
  - 前台需要记录多少个推理请求
- `--warmup`
  - 前台推理预热次数
- `--foreground-workload`
  - `resnet50`
  - `transformer`
- `--batch-size`
  - 前台高优先级推理任务的 batch size
- `--train-batch-size`
  - MobileNetV2 background training batch size
- `--gpu`
  - CUDA 设备编号
- `--hp-delay`
  - 后台训练先运行多久，再启动前台推理
- `--background-count`
  - 低优先级训练进程并发数
- `--xqueue-level`
  - `xsched` 场景使用的默认 level，`xsched_lv2` 会强制用 level 2
- `--threshold`
  - 自动提交阈值
- `--batch-commands`
  - 自动批量提交命令数
- `--policy`
  - 调度策略
- `--port`
  - `xserver` 端口
- `--real-libcuda`
  - 真实 `libcuda.so.1` 路径
- `--no-start-xserver`
  - 不自动启动 `xserver`
- `--restart-xserver`
  - 自动重启干净的 `xserver`

### 3.3 常用变体

只跑 native：

```bash
python3 benchmarks/realtime_inference_latency.py \
  --scenario native \
  --foreground-workload resnet50 \
  --requests 500 \
  --warmup 50 \
  --batch-size 32 \
  --train-batch-size 16 \
  --background-count 1
```

只跑 LV1：

```bash
python3 benchmarks/realtime_inference_latency.py \
  --scenario xsched \
  --foreground-workload resnet50 \
  --requests 500 \
  --warmup 50 \
  --batch-size 32 \
  --train-batch-size 16 \
  --background-count 1 \
  --restart-xserver
```

只跑 LV2：

```bash
python3 benchmarks/realtime_inference_latency.py \
  --scenario xsched_lv2 \
  --foreground-workload resnet50 \
  --requests 500 \
  --warmup 50 \
  --batch-size 32 \
  --train-batch-size 16 \
  --background-count 1 \
  --restart-xserver
```

将低优先级并发数改为 2：

```bash
python3 benchmarks/realtime_inference_latency.py \
  --scenario all \
  --foreground-workload resnet50 \
  --requests 500 \
  --warmup 50 \
  --batch-size 32 \
  --train-batch-size 16 \
  --background-count 2 \
  --restart-xserver
```

切换为 Transformer 前台推理：

```bash
python3 benchmarks/realtime_inference_latency.py \
  --scenario all \
  --foreground-workload transformer \
  --requests 500 \
  --warmup 50 \
  --batch-size 8 \
  --train-batch-size 16 \
  --background-count 1 \
  --restart-xserver
```

## 4. 结果文件

`priority_preemption_throughput.py` 和 `realtime_inference_latency.py` 的结果默认都会写入：

```text
/home/cyk/xsched/benchmark_results/
```

其中：

- `priority_preemption_throughput.py`
  - 生成吞吐记录与汇总 JSON/CSV
- `realtime_inference_latency.py`
  - 生成：
    - `config.json`
    - `comparison.json`
    - `*/latency.csv`
    - `*/summary.json`
    - `*.log`

`realtime_inference_latency.py` 的结果图可通过如下命令生成：

```bash
python3 tools/plot_realtime_latency.py \
  benchmark_results/<your_result_dir> \
  --title "Foreground inference latency under MobileNetV2 background"
```

将生成：

- `latency_percentiles.png`
- `latency_cdf.png`
- `p99_slowdown.png`
