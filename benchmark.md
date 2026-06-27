# XSched Benchmark Guide

本文档用于说明当前测试环境下的手工演示方式，以及两个基准脚本的标准运行方法。

当前赛题测试用例支持在终端中直接自定义：

- 高优先级与低优先级任务的优先级配置
- 高优先级与低优先级任务的并发数
- 不同 workload 的类型选择
- XQueue level、threshold、batch size 等调度相关参数

因此，本文中的命令示例是推荐配置和最小演示配置，不是固定写死的唯一运行方式。

## 0. 环境搭建（首次运行前必做）

以下步骤适用于全新 Linux 环境的首次搭建，支持原生 Ubuntu、WSL2、CentOS、Fedora 等主流发行版。如果环境已就绪，可跳过本节。

### 0.1 系统基础依赖

根据你的 Linux 发行版选择对应的安装命令：

**Ubuntu / Debian：**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential gcc g++ make cmake git wget curl
```

**CentOS / RHEL / Fedora：**
```bash
sudo yum update -y
sudo yum groupinstall -y "Development Tools"
sudo yum install -y gcc gcc-c++ make cmake git wget curl
```

**Arch Linux：**
```bash
sudo pacman -Syu
sudo pacman -S base-devel gcc make cmake git wget curl
```

验证 NVIDIA 驱动：
```bash
nvidia-smi
# 应该能看到 GPU 信息和驱动版本
```

### 0.2 安装 CUDA Toolkit

CUDA Toolkit 用于编译 XSched。运行时使用系统自带的 `libcuda.so.1`。

**方法1：使用 NVIDIA 官方安装器（推荐）**
```bash
# 下载 CUDA 12.4（与 PyTorch 匹配）
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# 安装（仅安装 toolkit，不安装驱动）
sudo sh cuda_12.4.0_550.54.14_linux.run --silent --toolkit

# 添加 CUDA 到 PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**方法2：使用包管理器（Ubuntu）**
```bash
sudo apt install -y nvidia-cuda-toolkit
```

**方法3：使用 conda（如果不想安装系统级 CUDA）**
```bash
conda install -n uxsched cudatoolkit=12.4 -c nvidia
```

验证安装：
```bash
nvcc --version
```

### 0.3 安装 Miniconda

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装（按提示操作，安装到 ~/miniconda3）
bash Miniconda3-latest-Linux-x86_64.sh

# 初始化 conda
conda init bash
source ~/.bashrc
```

### 0.4 创建 Python 虚拟环境

```bash
# 创建虚拟环境
conda create -n uxsched python=3.11 -y

# 激活环境
conda activate uxsched
```

### 0.5 安装 Python 依赖

```bash
# 安装 PyTorch（CUDA 12.4 版本）
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install numpy pillow matplotlib pandas

# 验证 PyTorch CUDA 可用
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

### 0.6 编译 XSched

```bash
cd /home/cyk/xsched

# 清理旧构建（如果有）
make clean

# 配置并编译（CUDA 平台）
make cuda

# 验证编译产物
ls -la output/bin/    # 应该有 xserver, xcli
ls -la output/lib/    # 应该有 libpreempt.so, libshimrtx4060.so 等
```

### 0.7 确定 libcuda.so.1 路径

不同环境下 `libcuda.so.1` 的位置不同，需要根据你的环境确定：

```bash
# 查找 libcuda.so.1 的位置
find /usr -name "libcuda.so.1" 2>/dev/null
find /lib -name "libcuda.so.1" 2>/dev/null

# 常见路径：
# - 原生 Ubuntu: /usr/lib/x86_64-linux-gnu/libcuda.so.1
# - WSL2: /usr/lib/wsl/lib/libcuda.so.1
# - CentOS/RHEL: /usr/lib64/libcuda.so.1
# - 自定义安装: /usr/local/cuda/lib64/libcuda.so.1
```

记录这个路径，后续环境变量配置需要用到。

### 0.8 环境变量配置（可选）

创建环境设置脚本以便后续使用。请根据你的实际环境修改 `REAL_LIBCUDA` 路径：

```bash
cat > ~/xsched_env.sh << 'EOF'
#!/bin/bash
# XSched 环境变量设置
# 请根据你的环境修改此路径

# CUDA 库路径（根据你的环境修改）
# 原生 Ubuntu: /usr/lib/x86_64-linux-gnu/libcuda.so.1
# WSL2: /usr/lib/wsl/lib/libcuda.so.1
# CentOS/RHEL: /usr/lib64/libcuda.so.1
REAL_LIBCUDA="/usr/lib/wsl/lib/libcuda.so.1"

export XSCHED_CUDA_LIB="$REAL_LIBCUDA"
export CUXTRA_CUDA_LIB="$REAL_LIBCUDA"

# XSched 库路径
export LD_LIBRARY_PATH=/home/cyk/xsched/output/lib:${LD_LIBRARY_PATH:-}

# 确保不使用 LD_PRELOAD（避免冲突）
unset LD_PRELOAD

# XSched 调度配置
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_PRIORITY=1
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=16
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8
EOF

chmod +x ~/xsched_env.sh
```

后续使用时执行 `source ~/xsched_env.sh` 即可快速设置环境变量。

### 0.9 快速检查清单

运行以下命令验证环境是否就绪：

```bash
# 1. 检查 GPU
nvidia-smi

# 2. 检查 CUDA 编译器
nvcc --version

# 3. 检查 Python 环境（需要先 conda activate uxsched）
python3 -c "import torch; print(torch.cuda.is_available())"

# 4. 检查 XSched 编译产物
ls -la /home/cyk/xsched/output/bin/xserver
ls -la /home/cyk/xsched/output/lib/libpreempt.so

# 5. 查找 libcuda.so.1 路径
find /usr -name "libcuda.so.1" 2>/dev/null

# 6. 测试 xserver 启动
timeout 2 /home/cyk/xsched/output/bin/xserver HPF 50000 || echo "xserver 正常"
```

---

环境假设：

- 当前工程根目录为 `/home/cyk/xsched`
- 已完成编译，且存在：
  - `/home/cyk/xsched/output/bin/xserver`
  - `/home/cyk/xsched/output/bin/xcli`
  - `/home/cyk/xsched/output/lib`
- 已安装 NVIDIA 驱动和 CUDA Toolkit，`nvidia-smi` 和 `nvcc` 可用
- 已激活 `uxsched` conda 环境（`conda activate uxsched`）
- 已确定当前环境的 `libcuda.so.1` 路径，并设置到环境变量 `XSCHED_CUDA_LIB` 和 `CUXTRA_CUDA_LIB`

**libcuda.so.1 常见路径：**
- 原生 Ubuntu: `/usr/lib/x86_64-linux-gnu/libcuda.so.1`
- WSL2: `/usr/lib/wsl/lib/libcuda.so.1`
- CentOS/RHEL: `/usr/lib64/libcuda.so.1`
- 自定义安装: `/usr/local/cuda/lib64/libcuda.so.1`

后续文档中的命令示例使用 `--real-libcuda` 参数指定真实 `libcuda.so.1` 路径，请根据你的环境替换。

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
  - `HPF`：Highest Priority First，严格优先级策略
  - `SLO`：SLO-Constrained Adaptive，基于 deadline 的三层自适应调度
  - `IAH`：Interference-Aware Heterogeneous，跨设备干扰感知调度（扩展 HHPF，低干扰时允许不同设备上的任务并行）
  - `HHPF`：Heterogeneous HPF，全局跨设备优先级调度
- 端口可以修改，例如：
  - `50000`
  - `50001`

如果修改了端口，后续所有需要连接 `xserver` 的脚本或 `xcli` 都要同步改成同一端口。

### 1.2 终端二：高优先级 ResNet50 任务

```bash
cd /home/cyk/xsched

# 请根据你的环境修改 libcuda.so.1 路径
# 原生 Ubuntu: /usr/lib/x86_64-linux-gnu/libcuda.so.1
# WSL2: /usr/lib/wsl/lib/libcuda.so.1
# CentOS/RHEL: /usr/lib64/libcuda.so.1
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

# 请根据你的环境修改 libcuda.so.1 路径（同终端二）
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

# 请根据你的环境修改 libcuda.so.1 路径
# 原生 Ubuntu: /usr/lib/x86_64-linux-gnu/libcuda.so.1
# WSL2: /usr/lib/wsl/lib/libcuda.so.1
# CentOS/RHEL: /usr/lib64/libcuda.so.1
REAL_LIBCUDA=/usr/lib/wsl/lib/libcuda.so.1

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
  --real-libcuda "$REAL_LIBCUDA" \
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

# 请根据你的环境修改 libcuda.so.1 路径
REAL_LIBCUDA=/usr/lib/wsl/lib/libcuda.so.1

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
  --real-libcuda "$REAL_LIBCUDA" \
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
- `--slo-factor`
  - SLO 策略专用：deadline = baseline p99 × factor，例如 1.25 表示 deadline 为独占时 p99 的 1.25 倍
  - 脚本会自动先跑 alone 场景测得 baseline，再算 deadline 注入到后续 xsched 场景
  - 不需要手动指定 deadline 绝对值
- `--slo-target-us`
  - SLO 策略专用：直接指定 deadline 绝对值，单位微秒
  - 有此项时 `--slo-factor` 不生效
  - 适用于已知业务场景 deadline 的场景，跳过 profiling 流程
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

### 3.4 SLO 策略的使用

SLO（SLO-Constrained Adaptive）策略使用 deadline 驱动三层调度（Urgent / Latency-Sensitive / Batch），在延迟和吞吐之间做量化权衡。使用时需要传入 SLO deadline 参数。

**指定 `--policy SLO`，配合以下两种 deadline 设定方式之一：**

**方式一：自动 profiling + factor**

先跑 alone 场景测得独占时 foreground 的 p99 latency 作为 baseline，再乘以容忍因子得到 deadline。

```bash
cd /home/cyk/xsched

LD_PRELOAD=/home/cyk/miniconda3/lib/libstdc++.so.6 \
LD_LIBRARY_PATH=/home/cyk/miniconda3/lib:$LD_LIBRARY_PATH \
/home/cyk/miniconda3/bin/python3 benchmarks/realtime_inference_latency.py \
  --policy SLO --slo-factor 1.25 \
  --requests 500 --warmup 50 \
  --restart-xserver
```

脚本执行流程：

1. 先跑 `alone`，测得 foreground 独占加速器时的 p99 latency
2. 计算 deadline = p99 × 1.25 × 1000（微秒）
3. 后续 `xsched` / `xsched_lv2` 场景自动注入 deadline 到 foreground 进程
4. SLO 策略根据 deadline 调度：slack 充足时 Batch 层填满加速器吞吐，deadline 逼近时进入 Urgent 层保延迟

factor 越小，SLO 约束越紧，延迟保护越好但低优任务的执行窗口越窄；factor 越大，Batch 层获得更多空闲运行时间，但延迟保护相应放宽。推荐在 1.2 ~ 2.0 之间选择。

**方式二：直接指定 deadline（跳过 profiling）**

业务场景已知 deadline 时，直接赋值即可：

```bash
cd /home/cyk/xsched

LD_PRELOAD=/home/cyk/miniconda3/lib/libstdc++.so.6 \
LD_LIBRARY_PATH=/home/cyk/miniconda3/lib:$LD_LIBRARY_PATH \
/home/cyk/miniconda3/bin/python3 benchmarks/realtime_inference_latency.py \
  --policy SLO --slo-target-us 70000 \
  --requests 500 --warmup 50 \
  --restart-xserver
```

此时跳过 alone profiling，直接以 deadline=70ms 调度。

**SLO 策略观察要点：**

- 与 HPF 不同，SLO 在 foreground slack 充足时不会激进取缔 background，background 吞吐会明显高于 HPF
- `xsched_lv2` 场景下 SLO 比 HPF 更稳定——deadline 兜底使调度器不需要频繁切换，避免了 HPF 在细粒度抢占下的过度调度退化

## 4. `preemption_latency_scaling.py`

脚本路径：

```text
/home/cyk/xsched/benchmarks/preemption_latency_scaling.py
```

作用：

- 这是机制型 benchmark，不是应用型 workload benchmark。
- 它在运行时使用 NVRTC 编译固定时长 sleep kernel，并直接调用 XSched 运行库。
- 通过持续提交固定执行时长的 CUDA command，测量 `Lv1 / Lv2` 的抢占延迟。
- 用于验证 `Lv2` 相比 `Lv1` 是否明显降低 `P99 preemption latency`。

### 4.1 标准运行命令

```bash
cd /home/cyk/xsched

# 请根据你的环境修改 libcuda.so.1 路径
REAL_LIBCUDA=/usr/lib/wsl/lib/libcuda.so.1

python3 benchmarks/preemption_latency_scaling.py \
  --levels 1 2 \
  --kernel-us 50 100 250 500 1000 2000 \
  --threshold 8 \
  --batch-size 4 \
  --iters 40 \
  --real-libcuda "$REAL_LIBCUDA"
```

说明：

- `levels 1 2`：对比 Lv1 和 Lv2
- `kernel-us`：每个 command 的目标执行时长，单位微秒
- `threshold=8`：对应论文里常见的 in-flight command threshold 设置
- `batch-size=4`：每次批量提交 command 的数量
- `iters=40`：每个 case 记录 40 个有效 suspend/resume 样本

### 4.2 可修改参数

查看帮助：

```bash
python3 benchmarks/preemption_latency_scaling.py --help
```

常用参数：

- `--levels`
  - 需要测试的 XSched level，例如 `1 2`
- `--arch`
  - NVRTC 使用的 CUDA 架构，例如 RTX4060 对应 `89`
- `--kernel-us`
  - 需要扫描的 command 执行时长列表，单位微秒
- `--threshold`
  - XQueue in-flight command threshold
- `--batch-size`
  - XQueue command batch size
- `--burst`
  - 每轮连续发射多少个 command 后再同步
- `--iters`
  - 每个 command 时长的有效采样次数
- `--warmup-s`
  - 启动持续提交线程后，正式测量前的预热时间
- `--jitter-us`
  - 每次 suspend/resume 前随机扰动范围，避免总在相同相位点采样
- `--real-libcuda`
  - 真实 `libcuda.so.1` 路径
- `--result-dir`
  - 自定义结果输出目录

输出结果包含：

- `preemption_latency_scaling.csv`
- `summary.json`

结果图可通过如下命令生成：

```bash
python3 tools/plot_preemption_latency.py \
  benchmark_results/<your_result_dir> \
  --title "XSched preemption latency scaling"
```

将生成：

- `preemption_p99_scaling.png`
- `preemption_percentiles.png`
- `preemption_lv2_gain.png`

其中重点关注：

- `preempt_avg_us`
- `preempt_p95_us`
- `preempt_p99_us`

如果机制正常，通常应看到：

- `Lv2` 的 `p99` 明显低于 `Lv1`
- 随着 `kernel_us` 增大，`Lv1` 的增长更明显

## 5. 结果文件

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
- `preemption_latency_scaling.py`
  - 生成：
    - `preemption_latency_scaling.csv`
    - `summary.json`

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
