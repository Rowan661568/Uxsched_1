# RTX4060 Platform (NVIDIA GPU)

与 [Ascend 平台](../ascend) 相同的目录结构（`hal/`、`shim/`、`test/`），面向 NVIDIA GPU（如 RTX 4060）。`hal` 与 `shim` 复用 [CUDA 平台](../cuda) 实现（通过符号链接），拦截 `libcuda` 以接入 XSched 调度。

## 目录结构

```
platforms/RTX4060/
├── CMakeLists.txt
├── README.md
├── hal/          -> ../cuda/hal
├── shim/         -> ../cuda/shim
└── test/
    └── resnet152.py
```

## 构建

```bash
cmake -B build -DPLATFORM_RTX4060=ON
cmake --build build
```

## ResNet152 压测（与 Ascend 用例逻辑一致）

```bash
# 可选：通过 LD_PRELOAD 加载 shim 以启用 XSched 拦截
export LD_PRELOAD=/path/to/install/lib/libshimrtx4060.so

python3 platforms/RTX4060/test/resnet152.py -c 10
```

依赖：`torch`、`torchvision`，且本机需有可用 CUDA 与 NVIDIA 驱动。

## 与 Ascend 测试的差异

| 项目 | Ascend (`ascend/test/resnet152.py`) | RTX4060 |
|------|-------------------------------------|---------|
| 设备后端 | `torch_npu` / `.npu()` | `torch.cuda` / `.cuda()` |
| Shim 库 | `libshimascend.so` | `libshimrtx4060.so` |
| 驱动拦截 | ACL (`libascendcl.so` 等) | CUDA Driver (`libcuda.so`) |
