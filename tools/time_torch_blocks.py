#!/usr/bin/env python3
"""Measure coarse CUDA block times for benchmark PyTorch models."""

from __future__ import annotations

import torch
import torchvision


def cuda_ms(fn, repeat: int = 20, warmup: int = 5) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def summary(name: str, values: list[float]) -> None:
    data = sorted(values)
    p50 = data[len(data) // 2]
    p95 = data[int((len(data) - 1) * 0.95)]
    print(
        f"{name:42s} avg_ms={sum(values) / len(values):8.3f} "
        f"p50_ms={p50:8.3f} p95_ms={p95:8.3f} "
        f"min_ms={min(values):8.3f} max_ms={max(values):8.3f}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    resnet = torchvision.models.resnet50(weights=None).eval().to(device)
    x_resnet = torch.randn(8, 3, 224, 224, device=device)
    res_blocks = [
        ("resnet stem", lambda: resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(x_resnet))))),
    ]
    with torch.no_grad():
        y = resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(x_resnet))))
        for i, layer in enumerate([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4], 1):
            block_input = y
            res_blocks.append((f"resnet layer{i}", lambda layer=layer, x=block_input: layer(x)))
            y = layer(y)
        res_blocks.append(("resnet full forward", lambda: resnet(x_resnet)))
        for name, fn in res_blocks:
            summary(name, cuda_ms(fn))

    layer = torch.nn.TransformerEncoderLayer(
        d_model=768,
        nhead=12,
        dim_feedforward=3072,
        batch_first=True,
        dropout=0.0,
        activation="gelu",
    )
    transformer = torch.nn.TransformerEncoder(layer, num_layers=6).eval().to(device)
    x_transformer = torch.randn(4, 128, 768, device=device)
    with torch.no_grad():
        summary("transformer one encoder layer", cuda_ms(lambda: transformer.layers[0](x_transformer)))
        summary("transformer full forward", cuda_ms(lambda: transformer(x_transformer)))

    mobilenet = torchvision.models.mobilenet_v2(weights=None, num_classes=1000).train().to(device)
    opt = torch.optim.SGD(mobilenet.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    x_mobilenet = torch.randn(16, 3, 224, 224, device=device)
    y_mobilenet = torch.randint(0, 1000, (16,), device=device)

    def mobilenet_forward() -> None:
        mobilenet(x_mobilenet)

    def mobilenet_step() -> None:
        opt.zero_grad(set_to_none=True)
        out = mobilenet(x_mobilenet)
        loss = loss_fn(out, y_mobilenet)
        loss.backward()
        opt.step()

    summary("mobilenet train forward only", cuda_ms(mobilenet_forward))
    summary("mobilenet train full step", cuda_ms(mobilenet_step))


if __name__ == "__main__":
    main()
