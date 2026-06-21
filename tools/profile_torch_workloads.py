#!/usr/bin/env python3
"""Profile CUDA op timing for the benchmark PyTorch workloads."""

from __future__ import annotations

import argparse

import torch
import torchvision
from torch.profiler import ProfilerActivity, profile


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    pos = (len(data) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    frac = pos - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def summarize(name: str, fn, warmup: int) -> None:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()

    events = [
        e for e in prof.key_averages()
        if getattr(e, "cuda_time_total", 0) > 0
    ]
    vals = sorted(e.cuda_time_total / 1000.0 for e in events)
    print(f"\n{name}")
    print(
        "cuda_op_count={count} sum_ms={sum_ms:.3f} "
        "p50_ms={p50:.3f} p90_ms={p90:.3f} max_ms={max_ms:.3f}".format(
            count=len(vals),
            sum_ms=sum(vals),
            p50=percentile(vals, 50),
            p90=percentile(vals, 90),
            max_ms=max(vals) if vals else 0.0,
        )
    )
    for e in sorted(events, key=lambda x: x.cuda_time_total, reverse=True)[:16]:
        print(
            f"{e.key[:64]:64s} "
            f"cuda_total_ms={e.cuda_time_total / 1000.0:.3f} calls={e.count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    resnet = torchvision.models.resnet50(weights=None).eval().to(device)
    x_resnet = torch.randn(8, 3, 224, 224, device=device)

    def resnet_once() -> None:
        with torch.no_grad():
            resnet(x_resnet)

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

    def transformer_once() -> None:
        with torch.no_grad():
            transformer(x_transformer)

    mobilenet = torchvision.models.mobilenet_v2(
        weights=None,
        num_classes=1000,
    ).train().to(device)
    opt = torch.optim.SGD(mobilenet.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    x_mobilenet = torch.randn(16, 3, 224, 224, device=device)
    y_mobilenet = torch.randint(0, 1000, (16,), device=device)

    def mobilenet_step() -> None:
        opt.zero_grad(set_to_none=True)
        out = mobilenet(x_mobilenet)
        loss = loss_fn(out, y_mobilenet)
        loss.backward()
        opt.step()

    summarize("ResNet50 inference batch=8", resnet_once, args.warmup)
    summarize("Transformer inference batch=4", transformer_once, args.warmup)
    summarize("MobileNetV2 train batch=16 one step", mobilenet_step, args.warmup)


if __name__ == "__main__":
    main()
