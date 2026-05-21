#!/usr/bin/env python3
"""ResNet50 CNN inference on a single CUDA GPU.

Each forward pass is one schedulable task. Wall-clock time is printed per task
(same style as examples/Linux/1_transparent_sched/app.cu).

Use run_with_xsched.sh to enable XSched transparent scheduling via libshimcuda.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet50 inference with per-task timing")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0)")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    p.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input spatial size H=W (default: 224)",
    )
    p.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="Number of timed inference tasks (default: 100)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before timing (default: 10)",
    )
    p.add_argument(
        "--sleep-ms",
        type=int,
        default=0,
        help="Sleep between tasks in ms, 0 = no sleep (default: 0)",
    )
    p.add_argument(
        "--no-dedicated-stream",
        action="store_true",
        help="Use default stream instead of a dedicated cuda.Stream",
    )
    return p.parse_args()


def check_cuda(gpu: int) -> torch.device:
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Install PyTorch with CUDA support.", file=sys.stderr)
        sys.exit(1)
    if gpu < 0 or gpu >= torch.cuda.device_count():
        print(
            f"ERROR: --gpu {gpu} is invalid ({torch.cuda.device_count()} device(s) visible).",
            file=sys.stderr,
        )
        sys.exit(1)
    return torch.device(f"cuda:{gpu}")


def run_forward(model: torch.nn.Module, data: torch.Tensor, stream: torch.cuda.Stream | None) -> None:
    if stream is not None:
        with torch.cuda.stream(stream):
            model(data)
        stream.synchronize()
    else:
        model(data)
        torch.cuda.synchronize(data.device)


def main() -> None:
    args = parse_args()
    device = check_cuda(args.gpu)

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    if os.environ.get("XSCHED_AUTO_XQUEUE", "").upper() == "ON":
        print(
            "XSched: transparent mode "
            f"(scheduler={os.environ.get('XSCHED_SCHEDULER', 'GLB')}, "
            f"priority={os.environ.get('XSCHED_AUTO_XQUEUE_PRIORITY', '0')})"
        )

    print("Loading ResNet50...")
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    model.to(device)

    data = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size, device=device
    )
    stream: torch.cuda.Stream | None = None
    if not args.no_dedicated_stream:
        stream = torch.cuda.Stream(device=device)

    print(
        f"Config: batch={args.batch_size}, size={args.image_size}x{args.image_size}, "
        f"tasks={args.num_tasks}, warmup={args.warmup}, "
        f"stream={'dedicated' if stream else 'default'}"
    )

    with torch.no_grad():
        print(f"Warming up ({args.warmup} iterations)...")
        for _ in range(args.warmup):
            run_forward(model, data, stream)

        print("Starting timed inference...")
        for i in range(args.num_tasks):
            t0 = time.perf_counter()
            run_forward(model, data, stream)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Task {i} completed in {elapsed_ms:.2f} ms")

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    print("Done.")


if __name__ == "__main__":
    main()
