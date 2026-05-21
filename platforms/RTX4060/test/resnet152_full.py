#!/usr/bin/env python3
"""ResNet152 full benchmark for NVIDIA GPU + XSched transparent scheduling.

Compared to resnet152.py:
  - Uses a dedicated torch.cuda.Stream by default so kernels go through XQueue
    (required for GLB/HPF Suspend-Resume; default stream bypasses the shim).
  - Prints device / XSched env / stream mode.
  - Supports per-round thpt loop (ascend-style) or per-task timing.

Example (high priority, with xserver HPF running):
  export XSCHED_SCHEDULER=GLB XSCHED_AUTO_XQUEUE=ON XSCHED_AUTO_XQUEUE_PRIORITY=1
  export LD_LIBRARY_PATH=/path/to/xsched/output/lib:$LD_LIBRARY_PATH
  export XSCHED_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1
  export CUXTRA_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1
  python3 platforms/RTX4060/test/resnet152_full.py -c 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torchvision

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ResNet152 inference on NVIDIA GPU (RTX4060, XSched-ready)"
    )
    p.add_argument("-c", "--run-cnt", type=int, default=10,
                   help="Inferences per thpt round in loop mode (default: 10)")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
                   help=f"Input H=W (default: {DEFAULT_IMAGE_SIZE})")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warmup inferences before benchmark (default: 1)")
    p.add_argument(
        "--per-task",
        action="store_true",
        help="Print per-task latency instead of infinite thpt loop",
    )
    p.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="Timed tasks when --per-task (default: 100)",
    )
    p.add_argument(
        "--no-dedicated-stream",
        action="store_true",
        help="Use default stream (XSched scheduling will NOT apply to kernels)",
    )
    return p.parse_args()


def check_cuda(gpu: int) -> torch.device:
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Install PyTorch with CUDA support.",
              file=sys.stderr)
        sys.exit(1)
    if gpu < 0 or gpu >= torch.cuda.device_count():
        print(
            f"ERROR: --gpu {gpu} invalid ({torch.cuda.device_count()} device(s)).",
            file=sys.stderr,
        )
        sys.exit(1)
    return torch.device(f"cuda:{gpu}")


def infer_on_gpu(
    model: torch.nn.Module,
    data: torch.Tensor,
    stream: torch.cuda.Stream | None,
) -> None:
    """Forward on GPU only. Avoid .cpu() in the hot path (D2H often uses default stream)."""
    with torch.no_grad():
        if stream is not None:
            with torch.cuda.stream(stream):
                model(data)
            stream.synchronize()
        else:
            model(data)
            torch.cuda.synchronize(data.device)


def infer_once_to_cpu(
    model: torch.nn.Module,
    data: torch.Tensor,
    stream: torch.cuda.Stream | None,
) -> torch.Tensor:
    with torch.no_grad():
        if stream is not None:
            with torch.cuda.stream(stream):
                out = model(data)
            stream.synchronize()
        else:
            out = model(data)
            torch.cuda.synchronize(data.device)
        return out.cpu()


def run_thpt_loop(
    model: torch.nn.Module,
    data: torch.Tensor,
    stream: torch.cuda.Stream | None,
    batch_size: int,
    run_cnt: int,
    warmup: int,
) -> None:
    for _ in range(warmup):
        infer_on_gpu(model, data, stream)
    print(infer_once_to_cpu(model, data, stream))

    while True:
        start = time.time()
        for _ in range(run_cnt):
            infer_on_gpu(model, data, stream)
        end = time.time()
        print(f"thpt: {batch_size * run_cnt / (end - start):.2f} img/s")


def run_per_task(
    model: torch.nn.Module,
    data: torch.Tensor,
    stream: torch.cuda.Stream | None,
    num_tasks: int,
    warmup: int,
) -> None:
    for _ in range(warmup):
        infer_on_gpu(model, data, stream)

    print("Starting timed inference...")
    for i in range(num_tasks):
        t0 = time.perf_counter()
        infer_on_gpu(model, data, stream)
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"Task {i} completed in {ms:.2f} ms")


def main() -> None:
    args = parse_args()
    device = check_cuda(args.gpu)

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    if os.environ.get("XSCHED_AUTO_XQUEUE", "").upper() == "ON":
        print(
            "XSched: transparent mode "
            f"(scheduler={os.environ.get('XSCHED_SCHEDULER', 'GLB')}, "
            f"priority={os.environ.get('XSCHED_AUTO_XQUEUE_PRIORITY', '0')}, "
            f"level={os.environ.get('XSCHED_AUTO_XQUEUE_LEVEL', '1')})"
        )
        if os.environ.get("XSCHED_SCHEDULER", "").upper() != "GLB":
            print("WARN: multi-process HPF needs XSCHED_SCHEDULER=GLB and xserver HPF")
    else:
        print("XSched: not enabled (set XSCHED_AUTO_XQUEUE=ON and LD_LIBRARY_PATH)")

    cuda_lib = os.environ.get("XSCHED_CUDA_LIB") or os.environ.get("CUXTRA_CUDA_LIB")
    if cuda_lib:
        print(f"Real libcuda: {cuda_lib}")
    else:
        print(
            "WARN: set XSCHED_CUDA_LIB and CUXTRA_CUDA_LIB to real driver, e.g.\n"
            "  export XSCHED_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1\n"
            "  export CUXTRA_CUDA_LIB=/usr/lib/wsl/lib/libcuda.so.1"
        )

    stream: torch.cuda.Stream | None = None
    if not args.no_dedicated_stream:
        stream = torch.cuda.Stream(device=device)
        torch.cuda.set_stream(stream)

    print(
        f"Config: model=ResNet152, batch={args.batch_size}, "
        f"size={args.image_size}x{args.image_size}, "
        f"stream={'dedicated' if stream else 'default (no XSched on kernels)'}"
    )

    print("Loading ResNet152...")
    model = torchvision.models.resnet152(
        weights=torchvision.models.ResNet152_Weights.DEFAULT
    )
    model.eval().to(device)

    data = torch.ones(
        args.batch_size, 3, args.image_size, args.image_size, device=device
    )

    if args.per_task:
        run_per_task(model, data, stream, args.num_tasks, args.warmup)
        print("Done.")
    else:
        print(f"Loop mode: run_cnt={args.run_cnt} per round (Ctrl+C to stop)")
        run_thpt_loop(
            model, data, stream, args.batch_size, args.run_cnt, args.warmup
        )


if __name__ == "__main__":
    main()
