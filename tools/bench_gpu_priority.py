#!/usr/bin/env python3
"""GPU priority benchmark for XSched.

This script builds contest-like workloads:
  - CNN inference: ResNet50
  - Transformer inference: synthetic TransformerEncoder
  - Batch training: MobileNetV2

It runs baseline, colocated baseline, XSched HPF, and XSched+SchedUM scenarios,
then writes CSV/JSON summaries and PNG charts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
RESULTS = ROOT / "benchmark_results"
REAL_LIBCUDA = "/usr/lib/wsl/lib/libcuda.so.1"
SUSPEND_RE = re.compile(r"suspend_us=(\d+)")


def ensure_cuda_shim_links() -> None:
    shim_dir = BUILD / "platforms/RTX4060"
    shim_name = "libshimrtx4060.so"
    for name in ["libcuda.so.1", "libcuda.so"]:
        link = shim_dir / name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(shim_name)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    pos = (len(data) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    frac = pos - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def check_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")


def dedicated_stream() -> torch.cuda.Stream:
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    return stream


def sync_stream(stream: torch.cuda.Stream | None) -> None:
    if stream is None:
        torch.cuda.synchronize()
    else:
        stream.synchronize()


def wait_start(ready_file: str = "", start_file: str = "") -> None:
    if ready_file:
        Path(ready_file).write_text("ready\n")
    if start_file:
        start = Path(start_file)
        while not start.exists():
            time.sleep(0.01)


def run_periodic_requests(duration: float, period: float, discard: int, run_once) -> list[float]:
    latencies = []
    measured = []
    end = time.perf_counter() + duration
    next_release = time.perf_counter()
    while time.perf_counter() < end:
        now = time.perf_counter()
        if now < next_release:
            time.sleep(next_release - now)
        t0 = time.perf_counter()
        run_once()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)
        if len(latencies) > discard:
            measured.append(latency_ms)
        next_release += period
    return measured


def run_resnet50_infer(duration: float, batch_size: int, warmup: int,
                       ready_file: str = "", start_file: str = "",
                       period: float = 0.0, discard: int = 0) -> dict[str, Any]:
    check_cuda()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    stream = dedicated_stream()
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval().to(device)
    data = torch.randn(batch_size, 3, 224, 224, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            with torch.cuda.stream(stream):
                model(data)
            stream.synchronize()

        wait_start(ready_file, start_file)

        def once() -> None:
            with torch.cuda.stream(stream):
                model(data)
            stream.synchronize()

        if period > 0:
            latencies = run_periodic_requests(duration, period, discard, once)
        else:
            latencies = []
            end = time.perf_counter() + duration
            while time.perf_counter() < end:
                t0 = time.perf_counter()
                once()
                latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "role": "hp_resnet50",
        "period_s": period,
        "discard": discard,
        "iterations": len(latencies),
        "latencies_ms": latencies,
        "avg_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "throughput_img_s": len(latencies) * batch_size / duration,
    }


def run_transformer_infer(duration: float, batch_size: int, warmup: int,
                          ready_file: str = "", start_file: str = "",
                          period: float = 0.0, discard: int = 0) -> dict[str, Any]:
    check_cuda()
    device = torch.device("cuda:0")
    stream = dedicated_stream()
    torch.manual_seed(1)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=768, nhead=12, dim_feedforward=3072, batch_first=True,
        dropout=0.0, activation="gelu",
    )
    model = torch.nn.TransformerEncoder(layer, num_layers=6).eval().to(device)
    data = torch.randn(batch_size, 128, 768, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            with torch.cuda.stream(stream):
                model(data)
            stream.synchronize()

        wait_start(ready_file, start_file)

        def once() -> None:
            with torch.cuda.stream(stream):
                model(data)
            stream.synchronize()

        if period > 0:
            latencies = run_periodic_requests(duration, period, discard, once)
        else:
            latencies = []
            end = time.perf_counter() + duration
            while time.perf_counter() < end:
                t0 = time.perf_counter()
                once()
                latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "role": "hp_transformer",
        "period_s": period,
        "discard": discard,
        "iterations": len(latencies),
        "latencies_ms": latencies,
        "avg_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "throughput_seq_s": len(latencies) * batch_size / duration,
    }


def run_mobilenet_train(duration: float, batch_size: int, warmup: int,
                        sync_interval: int) -> dict[str, Any]:
    check_cuda()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    stream = dedicated_stream()
    model = torchvision.models.mobilenet_v2(weights=None, num_classes=1000).train().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    data = torch.randn(batch_size, 3, 224, 224, device=device)
    target = torch.randint(0, 1000, (batch_size,), device=device)

    def step() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
        return loss.detach()

    last_loss = None
    for _ in range(warmup):
        with torch.cuda.stream(stream):
            last_loss = step()
        stream.synchronize()

    steps = 0
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        for _ in range(sync_interval):
            with torch.cuda.stream(stream):
                last_loss = step()
            steps += 1
            if time.perf_counter() >= end:
                break
        stream.synchronize()

    last_loss_value = float(last_loss.cpu()) if last_loss is not None else 0.0

    return {
        "role": "lp_mobilenetv2_train",
        "sync_interval": sync_interval,
        "iterations": steps,
        "throughput_img_s": steps * batch_size / duration,
        "last_loss": last_loss_value,
    }


def worker_main(args: argparse.Namespace) -> None:
    if args.role == "hp_resnet50":
        result = run_resnet50_infer(args.duration, args.batch_size, args.warmup,
                                    args.ready_file, args.start_file, args.period, args.discard)
    elif args.role == "hp_transformer":
        result = run_transformer_infer(args.duration, args.batch_size, args.warmup,
                                       args.ready_file, args.start_file, args.period, args.discard)
    elif args.role == "lp_mobilenetv2_train":
        result = run_mobilenet_train(args.duration, args.batch_size, args.warmup,
                                     args.sync_interval)
    else:
        raise ValueError(f"unknown role {args.role}")
    print("JSON_RESULT " + json.dumps(result), flush=True)


class Monitor:
    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        self.stop.set()
        self.thread.join()

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ], text=True, stderr=subprocess.DEVNULL, timeout=2)
                fields = out.strip().splitlines()[0].split(",")
                self.samples.append({
                    "gpu_util_pct": float(fields[0].strip()),
                    "mem_used_mib": float(fields[1].strip()),
                })
            except Exception:
                pass
            time.sleep(0.5)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {
                "gpu_util_avg_pct": 0.0,
                "mem_used_avg_mib": 0.0,
                "mem_used_max_mib": 0.0,
            }
        return {
            "gpu_util_avg_pct": statistics.mean(s["gpu_util_pct"] for s in self.samples),
            "mem_used_avg_mib": statistics.mean(s["mem_used_mib"] for s in self.samples),
            "mem_used_max_mib": max(s["mem_used_mib"] for s in self.samples),
        }


def merged_env(use_xsched: bool, priority: int, schedum: bool,
               threshold: int | None = None, batch_size: int | None = None,
               xqueue_level: int = 1, sync_suspend: bool = False,
               profile_suspend: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    if not use_xsched:
        env.pop("LD_PRELOAD", None)
        env.pop("XSCHED_AUTO_XQUEUE", None)
        env.pop("XSCHED_SCHEDULER", None)
        env.pop("XSCHED_CUDA_MEM_OVERSUB", None)
        return env

    ensure_cuda_shim_links()
    lib_paths = [
        str(BUILD / "preempt"),
        str(BUILD / "platforms/RTX4060"),
        "/home/cyk/miniconda3/lib",
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(p for p in lib_paths if p)
    env["LD_PRELOAD"] = str(BUILD / "platforms/RTX4060/libshimrtx4060.so")
    env["XSCHED_SCHEDULER"] = "GLB"
    env["XSCHED_AUTO_XQUEUE"] = "ON"
    env["XSCHED_AUTO_XQUEUE_LEVEL"] = str(xqueue_level)
    env["XSCHED_AUTO_XQUEUE_PRIORITY"] = str(priority)
    if sync_suspend:
        env["XSCHED_SYNC_SUSPEND"] = "1"
    else:
        env.pop("XSCHED_SYNC_SUSPEND", None)
    if profile_suspend:
        env["XSCHED_PROFILE_SUSPEND"] = "1"
    else:
        env.pop("XSCHED_PROFILE_SUSPEND", None)
    if threshold is not None:
        env["XSCHED_AUTO_XQUEUE_THRESHOLD"] = str(threshold)
    else:
        env.pop("XSCHED_AUTO_XQUEUE_THRESHOLD", None)
    if batch_size is not None:
        env["XSCHED_AUTO_XQUEUE_BATCH_SIZE"] = str(batch_size)
    else:
        env.pop("XSCHED_AUTO_XQUEUE_BATCH_SIZE", None)
    env["XSCHED_CUDA_LIB"] = REAL_LIBCUDA
    env["CUXTRA_CUDA_LIB"] = REAL_LIBCUDA
    if schedum:
        env["XSCHED_CUDA_MEM_OVERSUB"] = "1"
    else:
        env.pop("XSCHED_CUDA_MEM_OVERSUB", None)
    return env


def start_xserver(log_path: Path | None = None) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([
        str(BUILD / "preempt"),
        str(BUILD / "platforms/RTX4060"),
        env.get("LD_LIBRARY_PATH", ""),
    ])
    log_file = None
    stdout = subprocess.PIPE
    if log_path is not None:
        log_file = log_path.open("w")
        stdout = log_file
    proc = subprocess.Popen(
        [str(BUILD / "service/xserver"), "HPF", "50000"],
        cwd=str(ROOT),
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    proc._xsched_log_file = log_file  # type: ignore[attr-defined]
    time.sleep(1.0)
    return proc


def stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        if proc is not None:
            log_file = getattr(proc, "_xsched_log_file", None)
            if log_file is not None:
                log_file.close()
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    log_file = getattr(proc, "_xsched_log_file", None)
    if log_file is not None:
        log_file.close()


def run_worker(role: str, duration: float, batch_size: int, warmup: int,
               env: dict[str, str], ready_file: Path | None = None,
               start_file: Path | None = None,
               period: float = 0.0, discard: int = 0,
               sync_interval: int = 1) -> subprocess.Popen[str]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "worker",
        "--role", role, "--duration", str(duration),
        "--batch-size", str(batch_size), "--warmup", str(warmup),
    ]
    if ready_file is not None:
        cmd += ["--ready-file", str(ready_file)]
    if start_file is not None:
        cmd += ["--start-file", str(start_file)]
    if period > 0:
        cmd += ["--period", str(period)]
    if discard > 0:
        cmd += ["--discard", str(discard)]
    if role == "lp_mobilenetv2_train":
        cmd += ["--sync-interval", str(sync_interval)]
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def wait_file(path: Path, proc: subprocess.Popen[str], timeout: float) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if path.exists():
            return
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"process exited before {path} appeared\n{out}")
        time.sleep(0.05)
    raise TimeoutError(f"timeout waiting for {path}")


def collect_result(proc: subprocess.Popen[str]) -> tuple[dict[str, Any], str]:
    out, _ = proc.communicate()
    result: dict[str, Any] = {}
    for line in out.splitlines():
        if line.startswith("JSON_RESULT "):
            result = json.loads(line[len("JSON_RESULT "):])
    if not result:
        raise RuntimeError(f"worker failed or produced no JSON_RESULT\n{out}")
    return result, out


def suspend_summary(*logs: str) -> dict[str, float]:
    values = [float(m.group(1)) for log in logs for m in SUSPEND_RE.finditer(log)]
    if not values:
        return {
            "suspend_count": 0.0,
            "suspend_avg_us": 0.0,
            "suspend_p50_us": 0.0,
            "suspend_p95_us": 0.0,
            "suspend_p99_us": 0.0,
        }
    return {
        "suspend_count": float(len(values)),
        "suspend_avg_us": statistics.mean(values),
        "suspend_p50_us": percentile(values, 50),
        "suspend_p95_us": percentile(values, 95),
        "suspend_p99_us": percentile(values, 99),
    }


def run_case(name: str, hp_role: str, use_xsched: bool, schedum: bool,
             with_lp: bool, duration: float, out_dir: Path) -> dict[str, Any]:
    xserver = start_xserver(out_dir / f"{name}.xserver.log") if use_xsched else None
    monitor = Monitor()
    monitor.start()
    try:
        lp_proc = None
        if with_lp:
            lp_proc = run_worker(
                "lp_mobilenetv2_train", duration + 3.0, 16, 2,
                merged_env(use_xsched, -10, schedum),
            )
            time.sleep(3.0)

        hp_batch = 32 if hp_role == "hp_resnet50" else 8
        hp_proc = run_worker(hp_role, duration, hp_batch, 3,
                             merged_env(use_xsched, 10, schedum))
        hp_result, hp_log = collect_result(hp_proc)
        lp_result = {}
        lp_log = ""
        if lp_proc is not None:
            lp_result, lp_log = collect_result(lp_proc)
    finally:
        monitor.join()
        stop_process(xserver)

    (out_dir / f"{name}.hp.log").write_text(hp_log)
    if lp_log:
        (out_dir / f"{name}.lp.log").write_text(lp_log)

    row = {
        "case": name,
        "hp_role": hp_role,
        "xsched": use_xsched,
        "schedum": schedum,
        "with_lp": with_lp,
        "hp_iterations": hp_result.get("iterations", 0),
        "hp_avg_ms": hp_result.get("avg_ms", 0.0),
        "hp_p50_ms": hp_result.get("p50_ms", 0.0),
        "hp_p95_ms": hp_result.get("p95_ms", 0.0),
        "hp_p99_ms": hp_result.get("p99_ms", 0.0),
        "hp_throughput": hp_result.get("throughput_img_s",
                                        hp_result.get("throughput_seq_s", 0.0)),
        "lp_iterations": lp_result.get("iterations", 0),
        "lp_throughput_img_s": lp_result.get("throughput_img_s", 0.0),
    }
    row.update(monitor.summary())
    row.update(suspend_summary(hp_log, lp_log))
    return row


def run_arrival_case(name: str, hp_role: str, use_xsched: bool, schedum: bool,
                     duration: float, out_dir: Path,
                     lp_threshold: int | None = None,
                     lp_batch_size: int | None = None) -> dict[str, Any]:
    xserver = start_xserver(out_dir / f"{name}.xserver.log") if use_xsched else None
    monitor = Monitor()
    monitor.start()
    ready_file = out_dir / f"{name}.hp.ready"
    start_file = out_dir / f"{name}.hp.start"
    ready_file.unlink(missing_ok=True)
    start_file.unlink(missing_ok=True)

    try:
        hp_batch = 32 if hp_role == "hp_resnet50" else 8
        hp_proc = run_worker(
            hp_role, duration, hp_batch, 3,
            merged_env(use_xsched, 10, schedum),
            ready_file=ready_file, start_file=start_file,
        )
        wait_file(ready_file, hp_proc, timeout=60.0)

        lp_proc = run_worker(
            "lp_mobilenetv2_train", duration + 8.0, 16, 2,
            merged_env(use_xsched, -10, schedum, lp_threshold, lp_batch_size),
        )
        time.sleep(5.0)

        trigger_time = time.perf_counter()
        start_file.write_text("start\n")
        hp_result, hp_log = collect_result(hp_proc)
        hp_done_time = time.perf_counter()
        lp_result, lp_log = collect_result(lp_proc)
    finally:
        monitor.join()
        stop_process(xserver)

    (out_dir / f"{name}.hp.log").write_text(hp_log)
    (out_dir / f"{name}.lp.log").write_text(lp_log)
    first_ms = 0.0
    if hp_result.get("latencies_ms"):
        first_ms = float(hp_result["latencies_ms"][0])
    row = {
        "case": name,
        "hp_role": hp_role,
        "xsched": use_xsched,
        "schedum": schedum,
        "lp_threshold": lp_threshold or 16,
        "lp_batch_size": lp_batch_size or 8,
        "with_lp": True,
        "mode": "arrival",
        "hp_iterations": hp_result.get("iterations", 0),
        "hp_first_ms": first_ms,
        "hp_wall_ms": (hp_done_time - trigger_time) * 1000.0,
        "hp_avg_ms": hp_result.get("avg_ms", 0.0),
        "hp_p50_ms": hp_result.get("p50_ms", 0.0),
        "hp_p95_ms": hp_result.get("p95_ms", 0.0),
        "hp_p99_ms": hp_result.get("p99_ms", 0.0),
        "hp_throughput": hp_result.get("throughput_img_s",
                                        hp_result.get("throughput_seq_s", 0.0)),
        "lp_iterations": lp_result.get("iterations", 0),
        "lp_throughput_img_s": lp_result.get("throughput_img_s", 0.0),
    }
    row.update(monitor.summary())
    row.update(suspend_summary(hp_log, lp_log))
    return row


def run_periodic_case(name: str, hp_role: str, use_xsched: bool, xqueue_level: int,
                      with_lp: bool, lp_duration: float, lp_warmup_s: float,
                      hp_period: float, hp_batch: int, lp_batch: int,
                      discard: int, lp_sync_interval: int,
                      sync_suspend: bool, profile_suspend: bool,
                      out_dir: Path) -> dict[str, Any]:
    xserver = start_xserver(out_dir / f"{name}.xserver.log") if use_xsched else None
    monitor = Monitor()
    monitor.start()
    ready_file = out_dir / f"{name}.hp.ready"
    start_file = out_dir / f"{name}.hp.start"
    ready_file.unlink(missing_ok=True)
    start_file.unlink(missing_ok=True)
    hp_duration = max(1.0, lp_duration - lp_warmup_s)
    lp_proc = None
    lp_result: dict[str, Any] = {}
    lp_log = ""

    try:
        hp_proc = run_worker(
            hp_role, hp_duration, hp_batch, 3,
            merged_env(use_xsched, 10, False, xqueue_level=xqueue_level,
                       sync_suspend=sync_suspend, profile_suspend=profile_suspend),
            ready_file=ready_file, start_file=start_file, period=hp_period,
            discard=discard,
        )
        wait_file(ready_file, hp_proc, timeout=90.0)

        if with_lp:
            lp_proc = run_worker(
                "lp_mobilenetv2_train", lp_duration, lp_batch, 2,
                merged_env(use_xsched, -10, False, xqueue_level=xqueue_level,
                           sync_suspend=sync_suspend, profile_suspend=profile_suspend),
                sync_interval=lp_sync_interval,
            )
            time.sleep(lp_warmup_s)

        trigger_time = time.perf_counter()
        start_file.write_text("start\n")
        hp_result, hp_log = collect_result(hp_proc)
        hp_done_time = time.perf_counter()
        if lp_proc is not None:
            lp_result, lp_log = collect_result(lp_proc)
    finally:
        monitor.join()
        stop_process(xserver)

    (out_dir / f"{name}.hp.log").write_text(hp_log)
    if lp_log:
        (out_dir / f"{name}.lp.log").write_text(lp_log)

    first_ms = 0.0
    if hp_result.get("latencies_ms"):
        first_ms = float(hp_result["latencies_ms"][0])
    row = {
        "case": name,
        "hp_role": hp_role,
        "mode": "periodic",
        "xsched": use_xsched,
        "xqueue_level": xqueue_level if use_xsched else 0,
        "schedum": False,
        "with_lp": with_lp,
        "hp_period_s": hp_period,
        "hp_batch": hp_batch,
        "hp_discard": discard,
        "lp_batch": lp_batch if with_lp else 0,
        "lp_sync_interval": lp_sync_interval if with_lp else 0,
        "sync_suspend": sync_suspend if use_xsched else False,
        "profile_suspend": profile_suspend if use_xsched else False,
        "lp_duration_s": lp_duration if with_lp else 0.0,
        "lp_warmup_s": lp_warmup_s if with_lp else 0.0,
        "hp_iterations": hp_result.get("iterations", 0),
        "hp_first_ms": first_ms,
        "hp_wall_ms": (hp_done_time - trigger_time) * 1000.0,
        "hp_avg_ms": hp_result.get("avg_ms", 0.0),
        "hp_p50_ms": hp_result.get("p50_ms", 0.0),
        "hp_p95_ms": hp_result.get("p95_ms", 0.0),
        "hp_p99_ms": hp_result.get("p99_ms", 0.0),
        "hp_throughput": hp_result.get("throughput_img_s",
                                        hp_result.get("throughput_seq_s", 0.0)),
        "lp_iterations": lp_result.get("iterations", 0),
        "lp_throughput_img_s": lp_result.get("throughput_img_s", 0.0),
    }
    row.update(monitor.summary())
    row.update(suspend_summary(hp_log, lp_log))
    return row


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN: matplotlib unavailable, skip charts: {exc}", file=sys.stderr)
        return

    for hp_role in sorted(set(r["hp_role"] for r in rows)):
        part = [r for r in rows if r["hp_role"] == hp_role]
        labels = [r["case"].replace(f"{hp_role}_", "") for r in part]
        x = range(len(part))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - 0.2 for i in x], [r["hp_avg_ms"] for r in part], width=0.4,
               label="avg")
        ax.bar([i + 0.2 for i in x], [r["hp_p95_ms"] for r in part], width=0.4,
               label="p95")
        ax.set_title(f"{hp_role} latency")
        ax.set_ylabel("latency (ms)")
        ax.set_xticks(list(x), labels, rotation=20, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{hp_role}_latency.png", dpi=160)
        plt.close(fig)

        if any("hp_first_ms" in r for r in part):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(labels, [r.get("hp_first_ms", 0.0) for r in part], color="#f58518")
            ax.set_title(f"{hp_role} first-request latency")
            ax.set_ylabel("first request latency (ms)")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            fig.savefig(out_dir / f"{hp_role}_first_latency.png", dpi=160)
            plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(labels, [r["lp_throughput_img_s"] for r in part], color="#4c78a8")
        ax1.set_ylabel("LP MobileNetV2 train throughput (img/s)")
        ax1.set_title(f"{hp_role} colocated LP throughput")
        ax1.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / f"{hp_role}_lp_throughput.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [r["case"] for r in rows]
    ax.bar(labels, [r["gpu_util_avg_pct"] for r in rows])
    ax.set_ylabel("average GPU util (%)")
    ax.set_title("GPU utilization")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "gpu_utilization.png", dpi=160)
    plt.close(fig)


def bench_main(args: argparse.Namespace) -> None:
    out_dir = RESULTS / time.strftime("gpu_priority_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    hp_roles = ["hp_resnet50", "hp_transformer"]
    for hp_role in hp_roles:
        rows.append(run_case(f"{hp_role}_alone", hp_role, False, False, False,
                             args.duration, out_dir))
        rows.append(run_case(f"{hp_role}_baseline_colocated", hp_role, False, False, True,
                             args.duration, out_dir))
        rows.append(run_case(f"{hp_role}_xsched_hpf", hp_role, True, False, True,
                             args.duration, out_dir))
        rows.append(run_case(f"{hp_role}_xsched_schedum", hp_role, True, True, True,
                             args.duration, out_dir))
        write_outputs(rows, out_dir)
    write_outputs(rows, out_dir)
    print(f"RESULT_DIR {out_dir}")
    print((out_dir / "summary.csv").read_text())


def arrival_main(args: argparse.Namespace) -> None:
    out_dir = RESULTS / time.strftime("gpu_arrival_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for hp_role in ["hp_resnet50", "hp_transformer"]:
        rows.append(run_arrival_case(f"{hp_role}_arrival_baseline",
                                     hp_role, False, False, args.duration, out_dir))
        rows.append(run_arrival_case(f"{hp_role}_arrival_xsched_hpf",
                                     hp_role, True, False, args.duration, out_dir))
        rows.append(run_arrival_case(f"{hp_role}_arrival_xsched_schedum",
                                     hp_role, True, True, args.duration, out_dir))
        write_outputs(rows, out_dir)
    write_outputs(rows, out_dir)
    print(f"RESULT_DIR {out_dir}")
    print((out_dir / "summary.csv").read_text())


def threshold_main(args: argparse.Namespace) -> None:
    out_dir = RESULTS / time.strftime("gpu_threshold_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    configs = [
        ("default_16_8", None, None),
        ("lp_8_4", 8, 4),
        ("lp_4_2", 4, 2),
        ("lp_2_1", 2, 1),
    ]
    for hp_role in ["hp_resnet50", "hp_transformer"]:
        rows.append(run_arrival_case(f"{hp_role}_baseline_no_xsched",
                                     hp_role, False, False, args.duration, out_dir))
        for suffix, threshold, batch_size in configs:
            rows.append(run_arrival_case(f"{hp_role}_xsched_{suffix}",
                                         hp_role, True, False, args.duration, out_dir,
                                         lp_threshold=threshold, lp_batch_size=batch_size))
            write_outputs(rows, out_dir)
    write_outputs(rows, out_dir)
    print(f"RESULT_DIR {out_dir}")
    print((out_dir / "summary.csv").read_text())


def periodic_main(args: argparse.Namespace) -> None:
    out_dir = RESULTS / time.strftime("gpu_periodic_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    hp_roles = ["hp_resnet50", "hp_transformer"]
    for hp_role in hp_roles:
        hp_batch = args.resnet_batch if hp_role == "hp_resnet50" else args.transformer_batch
        rows.append(run_periodic_case(
            f"{hp_role}_exclusive", hp_role, False, 0, False,
            args.lp_duration, args.lp_warmup, args.hp_period, hp_batch, args.lp_batch,
            args.discard, args.lp_sync_interval, args.sync_suspend, args.profile_suspend, out_dir,
        ))
        rows.append(run_periodic_case(
            f"{hp_role}_native", hp_role, False, 0, True,
            args.lp_duration, args.lp_warmup, args.hp_period, hp_batch, args.lp_batch,
            args.discard, args.lp_sync_interval, args.sync_suspend, args.profile_suspend, out_dir,
        ))
        rows.append(run_periodic_case(
            f"{hp_role}_xsched_lv1", hp_role, True, 1, True,
            args.lp_duration, args.lp_warmup, args.hp_period, hp_batch, args.lp_batch,
            args.discard, args.lp_sync_interval, args.sync_suspend, args.profile_suspend, out_dir,
        ))
        rows.append(run_periodic_case(
            f"{hp_role}_xsched_lv2", hp_role, True, 2, True,
            args.lp_duration, args.lp_warmup, args.hp_period, hp_batch, args.lp_batch,
            args.discard, args.lp_sync_interval, args.sync_suspend, args.profile_suspend, out_dir,
        ))
        write_outputs(rows, out_dir)
    write_outputs(rows, out_dir)
    print(f"RESULT_DIR {out_dir}")
    print((out_dir / "summary.csv").read_text())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--role", required=True,
                   choices=["hp_resnet50", "hp_transformer", "lp_mobilenetv2_train"])
    w.add_argument("--duration", type=float, default=20.0)
    w.add_argument("--batch-size", type=int, default=32)
    w.add_argument("--warmup", type=int, default=3)
    w.add_argument("--ready-file", default="")
    w.add_argument("--start-file", default="")
    w.add_argument("--period", type=float, default=0.0)
    w.add_argument("--discard", type=int, default=0)
    w.add_argument("--sync-interval", type=int, default=1)
    b = sub.add_parser("bench")
    b.add_argument("--duration", type=float, default=20.0)
    a = sub.add_parser("arrival")
    a.add_argument("--duration", type=float, default=12.0)
    t = sub.add_parser("threshold")
    t.add_argument("--duration", type=float, default=10.0)
    pe = sub.add_parser("periodic")
    pe.add_argument("--lp-duration", type=float, default=60.0)
    pe.add_argument("--lp-warmup", type=float, default=5.0)
    pe.add_argument("--hp-period", type=float, default=2.0)
    pe.add_argument("--resnet-batch", type=int, default=8)
    pe.add_argument("--transformer-batch", type=int, default=4)
    pe.add_argument("--lp-batch", type=int, default=16)
    pe.add_argument("--lp-sync-interval", type=int, default=8)
    pe.add_argument("--discard", type=int, default=5)
    pe.add_argument("--sync-suspend", action="store_true")
    pe.add_argument("--profile-suspend", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "worker":
        worker_main(args)
    elif args.cmd == "bench":
        bench_main(args)
    elif args.cmd == "arrival":
        arrival_main(args)
    elif args.cmd == "threshold":
        threshold_main(args)
    elif args.cmd == "periodic":
        periodic_main(args)


if __name__ == "__main__":
    main()
