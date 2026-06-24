#!/usr/bin/env python3
"""Foreground latency CDF benchmark for RTX4060 XSched experiments.

This follows the paper-style comparison:
1. foreground alone;
2. foreground colocated with a background workload on the native scheduler;
3. foreground colocated with a background workload under XSched LV1;
4. foreground colocated with a background workload under XSched LV2.

The foreground task records per-batch inference latency, so p50/p95/p99 and
slowdown versus the alone case can be reported directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_LIB = ROOT / "output/lib"
XSERVER = ROOT / "output/bin/xserver"
XCLI = ROOT / "output/bin/xcli"
RESULTS = ROOT / "benchmark_results"
DEFAULT_REAL_LIBCUDA = "/usr/lib/wsl/lib/libcuda.so.1"
LAT_RE = re.compile(r"latency_ms:\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass
class ProcSpec:
    role: str
    scenario: str
    priority: int
    index: int
    proc: subprocess.Popen[str]
    log_file: TextIO
    start_elapsed_s: float
    records: list[dict[str, object]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure foreground inference latency CDF with native vs XSched scheduling."
    )
    p.add_argument("--worker", choices=["foreground", "background"], help=argparse.SUPPRESS)
    p.add_argument("--scenario", choices=["alone", "native", "xsched", "xsched_lv2", "all"],
                   default="all")
    p.add_argument("--requests", type=int, default=200,
                   help="Number of foreground measured batches.")
    p.add_argument("--warmup", type=int, default=10,
                   help="Foreground warmup batches before recording latency.")
    p.add_argument("--foreground-workload", choices=["resnet50", "transformer"],
                   default="resnet50",
                   help="Foreground high-priority inference workload.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--hp-delay", type=float, default=10.0,
                   help="Seconds to let background training run before starting foreground.")
    p.add_argument("--background-count", type=int, default=1,
                   help="Number of low-priority background training processes.")
    p.add_argument("--xqueue-level", type=int, default=1)
    p.add_argument("--threshold", type=int, default=16)
    p.add_argument("--batch-commands", type=int, default=8)
    p.add_argument("--policy", default="HPF")
    p.add_argument("--port", type=int, default=50000)
    p.add_argument("--real-libcuda", default=DEFAULT_REAL_LIBCUDA)
    p.add_argument("--result-dir", type=Path)
    p.add_argument("--no-start-xserver", action="store_true")
    p.add_argument("--restart-xserver", action="store_true")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def make_result_dir(user_dir: Path | None) -> Path:
    if user_dir is not None:
        out = user_dir
    else:
        out = RESULTS / f"rtx4060_latency_cdf_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def prepend_path(value: str, prefix: Path) -> str:
    return f"{prefix}:{value}" if value else str(prefix)


def child_env(args: argparse.Namespace, scenario: str, priority: int) -> dict[str, str]:
    env = os.environ.copy()
    if not scenario.startswith("xsched"):
        for key in list(env):
            if key.startswith("XSCHED_") or key == "CUXTRA_CUDA_LIB":
                env.pop(key, None)
        env.pop("LD_PRELOAD", None)
        if "LD_LIBRARY_PATH" in env:
            paths = [p for p in env["LD_LIBRARY_PATH"].split(":") if p != str(OUTPUT_LIB)]
            env["LD_LIBRARY_PATH"] = ":".join(paths)
        return env

    env["XSCHED_CUDA_LIB"] = args.real_libcuda
    env["CUXTRA_CUDA_LIB"] = args.real_libcuda
    env["LD_LIBRARY_PATH"] = prepend_path(env.get("LD_LIBRARY_PATH", ""), OUTPUT_LIB)
    env.pop("LD_PRELOAD", None)
    env.setdefault("XLOG_LEVEL", "INFO")
    env["XSCHED_SCHEDULER"] = "GLB"
    env["XSCHED_AUTO_XQUEUE"] = "ON"
    env["XSCHED_AUTO_XQUEUE_PRIORITY"] = str(priority)
    level = 2 if scenario == "xsched_lv2" else args.xqueue_level
    env["XSCHED_AUTO_XQUEUE_LEVEL"] = str(level)
    env["XSCHED_AUTO_XQUEUE_THRESHOLD"] = str(args.threshold)
    env["XSCHED_AUTO_XQUEUE_BATCH_SIZE"] = str(args.batch_commands)
    return env


def check_xserver(args: argparse.Namespace) -> bool:
    try:
        res = subprocess.run(
            [str(XCLI), "--port", str(args.port), "policy", "-q"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
        return res.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def stop_existing_xserver() -> None:
    try:
        res = subprocess.run(
            ["pgrep", "-f", str(XSERVER)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return
    pids = [int(line) for line in res.stdout.splitlines() if line.strip().isdigit()]
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + 3.0
    while time.time() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(0.1)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def start_xserver(args: argparse.Namespace, result_dir: Path) -> subprocess.Popen[str] | None:
    if args.restart_xserver:
        stop_existing_xserver()
    if check_xserver(args):
        return None
    if args.no_start_xserver:
        raise RuntimeError("xserver is not responding; start output/bin/xserver first")

    log = (result_dir / "xserver.log").open("w", buffering=1)
    env = os.environ.copy()
    env.setdefault("XLOG_LEVEL", "INFO")
    proc = subprocess.Popen(
        [str(XSERVER), args.policy, str(args.port)],
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        proc.stdin.write("y\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError, AttributeError):
        pass

    deadline = time.time() + 15.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"xserver exited early with code {proc.returncode}")
        if check_xserver(args):
            return proc
        time.sleep(0.1)
    raise RuntimeError("xserver did not become ready within 15 seconds")


def terminate_proc(proc: subprocess.Popen[str], timeout: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=timeout)


def worker_cmd(args: argparse.Namespace, role: str, scenario: str) -> list[str]:
    cmd = [
        args.python,
        str(Path(__file__).resolve()),
        "--worker",
        role,
        "--scenario",
        scenario,
        "--requests",
        str(args.requests),
        "--warmup",
        str(args.warmup),
        "--batch-size",
        str(args.batch_size),
        "--train-batch-size",
        str(args.train_batch_size),
        "--image-size",
        str(args.image_size),
        "--gpu",
        str(args.gpu),
    ]
    return cmd


def launch_worker(args: argparse.Namespace, scenario: str, role: str, priority: int,
                  index: int,
                  result_dir: Path, t0: float, rows: list[dict[str, object]],
                  lock: threading.Lock) -> tuple[ProcSpec, threading.Thread]:
    log_file = (result_dir / f"{scenario}_{role}_{index}.log").open("w", buffering=1)
    proc = subprocess.Popen(
        worker_cmd(args, role, scenario),
        cwd=ROOT,
        env=child_env(args, scenario, priority),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    spec = ProcSpec(role=role, scenario=scenario, priority=priority, index=index, proc=proc,
                    log_file=log_file, start_elapsed_s=round(time.time() - t0, 3))
    thread = threading.Thread(target=stream_worker, args=(spec, t0, rows, lock), daemon=True)
    thread.start()
    print(f"started {scenario} {role}[{index}] priority={priority} pid={proc.pid}",
          flush=True)
    return spec, thread


def stream_worker(spec: ProcSpec, t0: float, rows: list[dict[str, object]],
                  lock: threading.Lock) -> None:
    assert spec.proc.stdout is not None
    seq = 0
    for line in spec.proc.stdout:
        spec.log_file.write(line)
        match = LAT_RE.search(line)
        if not match:
            continue
        seq += 1
        row = {
            "elapsed_s": round(time.time() - t0, 3),
            "scenario": spec.scenario,
            "role": spec.role,
            "index": spec.index,
            "priority": spec.priority,
            "pid": spec.proc.pid,
            "seq": seq,
            "latency_ms": float(match.group(1)),
        }
        spec.records.append(row)
        with lock:
            rows.append(row)
        print(
            f"{row['elapsed_s']:>7.3f}s {spec.scenario} {spec.role}[{spec.index}] "
            f"latency={row['latency_ms']:.3f} ms",
            flush=True,
        )


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def summarize_scenario(scenario: str, hp: ProcSpec,
                       backgrounds: list[ProcSpec]) -> dict[str, object]:
    vals = [float(r["latency_ms"]) for r in hp.records]
    return {
        "scenario": scenario,
        "foreground_samples": len(vals),
        "foreground_batch_size": None,
        "latency_avg_ms": round(sum(vals) / len(vals), 3) if vals else 0.0,
        "latency_p50_ms": round(percentile(vals, 50), 3),
        "latency_p95_ms": round(percentile(vals, 95), 3),
        "latency_p99_ms": round(percentile(vals, 99), 3),
        "latency_min_ms": round(min(vals), 3) if vals else 0.0,
        "latency_max_ms": round(max(vals), 3) if vals else 0.0,
        "foreground_returncode": hp.proc.returncode,
        "background_count": len(backgrounds),
        "background_pids": [bg.proc.pid for bg in backgrounds],
        "background_returncodes": [bg.proc.returncode for bg in backgrounds],
    }


def write_latency_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["elapsed_s", "scenario", "role", "index", "priority", "pid", "seq",
                        "latency_ms"],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (str(r["scenario"]), int(r["seq"]))))


def run_one(args: argparse.Namespace, scenario: str, result_dir: Path) -> dict[str, object]:
    scenario_dir = result_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []
    specs: list[ProcSpec] = []
    xserver_proc: subprocess.Popen[str] | None = None
    t0 = time.time()

    try:
        if scenario.startswith("xsched"):
            xserver_proc = start_xserver(args, scenario_dir)

        backgrounds: list[ProcSpec] = []
        if scenario != "alone":
            for i in range(args.background_count):
                background, thread = launch_worker(args, scenario, "background", 0, i,
                                                   scenario_dir, t0, rows, lock)
                backgrounds.append(background)
                specs.append(background)
                threads.append(thread)
                time.sleep(0.5)
            print(f"waiting {args.hp_delay:.1f}s before foreground", flush=True)
            time.sleep(args.hp_delay)

        foreground, thread = launch_worker(args, scenario, "foreground", 1, 0,
                                           scenario_dir, t0, rows, lock)
        specs.append(foreground)
        threads.append(thread)
        foreground.proc.wait()
    finally:
        for spec in specs:
            terminate_proc(spec.proc)
        for thread in threads:
            thread.join(timeout=2.0)
        for spec in specs:
            spec.log_file.close()
        if xserver_proc is not None:
            terminate_proc(xserver_proc)

    with lock:
        final_rows = list(rows)
    write_latency_csv(scenario_dir / "latency.csv", final_rows)
    hp = next(spec for spec in specs if spec.role == "foreground")
    bg = [spec for spec in specs if spec.role == "background"]
    summary = summarize_scenario(scenario, hp, bg)
    summary["foreground_batch_size"] = args.batch_size
    (scenario_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def add_slowdowns(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    alone = next((s for s in summaries if s["scenario"] == "alone"), None)
    if alone is None or alone["latency_p99_ms"] == 0:
        return summaries
    base_p50 = float(alone["latency_p50_ms"])
    base_p95 = float(alone["latency_p95_ms"])
    base_p99 = float(alone["latency_p99_ms"])
    for item in summaries:
        item["p50_slowdown_vs_alone"] = round(float(item["latency_p50_ms"]) / base_p50, 3)
        item["p95_slowdown_vs_alone"] = round(float(item["latency_p95_ms"]) / base_p95, 3)
        item["p99_slowdown_vs_alone"] = round(float(item["latency_p99_ms"]) / base_p99, 3)
    return summaries


def check_cuda(gpu: int):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise RuntimeError(f"invalid GPU {gpu}; visible device count is {torch.cuda.device_count()}")
    return torch.device(f"cuda:{gpu}")


def foreground_worker(args: argparse.Namespace) -> int:
    import torch
    import torchvision

    device = check_cuda(args.gpu)
    torch.backends.cudnn.benchmark = True
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)

    if args.foreground_workload == "resnet50":
        model = torchvision.models.resnet50(weights=None).eval().to(device)
        data = torch.ones(args.batch_size, 3, args.image_size, args.image_size, device=device)

        def step() -> None:
            with torch.no_grad(), torch.cuda.stream(stream):
                model(data)

    else:
        model = torch.nn.Transformer(
            d_model=1024,
            nhead=16,
            num_encoder_layers=12,
            num_decoder_layers=12,
            dim_feedforward=4096,
            dropout=0.0,
            batch_first=True,
        ).eval().to(device)
        src = torch.randn(args.batch_size, 128, 1024, device=device)
        tgt = torch.randn(args.batch_size, 128, 1024, device=device)

        def step() -> None:
            with torch.no_grad(), torch.cuda.stream(stream):
                model(src, tgt)

    for _ in range(args.warmup):
        step()
        stream.synchronize()

    for _ in range(args.requests):
        start = time.perf_counter()
        step()
        stream.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        print(f"latency_ms: {latency_ms:.3f}", flush=True)
    return 0


def background_worker(args: argparse.Namespace) -> int:
    import torch
    import torchvision

    device = check_cuda(args.gpu)
    torch.backends.cudnn.benchmark = True
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)
    model = torchvision.models.mobilenet_v2(weights=None, num_classes=1000).train().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    data = torch.randn(args.train_batch_size, 3, args.image_size, args.image_size, device=device)
    target = torch.randint(0, 1000, (args.train_batch_size,), device=device)

    def step() -> None:
        with torch.cuda.stream(stream):
            opt.zero_grad(set_to_none=True)
            out = model(data)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()

    for _ in range(args.warmup):
        step()
        stream.synchronize()
    while True:
        step()
        stream.synchronize()


def main() -> int:
    args = parse_args()
    if args.worker == "foreground":
        return foreground_worker(args)
    if args.worker == "background":
        return background_worker(args)
    if args.requests <= 0:
        raise SystemExit("--requests must be positive")
    if args.background_count < 0:
        raise SystemExit("--background-count must be non-negative")
    if args.scenario in ("xsched", "all") and not OUTPUT_LIB.exists():
        raise SystemExit(f"missing output lib directory: {OUTPUT_LIB}")

    result_dir = make_result_dir(args.result_dir)
    config = vars(args).copy()
    config["result_dir"] = str(result_dir)
    config["foreground"] = f"{args.foreground_workload} inference per-batch latency"
    config["background"] = "MobileNetV2 continuous batch training"
    (result_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    scenarios = ["alone", "native", "xsched", "xsched_lv2"] if args.scenario == "all" else [args.scenario]
    summaries = []
    for scenario in scenarios:
        print(f"\n=== scenario: {scenario} ===", flush=True)
        summaries.append(run_one(args, scenario, result_dir))
        time.sleep(2.0)

    comparison = add_slowdowns(summaries)
    (result_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    print(f"\nresults: {result_dir}")
    print(f"comparison: {result_dir / 'comparison.json'}")
    for item in comparison:
        print(
            f"{item['scenario']}: samples={item['foreground_samples']} "
            f"avg={item['latency_avg_ms']}ms p50={item['latency_p50_ms']}ms "
            f"p95={item['latency_p95_ms']}ms p99={item['latency_p99_ms']}ms "
            f"p99_slowdown={item.get('p99_slowdown_vs_alone')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
