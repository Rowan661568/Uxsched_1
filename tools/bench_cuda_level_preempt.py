#!/usr/bin/env python3

import argparse
import ctypes
import json
import os
import random
import statistics
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB_PREEMPT = ROOT / "build/preempt/libpreempt.so"
LIB_HAL = ROOT / "build/platforms/RTX4060/libhalrtx4060.so"
LIB_SHIM = ROOT / "build/platforms/RTX4060/libshimrtx4060.so"
RESULTS = ROOT / "benchmark_results"

CUDA_SUCCESS = 0
NVRTC_SUCCESS = 0
CU_STREAM_NON_BLOCKING = 1
K_QUEUE_CREATE_FLAG_NONE = 0
K_QUEUE_SUSPEND_FLAG_SYNC_HW_QUEUE = 1
K_QUEUE_RESUME_FLAG_NONE = 0

KERNEL_SRC = r'''
extern "C" __global__ void sleep_ns(unsigned long long duration_ns)
{
    unsigned long long start;
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < duration_ns);
}
'''


def find_nvrtc() -> str:
    candidates = [
        ROOT / "examples/Python/resnet50_infer/env/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
        Path("/home/cyk/miniconda3/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"),
        Path("/home/cyk/miniconda3/envs/hummingbird-torch/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return "libnvrtc.so"


def check_cuda(ret: int, what: str) -> None:
    if ret != CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed with CUresult={ret}")


def check_nvrtc(ret: int, what: str) -> None:
    if ret != NVRTC_SUCCESS:
        raise RuntimeError(f"{what} failed with NVRTC result={ret}")


def check_xs(ret: int, what: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{what} failed with XResult={ret}")


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * q)
    return ordered[idx]


def load_libs():
    os.environ.setdefault("XSCHED_CUDA_LIB", "/usr/lib/wsl/lib/libcuda.so.1")
    os.environ.setdefault("CUXTRA_CUDA_LIB", "/usr/lib/wsl/lib/libcuda.so.1")
    os.environ.setdefault("XSCHED_CUDA_MEM_OVERSUB", "0")
    mode = ctypes.RTLD_GLOBAL
    preempt = ctypes.CDLL(str(LIB_PREEMPT), mode=mode)
    hal = ctypes.CDLL(str(LIB_HAL), mode=mode)
    cuda = ctypes.CDLL(str(LIB_SHIM), mode=mode)
    nvrtc = ctypes.CDLL(find_nvrtc(), mode=mode)
    return preempt, hal, cuda, nvrtc


def setup(preempt, hal, cuda, nvrtc):
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    cuda.cuCtxSetCurrent.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cuda.cuStreamCreate.restype = ctypes.c_int
    cuda.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
    cuda.cuStreamSynchronize.restype = ctypes.c_int
    cuda.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuStreamDestroy_v2.restype = ctypes.c_int
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    cuda.cuModuleLoadData.restype = ctypes.c_int
    cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    cuda.cuModuleGetFunction.restype = ctypes.c_int
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuModuleUnload.restype = ctypes.c_int
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
    ]
    cuda.cuLaunchKernel.restype = ctypes.c_int

    hal.CudaQueueCreate.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p]
    hal.CudaQueueCreate.restype = ctypes.c_int

    preempt.XQueueCreate.argtypes = [
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.c_ulonglong,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]
    preempt.XQueueCreate.restype = ctypes.c_int
    preempt.XQueueSetLaunchConfig.argtypes = [ctypes.c_ulonglong, ctypes.c_longlong, ctypes.c_longlong]
    preempt.XQueueSetLaunchConfig.restype = ctypes.c_int
    preempt.XQueueSuspend.argtypes = [ctypes.c_ulonglong, ctypes.c_longlong]
    preempt.XQueueSuspend.restype = ctypes.c_int
    preempt.XQueueResume.argtypes = [ctypes.c_ulonglong, ctypes.c_longlong]
    preempt.XQueueResume.restype = ctypes.c_int
    preempt.XQueueDestroy.argtypes = [ctypes.c_ulonglong]
    preempt.XQueueDestroy.restype = ctypes.c_int

    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
    nvrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
    nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
    nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcGetPTX.restype = ctypes.c_int
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int


def compile_ptx(nvrtc, arch: str) -> bytes:
    prog = ctypes.c_void_p()
    check_nvrtc(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), KERNEL_SRC.encode(), b"sleep.cu", 0, None, None),
                "nvrtcCreateProgram")
    options = (ctypes.c_char_p * 2)(f"--gpu-architecture=compute_{arch}".encode(), b"--std=c++11")
    ret = nvrtc.nvrtcCompileProgram(prog, 2, options)
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    if log_size.value > 1:
        log = ctypes.create_string_buffer(log_size.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        print(log.value.decode(errors="replace").strip())
    check_nvrtc(ret, "nvrtcCompileProgram")

    ptx_size = ctypes.c_size_t()
    check_nvrtc(nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)), "nvrtcGetPTXSize")
    ptx = ctypes.create_string_buffer(ptx_size.value)
    check_nvrtc(nvrtc.nvrtcGetPTX(prog, ptx), "nvrtcGetPTX")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.raw


def create_context(cuda):
    check_cuda(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check_cuda(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check_cuda(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value), "cuCtxCreate_v2")
    return ctx


def load_kernel(cuda, ptx: bytes):
    module = ctypes.c_void_p()
    check_cuda(cuda.cuModuleLoadData(ctypes.byref(module), ptx), "cuModuleLoadData")
    func = ctypes.c_void_p()
    check_cuda(cuda.cuModuleGetFunction(ctypes.byref(func), module, b"sleep_ns"), "cuModuleGetFunction")
    return module, func


def create_xqueue(preempt, hal, cuda, level: int, threshold: int, batch_size: int):
    stream = ctypes.c_void_p()
    check_cuda(cuda.cuStreamCreate(ctypes.byref(stream), CU_STREAM_NON_BLOCKING), "cuStreamCreate")
    hwq = ctypes.c_ulonglong()
    check_xs(hal.CudaQueueCreate(ctypes.byref(hwq), stream), "CudaQueueCreate")
    xq = ctypes.c_ulonglong()
    check_xs(preempt.XQueueCreate(ctypes.byref(xq), hwq, level, K_QUEUE_CREATE_FLAG_NONE), "XQueueCreate")
    check_xs(preempt.XQueueSetLaunchConfig(xq, threshold, batch_size), "XQueueSetLaunchConfig")
    return stream, xq


def launch_sleep(cuda, func, stream, duration_ns: int):
    duration_arg = ctypes.c_ulonglong(duration_ns)
    params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(duration_arg), ctypes.c_void_p))
    check_cuda(cuda.cuLaunchKernel(func, 64, 1, 1, 64, 1, 1, 0, stream, params, None),
               "cuLaunchKernel(sleep_ns)")


def run_case(args) -> dict:
    preempt, hal, cuda, nvrtc = load_libs()
    setup(preempt, hal, cuda, nvrtc)
    ctx = create_context(cuda)
    ptx = compile_ptx(nvrtc, args.arch)
    module, func = load_kernel(cuda, ptx)
    stream, xq = create_xqueue(preempt, hal, cuda, args.level, args.threshold, args.batch_size)

    stop = threading.Event()
    launch_errors = []

    def launcher():
        try:
            check_cuda(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent(launcher)")
            while not stop.is_set():
                for _ in range(args.burst):
                    launch_sleep(cuda, func, stream, args.kernel_us * 1000)
                cuda.cuStreamSynchronize(stream)
        except Exception as exc:
            launch_errors.append(repr(exc))
            stop.set()

    thread = threading.Thread(target=launcher, daemon=True)
    thread.start()
    time.sleep(args.warmup_s)

    preempt_us = []
    restore_us = []
    try:
        for _ in range(args.iters):
            time.sleep(random.uniform(0, args.jitter_us) / 1_000_000.0)
            start = time.perf_counter_ns()
            check_xs(preempt.XQueueSuspend(xq, K_QUEUE_SUSPEND_FLAG_SYNC_HW_QUEUE), "XQueueSuspend")
            preempt_us.append((time.perf_counter_ns() - start) / 1000.0)
            time.sleep(random.uniform(args.jitter_us, args.jitter_us * 2) / 1_000_000.0)
            start = time.perf_counter_ns()
            check_xs(preempt.XQueueResume(xq, K_QUEUE_RESUME_FLAG_NONE), "XQueueResume")
            restore_us.append((time.perf_counter_ns() - start) / 1000.0)
            if launch_errors:
                raise RuntimeError(launch_errors[0])
    finally:
        stop.set()
        preempt.XQueueResume(xq, K_QUEUE_RESUME_FLAG_NONE)
        thread.join(timeout=5)
        cuda.cuStreamSynchronize(stream)
        preempt.XQueueDestroy(xq)
        cuda.cuStreamDestroy_v2(stream)
        cuda.cuModuleUnload(module)
        cuda.cuCtxDestroy_v2(ctx)

    return {
        "level": args.level,
        "arch": args.arch,
        "kernel_us": args.kernel_us,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "iters": len(preempt_us),
        "preempt_avg_us": statistics.mean(preempt_us),
        "preempt_p50_us": pct(preempt_us, 0.50),
        "preempt_p95_us": pct(preempt_us, 0.95),
        "preempt_p99_us": pct(preempt_us, 0.99),
        "restore_avg_us": statistics.mean(restore_us),
        "restore_p99_us": pct(restore_us, 0.99),
        "raw_preempt_us": preempt_us,
        "raw_restore_us": restore_us,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--arch", default="89")
    parser.add_argument("--kernel-us", type=int, default=500)
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--burst", type=int, default=64)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--jitter-us", type=int, default=800)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run_case(args)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("raw_")},
                     indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
