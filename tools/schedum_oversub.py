#!/usr/bin/env python3

import ctypes
import os
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_PREEMPT = os.path.join(ROOT, "build/preempt/libpreempt.so")
LIB_HAL = os.path.join(ROOT, "build/platforms/RTX4060/libhalrtx4060.so")
LIB_SHIM = os.path.join(ROOT, "build/platforms/RTX4060/libshimrtx4060.so")

CUDA_SUCCESS = 0
CU_STREAM_NON_BLOCKING = 1
K_PREEMPT_LEVEL_BLOCK = 1
K_QUEUE_CREATE_FLAG_NONE = 0
K_QUEUE_SUSPEND_FLAG_NONE = 0
K_QUEUE_RESUME_FLAG_NONE = 0

GIB = 1024 * 1024 * 1024

PTX = r"""
.version 7.0
.target sm_50
.address_size 64

.visible .entry touch_pages(
    .param .u64 ptr,
    .param .u64 page_count
)
{
    .reg .pred  %p;
    .reg .b32   %r<6>;
    .reg .b64   %rd<8>;

    ld.param.u64    %rd1, [ptr];
    ld.param.u64    %rd2, [page_count];

    mov.u32         %r1, %tid.x;
    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mad.lo.u32      %r4, %r2, %r3, %r1;
    cvt.u64.u32     %rd3, %r4;

    setp.ge.u64     %p, %rd3, %rd2;
    @%p bra         DONE;

    mul.lo.u64      %rd4, %rd3, 4096;
    add.u64         %rd5, %rd1, %rd4;
    st.global.u32   [%rd5], %r4;

DONE:
    ret;
}
"""


def load_libs():
    os.environ.setdefault("XSCHED_CUDA_LIB", "/usr/lib/wsl/lib/libcuda.so.1")
    os.environ.setdefault("CUXTRA_CUDA_LIB", "/usr/lib/wsl/lib/libcuda.so.1")
    os.environ.setdefault("XSCHED_CUDA_MEM_OVERSUB", "1")
    os.environ.setdefault("XSCHED_CUDA_MEM_HIGH_WATERMARK", "85")
    os.environ.setdefault("XSCHED_CUDA_MEM_LOW_WATERMARK", "75")

    mode = ctypes.RTLD_GLOBAL
    preempt = ctypes.CDLL(LIB_PREEMPT, mode=mode)
    hal = ctypes.CDLL(LIB_HAL, mode=mode)
    cuda = ctypes.CDLL(LIB_SHIM, mode=mode)
    return preempt, hal, cuda


def check_cuda(ret, what):
    if ret != CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed with CUresult={ret}")


def check_xs(ret, what):
    if ret != 0:
        raise RuntimeError(f"{what} failed with XResult={ret}")


def setup_prototypes(preempt, hal, cuda):
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cuda.cuStreamCreate.restype = ctypes.c_int
    cuda.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
    cuda.cuStreamSynchronize.restype = ctypes.c_int
    cuda.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuStreamDestroy_v2.restype = ctypes.c_int
    cuda.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.c_uint]
    cuda.cuMemAllocManaged.restype = ctypes.c_int
    cuda.cuMemFree_v2.argtypes = [ctypes.c_ulonglong]
    cuda.cuMemFree_v2.restype = ctypes.c_int
    cuda.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    cuda.cuMemGetInfo_v2.restype = ctypes.c_int
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
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
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
    preempt.XQueueSuspend.argtypes = [ctypes.c_ulonglong, ctypes.c_longlong]
    preempt.XQueueSuspend.restype = ctypes.c_int
    preempt.XQueueResume.argtypes = [ctypes.c_ulonglong, ctypes.c_longlong]
    preempt.XQueueResume.restype = ctypes.c_int
    preempt.XQueueDestroy.argtypes = [ctypes.c_ulonglong]
    preempt.XQueueDestroy.restype = ctypes.c_int


def mem_info(cuda, label):
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    check_cuda(cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo_v2")
    used = total.value - free.value
    print(f"{label}: used={used / GIB:.2f} GiB free={free.value / GIB:.2f} GiB total={total.value / GIB:.2f} GiB")


def alloc_managed(cuda, count, size):
    ptrs = []
    for i in range(count):
        ptr = ctypes.c_ulonglong()
        check_cuda(cuda.cuMemAllocManaged(ctypes.byref(ptr), size, 1), f"cuMemAllocManaged[{i}]")
        ptrs.append(ptr.value)
    return ptrs


def touch(cuda, func, stream, ptr, size):
    pages = size // 4096
    ptr_arg = ctypes.c_ulonglong(ptr)
    pages_arg = ctypes.c_ulonglong(pages)
    args = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(ptr_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(pages_arg), ctypes.c_void_p),
    )
    threads = 256
    blocks = (pages + threads - 1) // threads
    check_cuda(cuda.cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, stream, args, None),
               "cuLaunchKernel(touch_pages)")


def create_xqueue(preempt, hal, cuda):
    stream = ctypes.c_void_p()
    check_cuda(cuda.cuStreamCreate(ctypes.byref(stream), CU_STREAM_NON_BLOCKING), "cuStreamCreate")
    hwq = ctypes.c_ulonglong()
    check_xs(hal.CudaQueueCreate(ctypes.byref(hwq), stream), "CudaQueueCreate")
    xq = ctypes.c_ulonglong()
    check_xs(preempt.XQueueCreate(ctypes.byref(xq), hwq, K_PREEMPT_LEVEL_BLOCK,
                                  K_QUEUE_CREATE_FLAG_NONE), "XQueueCreate")
    return stream, xq


def main():
    preempt, hal, cuda = load_libs()
    setup_prototypes(preempt, hal, cuda)

    print("SchedUM oversubscription workload")
    print(f"XSCHED_CUDA_MEM_OVERSUB={os.environ['XSCHED_CUDA_MEM_OVERSUB']}")
    print(f"watermarks high={os.environ['XSCHED_CUDA_MEM_HIGH_WATERMARK']} low={os.environ['XSCHED_CUDA_MEM_LOW_WATERMARK']}")

    check_cuda(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check_cuda(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check_cuda(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value), "cuCtxCreate_v2")

    module = ctypes.c_void_p()
    check_cuda(cuda.cuModuleLoadData(ctypes.byref(module), PTX.encode("utf-8")), "cuModuleLoadData")
    func = ctypes.c_void_p()
    check_cuda(cuda.cuModuleGetFunction(ctypes.byref(func), module, b"touch_pages"), "cuModuleGetFunction")

    stream, xq = create_xqueue(preempt, hal, cuda)
    print(f"created XQueue=0x{xq.value:x}")

    mem_info(cuda, "initial")

    chunk = GIB
    print("allocating B working set: 7 x 1GiB managed regions")
    b_ptrs = alloc_managed(cuda, 7, chunk)
    print("touching B working set on GPU")
    for ptr in b_ptrs:
        touch(cuda, func, stream, ptr, chunk)
    check_cuda(cuda.cuStreamSynchronize(stream), "cuStreamSynchronize(B touch)")
    mem_info(cuda, "after B touch")

    print("suspending B XQueue: should trigger pressure-driven partial eviction")
    check_xs(preempt.XQueueSuspend(xq, K_QUEUE_SUSPEND_FLAG_NONE), "XQueueSuspend")
    mem_info(cuda, "after B suspend/evict")

    print("allocating and touching A pressure set: 3 x 1GiB managed regions")
    a_ptrs = alloc_managed(cuda, 3, chunk)
    check_xs(preempt.XQueueResume(xq, K_QUEUE_RESUME_FLAG_NONE), "XQueueResume for A touch")
    for ptr in a_ptrs:
        touch(cuda, func, stream, ptr, chunk)
    check_cuda(cuda.cuStreamSynchronize(stream), "cuStreamSynchronize(A touch)")
    mem_info(cuda, "after A touch")

    print("suspending again under oversubscription")
    check_xs(preempt.XQueueSuspend(xq, K_QUEUE_SUSPEND_FLAG_NONE), "XQueueSuspend(second)")
    mem_info(cuda, "after second suspend/evict")

    print("resuming: should prefetch evicted managed regions")
    check_xs(preempt.XQueueResume(xq, K_QUEUE_RESUME_FLAG_NONE), "XQueueResume(prefetch)")
    mem_info(cuda, "after resume/prefetch")

    for ptr in a_ptrs + b_ptrs:
        cuda.cuMemFree_v2(ptr)
    preempt.XQueueDestroy(xq)
    cuda.cuStreamDestroy_v2(stream)
    cuda.cuModuleUnload(module)
    cuda.cuCtxDestroy_v2(ctx)
    print("done")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"elapsed={time.time() - start:.2f}s")
