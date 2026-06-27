
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "xsched/types.h"
#include "xsched/xqueue.h"
#include "xsched/rtx4060/hal.h"

static inline void CheckCuda(cudaError_t err, const char *expr)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA failed: %s (%d)\n", expr, (int)err);
        std::fflush(stderr);
        std::exit(2);
    }
}

static inline void CheckXs(XResult res, const char *expr)
{
    if (res != kXSchedSuccess) {
        std::fprintf(stderr, "XSched failed: %s (%d)\n", expr, (int)res);
        std::fflush(stderr);
        std::exit(3);
    }
}

#define CUDA_ASSERT(cmd) CheckCuda((cmd), #cmd)
#define XS_ASSERT(cmd) CheckXs((cmd), #cmd)

__device__ __forceinline__ void BusyWait(uint64_t clock_cnt)
{
    if (clock_cnt == 0) return;
    uint64_t start = clock64();
    while ((clock64() - start) < clock_cnt) {}
}

__global__ void SleepKernel(uint64_t clock_cnt)
{
    BusyWait(clock_cnt);
}

static uint64_t ConvertClockCnt(uint64_t microseconds)
{
    cudaDeviceProp prop;
    CUDA_ASSERT(cudaGetDeviceProperties(&prop, 0));
    uint64_t clock_rate_hz = (uint64_t)prop.clockRate * 1000ULL;
    return (microseconds * clock_rate_hz) / 1000000ULL;
}

static double PercentileUs(std::vector<double> vals, double pct)
{
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    double pos = (vals.size() - 1) * pct;
    size_t lo = (size_t)pos;
    size_t hi = std::min(lo + 1, vals.size() - 1);
    double frac = pos - (double)lo;
    return vals[lo] * (1.0 - frac) + vals[hi] * frac;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        std::fprintf(stderr,
                     "Usage: %s <level> <threshold> <batch_size> <iters> <kernel_us> [kernel_us...]\n",
                     argv[0]);
        return 1;
    }

    int level = std::atoi(argv[1]);
    int threshold = std::atoi(argv[2]);
    int batch_size = std::atoi(argv[3]);
    int iters = std::atoi(argv[4]);

    std::vector<int> kernel_us_list;
    for (int i = 5; i < argc; ++i) kernel_us_list.push_back(std::atoi(argv[i]));

    cudaStream_t stream;
    CUDA_ASSERT(cudaStreamCreate(&stream));

    HwQueueHandle hwq;
    XQueueHandle xq;
    XS_ASSERT(CudaQueueCreate(&hwq, stream));
    XS_ASSERT(XQueueCreate(&xq, hwq, level, kQueueCreateFlagNone));
    XS_ASSERT(XQueueSetLaunchConfig(xq, threshold, batch_size));

    constexpr int kSubmitBurst = 64;
    constexpr int kBlocks = 64;
    constexpr int kThreads = 64;

    for (int kernel_us : kernel_us_list) {
        uint64_t clock_cnt = ConvertClockCnt((uint64_t)kernel_us);
        volatile bool stop = false;
        std::thread runner([&]() {
            while (!stop) {
                for (int i = 0; i < kSubmitBurst; ++i) {
                    SleepKernel<<<kBlocks, kThreads, 0, stream>>>(clock_cnt);
                }
                CUDA_ASSERT(cudaStreamSynchronize(stream));
            }
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        std::vector<double> preempt_us;
        std::vector<double> restore_us;
        for (int i = 0; i < iters * 3; ++i) {
            std::this_thread::sleep_for(std::chrono::microseconds(200 + std::rand() % 400));
            auto preempt_t0 = std::chrono::steady_clock::now();
            XS_ASSERT(XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue));
            auto preempt_t1 = std::chrono::steady_clock::now();

            std::this_thread::sleep_for(std::chrono::microseconds(600 + std::rand() % 400));
            auto restore_t0 = std::chrono::steady_clock::now();
            XS_ASSERT(XQueueResume(xq, kQueueResumeFlagNone));
            auto restore_t1 = std::chrono::steady_clock::now();

            if (i >= iters && i < iters * 2) {
                preempt_us.push_back(
                    std::chrono::duration<double, std::micro>(preempt_t1 - preempt_t0).count());
                restore_us.push_back(
                    std::chrono::duration<double, std::micro>(restore_t1 - restore_t0).count());
            }
        }

        stop = true;
        runner.join();
        CUDA_ASSERT(cudaStreamSynchronize(stream));

        double avg_us = 0.0;
        for (double v : preempt_us) avg_us += v;
        avg_us = preempt_us.empty() ? 0.0 : avg_us / (double)preempt_us.size();

        double restore_avg_us = 0.0;
        for (double v : restore_us) restore_avg_us += v;
        restore_avg_us = restore_us.empty() ? 0.0 : restore_avg_us / (double)restore_us.size();

        std::printf("%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%zu\n",
                    level,
                    kernel_us,
                    avg_us,
                    PercentileUs(preempt_us, 0.50),
                    PercentileUs(preempt_us, 0.95),
                    PercentileUs(preempt_us, 0.99),
                    restore_avg_us,
                    preempt_us.size());
        std::fflush(stdout);
    }

    XS_ASSERT(XQueueDestroy(xq));
    XS_ASSERT(HwQueueDestroy(hwq));
    CUDA_ASSERT(cudaStreamDestroy(stream));
    return 0;
}
