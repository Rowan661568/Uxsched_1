#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <typeinfo>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass_probe_common.h"

#ifndef CUTLASS_GIT_REVISION
#define CUTLASS_GIT_REVISION "unknown"
#endif

namespace
{

using Clock = std::chrono::steady_clock;

struct Options {
    std::string mode = "runtime";
    int m = 512;
    int n = 512;
    int k = 512;
    int iterations = 1;
    int warmup = 0;
    std::string stream = "explicit";
    bool correctness = true;
    std::string output;
    float alpha = 1.0f;
    float beta = 1.0f;
};

struct Metrics {
    bool cuda_available = false;
    int return_code = 0;
    std::string status = "not_run";
    std::string reason;
    std::string cuda_error;
    std::string cutlass_status;
    std::string stream_handle = "0";
    std::string kernel_symbol_hint;
    int device_count = 0;
    std::string device_name;
    int driver_version = 0;
    int runtime_version = 0;
    double checksum = 0.0;
    std::string output_hash = "0x0000000000000000";
    uint64_t output_element_count = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    uint64_t mismatch_count = 0;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
    bool correctness_pass = false;
    double cpu_request_us = 0.0;
    double gpu_event_us = 0.0;
    float abs_tolerance = 1.0e-2f;
    float rel_tolerance = 1.0e-4f;
};

using Element = float;
using Layout = cutlass::layout::RowMajor;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    Element, 1, Element, Element>;

using ProbeGemm = cutlass::gemm::device::Gemm<
    Element, Layout,
    Element, Layout,
    Element, Layout,
    Element,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm120,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

bool NextArg(int argc, char **argv, int *i, std::string *out)
{
    if (*i + 1 >= argc) return false;
    *out = argv[++(*i)];
    return true;
}

bool NextInt(int argc, char **argv, int *i, int *out)
{
    std::string value;
    if (!NextArg(argc, argv, i, &value)) return false;
    char *end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') return false;
    *out = static_cast<int>(parsed);
    return true;
}

bool ParseArgs(int argc, char **argv, Options *opts)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode") {
            if (!NextArg(argc, argv, &i, &opts->mode)) return false;
        } else if (arg == "--m") {
            if (!NextInt(argc, argv, &i, &opts->m)) return false;
        } else if (arg == "--n") {
            if (!NextInt(argc, argv, &i, &opts->n)) return false;
        } else if (arg == "--k") {
            if (!NextInt(argc, argv, &i, &opts->k)) return false;
        } else if (arg == "--iterations") {
            if (!NextInt(argc, argv, &i, &opts->iterations)) return false;
        } else if (arg == "--warmup") {
            if (!NextInt(argc, argv, &i, &opts->warmup)) return false;
        } else if (arg == "--stream") {
            if (!NextArg(argc, argv, &i, &opts->stream)) return false;
        } else if (arg == "--correctness") {
            opts->correctness = true;
        } else if (arg == "--no-correctness") {
            opts->correctness = false;
        } else if (arg == "--output") {
            if (!NextArg(argc, argv, &i, &opts->output)) return false;
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            return false;
        }
    }

    return (opts->mode == "runtime" || opts->mode == "driver") &&
           (opts->stream == "default" || opts->stream == "explicit") &&
           opts->m > 0 && opts->n > 0 && opts->k > 0 &&
           opts->iterations > 0 && opts->warmup >= 0;
}

void PrintUsage(const char *argv0)
{
    std::cerr
        << "usage: " << argv0 << " --mode runtime|driver --m M --n N --k K "
        << "[--iterations N] [--warmup N] [--stream default|explicit] "
        << "[--correctness|--no-correctness] [--output PATH]\n";
}

float ValueA(int row, int)
{
    return static_cast<float>((row % 13) - 6) * 0.03125f;
}

float ValueB(int, int col)
{
    return static_cast<float>((col % 17) - 8) * 0.015625f;
}

float ValueC(int row, int col)
{
    return static_cast<float>(((row + col) % 19) - 9) * 0.001f;
}

void InitializeInputs(const Options &opts,
                      std::vector<float> *a,
                      std::vector<float> *b,
                      std::vector<float> *c)
{
    a->resize(static_cast<size_t>(opts.m) * static_cast<size_t>(opts.k));
    b->resize(static_cast<size_t>(opts.k) * static_cast<size_t>(opts.n));
    c->resize(static_cast<size_t>(opts.m) * static_cast<size_t>(opts.n));

    for (int row = 0; row < opts.m; ++row) {
        for (int col = 0; col < opts.k; ++col) {
            (*a)[static_cast<size_t>(row) * opts.k + col] = ValueA(row, col);
        }
    }
    for (int row = 0; row < opts.k; ++row) {
        for (int col = 0; col < opts.n; ++col) {
            (*b)[static_cast<size_t>(row) * opts.n + col] = ValueB(row, col);
        }
    }
    for (int row = 0; row < opts.m; ++row) {
        for (int col = 0; col < opts.n; ++col) {
            (*c)[static_cast<size_t>(row) * opts.n + col] = ValueC(row, col);
        }
    }
}

double ReferenceValue(const Options &opts, int row, int col)
{
    return static_cast<double>(opts.alpha) * static_cast<double>(opts.k) *
           static_cast<double>(ValueA(row, 0)) * static_cast<double>(ValueB(0, col)) +
           static_cast<double>(opts.beta) * static_cast<double>(ValueC(row, col));
}

std::string CudaErrorString(cudaError_t err)
{
    return std::string(cudaGetErrorName(err)) + ":" + cudaGetErrorString(err);
}

void EmitJson(const Options &opts, const Metrics &m, std::ostream &os)
{
    using namespace uxsched::cutlass_probe;
    os << '{';
    JsonField(os, "mode", opts.mode);
    JsonField(os, "launch_api", opts.mode == "runtime" ? "cutlass_device_gemm_runtime" : "cutlass_driver_host_adapter");
    JsonField(os, "cutlass_revision", CUTLASS_GIT_REVISION);
    JsonNumberField(os, "cuda_toolkit", CUDART_VERSION);
    JsonNumberField(os, "cuda_runtime_version", m.runtime_version);
    JsonNumberField(os, "cuda_driver_version", m.driver_version);
    JsonNumberField(os, "cuda_arch", 120);
    JsonField(os, "cutlass_mode", "NATIVE_SM120");
    JsonNumberField(os, "M", opts.m);
    JsonNumberField(os, "N", opts.n);
    JsonNumberField(os, "K", opts.k);
    JsonField(os, "input_type", "fp32");
    JsonField(os, "output_type", "fp32");
    JsonField(os, "accumulator_type", "fp32");
    JsonFloatField(os, "alpha", opts.alpha);
    JsonFloatField(os, "beta", opts.beta);
    JsonField(os, "stream", opts.stream);
    JsonField(os, "stream_handle", m.stream_handle);
    JsonField(os, "kernel_symbol_hint", m.kernel_symbol_hint);
    JsonFloatField(os, "checksum", m.checksum);
    JsonField(os, "output_hash", m.output_hash);
    JsonNumberField(os, "output_element_count", m.output_element_count);
    JsonFloatField(os, "max_abs_error", m.max_abs_error);
    JsonFloatField(os, "max_rel_error", m.max_rel_error);
    JsonNumberField(os, "mismatch_count", m.mismatch_count);
    JsonNumberField(os, "nan_count", m.nan_count);
    JsonNumberField(os, "inf_count", m.inf_count);
    JsonFloatField(os, "absolute_tolerance", m.abs_tolerance);
    JsonFloatField(os, "relative_tolerance", m.rel_tolerance);
    JsonField(os, "correctness_pass", m.correctness_pass);
    JsonFloatField(os, "cpu_request_us", m.cpu_request_us);
    JsonFloatField(os, "gpu_event_us", m.gpu_event_us);
    JsonField(os, "cuda_available", m.cuda_available);
    JsonNumberField(os, "device_count", m.device_count);
    JsonField(os, "device", m.device_name);
    JsonField(os, "status", m.status);
    JsonField(os, "reason", m.reason);
    JsonField(os, "cuda_error", m.cuda_error);
    JsonField(os, "cutlass_status", m.cutlass_status);
    JsonNumberField(os, "return_code", m.return_code, false);
    os << "}\n";
}

void EmitResult(const Options &opts, const Metrics &metrics)
{
    EmitJson(opts, metrics, std::cout);
    if (!opts.output.empty()) {
        std::ofstream out(opts.output, std::ios::app);
        if (out) EmitJson(opts, metrics, out);
    }
}

int BlockedDriverMode(const Options &opts)
{
    Metrics metrics;
    metrics.return_code = 3;
    metrics.status = "BLOCKED";
    metrics.reason = "cutlass_driver_launch_integration_blocked";
    metrics.cutlass_status =
        "CudaHostAdapter is an abstract launch adapter; this probe has no official "
        "CUTLASS path that supplies CUmodule, CUfunction, kernelParams layout, "
        "dynamic shared memory, and stream to cuLaunchKernel for this GEMM.";
    metrics.kernel_symbol_hint = typeid(typename ProbeGemm::GemmKernel).name();
    EmitResult(opts, metrics);
    return metrics.return_code;
}

int RunRuntimeMode(const Options &opts)
{
    Metrics metrics;
    metrics.kernel_symbol_hint = typeid(typename ProbeGemm::GemmKernel).name();

    cudaError_t err = cudaGetDeviceCount(&metrics.device_count);
    if (err != cudaSuccess || metrics.device_count <= 0) {
        metrics.return_code = 0;
        metrics.status = "BLOCKED";
        metrics.reason = "CUDA_UNAVAILABLE";
        metrics.cuda_error = CudaErrorString(err);
        EmitResult(opts, metrics);
        return metrics.return_code;
    }
    metrics.cuda_available = true;
    cudaRuntimeGetVersion(&metrics.runtime_version);
    cudaDriverGetVersion(&metrics.driver_version);

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) metrics.device_name = prop.name;

    std::vector<float> host_a;
    std::vector<float> host_b;
    std::vector<float> host_c;
    std::vector<float> host_d(static_cast<size_t>(opts.m) * static_cast<size_t>(opts.n), 0.0f);
    InitializeInputs(opts, &host_a, &host_b, &host_c);

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;
    float *dev_d = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    auto fail = [&](const std::string &op, cudaError_t code) {
        metrics.return_code = 1;
        metrics.status = "FAILED";
        metrics.reason = op;
        metrics.cuda_error = CudaErrorString(code);
        EmitResult(opts, metrics);
        return metrics.return_code;
    };

    if (opts.stream == "explicit") {
        err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) return fail("cudaStreamCreateWithFlags", err);
    }
    {
        std::ostringstream ss;
        ss << stream;
        metrics.stream_handle = ss.str();
    }

    const size_t bytes_a = host_a.size() * sizeof(float);
    const size_t bytes_b = host_b.size() * sizeof(float);
    const size_t bytes_c = host_c.size() * sizeof(float);
    const size_t bytes_d = host_d.size() * sizeof(float);

    if ((err = cudaMalloc(&dev_a, bytes_a)) != cudaSuccess) return fail("cudaMalloc_A", err);
    if ((err = cudaMalloc(&dev_b, bytes_b)) != cudaSuccess) return fail("cudaMalloc_B", err);
    if ((err = cudaMalloc(&dev_c, bytes_c)) != cudaSuccess) return fail("cudaMalloc_C", err);
    if ((err = cudaMalloc(&dev_d, bytes_d)) != cudaSuccess) return fail("cudaMalloc_D", err);
    if ((err = cudaMemcpyAsync(dev_a, host_a.data(), bytes_a, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return fail("cudaMemcpyAsync_A", err);
    if ((err = cudaMemcpyAsync(dev_b, host_b.data(), bytes_b, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return fail("cudaMemcpyAsync_B", err);
    if ((err = cudaMemcpyAsync(dev_c, host_c.data(), bytes_c, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return fail("cudaMemcpyAsync_C", err);
    if ((err = cudaMemcpyAsync(dev_d, host_c.data(), bytes_d, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return fail("cudaMemcpyAsync_D", err);

    ProbeGemm gemm;
    typename ProbeGemm::Arguments args(
        cutlass::gemm::GemmCoord(opts.m, opts.n, opts.k),
        typename ProbeGemm::TensorRefA(dev_a, typename Layout::Stride(opts.k)),
        typename ProbeGemm::TensorRefB(dev_b, typename Layout::Stride(opts.n)),
        typename ProbeGemm::TensorRefC(dev_c, typename Layout::Stride(opts.n)),
        typename ProbeGemm::TensorRefD(dev_d, typename Layout::Stride(opts.n)),
        typename ProbeGemm::EpilogueOutputOp::Params(opts.alpha, opts.beta),
        1);

    cutlass::Status status = ProbeGemm::can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        metrics.return_code = 1;
        metrics.status = "FAILED";
        metrics.reason = "cutlass_can_implement_failed";
        metrics.cutlass_status = cutlassGetStatusString(status);
        EmitResult(opts, metrics);
        return metrics.return_code;
    }

    status = gemm.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        metrics.return_code = 1;
        metrics.status = "FAILED";
        metrics.reason = "cutlass_initialize_failed";
        metrics.cutlass_status = cutlassGetStatusString(status);
        EmitResult(opts, metrics);
        return metrics.return_code;
    }

    for (int i = 0; i < opts.warmup; ++i) {
        status = gemm.run(stream);
        if (status != cutlass::Status::kSuccess) {
            metrics.return_code = 1;
            metrics.status = "FAILED";
            metrics.reason = "cutlass_warmup_failed";
            metrics.cutlass_status = cutlassGetStatusString(status);
            EmitResult(opts, metrics);
            return metrics.return_code;
        }
    }
    if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) return fail("cudaStreamSynchronize_warmup", err);

    if ((err = cudaEventCreate(&start)) != cudaSuccess) return fail("cudaEventCreate_start", err);
    if ((err = cudaEventCreate(&stop)) != cudaSuccess) return fail("cudaEventCreate_stop", err);

    auto cpu_start = Clock::now();
    if ((err = cudaEventRecord(start, stream)) != cudaSuccess) return fail("cudaEventRecord_start", err);
    for (int i = 0; i < opts.iterations; ++i) {
        status = gemm.run(stream);
        if (status != cutlass::Status::kSuccess) {
            metrics.return_code = 1;
            metrics.status = "FAILED";
            metrics.reason = "cutlass_run_failed";
            metrics.cutlass_status = cutlassGetStatusString(status);
            EmitResult(opts, metrics);
            return metrics.return_code;
        }
    }
    if ((err = cudaEventRecord(stop, stream)) != cudaSuccess) return fail("cudaEventRecord_stop", err);
    if ((err = cudaEventSynchronize(stop)) != cudaSuccess) return fail("cudaEventSynchronize_stop", err);
    auto cpu_stop = Clock::now();

    float elapsed_ms = 0.0f;
    if ((err = cudaEventElapsedTime(&elapsed_ms, start, stop)) != cudaSuccess) return fail("cudaEventElapsedTime", err);
    metrics.gpu_event_us = static_cast<double>(elapsed_ms) * 1000.0 / static_cast<double>(opts.iterations);
    metrics.cpu_request_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_stop - cpu_start).count()) /
        1000.0 / static_cast<double>(opts.iterations);

    if ((err = cudaMemcpyAsync(host_d.data(), dev_d, bytes_d, cudaMemcpyDeviceToHost, stream)) != cudaSuccess) return fail("cudaMemcpyAsync_DtoH", err);
    if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) return fail("cudaStreamSynchronize_final", err);

    metrics.output_element_count = host_d.size();
    metrics.output_hash = uxsched::cutlass_probe::Hex64(
        uxsched::cutlass_probe::Fnva64(host_d.data(), bytes_d));

    double checksum = 0.0;
    uint64_t mismatch_count = 0;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;

    for (int row = 0; row < opts.m; ++row) {
        for (int col = 0; col < opts.n; ++col) {
            const size_t idx = static_cast<size_t>(row) * opts.n + col;
            const double got = static_cast<double>(host_d[idx]);
            checksum += got;
            if (std::isnan(got)) {
                ++nan_count;
                continue;
            }
            if (std::isinf(got)) {
                ++inf_count;
                continue;
            }
            if (opts.correctness) {
                const double expected = ReferenceValue(opts, row, col);
                const double abs_error = std::abs(got - expected);
                const double rel_error = abs_error / std::max(std::abs(expected), 1.0e-12);
                max_abs_error = std::max(max_abs_error, abs_error);
                max_rel_error = std::max(max_rel_error, rel_error);
                if (abs_error > metrics.abs_tolerance &&
                    rel_error > metrics.rel_tolerance) {
                    ++mismatch_count;
                }
            }
        }
    }

    metrics.checksum = checksum;
    metrics.max_abs_error = max_abs_error;
    metrics.max_rel_error = max_rel_error;
    metrics.mismatch_count = mismatch_count;
    metrics.nan_count = nan_count;
    metrics.inf_count = inf_count;
    metrics.correctness_pass = opts.correctness &&
        mismatch_count == 0 && nan_count == 0 && inf_count == 0;
    metrics.status = metrics.correctness_pass || !opts.correctness ? "RAN" : "FAILED";
    metrics.reason = metrics.correctness_pass || !opts.correctness ? "" : "cutlass_correctness_failed";
    metrics.return_code = metrics.correctness_pass || !opts.correctness ? 0 : 1;
    metrics.cutlass_status = "kSuccess";

    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (dev_a != nullptr) cudaFree(dev_a);
    if (dev_b != nullptr) cudaFree(dev_b);
    if (dev_c != nullptr) cudaFree(dev_c);
    if (dev_d != nullptr) cudaFree(dev_d);

    EmitResult(opts, metrics);
    return metrics.return_code;
}

} // namespace

int main(int argc, char **argv)
{
    Options opts;
    if (!ParseArgs(argc, argv, &opts)) {
        PrintUsage(argv[0]);
        return 2;
    }

    if (opts.mode == "driver") {
        return BlockedDriverMode(opts);
    }
    return RunRuntimeMode(opts);
}
