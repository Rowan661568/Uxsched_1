#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

#include "cutlass_probe_common.h"

#ifndef CUTLASS_GIT_REVISION
#define CUTLASS_GIT_REVISION "unknown"
#endif

namespace
{

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Options {
    std::string role = "hp";
    int m = 2048;
    int n = 2048;
    int k = 2048;
    int warmup = 5;
    int requests = 0;
    int duration_ms = 1500;
    int hp_period_us = 30000;
    std::string stream = "explicit";
    bool correctness = true;
    std::string output;
    std::string barrier_dir;
    int barrier_timeout_ms = 120000;
    float alpha = 1.0f;
    float beta = 1.0f;
};

struct Correctness {
    double checksum = 0.0;
    std::string output_hash = "0x0000000000000000";
    uint64_t output_element_count = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    uint64_t mismatch_count = 0;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
    bool correctness_pass = false;
    float abs_tolerance = 1.0e-2f;
    float rel_tolerance = 1.0e-4f;
};

struct RequestResult {
    bool ok = false;
    std::string reason;
    std::string cuda_error;
    std::string cutlass_status;
    double latency_us = 0.0;
    double gpu_event_us = 0.0;
    int64_t start_us = 0;
    int64_t finish_us = 0;
};

using Element = float;
using Layout = cutlass::layout::RowMajor;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    Element, 1, Element, Element>;

using WorkerGemm = cutlass::gemm::device::Gemm<
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

TimePoint g_process_start = Clock::now();

int64_t UsSinceStart(TimePoint t)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(t - g_process_start).count();
}

std::string CudaErrorString(cudaError_t err)
{
    return std::string(cudaGetErrorName(err)) + ":" + cudaGetErrorString(err);
}

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
        if (arg == "--role") {
            if (!NextArg(argc, argv, &i, &opts->role)) return false;
        } else if (arg == "--m") {
            if (!NextInt(argc, argv, &i, &opts->m)) return false;
        } else if (arg == "--n") {
            if (!NextInt(argc, argv, &i, &opts->n)) return false;
        } else if (arg == "--k") {
            if (!NextInt(argc, argv, &i, &opts->k)) return false;
        } else if (arg == "--warmup") {
            if (!NextInt(argc, argv, &i, &opts->warmup)) return false;
        } else if (arg == "--requests") {
            if (!NextInt(argc, argv, &i, &opts->requests)) return false;
        } else if (arg == "--duration-ms") {
            if (!NextInt(argc, argv, &i, &opts->duration_ms)) return false;
        } else if (arg == "--hp-period-us") {
            if (!NextInt(argc, argv, &i, &opts->hp_period_us)) return false;
        } else if (arg == "--stream") {
            if (!NextArg(argc, argv, &i, &opts->stream)) return false;
        } else if (arg == "--output") {
            if (!NextArg(argc, argv, &i, &opts->output)) return false;
        } else if (arg == "--barrier-dir") {
            if (!NextArg(argc, argv, &i, &opts->barrier_dir)) return false;
        } else if (arg == "--barrier-timeout-ms") {
            if (!NextInt(argc, argv, &i, &opts->barrier_timeout_ms)) return false;
        } else if (arg == "--correctness") {
            opts->correctness = true;
        } else if (arg == "--no-correctness") {
            opts->correctness = false;
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            return false;
        }
    }

    if (opts->role != "hp" && opts->role != "lp") return false;
    if (opts->stream != "explicit") return false;
    if (opts->m <= 0 || opts->n <= 0 || opts->k <= 0) return false;
    if (opts->warmup < 0 || opts->requests < 0 || opts->duration_ms < 0) return false;
    if (opts->role == "hp" && opts->requests <= 0) return false;
    if (opts->role == "lp" && opts->requests == 0 && opts->duration_ms <= 0) return false;
    return true;
}

void PrintUsage(const char *argv0)
{
    std::cerr
        << "usage: " << argv0
        << " --role hp|lp --m M --n N --k K --stream explicit"
        << " [--warmup N] [--requests N] [--duration-ms N]"
        << " [--hp-period-us US] [--barrier-dir DIR] [--output JSONL]"
        << " [--correctness|--no-correctness]\n";
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

double ReferenceValue(const Options &opts, int row, int col)
{
    return static_cast<double>(opts.alpha) * static_cast<double>(opts.k) *
           static_cast<double>(ValueA(row, 0)) * static_cast<double>(ValueB(0, col)) +
           static_cast<double>(opts.beta) * static_cast<double>(ValueC(row, col));
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

class JsonEmitter {
public:
    explicit JsonEmitter(std::string path) : path_(std::move(path)) {}

    template <typename Fn>
    void Emit(Fn fn)
    {
        fn(std::cout);
        std::cout << "\n";
        std::cout.flush();
        if (!path_.empty()) {
            std::ofstream out(path_, std::ios::app);
            if (out) {
                fn(out);
                out << "\n";
            }
        }
    }

private:
    std::string path_;
};

void EmitRunConfig(JsonEmitter *emitter, const Options &opts, const std::string &stream_handle)
{
    using namespace uxsched::cutlass_probe;
    emitter->Emit([&](std::ostream &os) {
        os << '{';
        JsonField(os, "type", "config");
        JsonField(os, "role", opts.role);
        JsonField(os, "launch_api", "cutlass_device_gemm_runtime");
        JsonField(os, "cutlass_revision", CUTLASS_GIT_REVISION);
        JsonNumberField(os, "cuda_toolkit", CUDART_VERSION);
        JsonNumberField(os, "cuda_arch", 120);
        JsonField(os, "cutlass_mode", "NATIVE_SM120");
        JsonField(os, "kernel_symbol_hint", typeid(typename WorkerGemm::GemmKernel).name());
        JsonNumberField(os, "M", opts.m);
        JsonNumberField(os, "N", opts.n);
        JsonNumberField(os, "K", opts.k);
        JsonField(os, "input_type", "fp32");
        JsonField(os, "output_type", "fp32");
        JsonField(os, "accumulator_type", "fp32");
        JsonFloatField(os, "alpha", opts.alpha);
        JsonFloatField(os, "beta", opts.beta);
        JsonField(os, "stream", opts.stream);
        JsonField(os, "stream_handle", stream_handle);
        JsonNumberField(os, "warmup", opts.warmup);
        JsonNumberField(os, "requests", opts.requests);
        JsonNumberField(os, "duration_ms", opts.duration_ms);
        JsonNumberField(os, "hp_period_us", opts.hp_period_us, false);
        os << '}';
    });
}

void EmitCorrectnessFields(std::ostream &os, const Correctness &c, bool final_comma)
{
    using namespace uxsched::cutlass_probe;
    JsonFloatField(os, "checksum", c.checksum);
    JsonField(os, "output_hash", c.output_hash);
    JsonNumberField(os, "output_element_count", c.output_element_count);
    JsonFloatField(os, "max_abs_error", c.max_abs_error);
    JsonFloatField(os, "max_rel_error", c.max_rel_error);
    JsonNumberField(os, "mismatch_count", c.mismatch_count);
    JsonNumberField(os, "nan_count", c.nan_count);
    JsonNumberField(os, "inf_count", c.inf_count);
    JsonFloatField(os, "absolute_tolerance", c.abs_tolerance);
    JsonFloatField(os, "relative_tolerance", c.rel_tolerance);
    JsonField(os, "correctness_pass", c.correctness_pass, final_comma);
}

RequestResult RunOne(WorkerGemm *gemm, cudaStream_t stream,
                     cudaEvent_t start_event, cudaEvent_t stop_event)
{
    RequestResult result;
    auto cpu_start = Clock::now();
    result.start_us = UsSinceStart(cpu_start);
    cudaError_t err = cudaEventRecord(start_event, stream);
    if (err != cudaSuccess) {
        result.reason = "cudaEventRecord_start";
        result.cuda_error = CudaErrorString(err);
        return result;
    }

    cutlass::Status status = gemm->run(stream);
    if (status != cutlass::Status::kSuccess) {
        result.reason = "cutlass_run_failed";
        result.cutlass_status = cutlassGetStatusString(status);
        return result;
    }

    err = cudaEventRecord(stop_event, stream);
    if (err != cudaSuccess) {
        result.reason = "cudaEventRecord_stop";
        result.cuda_error = CudaErrorString(err);
        return result;
    }
    err = cudaEventSynchronize(stop_event);
    auto cpu_stop = Clock::now();
    result.finish_us = UsSinceStart(cpu_stop);
    if (err != cudaSuccess) {
        result.reason = "cudaEventSynchronize_stop";
        result.cuda_error = CudaErrorString(err);
        return result;
    }

    float elapsed_ms = 0.0f;
    err = cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
    if (err != cudaSuccess) {
        result.reason = "cudaEventElapsedTime";
        result.cuda_error = CudaErrorString(err);
        return result;
    }

    result.ok = true;
    result.gpu_event_us = static_cast<double>(elapsed_ms) * 1000.0;
    result.latency_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                cpu_stop - cpu_start).count()) / 1000.0;
    result.cutlass_status = "kSuccess";
    return result;
}

Correctness CheckCorrectness(const Options &opts, cudaStream_t stream, float *dev_d,
                             size_t bytes_d)
{
    Correctness c;
    std::vector<float> host_d(static_cast<size_t>(opts.m) * static_cast<size_t>(opts.n), 0.0f);
    cudaError_t err = cudaMemcpyAsync(host_d.data(), dev_d, bytes_d,
                                      cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        c.mismatch_count = 1;
        return c;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        c.mismatch_count = 1;
        return c;
    }

    c.output_element_count = host_d.size();
    c.output_hash = uxsched::cutlass_probe::Hex64(
        uxsched::cutlass_probe::Fnva64(host_d.data(), bytes_d));

    for (int row = 0; row < opts.m; ++row) {
        for (int col = 0; col < opts.n; ++col) {
            const size_t idx = static_cast<size_t>(row) * opts.n + col;
            const double got = static_cast<double>(host_d[idx]);
            c.checksum += got;
            if (std::isnan(got)) {
                ++c.nan_count;
                continue;
            }
            if (std::isinf(got)) {
                ++c.inf_count;
                continue;
            }
            if (opts.correctness) {
                const double expected = ReferenceValue(opts, row, col);
                const double abs_error = std::abs(got - expected);
                const double rel_error = abs_error / std::max(std::abs(expected), 1.0e-12);
                c.max_abs_error = std::max(c.max_abs_error, abs_error);
                c.max_rel_error = std::max(c.max_rel_error, rel_error);
                if (abs_error > c.abs_tolerance && rel_error > c.rel_tolerance) {
                    ++c.mismatch_count;
                }
            }
        }
    }
    c.correctness_pass = opts.correctness &&
        c.mismatch_count == 0 && c.nan_count == 0 && c.inf_count == 0;
    return c;
}

bool FileExists(const std::string &path)
{
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
}

bool WriteReadyFile(const Options &opts)
{
    if (opts.barrier_dir.empty()) return true;
    const std::string ready = opts.barrier_dir + "/" + opts.role + ".ready";
    const std::string tmp = ready + ".tmp." + std::to_string(static_cast<long long>(getpid()));
    {
        std::ofstream out(tmp);
        if (!out) return false;
        out << "pid=" << getpid() << "\n";
        out << "role=" << opts.role << "\n";
        out << "ready_timestamp_us=" << UsSinceStart(Clock::now()) << "\n";
    }
    return rename(tmp.c_str(), ready.c_str()) == 0;
}

bool WaitForStartFile(const Options &opts)
{
    if (opts.barrier_dir.empty()) return true;
    const std::string start = opts.barrier_dir + "/start";
    const auto deadline = Clock::now() + std::chrono::milliseconds(opts.barrier_timeout_ms);
    while (Clock::now() < deadline) {
        if (FileExists(start)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

void EmitPhaseMarker(const char *phase)
{
    std::cerr << "UXSCHED_CUTLASS_PHASE=" << phase
              << " timestamp_us=" << UsSinceStart(Clock::now()) << std::endl;
}

void EmitRequest(JsonEmitter *emitter, const Options &opts, int index,
                 int64_t scheduled_time_us, int64_t actual_start_us,
                 const RequestResult &result)
{
    using namespace uxsched::cutlass_probe;
    emitter->Emit([&](std::ostream &os) {
        os << '{';
        JsonField(os, "type", "request");
        JsonField(os, "role", opts.role);
        JsonNumberField(os, "request_index", index);
        JsonNumberField(os, "scheduled_time_us", scheduled_time_us);
        JsonNumberField(os, "actual_start_time_us", actual_start_us);
        JsonNumberField(os, "finish_time_us", result.finish_us);
        JsonFloatField(os, "release_lateness_us",
                       static_cast<double>(actual_start_us - scheduled_time_us));
        JsonFloatField(os, "latency_us", result.latency_us);
        JsonFloatField(os, "gpu_event_us", result.gpu_event_us);
        JsonField(os, "status", result.ok ? "RAN" : "FAILED");
        JsonField(os, "reason", result.reason);
        JsonField(os, "cuda_error", result.cuda_error);
        JsonField(os, "cutlass_status", result.cutlass_status, false);
        os << '}';
    });
}

int RunWorker(const Options &opts)
{
    JsonEmitter emitter(opts.output);

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        using namespace uxsched::cutlass_probe;
        emitter.Emit([&](std::ostream &os) {
            os << '{';
            JsonField(os, "type", "summary");
            JsonField(os, "role", opts.role);
            JsonField(os, "status", "BLOCKED");
            JsonField(os, "reason", "CUDA_UNAVAILABLE");
            JsonField(os, "cuda_error", CudaErrorString(err));
            JsonNumberField(os, "return_code", 3, false);
            os << '}';
        });
        return 3;
    }

    std::vector<float> host_a;
    std::vector<float> host_b;
    std::vector<float> host_c;
    InitializeInputs(opts, &host_a, &host_b, &host_c);

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;
    float *dev_d = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    auto fail = [&](const std::string &reason, cudaError_t code) {
        using namespace uxsched::cutlass_probe;
        emitter.Emit([&](std::ostream &os) {
            os << '{';
            JsonField(os, "type", "summary");
            JsonField(os, "role", opts.role);
            JsonField(os, "status", "FAILED");
            JsonField(os, "reason", reason);
            JsonField(os, "cuda_error", CudaErrorString(code));
            JsonNumberField(os, "return_code", 1, false);
            os << '}';
        });
        return 1;
    };

    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) return fail("cudaStreamCreateWithFlags", err);
    err = cudaEventCreate(&start_event);
    if (err != cudaSuccess) return fail("cudaEventCreate_start", err);
    err = cudaEventCreate(&stop_event);
    if (err != cudaSuccess) return fail("cudaEventCreate_stop", err);

    std::ostringstream stream_ss;
    stream_ss << stream;
    EmitRunConfig(&emitter, opts, stream_ss.str());

    const size_t bytes_a = host_a.size() * sizeof(float);
    const size_t bytes_b = host_b.size() * sizeof(float);
    const size_t bytes_c = host_c.size() * sizeof(float);
    const size_t bytes_d = static_cast<size_t>(opts.m) * static_cast<size_t>(opts.n) * sizeof(float);

    if ((err = cudaMalloc(&dev_a, bytes_a)) != cudaSuccess) return fail("cudaMalloc_A", err);
    if ((err = cudaMalloc(&dev_b, bytes_b)) != cudaSuccess) return fail("cudaMalloc_B", err);
    if ((err = cudaMalloc(&dev_c, bytes_c)) != cudaSuccess) return fail("cudaMalloc_C", err);
    if ((err = cudaMalloc(&dev_d, bytes_d)) != cudaSuccess) return fail("cudaMalloc_D", err);
    if ((err = cudaMemcpyAsync(dev_a, host_a.data(), bytes_a, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return fail("cudaMemcpyAsync_A", err);
    if ((err = cudaMemcpyAsync(dev_b, host_b.data(), bytes_b, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return fail("cudaMemcpyAsync_B", err);
    if ((err = cudaMemcpyAsync(dev_c, host_c.data(), bytes_c, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return fail("cudaMemcpyAsync_C", err);
    if ((err = cudaMemcpyAsync(dev_d, host_c.data(), bytes_d, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return fail("cudaMemcpyAsync_D", err);

    WorkerGemm gemm;
    typename WorkerGemm::Arguments args(
        cutlass::gemm::GemmCoord(opts.m, opts.n, opts.k),
        typename WorkerGemm::TensorRefA(dev_a, typename Layout::Stride(opts.k)),
        typename WorkerGemm::TensorRefB(dev_b, typename Layout::Stride(opts.n)),
        typename WorkerGemm::TensorRefC(dev_c, typename Layout::Stride(opts.n)),
        typename WorkerGemm::TensorRefD(dev_d, typename Layout::Stride(opts.n)),
        typename WorkerGemm::EpilogueOutputOp::Params(opts.alpha, opts.beta),
        1);

    cutlass::Status status = WorkerGemm::can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        using namespace uxsched::cutlass_probe;
        emitter.Emit([&](std::ostream &os) {
            os << '{';
            JsonField(os, "type", "summary");
            JsonField(os, "role", opts.role);
            JsonField(os, "status", "FAILED");
            JsonField(os, "reason", "cutlass_can_implement_failed");
            JsonField(os, "cutlass_status", cutlassGetStatusString(status));
            JsonNumberField(os, "return_code", 1, false);
            os << '}';
        });
        return 1;
    }
    status = gemm.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        using namespace uxsched::cutlass_probe;
        emitter.Emit([&](std::ostream &os) {
            os << '{';
            JsonField(os, "type", "summary");
            JsonField(os, "role", opts.role);
            JsonField(os, "status", "FAILED");
            JsonField(os, "reason", "cutlass_initialize_failed");
            JsonField(os, "cutlass_status", cutlassGetStatusString(status));
            JsonNumberField(os, "return_code", 1, false);
            os << '}';
        });
        return 1;
    }

    EmitPhaseMarker("WARMUP_START");
    const auto warmup_start = Clock::now();
    double cold_start_us = 0.0;
    for (int i = 0; i < opts.warmup; ++i) {
        RequestResult warmup = RunOne(&gemm, stream, start_event, stop_event);
        if (!warmup.ok) {
            using namespace uxsched::cutlass_probe;
            emitter.Emit([&](std::ostream &os) {
                os << '{';
                JsonField(os, "type", "summary");
                JsonField(os, "role", opts.role);
                JsonField(os, "status", "FAILED");
                JsonField(os, "reason", warmup.reason);
                JsonField(os, "cuda_error", warmup.cuda_error);
                JsonField(os, "cutlass_status", warmup.cutlass_status);
                JsonNumberField(os, "return_code", 1, false);
                os << '}';
            });
            return 1;
        }
        if (i == 0) cold_start_us = warmup.latency_us;
    }
    if (opts.warmup == 0) {
        RequestResult init = RunOne(&gemm, stream, start_event, stop_event);
        if (!init.ok) return 1;
        cold_start_us = init.latency_us;
    }
    const auto warmup_stop = Clock::now();
    const double warmup_total_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                warmup_stop - warmup_start).count()) / 1000.0;

    Correctness correctness = CheckCorrectness(opts, stream, dev_d, bytes_d);
    if (opts.correctness && !correctness.correctness_pass) {
        using namespace uxsched::cutlass_probe;
        emitter.Emit([&](std::ostream &os) {
            os << '{';
            JsonField(os, "type", "warmup_summary");
            JsonField(os, "role", opts.role);
            JsonField(os, "status", "FAILED");
            JsonFloatField(os, "cold_start_us", cold_start_us);
            JsonFloatField(os, "warmup_total_us", warmup_total_us);
            EmitCorrectnessFields(os, correctness, true);
            JsonField(os, "reason", "cutlass_correctness_failed");
            JsonNumberField(os, "return_code", 1, false);
            os << '}';
        });
        return 1;
    }

    EmitPhaseMarker("AFTER_WARMUP");
    using namespace uxsched::cutlass_probe;
    emitter.Emit([&](std::ostream &os) {
        os << '{';
        JsonField(os, "type", "warmup_summary");
        JsonField(os, "role", opts.role);
        JsonField(os, "status", "READY");
        JsonFloatField(os, "cold_start_us", cold_start_us);
        JsonFloatField(os, "warmup_total_us", warmup_total_us);
        JsonNumberField(os, "steady_state_ready_timestamp_us", UsSinceStart(Clock::now()));
        EmitCorrectnessFields(os, correctness, true);
        JsonNumberField(os, "return_code", 0, false);
        os << '}';
    });

    if (!WriteReadyFile(opts)) {
        emitter.Emit([&](std::ostream &os) {
            os << '{';
            JsonField(os, "type", "summary");
            JsonField(os, "role", opts.role);
            JsonField(os, "status", "FAILED");
            JsonField(os, "reason", "ready_file_write_failed");
            JsonNumberField(os, "return_code", 1, false);
            os << '}';
        });
        return 1;
    }
    emitter.Emit([&](std::ostream &os) {
        os << '{';
        JsonField(os, "type", "ready");
        JsonField(os, "role", opts.role);
        JsonNumberField(os, "timestamp_us", UsSinceStart(Clock::now()), false);
        os << '}';
    });
    if (!WaitForStartFile(opts)) {
        emitter.Emit([&](std::ostream &os) {
            os << '{';
            JsonField(os, "type", "summary");
            JsonField(os, "role", opts.role);
            JsonField(os, "status", "FAILED");
            JsonField(os, "reason", "start_barrier_timeout");
            JsonNumberField(os, "return_code", 1, false);
            os << '}';
        });
        return 1;
    }

    EmitPhaseMarker("MEASUREMENT_START");
    const TimePoint measurement_start = Clock::now();
    emitter.Emit([&](std::ostream &os) {
        os << '{';
        JsonField(os, "type", "start_seen");
        JsonField(os, "role", opts.role);
        JsonNumberField(os, "timestamp_us", UsSinceStart(measurement_start), false);
        os << '}';
    });

    int completed = 0;
    double total_latency_us = 0.0;
    bool ok = true;
    std::string fail_reason;
    if (opts.role == "hp") {
        for (int i = 0; i < opts.requests; ++i) {
            const TimePoint scheduled = measurement_start +
                std::chrono::microseconds(static_cast<int64_t>(i) * opts.hp_period_us);
            std::this_thread::sleep_until(scheduled);
            const TimePoint actual_start = Clock::now();
            RequestResult result = RunOne(&gemm, stream, start_event, stop_event);
            EmitRequest(&emitter, opts, i, UsSinceStart(scheduled),
                        UsSinceStart(actual_start), result);
            if (!result.ok) {
                ok = false;
                fail_reason = result.reason;
                break;
            }
            ++completed;
            total_latency_us += result.latency_us;
        }
    } else {
        const TimePoint deadline = measurement_start + std::chrono::milliseconds(opts.duration_ms);
        int i = 0;
        while ((opts.requests > 0 && i < opts.requests) ||
               (opts.requests == 0 && Clock::now() < deadline)) {
            const TimePoint actual_start = Clock::now();
            RequestResult result = RunOne(&gemm, stream, start_event, stop_event);
            EmitRequest(&emitter, opts, i, UsSinceStart(actual_start),
                        UsSinceStart(actual_start), result);
            if (!result.ok) {
                ok = false;
                fail_reason = result.reason;
                break;
            }
            ++completed;
            ++i;
            total_latency_us += result.latency_us;
        }
    }

    const TimePoint measurement_stop = Clock::now();
    EmitPhaseMarker("MEASUREMENT_END");
    const double duration_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                measurement_stop - measurement_start).count()) / 1000.0;
    const double mean_us = completed > 0 ? total_latency_us / static_cast<double>(completed) : 0.0;
    const double throughput =
        duration_us > 0.0 ? static_cast<double>(completed) * 1000000.0 / duration_us : 0.0;

    emitter.Emit([&](std::ostream &os) {
        os << '{';
        JsonField(os, "type", "summary");
        JsonField(os, "role", opts.role);
        JsonField(os, "status", ok ? "RAN" : "FAILED");
        JsonField(os, "reason", fail_reason);
        JsonNumberField(os, "completed_count", completed);
        JsonNumberField(os, "request_count", opts.role == "hp" ? opts.requests : completed);
        JsonFloatField(os, "duration_us", duration_us);
        JsonFloatField(os, "throughput_requests_per_second", throughput);
        JsonFloatField(os, "mean_request_us", mean_us);
        JsonNumberField(os, "return_code", ok ? 0 : 1, false);
        os << '}';
    });

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaStreamDestroy(stream);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char **argv)
{
    Options opts;
    if (!ParseArgs(argc, argv, &opts)) {
        PrintUsage(argv[0]);
        return 2;
    }
    return RunWorker(opts);
}
