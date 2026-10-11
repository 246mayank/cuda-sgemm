// common.h — error checking, timing statistics, verification, device query and environment
// record, shared by bench.cu and (later) bandwidth.cu.
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

#ifndef ARCH_STR
#define ARCH_STR "unknown"
#endif

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,     \
                         __LINE__, cudaGetErrorString(err_));                         \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

#define CUBLAS_CHECK(call)                                                            \
    do {                                                                              \
        cublasStatus_t st_ = (call);                                                  \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                           \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d: %s\n", (int)st_,          \
                         __FILE__, __LINE__, #call);                                  \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Times one launch with CUDA events. The events bracket only GPU work on the default
// stream, so host-side launch overhead between launches is not included.
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

struct TimingStats {
    double median_ms, min_ms, p90_ms;
};

// median: mean of the two middle samples for even n.
// p90: nearest-rank percentile, i.e. the ceil(0.9 n)-th smallest sample.
inline TimingStats summarize(std::vector<float> t) {
    std::sort(t.begin(), t.end());
    const size_t n = t.size();
    TimingStats s;
    s.min_ms = t.front();
    s.median_ms = (n % 2) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
    size_t rank = (size_t)std::ceil(0.9 * (double)n);
    s.p90_ms = t[std::max<size_t>(rank, 1) - 1];
    return s;
}

// Normwise max relative error:  max_i |C_i - R_i| / max_i |R_i|.
// Per-element relative error is avoided on purpose: with random inputs in [-1, 1] some
// reference entries are ~0, and dividing by them reports huge "errors" for correct
// kernels. Returns NaN if any element of C is NaN/Inf, so the caller's `!(err <= tol)`
// check fails.
inline double max_rel_err(const float* C, const float* R, size_t n) {
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(C[i])) return NAN;
        max_diff = std::max(max_diff, std::fabs((double)C[i] - (double)R[i]));
        max_ref = std::max(max_ref, std::fabs((double)R[i]));
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

// ------------------------------------------------------------------ environment record

// Runs a shell command and returns its stdout (trailing newlines stripped).
inline std::string run_cmd(const char* cmd) {
    std::string out;
    FILE* p = popen(cmd, "r");
    if (!p) return "unavailable";
    char buf[512];
    while (std::fgets(buf, sizeof buf, p)) out += buf;
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out.empty() ? "unavailable" : out;
}

// Clocks, temperature and power at this moment; taken at the start and end of a run so
// throttling during the run is visible in the env JSON.
inline std::string nvidia_smi_snapshot() {
    return run_cmd(
        "nvidia-smi --query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,clocks.max.sm,"
        "clocks.max.mem,temperature.gpu,power.draw,power.limit --format=csv 2>&1");
}

inline std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if ((unsigned char)c < 0x20) o += ' ';
        else o += c;
    }
    return o + "\"";
}

struct DeviceEnv {
    int dev;
    cudaDeviceProp prop;
    int rt_ver, drv_ver;
    int sm_clock_khz, mem_clock_khz, mem_bus_width_bits;  // max (boost) clocks
    std::string date_utc;  // 2026-10-11T08:00:00Z
    std::string stamp;     // 2026-10-11_080000, used in result file names
    std::string gpu_tag;   // GPU name with non-alphanumerics replaced: Tesla_T4
};

// Clock fields come from cudaDeviceGetAttribute rather than cudaDeviceProp, because the
// cudaDeviceProp clock fields are deprecated in recent CUDA versions.
inline DeviceEnv query_device(int dev = 0) {
    DeviceEnv e;
    e.dev = dev;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaGetDeviceProperties(&e.prop, dev));
    CUDA_CHECK(cudaRuntimeGetVersion(&e.rt_ver));
    CUDA_CHECK(cudaDriverGetVersion(&e.drv_ver));
    CUDA_CHECK(cudaDeviceGetAttribute(&e.sm_clock_khz, cudaDevAttrClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&e.mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&e.mem_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev));

    char buf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    e.date_utc = buf;
    std::strftime(buf, sizeof buf, "%Y-%m-%d_%H%M%S", std::gmtime(&now));
    e.stamp = buf;
    for (const char* c = e.prop.name; *c; ++c)
        e.gpu_tag += std::isalnum((unsigned char)*c) ? *c : '_';
    return e;
}

inline void print_device(const DeviceEnv& e) {
    std::printf("GPU: %s  SMs: %d  cc %d.%d  max SM clock %d MHz  CUDA runtime %d  driver %d\n",
                e.prop.name, e.prop.multiProcessorCount, e.prop.major, e.prop.minor,
                e.sm_clock_khz / 1000, e.rt_ver, e.drv_ver);
}

// Writes the device and toolchain fields as JSON object members, each followed by ",\n".
// The caller writes the opening "{", its own fields, and the closing "}".
inline void write_env_fields(FILE* f, const DeviceEnv& e) {
    const cudaDeviceProp& p = e.prop;
    std::fprintf(f, "  \"date_utc\": %s,\n", json_str(e.date_utc).c_str());
    std::fprintf(f, "  \"gpu_name\": %s,\n", json_str(p.name).c_str());
    std::fprintf(f, "  \"compute_capability\": \"%d.%d\",\n", p.major, p.minor);
    std::fprintf(f, "  \"sm_count\": %d,\n", p.multiProcessorCount);
    std::fprintf(f, "  \"sm_clock_max_khz\": %d,\n", e.sm_clock_khz);
    std::fprintf(f, "  \"mem_clock_max_khz\": %d,\n", e.mem_clock_khz);
    std::fprintf(f, "  \"mem_bus_width_bits\": %d,\n", e.mem_bus_width_bits);
    std::fprintf(f, "  \"global_mem_bytes\": %zu,\n", p.totalGlobalMem);
    std::fprintf(f, "  \"l2_cache_bytes\": %d,\n", p.l2CacheSize);
    std::fprintf(f, "  \"smem_per_sm_bytes\": %zu,\n", p.sharedMemPerMultiprocessor);
    std::fprintf(f, "  \"smem_per_block_optin_bytes\": %zu,\n", p.sharedMemPerBlockOptin);
    std::fprintf(f, "  \"regs_per_sm\": %d,\n", p.regsPerMultiprocessor);
    std::fprintf(f, "  \"max_threads_per_sm\": %d,\n", p.maxThreadsPerMultiProcessor);
    std::fprintf(f, "  \"cuda_runtime_version\": %d,\n", e.rt_ver);
    std::fprintf(f, "  \"cuda_driver_version\": %d,\n", e.drv_ver);
    std::fprintf(f, "  \"nvcc_version\": \"%d.%d.%d\",\n", __CUDACC_VER_MAJOR__,
                 __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
    std::fprintf(f, "  \"compiled_arch\": %s,\n", json_str(ARCH_STR).c_str());
}
