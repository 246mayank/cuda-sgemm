// common.h — error checking, timing statistics, verification helpers shared by bench.cu
// and (later) bandwidth.cu.
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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
