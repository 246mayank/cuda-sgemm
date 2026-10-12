// reduce.cu — side study: parallel sum reduction, step by step, against CUB and the DRAM roof.
//
// A reduction reads every input once and does one add per element, so it is purely
// bandwidth-bound: the target is not a FLOP count but the measured DRAM bandwidth. The kernel
// sequence is the classic one (Harris, "Optimizing Parallel Reduction in CUDA"), one idea each:
//   r0_interleaved   tree in shared memory, active threads tid % (2s) == 0: the active lanes of
//                    a warp are scattered, so most warps stay busy doing little (divergence),
//                    and % is slow.
//   r1_strided_index same tree, active thread t works on index 2*s*t: contiguous active
//                    threads (no divergence), but the strided indices hit the same shared-memory
//                    banks (2-way at s = 1, 4-way at s = 2, ...).
//   r2_sequential    halve the active range each step (t < s adds s[t + s]): contiguous
//                    threads and contiguous addresses, no divergence and no bank conflicts.
//   r3_first_add     each thread adds two inputs while loading: half the blocks, and no
//                    thread sits idle in the first tree step.
//   r4_warp_shuffle  once 32 values remain, finish with __shfl_down_sync inside one warp:
//                    no shared memory and no __syncthreads for the last 5 steps.
//   r5_grid_stride   a fixed grid (SMs x 8 blocks) in which each thread sums many float4
//                    loads into a register first, then one block reduction and one atomicAdd
//                    per block: a single pass, and nearly all time spent streaming memory.
// r0-r4 need several passes (the per-block partial sums are reduced again by the same kernel
// until one value is left); all passes are inside the timed region. cub::DeviceReduce::Sum is
// the library reference, and bandwidth = input bytes / time.
//
// Verification: inputs uniform in [0, 1) (no cancellation), result vs a double-precision host
// sum, relative tolerance 1e-4.
// Output: results/reduce_<gpu>_<stamp>.csv (+ reduce_env_*.json)

#include "common.h"

#include <cub/cub.cuh>
#include <cstring>
#include <functional>
#include <random>

namespace {

constexpr int B = 256;  // threads per block for r0-r4

__global__ void r0_interleaved(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    __shared__ float s[B];
    const unsigned t = threadIdx.x;
    const size_t i = blockIdx.x * (size_t)B + t;
    s[t] = i < n ? in[i] : 0.f;
    __syncthreads();
    for (unsigned st = 1; st < B; st *= 2) {
        if (t % (2 * st) == 0) s[t] += s[t + st];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

__global__ void r1_strided_index(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    __shared__ float s[B];
    const unsigned t = threadIdx.x;
    const size_t i = blockIdx.x * (size_t)B + t;
    s[t] = i < n ? in[i] : 0.f;
    __syncthreads();
    for (unsigned st = 1; st < B; st *= 2) {
        const unsigned idx = 2 * st * t;
        if (idx < B) s[idx] += s[idx + st];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

__global__ void r2_sequential(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    __shared__ float s[B];
    const unsigned t = threadIdx.x;
    const size_t i = blockIdx.x * (size_t)B + t;
    s[t] = i < n ? in[i] : 0.f;
    __syncthreads();
    for (unsigned st = B / 2; st > 0; st >>= 1) {
        if (t < st) s[t] += s[t + st];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

__global__ void r3_first_add(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    __shared__ float s[B];
    const unsigned t = threadIdx.x;
    const size_t i = blockIdx.x * (size_t)(2 * B) + t;
    s[t] = (i < n ? in[i] : 0.f) + (i + B < n ? in[i + B] : 0.f);
    __syncthreads();
    for (unsigned st = B / 2; st > 0; st >>= 1) {
        if (t < st) s[t] += s[t + st];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__global__ void r4_warp_shuffle(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    __shared__ float s[B];
    const unsigned t = threadIdx.x;
    const size_t i = blockIdx.x * (size_t)(2 * B) + t;
    s[t] = (i < n ? in[i] : 0.f) + (i + B < n ? in[i + B] : 0.f);
    __syncthreads();
    for (unsigned st = B / 2; st > 32; st >>= 1) {
        if (t < st) s[t] += s[t + st];
        __syncthreads();
    }
    if (t < 32) {
        const float v = warp_sum(s[t] + s[t + 32]);
        if (t == 0) out[blockIdx.x] = v;
    }
}

__global__ void r5_grid_stride(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    __shared__ float warps[B / 32];
    const unsigned t = threadIdx.x;
    float v = 0.f;
    const size_t n4 = n / 4;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + t; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        const float4 x = in4[i];
        v += (x.x + x.y) + (x.z + x.w);
    }
    for (size_t i = n4 * 4 + blockIdx.x * (size_t)blockDim.x + t; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        v += in[i];  // tail when n % 4 != 0
    v = warp_sum(v);
    if (t % 32 == 0) warps[t / 32] = v;
    __syncthreads();
    if (t < 32) {
        v = t < B / 32 ? warps[t] : 0.f;
        v = warp_sum(v);
        if (t == 0) atomicAdd(out, v);
    }
}

using TreeKernel = void (*)(const float*, float*, size_t);

}  // namespace

int main(int argc, char** argv) {
    size_t n = size_t(1) << 26;  // 64 Mi floats = 256 MiB, far beyond L2
    if (argc == 3 && !std::strcmp(argv[1], "--n")) n = std::strtoull(argv[2], nullptr, 10);
    const int warmup = 3, reps = 20;
    const double tol = 1e-4;

    const DeviceEnv env = query_device(0);
    print_device(env);

    std::vector<float> h(n);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    double ref = 0.0;
    for (auto& x : h) { x = dist(rng); ref += x; }

    float *din, *t1, *t2, *dout;
    CUDA_CHECK(cudaMalloc(&din, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&t1, (n / B + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&t2, (n / B + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(din, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Multi-pass tree: apply kernel k until one value is left; returns the pointer holding it.
    auto tree = [&](TreeKernel k, size_t per_block) -> const float* {
        size_t m = n;
        const float* src = din;
        float* dst = t1;
        while (m > 1) {
            const size_t blocks = (m + per_block - 1) / per_block;
            k<<<(unsigned)blocks, B>>>(src, dst, m);
            src = dst;
            dst = (dst == t1) ? t2 : t1;
            m = blocks;
        }
        return src;
    };

    void* cub_tmp = nullptr;
    size_t cub_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(cub_tmp, cub_bytes, din, dout, (int)n));
    CUDA_CHECK(cudaMalloc(&cub_tmp, cub_bytes));
    const int grid5 = env.prop.multiProcessorCount * 8;

    struct Method {
        const char* name;
        std::function<const float*()> run;
    };
    const std::vector<Method> methods = {
        {"cub", [&]() -> const float* {
             CUDA_CHECK(cub::DeviceReduce::Sum(cub_tmp, cub_bytes, din, dout, (int)n));
             return dout;
         }},
        {"r0_interleaved", [&] { return tree(r0_interleaved, B); }},
        {"r1_strided_index", [&] { return tree(r1_strided_index, B); }},
        {"r2_sequential", [&] { return tree(r2_sequential, B); }},
        {"r3_first_add", [&] { return tree(r3_first_add, 2 * B); }},
        {"r4_warp_shuffle", [&] { return tree(r4_warp_shuffle, 2 * B); }},
        {"r5_grid_stride", [&]() -> const float* {
             CUDA_CHECK(cudaMemsetAsync(dout, 0, sizeof(float)));
             r5_grid_stride<<<grid5, B>>>(din, dout, n);
             return dout;
         }},
    };

    const std::string path = "results/reduce_" + env.gpu_tag + "_" + env.stamp + ".csv";
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot open %s (run from the repo root)\n", path.c_str()); return 1; }
    std::fprintf(f, "kernel,n,median_ms,min_ms,p90_ms,gbps,pct_of_cub,rel_err,status\n");

    GpuTimer timer;
    double cub_ms = 0.0;
    std::printf("%-18s %10s %9s %8s %10s %s\n", "kernel", "median_ms", "GB/s", "%CUB", "rel_err", "status");
    for (const Method& m : methods) {
        const float* res = nullptr;
        for (int i = 0; i < warmup; ++i) res = m.run();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float got = 0.f;
        CUDA_CHECK(cudaMemcpy(&got, res, sizeof(float), cudaMemcpyDeviceToHost));
        const double err = std::fabs((double)got - ref) / ref;
        std::vector<float> t(reps);
        for (int r = 0; r < reps; ++r) {
            timer.start();
            m.run();
            t[r] = timer.stop_ms();
        }
        CUDA_CHECK(cudaGetLastError());
        const TimingStats s = summarize(t);
        if (std::string(m.name) == "cub") cub_ms = s.median_ms;
        const double gbps = n * sizeof(float) / (s.median_ms * 1e6);
        const char* status = std::string(m.name) == "cub" ? (err <= tol ? "REF" : "FAILED")
                                                           : (err <= tol ? "PASS" : "FAILED");
        std::printf("%-18s %10.4f %9.1f %8.1f %10.2e %s\n", m.name, s.median_ms, gbps,
                    100.0 * cub_ms / s.median_ms, err, status);
        std::fprintf(f, "%s,%zu,%.6f,%.6f,%.6f,%.3f,%.3f,%.3e,%s\n", m.name, n, s.median_ms,
                     s.min_ms, s.p90_ms, gbps, 100.0 * cub_ms / s.median_ms, err, status);
    }
    std::fclose(f);

    const std::string env_path = "results/reduce_env_" + env.gpu_tag + "_" + env.stamp + ".json";
    f = std::fopen(env_path.c_str(), "w");
    if (f) {
        std::fprintf(f, "{\n");
        write_env_fields(f, env);
        std::fprintf(f, "  \"n\": %zu,\n  \"warmup_runs\": %d,\n  \"timed_runs\": %d,\n", n, warmup, reps);
        std::fprintf(f, "  \"tolerance_rel_err\": %g\n}\n", tol);
        std::fclose(f);
    }
    std::printf("wrote %s\n", path.c_str());
    return 0;
}
