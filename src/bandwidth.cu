// bandwidth.cu — measured DRAM bandwidth, the memory roof of the roofline.
//
// Copies a buffer far larger than the 4 MB L2 (512 MiB in, 512 MiB out), so every byte
// really goes to and from DRAM. Bandwidth = (bytes read + bytes written) / time. Three
// methods, so the roof does not depend on one kernel's quirks:
//   copy_float4   grid-stride kernel, one 128-bit load + one 128-bit store per iteration
//   copy_float    same with 32-bit accesses
//   cudaMemcpy    device-to-device cudaMemcpyAsync (the driver's own copy path)
// Each: 3 warm-ups, 20 timed runs with CUDA events, median / min / p90, like bench.cu.
// The theoretical number (memory clock x 2 for DDR x bus width) is written alongside and
// labelled as such.
//
// Output: results/bandwidth_<gpu>_<UTC timestamp>.csv
// Usage:  ./bandwidth [--mib <buffer size in MiB>]

#include "common.h"

#include <cstring>
#include <string>

namespace {

__global__ void copy_float4(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = in[i];
}

__global__ void copy_float(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = in[i];
}

}  // namespace

int main(int argc, char** argv) {
    size_t mib = 512;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--mib") && i + 1 < argc) mib = std::strtoull(argv[++i], nullptr, 10);
        else { std::fprintf(stderr, "usage: ./bandwidth [--mib N]\n"); return 1; }
    }
    const int warmup = 3, reps = 20;

    const DeviceEnv env = query_device(0);
    print_device(env);
    const double theoretical_gbps =
        env.mem_clock_khz * 1e3 * 2.0 * (env.mem_bus_width_bits / 8.0) / 1e9;

    const size_t bytes = mib << 20;
    const size_t n = bytes / sizeof(float);
    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMemset(in, 0, bytes));
    CUDA_CHECK(cudaMemset(out, 0, bytes));

    // Enough blocks to fill every SM several times over; the grid-stride loop covers the rest.
    const int threads = 256;
    const int blocks = env.prop.multiProcessorCount * 32;

    struct Method {
        const char* name;
        void (*run)(float*, float*, size_t, int, int, size_t);
    };
    const Method methods[] = {
        {"copy_float4", [](float* a, float* b, size_t n, int g, int t, size_t) {
             copy_float4<<<g, t>>>(reinterpret_cast<const float4*>(a),
                                   reinterpret_cast<float4*>(b), n / 4);
         }},
        {"copy_float", [](float* a, float* b, size_t n, int g, int t, size_t) {
             copy_float<<<g, t>>>(a, b, n);
         }},
        {"cudaMemcpy_d2d", [](float* a, float* b, size_t, int, int, size_t bytes) {
             CUDA_CHECK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice));
         }},
    };

    const std::string path = "results/bandwidth_" + env.gpu_tag + "_" + env.stamp + ".csv";
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot open %s (run from the repo root)\n", path.c_str()); return 1; }
    std::fprintf(f, "method,buffer_bytes,bytes_moved,median_ms,min_ms,p90_ms,gbps_median,gbps_best,"
                    "theoretical_gbps\n");

    GpuTimer timer;
    const double moved = 2.0 * bytes;  // read + write
    for (const Method& m : methods) {
        for (int i = 0; i < warmup; ++i) m.run(in, out, n, blocks, threads, bytes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> t(reps);
        for (int r = 0; r < reps; ++r) {
            timer.start();
            m.run(in, out, n, blocks, threads, bytes);
            t[r] = timer.stop_ms();
        }
        CUDA_CHECK(cudaGetLastError());
        const TimingStats s = summarize(t);
        const double med = moved / (s.median_ms * 1e6), best = moved / (s.min_ms * 1e6);
        std::printf("%-15s median %.3f ms  %.1f GB/s (best %.1f)  theoretical %.1f GB/s\n", m.name,
                    s.median_ms, med, best, theoretical_gbps);
        std::fprintf(f, "%s,%zu,%.0f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f\n", m.name, bytes, moved,
                     s.median_ms, s.min_ms, s.p90_ms, med, best, theoretical_gbps);
    }
    std::fclose(f);
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    std::printf("wrote %s\n", path.c_str());
    return 0;
}
