// bench.cu — SGEMM benchmark harness.
//
// For every size and every kernel: run, verify against cuBLAS, time, and append one CSV row.
//   results/sgemm_<gpu>_<UTC timestamp>.csv
//   results/env_<gpu>_<UTC timestamp>.json
//
// Usage: ./bench [--kernel <prefix>] [--size <n>] [--warmup <n>] [--reps <n>]
//                [--tol <x>] [--no-write]
//   --kernel v0     run only kernels whose name starts with "v0" (cuBLAS always runs; it is
//                   the reference)
//   --size 2048     run only this square size
//   --no-write      print results but write no files (used for the Nsight Compute probe)
//
// ROW-MAJOR vs COLUMN-MAJOR (the cuBLAS transpose trick)
//   Our matrices are row-major; cuBLAS assumes column-major. A row-major MxN matrix has the
//   same bytes as a column-major NxM matrix, i.e. its transpose. So the row-major product
//   C = A B is, in cuBLAS's view, C^T = B^T A^T, where B^T (NxK) and A^T (KxM) are just our
//   B and A buffers read column-major. We therefore call
//       cublasSgemm(N, N, m=N, n=M, k=K, B, ldb=N, A, lda=K, C, ldc=N)
//   i.e. swap the operands and the M/N dimensions. No data is transposed or copied.

#include "common.h"
#include "kernels/kernels.h"

#include <cstring>
#include <ctime>
#include <functional>
#include <random>
#include <string>

namespace {

using Launcher = std::function<void(int, int, int, const float*, const float*, float*)>;

struct Kernel {
    std::string name;
    Launcher launch;
};

struct Row {
    std::string kernel;
    int M, N, K;
    TimingStats t;
    double gflops, pct_cublas, err;
    std::string status;
};

}  // namespace

int main(int argc, char** argv) {
    std::string kernel_filter;
    int only_size = 0, warmup = 3, reps = 20;
    double tol = 1e-4;
    bool write = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "--kernel") kernel_filter = next();
        else if (a == "--size") only_size = std::atoi(next());
        else if (a == "--warmup") warmup = std::atoi(next());
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--tol") tol = std::atof(next());
        else if (a == "--no-write") write = false;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (reps < 1) { std::fprintf(stderr, "--reps must be >= 1\n"); return 1; }

    // ---------------------------------------------------------------- environment
    const DeviceEnv env = query_device(0);
    print_device(env);
    const std::string nvsmi_start = nvidia_smi_snapshot();

    int cublas_ver = 0;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasGetVersion(handle, &cublas_ver));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    std::printf("cuBLAS %d\n", cublas_ver);

    // ---------------------------------------------------------------- kernels
    const float one = 1.f, zero = 0.f;
    Launcher cublas = [&](int M, int N, int K, const float* A, const float* B, float* C) {
        // Row-major C = A B via column-major C^T = B^T A^T (see header comment).
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K,
                                 &zero, C, N));
    };
    std::vector<Kernel> all = {
        {"v0_naive", launch_v0_naive},
        {"v1_coalesced", launch_v1_coalesced},
    };
    std::vector<Kernel> kernels;
    for (auto& k : all)
        if (kernel_filter.empty() || k.name.rfind(kernel_filter, 0) == 0) kernels.push_back(k);

    std::vector<int> sizes = {256, 512, 1000, 1023, 1024, 2048, 4096};
    if (only_size) sizes = {only_size};

    // ---------------------------------------------------------------- run
    std::vector<Row> rows;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    GpuTimer timer;

    for (int S : sizes) {
        const int M = S, N = S, K = S;
        const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
        const double flop = 2.0 * M * N * K;

        std::vector<float> hA(nA), hB(nB), hC(nC), hRef(nC);
        for (auto& x : hA) x = dist(rng);
        for (auto& x : hB) x = dist(rng);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));

        // Runs warm-ups, checks the output (when ref != nullptr), then the timed reps.
        // C is filled with 0xFF bytes (NaN) first, so a kernel that skips elements cannot
        // pass on values left behind by a previous kernel.
        auto measure = [&](const Launcher& f, const float* ref, double& err) {
            CUDA_CHECK(cudaMemset(dC, 0xFF, nC * sizeof(float)));
            for (int i = 0; i < std::max(warmup, 1); ++i) f(M, N, K, dA, dB, dC);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
            err = ref ? max_rel_err(hC.data(), ref, nC) : 0.0;
            std::vector<float> t(reps);
            for (int r = 0; r < reps; ++r) {
                timer.start();
                f(M, N, K, dA, dB, dC);
                t[r] = timer.stop_ms();
            }
            CUDA_CHECK(cudaGetLastError());
            return summarize(t);
        };

        double err = 0.0;
        TimingStats tc = measure(cublas, nullptr, err);
        // Reference = cuBLAS output from its warm-up run (taken before its timed reps).
        hRef = hC;
        const double cublas_gflops = flop / (tc.median_ms * 1e6);
        rows.push_back({"cublas", M, N, K, tc, cublas_gflops, 100.0, 0.0, "REF"});

        for (auto& k : kernels) {
            TimingStats t = measure(k.launch, hRef.data(), err);
            const double g = flop / (t.median_ms * 1e6);
            rows.push_back({k.name, M, N, K, t, g, 100.0 * g / cublas_gflops, err,
                            (err <= tol) ? "PASS" : "FAILED"});
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    const std::string nvsmi_end = nvidia_smi_snapshot();
    CUBLAS_CHECK(cublasDestroy(handle));

    // ---------------------------------------------------------------- report
    std::printf("\n%-14s %5s %10s %10s %10s %9s %8s %10s %s\n", "kernel", "size", "median_ms",
                "min_ms", "p90_ms", "GFLOP/s", "%cuBLAS", "max_rel_err", "status");
    for (auto& r : rows)
        std::printf("%-14s %5d %10.4f %10.4f %10.4f %9.1f %8.1f %10.2e %s\n", r.kernel.c_str(),
                    r.M, r.t.median_ms, r.t.min_ms, r.t.p90_ms, r.gflops, r.pct_cublas, r.err,
                    r.status.c_str());

    if (!write) {
        std::printf("\n--no-write: no files written\n");
        return 0;
    }

    const std::string csv_path = "results/sgemm_" + env.gpu_tag + "_" + env.stamp + ".csv";
    const std::string env_path = "results/env_" + env.gpu_tag + "_" + env.stamp + ".json";

    FILE* f = std::fopen(csv_path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot open %s (run from the repo root)\n", csv_path.c_str()); return 1; }
    std::fprintf(f, "kernel,M,N,K,median_ms,min_ms,p90_ms,gflops,pct_cublas,max_rel_err,status\n");
    for (auto& r : rows)
        std::fprintf(f, "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.3f,%.3e,%s\n", r.kernel.c_str(), r.M,
                     r.N, r.K, r.t.median_ms, r.t.min_ms, r.t.p90_ms, r.gflops, r.pct_cublas,
                     r.err, r.status.c_str());
    std::fclose(f);

    f = std::fopen(env_path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", env_path.c_str()); return 1; }
    std::fprintf(f, "{\n");
    write_env_fields(f, env);
    std::fprintf(f, "  \"cublas_version\": %d,\n", cublas_ver);
    std::fprintf(f, "  \"cublas_math_mode\": \"CUBLAS_DEFAULT_MATH\",\n");
    std::fprintf(f, "  \"warmup_runs\": %d,\n", warmup);
    std::fprintf(f, "  \"timed_runs\": %d,\n", reps);
    std::fprintf(f, "  \"tolerance_max_rel_err\": %g,\n", tol);
    std::fprintf(f, "  \"input_distribution\": \"uniform[-1,1], mt19937 seed 42\",\n");
    std::fprintf(f, "  \"results_csv\": %s,\n", json_str(csv_path).c_str());
    std::fprintf(f, "  \"nvidia_smi_start\": %s,\n", json_str(nvsmi_start).c_str());
    std::fprintf(f, "  \"nvidia_smi_end\": %s\n", json_str(nvsmi_end).c_str());
    std::fprintf(f, "}\n");
    std::fclose(f);

    std::printf("\nwrote %s\nwrote %s\n", csv_path.c_str(), env_path.c_str());
    return 0;
}
