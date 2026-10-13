// bench.cu — SGEMM benchmark harness.
//
// For every size and every kernel: run, verify against cuBLAS, time, and append one CSV row.
//   results/sgemm_<gpu>_<UTC timestamp>.csv
//   results/env_<gpu>_<UTC timestamp>.json
//
// PAIRED cuBLAS TIMING (why pct_cublas is not just gflops / gflops of the cuBLAS row)
//   A T4 is capped at 70 W. Under sustained load its SM clock drops, so a cuBLAS number
//   measured at the start of a size and a kernel number measured a minute later are taken
//   at different clocks. To compare like with like, every timed rep of a kernel is
//   immediately followed by one timed cuBLAS call on the same inputs. pct_cublas is
//   (median paired cuBLAS time) / (median kernel time) x 100, and the paired cuBLAS median
//   is written in the extra column cublas_paired_median_ms. The "cublas" row itself is a
//   standalone measurement taken before any kernel at that size.
//
// SIZE CAP
//   v0_naive is only run up to 2048: at 4096 it takes ~2 s per launch, adds about a minute
//   of heat to the GPU, and teaches nothing new (its GFLOP/s is flat from 1024 up).
//
// Usage: ./bench [--kernel <prefix>] [--size <n>] [--warmup <n>] [--reps <n>]
//                [--tol <x>] [--no-write]
//   --kernel v0     run only kernels whose name starts with "v0" (cuBLAS always runs; it is
//                   the reference)
//   --size 2048     run only this square size
//   --no-write      print results but write no files (used for the Nsight Compute probe)
//   --tune          run the tuning study (12 tile shapes of v7, 6 warp-tile shapes of v8)
//                   instead of the v0-v8 series; files get the prefix "tune_" unless --prefix is given
//   --tc            tensor-core track: FP16 copies of A and B, FP32 accumulation (WMMA kernels
//                   tc0, tc1), verified and paired against cublasGemmEx with the same FP16
//                   inputs; sizes that are multiples of 128; prefix "tc_"; default tolerance
//                   1e-3. Also writes tc_precision_*.csv: how far the FP16-input result is
//                   from FP32 SGEMM (the precision cost of FP16 inputs, not a kernel error).
//   --tc-tune       tensor-core tuning sweep: the tc5 configurations (tc1's WMMA kernel over
//                   several tile shapes), run exactly like --tc; prefix "tctune_"
//   --shapes        rectangular GEMMs of the kind found in transformer layers (M = tokens,
//                   N/K = hidden or FFN width), for v4, v7, v9 vs cuBLAS; prefix "shapes_"
//   --epilogue      fused-epilogue study: C = act(A*B + bias) (ReLU, GELU) with the f1 kernel vs
//                   cublasSgemm followed by a separate bias+activation kernel, and vs cuBLASLt's own
//                   fused epilogue (one cublasLtMatmul with CUBLASLT_EPILOGUE_RELU_BIAS /
//                   GELU_BIAS), all three timed interleaved; writes results/epi_<gpu>_<stamp>.csv
//   --prefix p_     prepend p_ to every output file name (e.g. locked900_ for a clock-locked
//                   run), so special runs never mix with the headline results/sgemm_*.csv
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

#include <cublasLt.h>

#include <cstring>
#include <ctime>
#include <functional>
#include <random>
#include <string>

namespace {

__global__ void to_half(const float* __restrict__ in, half* __restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = __float2half(in[i]);
}

using Launcher = std::function<void(int, int, int, const float*, const float*, float*)>;

// Unfused epilogue: C = act(C + bias[col]) in a separate pass over C (grid-stride, float4 when
// N % 4 == 0). act: 1 = ReLU, 2 = GELU tanh approximation (same formula as the fused kernel).
__device__ __forceinline__ float act_fn(float x, int act) {
    if (act == 1) return fmaxf(x, 0.f);
    if (act == 2) return 0.5f * x * (1.f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    return x;
}

__global__ void bias_act(float* __restrict__ C, const float* __restrict__ bias, int M, int N,
                         int act) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    if (N % 4 == 0) {
        float4* C4 = reinterpret_cast<float4*>(C);
        const size_t n4 = (size_t)M * N / 4;
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += stride) {
            const int c = (int)((i * 4) % N);
            float4 v = C4[i];
            v.x = act_fn(v.x + bias[c + 0], act);
            v.y = act_fn(v.y + bias[c + 1], act);
            v.z = act_fn(v.z + bias[c + 2], act);
            v.w = act_fn(v.w + bias[c + 3], act);
            C4[i] = v;
        }
    } else {
        const size_t n = (size_t)M * N;
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride)
            C[i] = act_fn(C[i] + bias[i % N], act);
    }
}

struct Kernel {
    std::string name;
    Launcher launch;
    KernelResources (*resources)();
    int max_size;  // sizes above this are skipped (no CSV row)
};

struct Row {
    std::string kernel;
    int M, N, K;
    TimingStats t;
    double gflops, pct_cublas, err;
    std::string status;
    double cublas_paired_ms;  // NAN for the cuBLAS row itself
    GpuSample after;          // SM clock / power sampled right after the timed runs
};

// --epilogue: fused vs unfused act(A*B + bias). Returns the process exit code.
int run_epilogue(const DeviceEnv& env, cublasHandle_t handle, int warmup, int reps, double tol,
                 const std::string& prefix, bool write) {
    struct Shape {
        int M, N, K;
    };
    const std::vector<Shape> shapes = {{1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096},
                                       {2048, 3072, 768},  {2048, 768, 3072}};
    const float one = 1.f, zero = 0.f;
    std::mt19937 rng(42), brng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    GpuTimer timer;
    const int grid = env.prop.multiProcessorCount * 8;

    struct EpiRow {
        const char* act;
        Shape s;
        TimingStats u, f, l;
        double err, lerr;
        const char* status;
        const char* lstatus;  // PASS, FAILED, or UNSUPPORTED (no cuBLASLt algorithm found)
    };
    std::vector<EpiRow> rows;

    // cuBLASLt: the same column-major view as cublasSgemm above (C^T = B^T A^T), with the bias
    // added per row of the column-major D, i.e. per column of our row-major C.
    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));
    const size_t ws_bytes = 32u << 20;
    void* ws;
    CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
    for (const Shape& sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K;
        const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
        std::vector<float> hA(nA), hB(nB), hbias(N), hu(nC), hf(nC);
        for (auto& x : hA) x = dist(rng);
        for (auto& x : hB) x = dist(rng);
        for (auto& x : hbias) x = dist(brng);
        std::vector<float> hl(nC);
        float *dA, *dB, *dbias, *dCu, *dCf, *dCl;
        CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dbias, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dCu, nC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dCf, nC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dCl, nC * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dbias, hbias.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        for (int act = 1; act <= 2; ++act) {
            auto unfused = [&]() {  // cuBLAS SGEMM, then a separate bias + activation pass
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, dB, N,
                                         dA, K, &zero, dCu, N));
                bias_act<<<grid, 256>>>(dCu, dbias, M, N, act);
            };
            auto fused = [&]() {
                if (act == 1) launch_f1_bias_relu(M, N, K, dA, dB, dbias, dCf);
                else launch_f1_bias_gelu(M, N, K, dA, dB, dbias, dCf);
            };
            // cuBLASLt fused epilogue (vendor fusion). GELU_BIAS uses the tanh approximation,
            // the same formula as bias_act and f1.
            cublasLtMatmulDesc_t op;
            cublasLtMatrixLayout_t la, lb, lc;
            cublasLtMatmulPreference_t pref;
            CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            const cublasLtEpilogue_t epi =
                act == 1 ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_GELU_BIAS;
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi,
                                                        sizeof(epi)));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                        &dbias, sizeof(dbias)));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, N, K, N));  // B^T
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, K, M, K));  // A^T
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, N, M, N));  // C^T
            CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
            CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
            cublasLtMatmulHeuristicResult_t heur = {};
            int nres = 0;
            const bool lt_ok = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1,
                                                              &heur, &nres) ==
                                   CUBLAS_STATUS_SUCCESS &&
                               nres > 0;
            auto ltfused = [&]() {
                CUBLAS_CHECK(cublasLtMatmul(lt, op, &one, dB, la, dA, lb, &zero, dCl, lc, dCl,
                                            lc, &heur.algo, ws, ws_bytes, 0));
            };

            CUDA_CHECK(cudaMemset(dCf, 0xFF, nC * sizeof(float)));  // NaN: skipped writes fail
            CUDA_CHECK(cudaMemset(dCl, 0xFF, nC * sizeof(float)));
            for (int i = 0; i < std::max(warmup, 1); ++i) {
                unfused();
                fused();
                if (lt_ok) ltfused();
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hu.data(), dCu, nC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hf.data(), dCf, nC * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hl.data(), dCl, nC * sizeof(float), cudaMemcpyDeviceToHost));
            const double err = max_rel_err(hf.data(), hu.data(), nC);
            const double lerr = lt_ok ? max_rel_err(hl.data(), hu.data(), nC) : NAN;
            std::vector<float> tu(reps), tf(reps), tl(reps, NAN);
            for (int r = 0; r < reps; ++r) {  // interleaved, like the paired cuBLAS timing
                timer.start();
                unfused();
                tu[r] = timer.stop_ms();
                timer.start();
                fused();
                tf[r] = timer.stop_ms();
                if (lt_ok) {
                    timer.start();
                    ltfused();
                    tl[r] = timer.stop_ms();
                }
            }
            CUDA_CHECK(cudaGetLastError());
            CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(la));
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(lb));
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(lc));
            CUBLAS_CHECK(cublasLtMatmulDescDestroy(op));
            const TimingStats ls = lt_ok ? summarize(tl) : TimingStats{NAN, NAN, NAN};
            rows.push_back({act == 1 ? "relu" : "gelu", sh, summarize(tu), summarize(tf), ls, err,
                            lerr, err <= tol ? "PASS" : "FAILED",
                            !lt_ok ? "UNSUPPORTED" : (lerr <= tol ? "PASS" : "FAILED")});
        }
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dbias));
        CUDA_CHECK(cudaFree(dCu));
        CUDA_CHECK(cudaFree(dCf));
        CUDA_CHECK(cudaFree(dCl));
    }
    CUDA_CHECK(cudaFree(ws));
    CUBLAS_CHECK(cublasLtDestroy(lt));

    std::printf("\n%-5s %5s %5s %5s %12s %12s %8s %10s %-7s %12s %8s %10s %s\n", "act", "M", "N",
                "K", "unfused_ms", "fused_ms", "speedup", "rel_err", "status", "cublasLt_ms",
                "lt_spd", "lt_err", "lt_status");
    for (auto& r : rows)
        std::printf("%-5s %5d %5d %5d %12.4f %12.4f %8.3f %10.2e %-7s %12.4f %8.3f %10.2e %s\n",
                    r.act, r.s.M, r.s.N, r.s.K, r.u.median_ms, r.f.median_ms,
                    r.u.median_ms / r.f.median_ms, r.err, r.status, r.l.median_ms,
                    r.u.median_ms / r.l.median_ms, r.lerr, r.lstatus);
    if (!write) return 0;

    const std::string path = "results/" + prefix + "epi_" + env.gpu_tag + "_" + env.stamp + ".csv";
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot open %s (run from the repo root)\n", path.c_str()); return 1; }
    std::fprintf(f, "act,M,N,K,unfused_median_ms,unfused_min_ms,unfused_p90_ms,fused_median_ms,"
                    "fused_min_ms,fused_p90_ms,speedup,max_rel_err,status,lt_median_ms,lt_min_ms,"
                    "lt_p90_ms,lt_speedup,lt_max_rel_err,lt_status\n");
    for (auto& r : rows)
        std::fprintf(f,
                     "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.3e,%s,%.6f,%.6f,%.6f,%.4f,"
                     "%.3e,%s\n",
                     r.act, r.s.M, r.s.N, r.s.K, r.u.median_ms, r.u.min_ms, r.u.p90_ms,
                     r.f.median_ms, r.f.min_ms, r.f.p90_ms, r.u.median_ms / r.f.median_ms, r.err,
                     r.status, r.l.median_ms, r.l.min_ms, r.l.p90_ms,
                     r.u.median_ms / r.l.median_ms, r.lerr, r.lstatus);
    std::fclose(f);
    const std::string env_path =
        "results/" + prefix + "epi_env_" + env.gpu_tag + "_" + env.stamp + ".json";
    f = std::fopen(env_path.c_str(), "w");
    if (f) {
        std::fprintf(f, "{\n");
        write_env_fields(f, env);
        std::fprintf(f, "  \"warmup_runs\": %d,\n  \"timed_runs\": %d,\n", warmup, reps);
        std::fprintf(f, "  \"tolerance_max_rel_err\": %g,\n", tol);
        std::fprintf(f, "  \"unfused\": \"cublasSgemm + bias_act kernel\",\n");
        std::fprintf(f, "  \"cublaslt\": \"cublasLtMatmul, CUBLASLT_EPILOGUE_RELU_BIAS / GELU_BIAS, "
                        "top heuristic algorithm, %zu MiB workspace\"\n}\n", ws_bytes >> 20);
        std::fclose(f);
    }
    std::printf("wrote %s\n", path.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string kernel_filter;
    int only_size = 0, warmup = 3, reps = 20;
    double tol = 1e-4;
    bool write = true, tune = false, tc = false, tc_tune = false, shapes_mode = false,
         epilogue_mode = false,
         prefix_set = false, tol_set = false;
    std::string prefix;
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
        else if (a == "--tol") { tol = std::atof(next()); tol_set = true; }
        else if (a == "--no-write") write = false;
        else if (a == "--tune") tune = true;
        else if (a == "--tc") tc = true;
        else if (a == "--tc-tune") tc = tc_tune = true;  // same FP16 path as --tc
        else if (a == "--shapes") shapes_mode = true;
        else if (a == "--epilogue") epilogue_mode = true;
        else if (a == "--prefix") { prefix = next(); prefix_set = true; }
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
    if (epilogue_mode) {
        const int rc = run_epilogue(env, handle, warmup, reps, tol, prefix, write);
        CUBLAS_CHECK(cublasDestroy(handle));
        return rc;
    }

    // ---------------------------------------------------------------- kernels
    const float one = 1.f, zero = 0.f;
    Launcher cublas = [&](int M, int N, int K, const float* A, const float* B, float* C) {
        // Row-major C = A B via column-major C^T = B^T A^T (see header comment).
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K,
                                 &zero, C, N));
    };
    std::vector<Kernel> all = {
        {"v0_naive", launch_v0_naive, resources_v0_naive, 2048},
        {"v1_coalesced", launch_v1_coalesced, resources_v1_coalesced, 1 << 30},
        {"v2_smem_tiling", launch_v2_smem_tiling, resources_v2_smem_tiling, 1 << 30},
        {"v3_1d_blocktiling", launch_v3_1d_blocktiling, resources_v3_1d_blocktiling, 1 << 30},
        {"v4_2d_blocktiling", launch_v4_2d_blocktiling, resources_v4_2d_blocktiling, 1 << 30},
        {"v5_vectorized", launch_v5_vectorized, resources_v5_vectorized, 1 << 30},
        {"v6_double_buffered", launch_v6_double_buffered, resources_v6_double_buffered, 1 << 30},
        {"v7_conflict_free", launch_v7_conflict_free, resources_v7_conflict_free, 1 << 30},
        {"v8_warptiling", launch_v8_warptiling, resources_v8_warptiling, 1 << 30},
        {"v9_autotuned", launch_v9_autotuned, resources_v9_autotuned, 1 << 30},
        {"v11_splitk", launch_v11_splitk, resources_v11_splitk, 1 << 30},
        {"v12_dispatch", launch_v12_dispatch, resources_v12_dispatch, 1 << 30},
        {"v15_splitk_det", launch_v15_splitk_det, resources_v15_splitk_det, 1 << 30},
    };
    // Tensor-core track: launchers read the FP16 copies (dAh, dBh) made for each size; the
    // reference is cublasGemmEx on the same FP16 inputs with FP32 output and FP32 compute.
    half *dAh = nullptr, *dBh = nullptr;
    Launcher reference = cublas;
    if (tc) {
        reference = [&](int M, int N, int K, const float*, const float*, float* C) {
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, dBh,
                                      CUDA_R_16F, N, dAh, CUDA_R_16F, K, &zero, C, CUDA_R_32F,
                                      N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        };
        all = {
            {"tc0_wmma_naive",
             [&](int M, int N, int K, const float*, const float*, float* C) {
                 launch_tc0_wmma_naive(M, N, K, dAh, dBh, C);
             },
             resources_tc0_wmma_naive, 1 << 30},
            {"tc1_wmma_smem",
             [&](int M, int N, int K, const float*, const float*, float* C) {
                 launch_tc1_wmma_smem(M, N, K, dAh, dBh, C);
             },
             resources_tc1_wmma_smem, 1 << 30},
            {"tc2_wmma_db",
             [&](int M, int N, int K, const float*, const float*, float* C) {
                 launch_tc2_wmma_db(M, N, K, dAh, dBh, C);
             },
             resources_tc2_wmma_db, 1 << 30},
            {"tc3_ptx_mma",
             [&](int M, int N, int K, const float*, const float*, float* C) {
                 launch_tc3_ptx_mma(M, N, K, dAh, dBh, C);
             },
             resources_tc3_ptx_mma, 1 << 30},
            {"tc4_ptx_bk32",
             [&](int M, int N, int K, const float*, const float*, float* C) {
                 launch_tc4_ptx_bk32(M, N, K, dAh, dBh, C);
             },
             resources_tc4_ptx_bk32, 1 << 30},
        };
        if (!prefix_set) prefix = "tc_";
        if (!tol_set) tol = 1e-3;
        if (tc_tune) {  // same FP16 inputs / reference / sizes, different kernel list
            all.clear();
            for (const TcConfig& c : tc5_tune_configs())
                all.push_back({c.name,
                               [&, f = c.launch](int M, int N, int K, const float*, const float*,
                                                 float* C) { f(M, N, K, dAh, dBh, C); },
                               c.resources, 1 << 30});
            if (!prefix_set) prefix = "tctune_";
        }
    }
    if (shapes_mode) {
        std::vector<Kernel> keep;
        for (auto& k : all)
            if (k.name == "v4_2d_blocktiling" || k.name == "v7_conflict_free" || k.name == "v9_autotuned" ||
                k.name == "v11_splitk" || k.name == "v12_dispatch" ||
                k.name == "v15_splitk_det")
                keep.push_back(k);
        all = keep;
        if (!prefix_set) prefix = "shapes_";
    }
    if (tune) {
        all.clear();
        for (const auto* list : {&v7_tune_configs(), &v8_tune_configs(), &v10_tune_configs(),
                                  &v13_tune_configs(), &v14_tune_configs()})
            for (const TuneConfig& c : *list) all.push_back({c.name, c.launch, c.resources, 1 << 30});
        if (!prefix_set) prefix = "tune_";
    }
    std::vector<Kernel> kernels;
    for (auto& k : all)
        if (kernel_filter.empty() || k.name.rfind(kernel_filter, 0) == 0) kernels.push_back(k);

    struct Shape {
        int M, N, K;
    };
    std::vector<Shape> shapes;
    std::vector<int> sizes = {256, 512, 1000, 1023, 1024, 2048, 4096};
    if (tc) sizes = {256, 512, 1024, 2048, 4096};  // WMMA kernels have no edge handling
    if (only_size) sizes = {only_size};
    for (int S : sizes) shapes.push_back({S, S, S});
    if (shapes_mode)  // M = tokens; N, K = model / FFN widths (BERT-base 768/3072, larger 1024/4096)
        shapes = {{4096, 4096, 4096}, {2048, 768, 768},  {2048, 3072, 768}, {2048, 768, 3072},
                  {4096, 1024, 4096}, {512, 4096, 4096}, {128, 4096, 4096}, {4096, 4096, 128}};
    std::vector<std::pair<int, double>> precision;  // tc: FP16-input result vs FP32 SGEMM

    // ---------------------------------------------------------------- run
    std::vector<Row> rows;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    GpuTimer timer;

    for (const Shape& sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K;
        const int S = std::max(M, std::max(N, K));
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
        if (tc) {
            CUDA_CHECK(cudaMalloc(&dAh, nA * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&dBh, nB * sizeof(half)));
            to_half<<<env.prop.multiProcessorCount * 8, 256>>>(dA, dAh, nA);
            to_half<<<env.prop.multiProcessorCount * 8, 256>>>(dB, dBh, nB);
            CUDA_CHECK(cudaGetLastError());
        }

        // Runs warm-ups and copies the output to hC. C is filled with 0xFF bytes (NaN)
        // first, so a kernel that skips elements cannot pass on values left behind by a
        // previous kernel.
        auto run_and_fetch = [&](const Launcher& f) {
            CUDA_CHECK(cudaMemset(dC, 0xFF, nC * sizeof(float)));
            for (int i = 0; i < std::max(warmup, 1); ++i) f(M, N, K, dA, dB, dC);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
        };
        auto time_one = [&](const Launcher& f) {
            timer.start();
            f(M, N, K, dA, dB, dC);
            return timer.stop_ms();
        };

        // Precision cost of FP16 inputs (tc only): FP32 SGEMM result vs the FP16 reference.
        std::vector<float> hSgemm;
        if (tc) {
            run_and_fetch(cublas);
            hSgemm = hC;
        }

        // Reference = cuBLAS output from its warm-up run; then its standalone timing.
        run_and_fetch(reference);
        hRef = hC;
        if (tc) precision.push_back({S, max_rel_err(hRef.data(), hSgemm.data(), nC)});
        std::vector<float> tref(reps);
        for (int r = 0; r < reps; ++r) tref[r] = time_one(reference);
        CUDA_CHECK(cudaGetLastError());
        const TimingStats sc = summarize(tref);
        rows.push_back({"cublas", M, N, K, sc, flop / (sc.median_ms * 1e6), 100.0, 0.0, "REF",
                        NAN, sample_gpu()});

        for (auto& k : kernels) {
            if (S > k.max_size) continue;
            run_and_fetch(k.launch);
            const double err = max_rel_err(hC.data(), hRef.data(), nC);
            // Interleaved: kernel, cuBLAS, kernel, cuBLAS, ... (see PAIRED cuBLAS TIMING).
            std::vector<float> tk(reps), tp(reps);
            for (int r = 0; r < reps; ++r) {
                tk[r] = time_one(k.launch);
                tp[r] = time_one(reference);
            }
            CUDA_CHECK(cudaGetLastError());
            const TimingStats st = summarize(tk), sp = summarize(tp);
            rows.push_back({k.name, M, N, K, st, flop / (st.median_ms * 1e6),
                            100.0 * sp.median_ms / st.median_ms, err,
                            (err <= tol) ? "PASS" : "FAILED", sp.median_ms, sample_gpu()});
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        if (tc) {
            CUDA_CHECK(cudaFree(dAh));
            CUDA_CHECK(cudaFree(dBh));
        }
    }

    const std::string nvsmi_end = nvidia_smi_snapshot();
    CUBLAS_CHECK(cublasDestroy(handle));

    // ---------------------------------------------------------------- report
    std::printf("\n%-22s %5s %10s %10s %10s %9s %8s %10s %s\n", "kernel", "size", "median_ms",
                "min_ms", "p90_ms", "GFLOP/s", "%cuBLAS", "max_rel_err", "status");
    for (auto& r : rows)
        std::printf("%-22s %5d %10.4f %10.4f %10.4f %9.1f %8.1f %10.2e %s\n", r.kernel.c_str(),
                    r.M, r.t.median_ms, r.t.min_ms, r.t.p90_ms, r.gflops, r.pct_cublas, r.err,
                    r.status.c_str());

    if (!write) {
        std::printf("\n--no-write: no files written\n");
        return 0;
    }

    const std::string csv_path = "results/" + prefix + "sgemm_" + env.gpu_tag + "_" + env.stamp + ".csv";
    const std::string env_path = "results/" + prefix + "env_" + env.gpu_tag + "_" + env.stamp + ".json";

    FILE* f = std::fopen(csv_path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot open %s (run from the repo root)\n", csv_path.c_str()); return 1; }
    std::fprintf(f, "kernel,M,N,K,median_ms,min_ms,p90_ms,gflops,pct_cublas,max_rel_err,status,"
                    "cublas_paired_median_ms,sm_clock_mhz_after,power_w_after\n");
    auto opt = [&](double v, const char* fmt) {
        if (std::isnan(v)) std::fprintf(f, ",");
        else { std::fprintf(f, ","); std::fprintf(f, fmt, v); }
    };
    for (auto& r : rows) {
        std::fprintf(f, "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.3f,%.3e,%s", r.kernel.c_str(), r.M,
                     r.N, r.K, r.t.median_ms, r.t.min_ms, r.t.p90_ms, r.gflops, r.pct_cublas,
                     r.err, r.status.c_str());
        opt(r.cublas_paired_ms, "%.6f");
        opt(r.after.sm_mhz, "%.0f");
        opt(r.after.power_w, "%.2f");
        std::fprintf(f, "\n");
    }
    std::fclose(f);

    f = std::fopen(env_path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", env_path.c_str()); return 1; }
    std::fprintf(f, "{\n");
    write_env_fields(f, env);
    std::fprintf(f, "  \"cublas_version\": %d,\n", cublas_ver);
    std::fprintf(f, "  \"cublas_math_mode\": \"CUBLAS_DEFAULT_MATH\",\n");
    std::fprintf(f, "  \"reference\": \"%s\",\n",
                 tc ? "cublasGemmEx: FP16 A and B, FP32 C, CUBLAS_COMPUTE_32F"
                    : "cublasSgemm: FP32");
    std::fprintf(f, "  \"warmup_runs\": %d,\n", warmup);
    std::fprintf(f, "  \"timed_runs\": %d,\n", reps);
    std::fprintf(f, "  \"tolerance_max_rel_err\": %g,\n", tol);
    std::fprintf(f, "  \"input_distribution\": \"uniform[-1,1], mt19937 seed 42\",\n");
    std::fprintf(f, "  \"results_csv\": %s,\n", json_str(csv_path).c_str());
    std::fprintf(f, "  \"nvidia_smi_start\": %s,\n", json_str(nvsmi_start).c_str());
    std::fprintf(f, "  \"nvidia_smi_end\": %s\n", json_str(nvsmi_end).c_str());
    std::fprintf(f, "}\n");
    std::fclose(f);

    // Per-kernel resources of this build: registers, spills, shared memory, occupancy.
    const std::string res_path = "results/" + prefix + "kernels_" + env.gpu_tag + "_" + env.stamp + ".csv";
    f = std::fopen(res_path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", res_path.c_str()); return 1; }
    const int max_warps = env.prop.maxThreadsPerMultiProcessor / 32;
    std::fprintf(f, "kernel,regs_per_thread,local_bytes_per_thread,static_smem_bytes,"
                    "threads_per_block,blocks_per_sm,warps_per_sm,max_warps_per_sm,"
                    "theoretical_occupancy_pct\n");
    for (auto& k : kernels) {
        const KernelResources r = k.resources();
        const int warps = r.blocks_per_sm * r.threads_per_block / 32;
        std::fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%.1f\n", k.name.c_str(), r.regs_per_thread,
                     r.local_bytes_per_thread, r.static_smem_bytes, r.threads_per_block,
                     r.blocks_per_sm, warps, max_warps, 100.0 * warps / max_warps);
    }
    std::fclose(f);

    if (tc) {
        const std::string prec_path =
            "results/" + prefix + "precision_" + env.gpu_tag + "_" + env.stamp + ".csv";
        f = std::fopen(prec_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", prec_path.c_str()); return 1; }
        std::fprintf(f, "N,max_rel_err_fp16_inputs_vs_fp32_sgemm\n");
        for (auto& p : precision) std::fprintf(f, "%d,%.3e\n", p.first, p.second);
        std::fclose(f);
        std::printf("wrote %s\n", prec_path.c_str());
    }
    std::printf("\nwrote %s\nwrote %s\nwrote %s\n", csv_path.c_str(), env_path.c_str(),
                res_path.c_str());
    return 0;
}
