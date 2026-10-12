// v8_warptiling.cu — v7 with an explicit warp-level tile: each warp owns a contiguous WMxWN
// sub-tile of the block tile, and each thread computes WMITER x WNITER 4x4 pieces inside it.
//
// WHAT CHANGED vs v7
//   Same 128x128x8 block tile, 256 threads, 64 outputs per thread, float4 global loads,
//   transposed + padded A tile, double buffering, conflict-free 128-bit shared reads, same
//   k-order of the sum. Only the mapping from (warp, lane) to outputs:
//     v7: thread tid -> (tr = tid / 16, tc = tid % 16); its 4x4 pieces are 64 rows/cols apart,
//         so one warp (tr in {0,1}, tc in 0..15) touches rows {0..7, 64..71} and ALL 128 columns.
//     v8: the 8 warps form a 2 x 4 grid of 64 x 32 warp tiles. Inside its warp tile, lane l
//         -> (trw = l / 4, tcw = l % 4); it owns rows wm*32 + trw*4 + 0..3 (wm = 0, 1) and
//         columns wn*16 + tcw*4 + 0..3 (wn = 0, 1).
//
// WHY IT SHOULD BE FASTER
//   Per k step every thread still loads 8 A + 8 B values (two float4 each), but what matters
//   for shared-memory bandwidth is how many DISTINCT values a warp requests:
//     v7 warp: A 2 rows-groups x 2 strips x 4 = 16 floats, B 16 lanes x 2 strips x 4 = 128 floats
//              -> 144 distinct floats per k
//     v8 warp: A 8 x 2 x 4 = 64 floats, B 4 x 2 x 4 = 32 floats -> 96 distinct floats per k
//   A square-ish warp tile reuses each loaded value across more lanes (broadcast), cutting the
//   shared-memory traffic per FMA by a third. It also matches how the hardware schedules
//   work (per warp), which is how CUTLASS/cuBLAS organize SGEMM: block tile -> warp tile ->
//   thread tile.
//
// BANKS
//   One 8-lane phase of an A read covers trw in {0,1}: 2 consecutive float4, a broadcast pair.
//   One phase of a B read covers tcw 0..3 (x2 trw): 4 consecutive float4 = 64 contiguous bytes.
//   Neither has two different addresses in one bank, so the reads stay conflict-free.
//
// WHAT THE LIMIT IS NOW
//   Same as v7 outside shared memory: the throttled clock, 50% occupancy from 64
//   accumulators per thread, and wave quantization at small N.

#include "kernels.h"

namespace {

// TM = TN = 4 (one float4 per piece). WMITER x WNITER pieces per thread per warp tile.
template <int BM, int BN, int BK, int WM, int WN, int WMITER, int WNITER>
struct WCfg {
    static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int WSUBM = WM / WMITER, WSUBN = WN / WNITER;  // one sub-tile per iteration
    static constexpr int LANES_N = WSUBN / 4, LANES_M = WSUBM / 4;  // lane grid in a sub-tile
    static constexpr int TM = WMITER * 4, TN = WNITER * 4;          // outputs per thread
    static constexpr int BMP = BM + 4;
    static constexpr int MINB = THREADS >= 1024 ? 1 : 2;
    static_assert(BM % WM == 0 && BN % WN == 0, "warp tiles must tile the block tile");
    static_assert(WM % WMITER == 0 && WN % WNITER == 0, "sub-tiles");
    static_assert(LANES_M * LANES_N == 32, "a sub-tile must be exactly one warp of 4x4 pieces");
    static_assert(BK % 4 == 0 && THREADS <= 1024, "shape");
};

__device__ __forceinline__ float4 load4(const float* __restrict__ p, int r, int c, int rows,
                                        int cols, bool vec_ok) {
    if (vec_ok && r < rows && c + 3 < cols)
        return *reinterpret_cast<const float4*>(p + (size_t)r * cols + c);
    float4 v;
    const bool rr = r < rows;
    const float* q = p + (size_t)r * cols + c;
    v.x = (rr && c + 0 < cols) ? q[0] : 0.f;
    v.y = (rr && c + 1 < cols) ? q[1] : 0.f;
    v.z = (rr && c + 2 < cols) ? q[2] : 0.f;
    v.w = (rr && c + 3 < cols) ? q[3] : 0.f;
    return v;
}

template <int BM, int BN, int BK, int WM, int WN, int WMITER, int WNITER>
__global__ void __launch_bounds__(WCfg<BM, BN, BK, WM, WN, WMITER, WNITER>::THREADS,
                                  WCfg<BM, BN, BK, WM, WN, WMITER, WNITER>::MINB)
sgemm_v8(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
         float* __restrict__ C) {
    using G = WCfg<BM, BN, BK, WM, WN, WMITER, WNITER>;
    constexpr int THREADS = G::THREADS, BMP = G::BMP, TM = G::TM, TN = G::TN;
    constexpr int WSUBM = G::WSUBM, WSUBN = G::WSUBN, LANES_N = G::LANES_N;
    constexpr int A_CHUNKS = BM * BK / 4, B_CHUNKS = BK * BN / 4;
    constexpr int A_PER = (A_CHUNKS + THREADS - 1) / THREADS;
    constexpr int B_PER = (B_CHUNKS + THREADS - 1) / THREADS;

    __shared__ __align__(16) float As[2][BK * BMP];
    __shared__ __align__(16) float Bs[2][BK * BN];

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int warp = tid / 32, lane = tid % 32;
    const int wr = (warp / G::WARPS_N) * WM;   // warp tile origin inside the block tile
    const int wc = (warp % G::WARPS_N) * WN;
    const int trw = lane / LANES_N, tcw = lane % LANES_N;
    const bool vecA = (K % 4) == 0, vecB = (N % 4) == 0;

    // Global -> shared copies: identical to v7 (same chunk mapping, padding, double buffer).
    float4 ra[A_PER], rb[B_PER];
    auto load_tile = [&](int t) {
#pragma unroll
        for (int p = 0; p < A_PER; ++p) {
            const int c = tid + p * THREADS;
            if (A_CHUNKS % THREADS == 0 || c < A_CHUNKS) {
                const int r = c / (BK / 4), kc = (c % (BK / 4)) * 4;
                ra[p] = load4(A, row0 + r, t + kc, M, K, vecA);
            }
        }
#pragma unroll
        for (int p = 0; p < B_PER; ++p) {
            const int c = tid + p * THREADS;
            if (B_CHUNKS % THREADS == 0 || c < B_CHUNKS) {
                const int r = c / (BN / 4), cc = (c % (BN / 4)) * 4;
                rb[p] = load4(B, t + r, col0 + cc, K, N, vecB);
            }
        }
    };
    auto store_tile = [&](int buf) {
#pragma unroll
        for (int p = 0; p < A_PER; ++p) {
            const int c = tid + p * THREADS;
            if (A_CHUNKS % THREADS == 0 || c < A_CHUNKS) {
                const int r = c / (BK / 4), kc = (c % (BK / 4)) * 4;
                As[buf][(kc + 0) * BMP + r] = ra[p].x;
                As[buf][(kc + 1) * BMP + r] = ra[p].y;
                As[buf][(kc + 2) * BMP + r] = ra[p].z;
                As[buf][(kc + 3) * BMP + r] = ra[p].w;
            }
        }
#pragma unroll
        for (int p = 0; p < B_PER; ++p) {
            const int c = tid + p * THREADS;
            if (B_CHUNKS % THREADS == 0 || c < B_CHUNKS) {
                const int r = c / (BN / 4), cc = (c % (BN / 4)) * 4;
                *reinterpret_cast<float4*>(&Bs[buf][r * BN + cc]) = rb[p];
            }
        }
    };

    float acc[TM][TN] = {{0.f}};

    load_tile(0);
    store_tile(0);
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        const bool has_next = t + BK < K;
        if (has_next) load_tile(t + BK);

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int wm = 0; wm < WMITER; ++wm) {
                const float4 v = *reinterpret_cast<const float4*>(
                    &As[cur][k * BMP + wr + wm * WSUBM + trw * 4]);
                a[wm * 4 + 0] = v.x; a[wm * 4 + 1] = v.y; a[wm * 4 + 2] = v.z; a[wm * 4 + 3] = v.w;
            }
#pragma unroll
            for (int wn = 0; wn < WNITER; ++wn) {
                const float4 v = *reinterpret_cast<const float4*>(
                    &Bs[cur][k * BN + wc + wn * WSUBN + tcw * 4]);
                b[wn * 4 + 0] = v.x; b[wn * 4 + 1] = v.y; b[wn * 4 + 2] = v.z; b[wn * 4 + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }

        if (has_next) store_tile(cur ^ 1);
        __syncthreads();
        cur ^= 1;
    }

    // acc[wm*4 + i][wn*4 + j] is C[row0 + wr + wm*WSUBM + trw*4 + i][col0 + wc + wn*WSUBN + tcw*4 + j]
#pragma unroll
    for (int wm = 0; wm < WMITER; ++wm) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = row0 + wr + wm * WSUBM + trw * 4 + i;
            if (r >= M) continue;
#pragma unroll
            for (int wn = 0; wn < WNITER; ++wn) {
                const int c = col0 + wc + wn * WSUBN + tcw * 4;
                const float4 v = make_float4(acc[wm * 4 + i][wn * 4 + 0], acc[wm * 4 + i][wn * 4 + 1],
                                             acc[wm * 4 + i][wn * 4 + 2], acc[wm * 4 + i][wn * 4 + 3]);
                float* out = C + (size_t)r * N + c;
                if (vecB && c + 3 < N) {
                    *reinterpret_cast<float4*>(out) = v;
                } else {
                    if (c + 0 < N) out[0] = v.x;
                    if (c + 1 < N) out[1] = v.y;
                    if (c + 2 < N) out[2] = v.z;
                    if (c + 3 < N) out[3] = v.w;
                }
            }
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int WMITER, int WNITER>
void launch_wcfg(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_v8<BM, BN, BK, WM, WN, WMITER, WNITER>
        <<<grid, WCfg<BM, BN, BK, WM, WN, WMITER, WNITER>::THREADS>>>(M, N, K, A, B, C);
}

template <int BM, int BN, int BK, int WM, int WN, int WMITER, int WNITER>
KernelResources resources_wcfg() {
    return kernel_resources(sgemm_v8<BM, BN, BK, WM, WN, WMITER, WNITER>,
                            WCfg<BM, BN, BK, WM, WN, WMITER, WNITER>::THREADS);
}

}  // namespace

void launch_v8_warptiling(int M, int N, int K, const float* A, const float* B, float* C) {
    launch_wcfg<128, 128, 8, 64, 32, 2, 2>(M, N, K, A, B, C);
}

KernelResources resources_v8_warptiling() { return resources_wcfg<128, 128, 8, 64, 32, 2, 2>(); }

// Warp-tile variants for the tuning study.
// Name: v8_<BM>x<BN>x<BK>_w<WM>x<WN>_<WMITER>x<WNITER>.
#define V8_CFG(bm, bn, bk, wm, wn, mi, ni)                                                    \
    TuneConfig {                                                                              \
        "v8_" #bm "x" #bn "x" #bk "_w" #wm "x" #wn "_" #mi "x" #ni,                           \
            launch_wcfg<bm, bn, bk, wm, wn, mi, ni>, resources_wcfg<bm, bn, bk, wm, wn, mi, ni> \
    }

const std::vector<TuneConfig>& v8_tune_configs() {
    static const std::vector<TuneConfig> configs = {
        V8_CFG(128, 128, 8, 64, 32, 2, 2),   // = v8: 8 warps, 64 outputs/thread
        V8_CFG(128, 128, 8, 32, 64, 2, 2),   // warp tile transposed
        V8_CFG(128, 128, 16, 64, 32, 2, 2),  // deeper K tile
        V8_CFG(128, 64, 8, 64, 32, 2, 2),    // 4 warps
        V8_CFG(64, 64, 8, 32, 32, 1, 2),     // 4 warps, 32 outputs/thread
        V8_CFG(64, 64, 16, 32, 32, 1, 2),
    };
    return configs;
}
