// v10_tuned.cu — the v7 kernel with two more tuning knobs, for the tuning study only.
//
// WHAT CHANGED vs v7
//   The device code is v7's (generated from it, so the inner loop is identical), plus:
//     MINB  the __launch_bounds__ minimum-blocks-per-SM hint becomes a template parameter.
//           It caps registers per thread at 65536 / (MINB x threads) so that MINB blocks fit
//           on an SM: more resident warps to hide latency, at the risk of register spills
//           (spills show up as local_bytes_per_thread in results/tune_kernels_*.csv).
//     SWZ   block swizzling. v7 walks the grid row by row, so the ~100 blocks resident at once
//           span few block-rows but every block-column, i.e. every column-panel of B. With
//           SWZ = 8 the grid is walked in groups of 8 block-columns (all rows of a group
//           first), so resident blocks share far fewer B panels in L2. CUTLASS calls this a
//           threadblock swizzle.
//
// WHY: the tuning study showed the best v7 shape (64x128, 128 threads) wins through occupancy
// (3 blocks per SM instead of 2). These knobs push occupancy and L2 locality further.
//
// WHAT THE LIMIT IS NOW
//   Measured, not assumed: see the tuning table. v9 dispatches across v7, v8 and v10 configs.

#include "kernels.h"

namespace {

template <int BM, int BN, int BK, int TM, int TN>
struct Cfg {
    static constexpr int TR = BM / TM;       // threads along M
    static constexpr int TC = BN / TN;       // threads along N
    static constexpr int THREADS = TR * TC;
    static constexpr int BMP = BM + 4;       // padded row stride of the transposed A tile
    static constexpr int MINB = THREADS >= 1024 ? 1 : 2;  // blocks/SM asked of the compiler
    static_assert(TM % 4 == 0 && TN % 4 == 0 && BK % 4 == 0, "float4 strips");
    static_assert(BM % TM == 0 && BN % TN == 0, "tile divisibility");
    static_assert(THREADS % 32 == 0 && THREADS <= 1024, "block size");
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

template <int BM, int BN, int BK, int TM, int TN, int MINB_, int SWZ>
__global__ void __launch_bounds__(Cfg<BM, BN, BK, TM, TN>::THREADS, MINB_)
sgemm_v10(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
         float* __restrict__ C) {
    using G = Cfg<BM, BN, BK, TM, TN>;
    constexpr int TR = G::TR, TC = G::TC, THREADS = G::THREADS, BMP = G::BMP;
    constexpr int SM_ = TM / 4, SN = TN / 4;            // 4-wide strips per thread
    constexpr int A_CHUNKS = BM * BK / 4, B_CHUNKS = BK * BN / 4;
    constexpr int A_PER = (A_CHUNKS + THREADS - 1) / THREADS;
    constexpr int B_PER = (B_CHUNKS + THREADS - 1) / THREADS;

    __shared__ __align__(16) float As[2][BK * BMP];  // transposed, padded: [buf][BK][BMP]
    __shared__ __align__(16) float Bs[2][BK * BN];   // [buf][BK][BN]

    const int tid = threadIdx.x;
    // Block swizzle: walk the grid in groups of SWZ block-columns (all rows of a group before
    // the next group), so blocks resident at the same time share B column-panels in L2.
    // SWZ = 0 keeps the plain row-major order.
    int bx = blockIdx.x, by = blockIdx.y;
    if (SWZ > 0) {
        const int gx = gridDim.x, gy = gridDim.y;
        const int lin = blockIdx.y * gx + blockIdx.x;
        const int group = lin / (SWZ * gy);
        const int first = group * SWZ;
        const int cols = min(gx - first, SWZ);
        const int in = lin - group * SWZ * gy;
        by = in / cols;
        bx = first + in % cols;
    }
    const int row0 = by * BM, col0 = bx * BN;
    const int tr = tid / TC, tc = tid % TC;
    const bool vecA = (K % 4) == 0, vecB = (N % 4) == 0;

    float4 ra[A_PER], rb[B_PER];  // next tile, staged in registers (double buffering)
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
            for (int s = 0; s < SM_; ++s) {
                const float4 v =
                    *reinterpret_cast<const float4*>(&As[cur][k * BMP + s * 4 * TR + tr * 4]);
                a[s * 4 + 0] = v.x; a[s * 4 + 1] = v.y; a[s * 4 + 2] = v.z; a[s * 4 + 3] = v.w;
            }
#pragma unroll
            for (int s = 0; s < SN; ++s) {
                const float4 v =
                    *reinterpret_cast<const float4*>(&Bs[cur][k * BN + s * 4 * TC + tc * 4]);
                b[s * 4 + 0] = v.x; b[s * 4 + 1] = v.y; b[s * 4 + 2] = v.z; b[s * 4 + 3] = v.w;
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

    // Output: acc[si*4 + i][sj*4 + j] is C[row0 + si*4*TR + tr*4 + i][col0 + sj*4*TC + tc*4 + j].
#pragma unroll
    for (int si = 0; si < SM_; ++si) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = row0 + si * 4 * TR + tr * 4 + i;
            if (r >= M) continue;
#pragma unroll
            for (int sj = 0; sj < SN; ++sj) {
                const int c = col0 + sj * 4 * TC + tc * 4;
                // Direct constant indices (no pointer into acc), so acc stays in registers.
                const float4 v = make_float4(acc[si * 4 + i][sj * 4 + 0], acc[si * 4 + i][sj * 4 + 1],
                                             acc[si * 4 + i][sj * 4 + 2], acc[si * 4 + i][sj * 4 + 3]);
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

template <int BM, int BN, int BK, int TM, int TN, int MINB_, int SWZ>
void launch_cfg(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_v10<BM, BN, BK, TM, TN, MINB_, SWZ><<<grid, Cfg<BM, BN, BK, TM, TN>::THREADS>>>(M, N, K, A, B, C);
}

template <int BM, int BN, int BK, int TM, int TN, int MINB_, int SWZ>
KernelResources resources_cfg() {
    return kernel_resources(sgemm_v10<BM, BN, BK, TM, TN, MINB_, SWZ>,
                            Cfg<BM, BN, BK, TM, TN>::THREADS);
}

}  // namespace

// Name: v10_<BM>x<BN>x<BK>_<TM>x<TN>_b<MINB>_s<SWZ>.
#define V10_CFG(bm, bn, bk, tm, tn, mb, sw)                                                     \
    TuneConfig {                                                                                \
        "v10_" #bm "x" #bn "x" #bk "_" #tm "x" #tn "_b" #mb "_s" #sw,                          \
            launch_cfg<bm, bn, bk, tm, tn, mb, sw>, resources_cfg<bm, bn, bk, tm, tn, mb, sw>   \
    }

const std::vector<TuneConfig>& v10_tune_configs() {
    static const std::vector<TuneConfig> configs = {
        // the v7 tuning winner, then each knob on its own and combined
        V10_CFG(64, 128, 8, 8, 8, 2, 0),
        V10_CFG(64, 128, 8, 8, 8, 2, 8),
        V10_CFG(64, 128, 8, 8, 8, 4, 0),   // 128 regs/thread cap -> 4 blocks/SM
        V10_CFG(64, 128, 8, 8, 8, 4, 8),
        V10_CFG(128, 64, 8, 8, 8, 2, 8),
        V10_CFG(128, 64, 8, 8, 8, 4, 0),
        // non-square thread tiles, 256 threads
        V10_CFG(64, 128, 8, 8, 4, 2, 0),
        V10_CFG(64, 128, 8, 4, 8, 2, 0),
        V10_CFG(64, 128, 8, 4, 8, 2, 8),
        // 128x128 with swizzle, 64x64 with more blocks
        V10_CFG(128, 128, 8, 8, 8, 2, 8),
        V10_CFG(64, 64, 8, 8, 8, 4, 0),
        V10_CFG(64, 64, 8, 8, 8, 6, 8),
        // tall-skinny tiles
        V10_CFG(32, 128, 8, 4, 8, 4, 0),
        V10_CFG(128, 32, 8, 8, 4, 4, 0),
        // shallower K tile: half the shared memory per block, twice the barriers
        V10_CFG(64, 128, 4, 8, 8, 4, 8),
        // round 2, around the round-1 winner (64x128, MINB 4)
        V10_CFG(64, 128, 8, 8, 8, 5, 0),   // 102-register cap: where do spills start to hurt?
        V10_CFG(64, 128, 8, 8, 8, 4, 4),
        V10_CFG(64, 128, 8, 8, 8, 4, 16),
        V10_CFG(96, 128, 8, 8, 8, 3, 0),   // 192 threads
        V10_CFG(64, 192, 8, 8, 8, 3, 0),   // 192 threads, wider B panel
        V10_CFG(64, 256, 8, 8, 8, 2, 0),   // 256 threads
        V10_CFG(128, 64, 8, 8, 8, 4, 8),
    };
    return configs;
}
