// v14_prefetch.cu — v10 plus register double buffering of the shared->register fragments.
//
// WHAT CHANGED vs v10
//   The device code is v10's (generated from it by a script: same tiles, same conflict-free
//   layout and padded transposed A tile, same global->shared double buffering, same MINB / SWZ
//   template knobs, same output stores). Only the inner k-loop changes. v10 loads the TM + TN
//   fragment values for step k from shared memory and then immediately runs that step's
//   TM x TN FFMAs, so every step starts with LDS latency exposed. v14 keeps two register sets:
//   while the FFMAs of step k run on one set, the other is filled with step k + 1's fragments.
//   This is the "prefetch the innermost dimension" step Boehm lists as remaining work, and what
//   CUTLASS calls fragment (register) double buffering.
//
// WHY IT SHOULD HELP
//   An LDS.128 takes tens of cycles; with 64 FFMAs per step the next step's loads can be issued
//   under the current step's math, so a warp stalls less often and needs fewer other resident
//   warps to hide latency. The compiler may already hoist some loads across steps of the
//   unrolled loop on its own; the measurement says whether explicit double buffering adds
//   anything.
//
// COST
//   TM + TN more live registers per thread (16 for 8x8). With MINB = 4 the compiler is capped
//   at 128 registers for 128-thread blocks, the same cap v10's best configuration already sits
//   at, so the b4 configs may spill: check local_bytes_per_thread in results/tune_kernels_*.csv.
//   The b2 / b3 configs give the compiler room (up to 255 / 170 registers) at the price of fewer
//   resident blocks. Which side wins is the experiment.
//
// WHAT TO LOOK AT IN THE RESULTS
//   v14_* vs the same-shaped v10_* in the tuning table (same MINB): equal or lower means the
//   compiler was already overlapping the loads, or spills ate the gain.

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
sgemm_v14(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
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

        // Fragment prefetch (register double buffering): two register sets. While the FFMAs for
        // step k run on set k & 1, set (k + 1) & 1 is filled from shared memory with the fragments
        // of step k + 1, so the shared-load latency hides behind the math. The loop is fully
        // unrolled, so k & 1 is a compile-time constant and both sets stay in registers.
        float a[2][TM], b[2][TN];
        auto load_frag = [&](int kk, int r) {
#pragma unroll
                for (int s = 0; s < SM_; ++s) {
                    const float4 v =
                        *reinterpret_cast<const float4*>(&As[cur][kk * BMP + s * 4 * TR + tr * 4]);
                    a[r][s * 4 + 0] = v.x; a[r][s * 4 + 1] = v.y; a[r][s * 4 + 2] = v.z; a[r][s * 4 + 3] = v.w;
                }
#pragma unroll
                for (int s = 0; s < SN; ++s) {
                    const float4 v =
                        *reinterpret_cast<const float4*>(&Bs[cur][kk * BN + s * 4 * TC + tc * 4]);
                    b[r][s * 4 + 0] = v.x; b[r][s * 4 + 1] = v.y; b[r][s * 4 + 2] = v.z; b[r][s * 4 + 3] = v.w;
                }
        };
        load_frag(0, 0);  // step 0 of this tile, right after the barrier
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            if (k + 1 < BK) load_frag(k + 1, (k + 1) & 1);  // never reads past this tile's BK
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[k & 1][i] * b[k & 1][j];
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
    sgemm_v14<BM, BN, BK, TM, TN, MINB_, SWZ><<<grid, Cfg<BM, BN, BK, TM, TN>::THREADS>>>(M, N, K, A, B, C);
}

template <int BM, int BN, int BK, int TM, int TN, int MINB_, int SWZ>
KernelResources resources_cfg() {
    return kernel_resources(sgemm_v14<BM, BN, BK, TM, TN, MINB_, SWZ>,
                            Cfg<BM, BN, BK, TM, TN>::THREADS);
}

}  // namespace

// Name: v14_<BM>x<BN>x<BK>_<TM>x<TN>_b<MINB>_s<SWZ>.
#define V14_CFG(bm, bn, bk, tm, tn, mb, sw)                                                     \
    TuneConfig {                                                                                \
        "v14_" #bm "x" #bn "x" #bk "_" #tm "x" #tn "_b" #mb "_s" #sw,                          \
            launch_cfg<bm, bn, bk, tm, tn, mb, sw>, resources_cfg<bm, bn, bk, tm, tn, mb, sw>   \
    }

const std::vector<TuneConfig>& v14_tune_configs() {
    static const std::vector<TuneConfig> configs = {
        V14_CFG(64, 128, 8, 8, 8, 2, 0),
        V14_CFG(64, 128, 8, 8, 8, 3, 0),
        V14_CFG(64, 128, 8, 8, 8, 4, 0),
        V14_CFG(128, 64, 8, 8, 8, 3, 0),
        V14_CFG(128, 64, 8, 8, 8, 4, 0),
        V14_CFG(128, 128, 8, 8, 8, 2, 0),
    };
    return configs;
}
