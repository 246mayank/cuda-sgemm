// v7_conflict_free.cu — v6 with a thread-fragment layout that has no shared-memory bank conflicts.
//
// WHAT CHANGED vs v6
//   Same 128x128x8 block tile, 256 threads, 64 outputs per thread, float4 global loads,
//   transposed A tile, double buffering, same k-order of the sum. Only WHICH outputs a thread
//   owns, and the row stride of the transposed A tile:
//     v6: thread (tr, tc) owns an 8x8 square: rows tr*8..tr*8+7, columns tc*8..tc*8+7.
//     v7: thread (tr, tc) owns four 4x4 pieces: rows {tr*4.., 64+tr*4..} x
//         columns {tc*4.., 64+tc*4..} (in general: TM/4 x TN/4 strips, 4*threads apart).
//     The transposed A tile is padded from 128 to 132 floats per row.
//
// WHY: SHARED-MEMORY BANKS AND 128-BIT LOADS
//   Shared memory has 32 banks of 4 bytes. A warp's 128-bit load (LDS.128) is served in 4
//   phases of 8 threads; a phase takes one cycle only if its 8 threads touch 8 different
//   16-byte slots spanning all 32 banks (or the same slot, a broadcast). In v6, the 8 threads
//   of a phase read B at columns tc*8 = 0, 8, 16, ... 56: slots 32 bytes apart, so threads
//   0 and 4, 1 and 5, ... hit the same banks and every phase needs 2 cycles. Nsight Compute
//   counted exactly 2 bank conflicts per shared-load instruction in v4-v6. In v7 the same 8
//   threads read columns tc*4 = 0, 4, 8, ... 28: 8 consecutive 16-byte slots = 128 contiguous
//   bytes, conflict-free. A reads are a broadcast (one tr per phase) in both layouts.
//   The transposed A store had a 2-way conflict in v5/v6 (rows 4 apart in k are 4*128 floats
//   apart, the same bank). With a row stride of 132 they are 4*132 = 528 floats apart, i.e.
//   16 banks apart. 132 keeps every row 16-byte aligned for the float4 reads.
//   Bonus: a thread's 4 output columns in a strip are now contiguous, so C is written with
//   128-bit stores (scalar fallback at edges or when N % 4 != 0).
//
// TEMPLATE
//   BM, BN, BK, TM, TN are template parameters so every configuration is fully unrolled with
//   its accumulators in registers. bench runs the 128x128x8 / 8x8 instance as v7; the tuning
//   study (bench --tune) runs the other instances listed at the end of this file.
//
// WHAT THE LIMIT IS NOW
//   With the conflicts gone, the remaining limits are the SM clock under the 70 W cap, 50%
//   occupancy (64 accumulators per thread), and tile-size effects at small N (wave
//   quantization: too few blocks for 40 SMs). The tuning study addresses the last one.

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

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(Cfg<BM, BN, BK, TM, TN>::THREADS, Cfg<BM, BN, BK, TM, TN>::MINB)
sgemm_v7(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
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
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
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

template <int BM, int BN, int BK, int TM, int TN>
void launch_cfg(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_v7<BM, BN, BK, TM, TN><<<grid, Cfg<BM, BN, BK, TM, TN>::THREADS>>>(M, N, K, A, B, C);
}

template <int BM, int BN, int BK, int TM, int TN>
KernelResources resources_cfg() {
    return kernel_resources(sgemm_v7<BM, BN, BK, TM, TN>, Cfg<BM, BN, BK, TM, TN>::THREADS);
}

}  // namespace

void launch_v7_conflict_free(int M, int N, int K, const float* A, const float* B, float* C) {
    launch_cfg<128, 128, 8, 8, 8>(M, N, K, A, B, C);
}

KernelResources resources_v7_conflict_free() { return resources_cfg<128, 128, 8, 8, 8>(); }

// Tuning study: every instance below is the same kernel with a different tile shape.
// Name format: v7_<BM>x<BN>x<BK>_<TM>x<TN> (block tile, thread tile).
#define V7_CFG(bm, bn, bk, tm, tn)                                                        \
    TuneConfig {                                                                          \
        "v7_" #bm "x" #bn "x" #bk "_" #tm "x" #tn, launch_cfg<bm, bn, bk, tm, tn>,        \
            resources_cfg<bm, bn, bk, tm, tn>                                             \
    }

const std::vector<TuneConfig>& v7_tune_configs() {
    static const std::vector<TuneConfig> configs = {
        V7_CFG(128, 128, 8, 8, 8),    // = v7
        V7_CFG(128, 128, 16, 8, 8),   // deeper K tile: half the barriers, 2x smem
        V7_CFG(128, 64, 8, 8, 8),     // 128 threads
        V7_CFG(64, 128, 8, 8, 8),
        V7_CFG(128, 64, 16, 8, 8),
        V7_CFG(64, 64, 8, 8, 8),      // 64 threads, 4x the blocks of 128x128
        V7_CFG(64, 64, 16, 8, 8),
        V7_CFG(64, 64, 8, 4, 4),      // 256 threads, 16 accumulators each
        V7_CFG(64, 64, 16, 4, 4),
        V7_CFG(128, 128, 8, 4, 4),    // 1024 threads
        V7_CFG(64, 32, 8, 4, 4),      // small tiles for small N
        V7_CFG(32, 32, 8, 4, 4),
    };
    return configs;
}
