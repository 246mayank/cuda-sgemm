// v15_splitk_det.cu — v11's split-K made deterministic: workspace slabs + fixed-order sum.
//
// WHAT CHANGED vs v11
//   Same tiles, same slicing heuristic, same device code up to the epilogue (generated from v11).
//   v11 adds the K-slices' partial tiles into a zeroed C with atomicAdd, so the order of the
//   floating-point additions depends on which block finishes first: results can differ in the
//   last bits from run to run. v15 writes each slice's partial tile to its own slab of a
//   workspace W[z][M][N] with plain (128-bit where possible) stores, then a second kernel sums
//   the slabs in the fixed order z = 0, 1, ..., s-1 into C. With one slice (grid already full)
//   it writes C directly, exactly like v10.
//
// WHY
//   Bit-reproducible results: the same inputs give the same output every run, which is what
//   cuBLAS's deterministic mode and most training/inference frameworks require. No memset of C
//   and no atomics.
//
// COST
//   Workspace of s x M x N floats, and an extra pass that reads s x M x N floats and writes
//   M x N. For the skinny shapes split-K targets (small M), that extra traffic is small next to
//   the K-long main loop, but it is not free; compare v15 with v11 in the tables.

#include "kernels.h"

#include <algorithm>

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
sgemm_v15(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
          float* __restrict__ C, int kchunk, float* __restrict__ W) {
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
    // Split-K: this block sums only k in [kb, ke). kchunk is a multiple of BK, so every slice
    // except the last ends exactly on a tile boundary, and the last ends at K: tiles never
    // straddle a slice, and the existing bounds checks (against K) stay correct.
    const int kb = blockIdx.z * kchunk, ke = min(K, kb + kchunk);
    // With split-K, slice z writes its partial tile to its own workspace slab W[z]; a second
    // kernel sums the slabs in a fixed order. Without split, write C directly (= v10).
    float* const dst = gridDim.z > 1 ? W + (size_t)blockIdx.z * M * N : C;
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

    load_tile(kb);
    store_tile(0);
    __syncthreads();

    int cur = 0;
    for (int t = kb; t < ke; t += BK) {
        const bool has_next = t + BK < ke;
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
                float* out = dst + (size_t)r * N + c;
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

// C = sum over z = 0..s-1 of W[z], always in that order, so the result is bit-identical run to
// run. float4 when M*N % 4 == 0 (every slab then starts 16-byte aligned), scalar otherwise.
__global__ void sum_slices(const float* __restrict__ W, float* __restrict__ C, size_t mn, int s) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    if (mn % 4 == 0) {
        const size_t n4 = mn / 4;
        const float4* W4 = reinterpret_cast<const float4*>(W);
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += stride) {
            float4 acc = W4[i];
            for (int z = 1; z < s; ++z) {
                const float4 w = W4[(size_t)z * n4 + i];
                acc.x += w.x; acc.y += w.y; acc.z += w.z; acc.w += w.w;
            }
            reinterpret_cast<float4*>(C)[i] = acc;
        }
    } else {
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < mn; i += stride) {
            float acc = W[i];
            for (int z = 1; z < s; ++z) acc += W[(size_t)z * mn + i];
            C[i] = acc;
        }
    }
}

// Workspace for the slabs: grown on demand, reused across calls, never freed (a benchmark
// process; the OS reclaims it at exit). Growing frees the old buffer first.
float* workspace(size_t floats) {
    static float* buf = nullptr;
    static size_t cap = 0;
    if (floats > cap) {
        if (buf) CUDA_CHECK(cudaFree(buf));
        CUDA_CHECK(cudaMalloc(&buf, floats * sizeof(float)));
        cap = floats;
    }
    return buf;
}

template <int BM, int BN, int BK, int TM, int TN, int MINB_, int SWZ>
void launch_cfg(int M, int N, int K, const float* A, const float* B, float* C) {
    static int sms = 0;
    if (sms == 0) CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    // Enough K slices to give about two waves of blocks (sms x MINB slots per wave), at most 8,
    // and at least 256 of K per slice so each block still does real work.
    const int blocks = grid.x * grid.y, slots = sms * MINB_;
    int s = std::min(8, (2 * slots + blocks - 1) / blocks);
    s = std::max(1, std::min(s, K / 256));
    const int kchunk = ceil_div(ceil_div(K, s), BK) * BK;
    grid.z = ceil_div(K, kchunk);
    const size_t mn = (size_t)M * N;
    float* W = grid.z > 1 ? workspace(mn * grid.z) : nullptr;
    sgemm_v15<BM, BN, BK, TM, TN, MINB_, SWZ>
        <<<grid, Cfg<BM, BN, BK, TM, TN>::THREADS>>>(M, N, K, A, B, C, kchunk, W);
    if (grid.z > 1) sum_slices<<<sms * 8, 256>>>(W, C, mn, (int)grid.z);
}

template <int BM, int BN, int BK, int TM, int TN, int MINB_, int SWZ>
KernelResources resources_cfg() {
    return kernel_resources(sgemm_v15<BM, BN, BK, TM, TN, MINB_, SWZ>,
                            Cfg<BM, BN, BK, TM, TN>::THREADS);
}

}  // namespace

void launch_v15_splitk_det(int M, int N, int K, const float* A, const float* B, float* C) {
    launch_cfg<64, 128, 8, 8, 8, 4, 0>(M, N, K, A, B, C);
}

KernelResources resources_v15_splitk_det() { return resources_cfg<64, 128, 8, 8, 8, 4, 0>(); }
