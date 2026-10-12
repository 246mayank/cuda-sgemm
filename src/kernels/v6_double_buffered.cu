// v6_double_buffered.cu — v5 plus double buffering: load tile t+1 while computing tile t.
//
// WHAT CHANGED vs v5
//   Same 128x128x8 tiles, 256 threads, 8x8 outputs per thread, float4 loads, transposed A,
//   same k-order of the sum. Only the schedule of the tile loop changes:
//     v5: load tile t -> sync -> compute tile t -> sync -> load tile t+1 -> ...
//     v6: two shared-memory buffers. While computing tile t from buffer (t % 2), each thread
//         has already issued its global loads for tile t+1 into registers; after the math it
//         writes them into the other buffer, then one __syncthreads().
//
// WHY IT SHOULD BE FASTER
//   In v5 every block spends part of each phase waiting for its global loads (hundreds of
//   cycles of DRAM/L2 latency) before it can compute, and only 2 blocks per SM can cover
//   for each other. In v6 the loads for the next tile are in flight during the 512 FFMAs of
//   the current tile, so their latency overlaps with math instead of adding to it. It also
//   halves the barriers: one __syncthreads() per tile instead of two.
//
// WHY ONE BARRIER IS ENOUGH
//   In iteration i threads read buffer cur and write buffer cur^1. The barrier at the end of
//   iteration i-1 guarantees that (a) everyone finished reading cur^1 (it was "cur" in i-1)
//   before anyone overwrites it, and (b) the writes to cur made in i-1 are visible in i.
//
// COST
//   Twice the shared memory (16 KB per block instead of 8 KB; 2 blocks still fit in a T4
//   SM's 64 KB) and 8 more registers per thread for the prefetched float4s, which must stay
//   within the 128-register cap of __launch_bounds__(256, 2) to keep 2 blocks per SM. The
//   measured registers and spills are in results/kernels_*.csv.
//
// WHAT THE LIMIT IS NOW
//   The 50% occupancy imposed by 64 accumulators per thread, and the power cap: a T4 cannot
//   sustain its boost clock at 70 W under a dense FFMA load, so even a perfect kernel runs at
//   the throttled clock. Beyond this, cuBLAS-class kernels add warp-level tiling, larger BK,
//   and per-size tile selection.

#include "kernels.h"

namespace {

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256
static_assert(BM * BK / 4 == THREADS && BK * BN / 4 == THREADS, "one float4 of each tile per thread");

// Same loader as v5: one 128-bit load when in bounds and aligned, else guarded scalars.
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

__global__ void __launch_bounds__(THREADS, 2)
sgemm_v6_double_buffered(int M, int N, int K, const float* __restrict__ A,
                         const float* __restrict__ B, float* __restrict__ C) {
    __shared__ __align__(16) float As[2][BK * BM];  // transposed: [buf][BK][BM]
    __shared__ __align__(16) float Bs[2][BK * BN];  // [buf][BK][BN]

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);
    const int aRow = tid / (BK / 4), aCol = (tid % (BK / 4)) * 4;
    const int bRow = tid / (BN / 4), bCol = (tid % (BN / 4)) * 4;
    const bool vecA = (K % 4) == 0, vecB = (N % 4) == 0;

    auto store_tile = [&](int buf, const float4& a, const float4& b) {
        As[buf][(aCol + 0) * BM + aRow] = a.x;
        As[buf][(aCol + 1) * BM + aRow] = a.y;
        As[buf][(aCol + 2) * BM + aRow] = a.z;
        As[buf][(aCol + 3) * BM + aRow] = a.w;
        *reinterpret_cast<float4*>(&Bs[buf][bRow * BN + bCol]) = b;
    };

    float acc[TM][TN] = {{0.f}};
    float regM[TM], regN[TN];

    // Prologue: tile 0 into buffer 0.
    store_tile(0, load4(A, row0 + aRow, aCol, M, K, vecA), load4(B, bRow, col0 + bCol, K, N, vecB));
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        const bool has_next = t + BK < K;
        float4 na, nb;
        if (has_next) {  // issue next tile's global loads before the math
            na = load4(A, row0 + aRow, t + BK + aCol, M, K, vecA);
            nb = load4(B, t + BK + bRow, col0 + bCol, K, N, vecB);
        }

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float4 m0 = *reinterpret_cast<const float4*>(&As[cur][k * BM + threadRow * TM]);
            const float4 m1 = *reinterpret_cast<const float4*>(&As[cur][k * BM + threadRow * TM + 4]);
            const float4 n0 = *reinterpret_cast<const float4*>(&Bs[cur][k * BN + threadCol * TN]);
            const float4 n1 = *reinterpret_cast<const float4*>(&Bs[cur][k * BN + threadCol * TN + 4]);
            regM[0] = m0.x; regM[1] = m0.y; regM[2] = m0.z; regM[3] = m0.w;
            regM[4] = m1.x; regM[5] = m1.y; regM[6] = m1.z; regM[7] = m1.w;
            regN[0] = n0.x; regN[1] = n0.y; regN[2] = n0.z; regN[3] = n0.w;
            regN[4] = n1.x; regN[5] = n1.y; regN[6] = n1.z; regN[7] = n1.w;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }

        if (has_next) store_tile(cur ^ 1, na, nb);
        __syncthreads();
        cur ^= 1;
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row0 + threadRow * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = col0 + threadCol * TN + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
    }
}

}  // namespace

void launch_v6_double_buffered(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_v6_double_buffered<<<grid, THREADS>>>(M, N, K, A, B, C);
}

KernelResources resources_v6_double_buffered() {
    return kernel_resources(sgemm_v6_double_buffered, THREADS);
}
