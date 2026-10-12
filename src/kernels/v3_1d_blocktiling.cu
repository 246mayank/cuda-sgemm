// v3_1d_blocktiling.cu — each thread computes TM = 8 outputs in a column of C (1D register tile).
//
// WHAT CHANGED vs v2
//   v2: 32x32 block tile, 1024 threads, 1 output per thread.
//   v3: 64x64 block tile (BM x BN), BK = 8 deep, 512 threads, and each thread accumulates
//   TM = 8 vertically adjacent outputs in registers (acc[8]). Per k step a thread reads one
//   value of B from shared memory into a register and reuses it for 8 multiply-adds, one per
//   row of its column. Same k-order of the sum as v0-v2, same zero-padding at the edges.
//   (The tile shape had to change too: a 64x64 output tile with 8 outputs per thread is
//   what makes 8 outputs per thread possible with a sensible thread count.)
//
// WHY IT SHOULD BE FASTER
//   1. Fewer shared-memory loads per FMA. v2 did 2 LDS per FMA (one from each tile). v3 does
//      1 load of B + 8 loads of A for 8 FMAs = 1.125 per FMA, and the 8 A values for a given
//      k sit at a fixed stride, so the compiler can keep the inner loop compact.
//   2. Twice the global-memory reuse. Per tile phase a block loads 64x8 + 8x64 = 1024 floats
//      and performs 64x64x8 = 32768 FMAs: 32 FMAs per float loaded, vs 16 for v2's 32x32
//      tile (2048 floats for 32768 FMAs).
//   3. 512 threads per block means 2 blocks can share an SM on a T4 (1024 threads per SM),
//      so while one block waits at __syncthreads() the other can issue.
//
// SHARED-MEMORY ACCESS PATTERN (one warp = 32 consecutive tid)
//   threadCol = tid % 64 varies across the warp, threadRow = tid / 64 is fixed:
//     Bs[k][threadCol]          32 consecutive floats -> 32 different banks, conflict-free
//     As[threadRow*8 + i][k]    same address for the whole warp -> broadcast
//
// WHAT THE LIMIT IS NOW
//   Still mostly A: each output needs its own A value from shared memory, so the ratio is
//   only improved for B. And the thread tile is a 8x1 strip, so B values are reused 8 times
//   but A values only once per thread. v4 makes the thread tile 2D (8x8): 8 A values and 8 B
//   values in registers feed 64 FMAs, i.e. 16 loads per 64 FMAs = 0.25 per FMA.
//   Small sizes also suffer: at N = 256 there are only (256/64)^2 = 16 blocks for 40 SMs.

#include "kernels.h"

namespace {

constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
constexpr int THREADS = BM * BN / TM;  // 512
static_assert(THREADS == BM * BK && THREADS == BK * BN, "one element of each tile per thread");

__global__ void __launch_bounds__(THREADS)
sgemm_v3_1d_blocktiling(int M, int N, int K, const float* __restrict__ A,
                        const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[BM * BK];  // [BM][BK]
    __shared__ float Bs[BK * BN];  // [BK][BN]

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    // Compute mapping: this thread owns C[row0 + threadRow*TM + i][col0 + threadCol], i < TM.
    const int threadCol = tid % BN;
    const int threadRow = tid / BN;
    // Load mapping: one element of the A tile and one of the B tile per thread.
    const int aRow = tid / BK, aCol = tid % BK;
    const int bRow = tid / BN, bCol = tid % BN;

    float acc[TM] = {0.f};
    for (int t = 0; t < K; t += BK) {
        const int ar = row0 + aRow, ac = t + aCol;
        As[aRow * BK + aCol] = (ar < M && ac < K) ? A[ar * K + ac] : 0.f;
        const int br = t + bRow, bc = col0 + bCol;
        Bs[bRow * BN + bCol] = (br < K && bc < N) ? B[br * N + bc] : 0.f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float b = Bs[k * BN + threadCol];
#pragma unroll
            for (int i = 0; i < TM; ++i) acc[i] += As[(threadRow * TM + i) * BK + k] * b;
        }
        __syncthreads();
    }

    const int c = col0 + threadCol;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row0 + threadRow * TM + i;
        if (r < M && c < N) C[r * N + c] = acc[i];
    }
}

}  // namespace

void launch_v3_1d_blocktiling(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_v3_1d_blocktiling<<<grid, THREADS>>>(M, N, K, A, B, C);
}

KernelResources resources_v3_1d_blocktiling() {
    return kernel_resources(sgemm_v3_1d_blocktiling, THREADS);
}
