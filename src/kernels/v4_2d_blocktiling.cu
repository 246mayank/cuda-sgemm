// v4_2d_blocktiling.cu — each thread computes a TM x TN = 8x8 tile of C (2D register tile).
//
// WHAT CHANGED vs v3
//   The per-thread tile goes from an 8x1 column to an 8x8 square, and the block tile from
//   64x64 to 128x128 (BK stays 8), so a block is (128/8) x (128/8) = 256 threads, each
//   holding 64 accumulators. Per k step a thread copies 8 values of A (its 8 rows) and 8
//   values of B (its 8 columns) from shared memory into registers, then does the 8x8
//   outer product: 64 FMAs. Each thread loads 4 elements of each tile per phase (1024
//   elements / 256 threads). Same k-order of the sum as before, same zero-padding.
//
// WHY IT SHOULD BE FASTER
//   1. Shared-memory loads per FMA: v3 needed 9 for 8 FMAs (1.125 per FMA); v4 needs 16 for
//      64 FMAs (0.25 per FMA). The SM now spends most issue slots on FFMA, not LDS.
//   2. Global reuse doubles again: per phase 128x8 + 8x128 = 2048 floats for 128x128x8 =
//      131072 FMAs, i.e. 64 FMAs per float loaded (v3: 32).
//
// REGISTER PRESSURE AND OCCUPANCY (sm_75: 65536 registers and 1024 threads per SM)
//   64 accumulators + 16 operand registers + addressing is ~100 registers per thread.
//   __launch_bounds__(256, 2) asks the compiler to fit 2 blocks per SM, which caps it at
//   65536 / (2 x 256) = 128 registers per thread. That gives 16 resident warps per SM (50%
//   occupancy): fewer warps than v3, but each one has 8x more independent FMAs to issue,
//   which is how this kernel hides latency (instruction-level rather than thread-level
//   parallelism). The measured registers/spills/blocks per SM are written by bench to
//   results/kernels_*.csv; a nonzero local_bytes_per_thread there would mean spills.
//
// WHAT THE LIMIT IS NOW
//   - Shared-memory bank conflicts. In a warp, threadCol = tid % 16 and threadRow = tid / 16.
//     Bs[k][threadCol*8 + j] puts 16 different addresses 8 floats apart in one request, so
//     threads 0, 4, 8, 12 hit the same bank; As[threadRow*8 + i][k] gives 2 addresses 64
//     floats apart, also the same bank. Nsight Compute's bank-conflict counter checks this.
//   - Global loads of A are only 8 floats (32 bytes) wide per row, and all loads are scalar
//     (4 bytes per instruction). v5 uses 128-bit loads and transposes A in shared memory.
//   - Small sizes: at N = 256 there are only 4 blocks for 40 SMs.

#include "kernels.h"

namespace {

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256
constexpr int A_STRIDE = THREADS / BK;          // rows of the A tile loaded per pass: 32
constexpr int B_STRIDE = THREADS / BN;          // rows of the B tile loaded per pass: 2

__global__ void __launch_bounds__(THREADS, 2)
sgemm_v4_2d_blocktiling(int M, int N, int K, const float* __restrict__ A,
                        const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[BM * BK];  // [BM][BK]
    __shared__ float Bs[BK * BN];  // [BK][BN]

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    // Compute mapping: this thread owns the 8x8 tile starting at
    // C[row0 + threadRow*TM][col0 + threadCol*TN].
    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);
    // Load mapping.
    const int aRow = tid / BK, aCol = tid % BK;
    const int bRow = tid / BN, bCol = tid % BN;

    float acc[TM][TN] = {{0.f}};
    float regM[TM], regN[TN];

    for (int t = 0; t < K; t += BK) {
#pragma unroll
        for (int off = 0; off < BM; off += A_STRIDE) {
            const int r = aRow + off, gr = row0 + r, gc = t + aCol;
            As[r * BK + aCol] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
#pragma unroll
        for (int off = 0; off < BK; off += B_STRIDE) {
            const int r = bRow + off, gr = t + r, gc = col0 + bCol;
            Bs[r * BN + bCol] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + k];
#pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[k * BN + threadCol * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
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

void launch_v4_2d_blocktiling(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_v4_2d_blocktiling<<<grid, THREADS>>>(M, N, K, A, B, C);
}

KernelResources resources_v4_2d_blocktiling() {
    return kernel_resources(sgemm_v4_2d_blocktiling, THREADS);
}
