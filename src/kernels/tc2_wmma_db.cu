// tc2_wmma_db.cu — tensor-core track, step 2: tc1 with double-buffered shared memory.
//
// WHAT CHANGED vs tc1
//   Same 128x128 block tile, 8 warps of 64x32, 4x2 accumulator fragments per warp, FP16 inputs
//   and FP32 accumulation, padded shared tiles. Two changes, one idea (overlap loads with MMA):
//     - two shared buffers: while the warps run their MMAs on tile t, each thread has already
//       issued the global loads for tile t+1 into registers and writes them into the other
//       buffer afterwards; one __syncthreads() per tile instead of two (the v6 scheme);
//     - K tile 16 instead of 32, so both buffers together (about 21 KB) still let 2 blocks
//       share an SM. Double-buffering tc1's 32-deep tiles would need about 38 KB per block,
//       i.e. 1 block per SM.
//   __launch_bounds__(256, 2) keeps registers at <= 128 so 2 blocks stay resident; any spill
//   shows in results/tc_kernels_*.csv.
//
// WHY IT SHOULD BE FASTER
//   Tensor cores finish a tile's MMAs much faster than FP32 cores finish the same FLOPs, so
//   the exposed global-load latency between tiles is a larger fraction of tc1's time. Nsight
//   Compute measured tc1's tensor pipe busy only part of the time; this targets the idle part.
//
// WHAT THE LIMIT IS NOW
//   Fragment loads still go through generic shared loads (no ldmatrix), and on Turing there
//   is no asynchronous global->shared copy (cp.async arrives with Ampere), so the prefetch
//   costs registers. Sizes: multiples of 128 only.

#include "kernels.h"

#include <mma.h>

namespace {

using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 16;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 64 x 32 per warp
constexpr int FM = WM / 16, FN = WN / 16;            // 4 x 2 fragments
constexpr int THREADS = WARPS_M * WARPS_N * 32;      // 256
constexpr int PAD = 8;
constexpr int LDA = BK + PAD, LDB = BN + PAD;        // 24, 136 halves
constexpr int A_CHUNKS = BM * BK / 8, B_CHUNKS = BK * BN / 8;  // 16-byte chunks: 256, 256
static_assert(A_CHUNKS == THREADS && B_CHUNKS == THREADS, "one chunk of each tile per thread");

__global__ void __launch_bounds__(THREADS, 2)
tc2_wmma_db_kernel(int M, int N, int K, const half* __restrict__ A, const half* __restrict__ B,
                   float* __restrict__ C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __shared__ __align__(32) half As[2][BM * LDA];
    __shared__ __align__(32) half Bs[2][BK * LDB];

    const int tid = threadIdx.x, warp = tid / 32;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int wr = (warp / WARPS_N) * WM, wc = (warp % WARPS_N) * WN;
    const int ar = tid / (BK / 8), ak = (tid % (BK / 8)) * 8;    // A chunk: row, k offset
    const int br = tid / (BN / 8), bc = (tid % (BN / 8)) * 8;    // B chunk: k row, column

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    uint4 ra = *reinterpret_cast<const uint4*>(&A[(size_t)(row0 + ar) * K + ak]);
    uint4 rb = *reinterpret_cast<const uint4*>(&B[(size_t)br * N + col0 + bc]);
    *reinterpret_cast<uint4*>(&As[0][ar * LDA + ak]) = ra;
    *reinterpret_cast<uint4*>(&Bs[0][br * LDB + bc]) = rb;
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        const bool has_next = t + BK < K;
        if (has_next) {  // next tile's global loads, in flight during the MMAs below
            ra = *reinterpret_cast<const uint4*>(&A[(size_t)(row0 + ar) * K + t + BK + ak]);
            rb = *reinterpret_cast<const uint4*>(&B[(size_t)(t + BK + br) * N + col0 + bc]);
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[FM];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[FN];
#pragma unroll
        for (int i = 0; i < FM; ++i)
            wmma::load_matrix_sync(a[i], &As[cur][(wr + i * 16) * LDA], LDA);
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::load_matrix_sync(b[j], &Bs[cur][wc + j * 16], LDB);
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);

        if (has_next) {
            *reinterpret_cast<uint4*>(&As[cur ^ 1][ar * LDA + ak]) = ra;
            *reinterpret_cast<uint4*>(&Bs[cur ^ 1][br * LDB + bc]) = rb;
        }
        __syncthreads();
        cur ^= 1;
    }

#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
            wmma::store_matrix_sync(C + (size_t)(row0 + wr + i * 16) * N + col0 + wc + j * 16,
                                    acc[i][j], N, wmma::mem_row_major);
#endif
}

}  // namespace

void launch_tc2_wmma_db(int M, int N, int K, const half* A, const half* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    tc2_wmma_db_kernel<<<grid, THREADS>>>(M, N, K, A, B, C);
}

KernelResources resources_tc2_wmma_db() { return kernel_resources(tc2_wmma_db_kernel, THREADS); }
