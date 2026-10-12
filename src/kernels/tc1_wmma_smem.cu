// tc1_wmma_smem.cu — tensor-core track, step 1: WMMA with shared-memory tiles.
//
// WHAT CHANGED vs tc0
//   Same WMMA 16x16x16 FP16 fragments with FP32 accumulation. The block now owns a 128x128
//   tile of C, made of 8 warps in a 2 (M) x 4 (N) grid; each warp owns 64x32 = 4 x 2
//   fragments, so it keeps 8 accumulator fragments. Per 32-deep K step the block copies a
//   128x32 tile of A and a 32x128 tile of B into shared memory (16-byte vector loads, 8 halves
//   each), syncs, and every warp loads its fragments from shared memory: 4 A fragments and
//   2 B fragments feed 8 mma_sync calls.
//
// WHY IT SHOULD BE FASTER
//   Reuse, exactly as v2 and v4 did for FP32. Each A fragment loaded by a warp is used for
//   2 MMAs and each B fragment for 4, and each global element is loaded once per block
//   instead of once per warp. The tensor cores are fed from shared memory instead of L2.
//
// SHARED-MEMORY PADDING
//   Rows of the A tile are 32 halves = 64 bytes, rows of the B tile 128 halves = 256 bytes.
//   load_matrix_sync reads 16 rows of a fragment at that stride; with a 64- or 256-byte
//   stride many of those rows map to the same banks. Padding each row by 8 halves (16 bytes) shifts
//   consecutive rows by 4 banks. The padded strides (40 and 136 halves) keep the 32-byte
//   alignment and multiple-of-8 stride that WMMA requires.
//
// WHAT THE LIMIT IS NOW
//   One shared buffer and two barriers per K step (no double buffering), and fragment
//   loads go through generic shared loads rather than the ldmatrix instruction CUTLASS uses.
//   Sizes: multiples of 128 only (no edge handling).

#include "kernels.h"

#include <mma.h>

namespace {

using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WARPS_M = 2, WARPS_N = 4;            // warp grid
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 64 x 32 per warp
constexpr int FM = WM / 16, FN = WN / 16;            // 4 x 2 fragments per warp
constexpr int THREADS = WARPS_M * WARPS_N * 32;      // 256
constexpr int PAD = 8;                               // halves
constexpr int LDA = BK + PAD, LDB = BN + PAD;        // 40, 136

__global__ void __launch_bounds__(THREADS)
tc1_wmma_smem_kernel(int M, int N, int K, const half* __restrict__ A, const half* __restrict__ B,
                     float* __restrict__ C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __shared__ __align__(32) half As[BM * LDA];
    __shared__ __align__(32) half Bs[BK * LDB];

    const int tid = threadIdx.x, warp = tid / 32;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int wr = (warp / WARPS_N) * WM, wc = (warp % WARPS_N) * WN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    for (int t = 0; t < K; t += BK) {
        // A tile: 128 rows x 32 halves = 512 chunks of 8 halves (16 bytes); 2 per thread.
#pragma unroll
        for (int p = 0; p < (BM * BK / 8) / THREADS; ++p) {
            const int c = tid + p * THREADS;
            const int r = c / (BK / 8), kc = (c % (BK / 8)) * 8;
            *reinterpret_cast<uint4*>(&As[r * LDA + kc]) =
                *reinterpret_cast<const uint4*>(&A[(size_t)(row0 + r) * K + t + kc]);
        }
        // B tile: 32 rows x 128 halves = 512 chunks; 2 per thread.
#pragma unroll
        for (int p = 0; p < (BK * BN / 8) / THREADS; ++p) {
            const int c = tid + p * THREADS;
            const int r = c / (BN / 8), cc = (c % (BN / 8)) * 8;
            *reinterpret_cast<uint4*>(&Bs[r * LDB + cc]) =
                *reinterpret_cast<const uint4*>(&B[(size_t)(t + r) * N + col0 + cc]);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[FN];
#pragma unroll
            for (int i = 0; i < FM; ++i)
                wmma::load_matrix_sync(a[i], &As[(wr + i * 16) * LDA + kk], LDA);
#pragma unroll
            for (int j = 0; j < FN; ++j)
                wmma::load_matrix_sync(b[j], &Bs[kk * LDB + wc + j * 16], LDB);
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
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

void launch_tc1_wmma_smem(int M, int N, int K, const half* A, const half* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    tc1_wmma_smem_kernel<<<grid, THREADS>>>(M, N, K, A, B, C);
}

KernelResources resources_tc1_wmma_smem() { return kernel_resources(tc1_wmma_smem_kernel, THREADS); }
