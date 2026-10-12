// tc0_wmma_naive.cu — tensor-core track, step 0: one warp per 16x16 tile of C using WMMA.
//
// SEPARATE TRACK. The FP32 series (v0-v8) multiplies FP32 inputs on the FP32 cores. This track
// feeds FP16 (half) copies of A and B to the tensor cores and accumulates in FP32, which is a
// different numerical problem. It is therefore verified against cublasGemmEx with the SAME
// FP16 inputs, not against SGEMM, and compared only with that cuBLAS call. bench --tc also
// records how far FP16-input results are from FP32 SGEMM (the precision cost), separately.
//
// WHAT IT DOES
//   WMMA (warp matrix multiply-accumulate, <mma.h>) is the CUDA C++ API for tensor cores. A
//   warp cooperatively holds a 16x16 "fragment" of A, of B and of the accumulator; one
//   mma_sync computes acc += A_frag * B_frag (16x16x16 = 4096 FMAs) on the tensor cores.
//   Here every warp owns one 16x16 tile of C and walks K in steps of 16, loading its A and B
//   fragments straight from global memory (load_matrix_sync with the row stride K or N).
//
// WHY IT IS FASTER THAN FP32 (expected)
//   A T4 SM has 8 tensor cores; each does a 4x4x4 FP16 matrix FMA per clock, i.e. 64 FMAs, vs
//   64 FP32 FMAs per clock for the whole SM's CUDA cores. So peak math is 8x higher.
//
// WHAT THE LIMIT IS NOW
//   The same as v1 was for FP32: no data reuse. Each fragment of A is reloaded by every warp
//   in its row of C, straight from global memory/L2, so the tensor cores mostly wait for data.
//   tc1 stages tiles in shared memory.
//
// SIZES: WMMA fragments are 16x16, so this track runs only sizes that are multiples of 128
// (no edge handling). Requires sm_70+ (Volta or newer); on older GPUs the kernel is empty.

#include "kernels.h"

#include <mma.h>

namespace {

using namespace nvcuda;

constexpr int WARPS_M = 2, WARPS_N = 2;  // 4 warps -> a 32x32 tile of C per block

__global__ void tc0_wmma_naive_kernel(int M, int N, int K, const half* __restrict__ A,
                                      const half* __restrict__ B, float* __restrict__ C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const int warp = threadIdx.x / 32;
    const int row = (blockIdx.y * WARPS_M + warp / WARPS_N) * 16;
    const int col = (blockIdx.x * WARPS_N + warp % WARPS_N) * 16;
    if (row >= M || col >= N) return;  // whole warp exits together; no barriers below

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + (size_t)row * K + k, K);
        wmma::load_matrix_sync(b, B + (size_t)k * N + col, N);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(C + (size_t)row * N + col, acc, N, wmma::mem_row_major);
#endif
}

}  // namespace

void launch_tc0_wmma_naive(int M, int N, int K, const half* A, const half* B, float* C) {
    dim3 grid(ceil_div(N, 16 * WARPS_N), ceil_div(M, 16 * WARPS_M));
    tc0_wmma_naive_kernel<<<grid, 32 * WARPS_M * WARPS_N>>>(M, N, K, A, B, C);
}

KernelResources resources_tc0_wmma_naive() {
    return kernel_resources(tc0_wmma_naive_kernel, 32 * WARPS_M * WARPS_N);
}
