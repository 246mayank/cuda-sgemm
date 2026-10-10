// v0_naive.cu — baseline: one thread per element of C.
//
// WHAT CHANGED vs previous version
//   Nothing; this is the starting point. Thread (threadIdx.x, threadIdx.y) of a 32x32 block
//   computes C[row][col] with row = blockIdx.x*32 + threadIdx.x and
//   col = blockIdx.y*32 + threadIdx.y, looping over all K products.
//
// WHY IT IS SLOW (expected, to be confirmed by measurement)
//   A warp is 32 threads with consecutive threadIdx.x and the same threadIdx.y, so within a
//   warp *row* varies and *col* is fixed:
//     - A[row*K + k]: 32 threads read 32 different rows, K floats apart. Each 4-byte value
//       sits in a different 32-byte sector, so one warp-wide load touches 32 sectors to
//       deliver 128 useful bytes (uncoalesced). Some of those sectors are reused from L1
//       on the next few values of k, but only if they survive in cache.
//     - B[k*N + col]: all 32 threads read the same address, a single broadcast.
//     - C[row*N + col]: the final store is also strided by N, so also uncoalesced.
//   Every operand is re-fetched from global memory (L1/L2 at best) for every
//   multiply-add, with no explicit reuse: 2 loads per FMA.
//
// WHAT THE LIMIT IS NOW
//   Memory transactions on A, not arithmetic. The fix (v1) is to swap the mapping so that
//   threadIdx.x walks along a row of C: then B and C accesses become contiguous across a
//   warp and A becomes the broadcast.

#include "kernels.h"
#include "../common.h"

namespace {

constexpr int BLOCK = 32;

__global__ void sgemm_v0_naive(int M, int N, int K, const float* __restrict__ A,
                               const float* __restrict__ B, float* __restrict__ C) {
    const int row = blockIdx.x * BLOCK + threadIdx.x;
    const int col = blockIdx.y * BLOCK + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

}  // namespace

void launch_v0_naive(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid(ceil_div(M, BLOCK), ceil_div(N, BLOCK));
    sgemm_v0_naive<<<grid, block>>>(M, N, K, A, B, C);
}
