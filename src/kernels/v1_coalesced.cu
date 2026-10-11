// v1_coalesced.cu — same algorithm as v0, threads remapped so a warp walks along a row of C.
//
// WHAT CHANGED vs v0
//   Only the thread-to-output mapping. v0 used row <- threadIdx.x, col <- threadIdx.y;
//   v1 uses col <- threadIdx.x, row <- threadIdx.y (and the grid's x/y swap to match).
//   Same 32x32 block, same loop, same number of loads and FMAs.
//
// WHY IT SHOULD BE FASTER
//   A warp is still 32 consecutive threadIdx.x with one threadIdx.y, but now that means one
//   row and 32 consecutive columns:
//     - B[k*N + col]: 32 consecutive floats = 128 bytes = 4 full 32-byte sectors per warp
//       load (coalesced). In v0 this was a broadcast; now it is the fully used wide load.
//     - A[row*K + k]: all 32 threads read the same address, a single broadcast. In v0 this
//       was the 32-sector strided load.
//     - C[row*N + col]: the store is contiguous, 4 sectors instead of 32.
//   Sectors fetched per warp per k drop from ~33 (32 for A + 1 for B) to ~5 (4 for B + 1
//   for A), so far less L1/L2 traffic for the same useful data.
//
// WHAT THE LIMIT IS NOW
//   Data reuse. Every FMA still needs two loads from the memory hierarchy: each element of
//   A is loaded by N different threads and each element of B by M different threads, and
//   only the caches stand between that and DRAM. Arithmetic intensity is still ~1 FMA per
//   two loads, so the kernel stays memory-bound. v2 stages tiles in shared memory so each
//   global element is loaded once per block instead of once per thread.

#include "kernels.h"
#include "../common.h"

namespace {

constexpr int BLOCK = 32;

__global__ void sgemm_v1_coalesced(int M, int N, int K, const float* __restrict__ A,
                                   const float* __restrict__ B, float* __restrict__ C) {
    const int row = blockIdx.y * BLOCK + threadIdx.y;
    const int col = blockIdx.x * BLOCK + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

}  // namespace

void launch_v1_coalesced(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid(ceil_div(N, BLOCK), ceil_div(M, BLOCK));
    sgemm_v1_coalesced<<<grid, block>>>(M, N, K, A, B, C);
}
