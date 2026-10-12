// v2_smem_tiling.cu — v1 plus 32x32 tiles of A and B staged in shared memory.
//
// WHAT CHANGED vs v1
//   Same 32x32 block, same thread-to-output mapping (threadIdx.x -> col, threadIdx.y -> row),
//   same k-order of the sum. The K loop is now split into K/32 phases. In each phase the
//   block cooperatively copies one 32x32 tile of A and one of B into shared memory (one
//   element per thread per tile, both copies coalesced), synchronizes, and then every thread
//   runs its 32 multiply-adds out of shared memory.
//
// WHY IT SHOULD BE FASTER
//   Reuse. In v1 each element of A was requested from the memory system by N different
//   threads and each element of B by M threads; only L1/L2 hits kept that off DRAM. Now each
//   element is fetched from global memory once per *block* that needs it: an A element by
//   N/32 blocks and a B element by M/32 blocks. Global load instructions drop 32x, from 2 per
//   FMA to 2 per 32 FMAs. The inner loop reads shared memory instead:
//     - As[ty][k]: all 32 threads of a warp share ty, so this is a broadcast.
//     - Bs[k][tx]: 32 consecutive floats map to the 32 different banks: conflict-free.
//   Boundary handling: out-of-range tile elements are written as 0, so the extra terms add
//   exactly 0 and sizes like 1000 and 1023 need no special case in the inner loop.
//
// WHY TWO __syncthreads() PER PHASE
//   The first makes sure the whole tile is written before anyone reads it. The second makes
//   sure everyone has finished reading the tile before the next phase overwrites it.
//
// WHAT THE LIMIT IS NOW
//   Shared-memory traffic and instruction mix. Every FMA still needs two shared-memory loads
//   (one is a broadcast), so the inner loop is ~2 LDS per FFMA and the SM spends as many
//   issue slots on loads as on math. 1024 threads per block also means at most one block per
//   SM on a T4 (1024 threads/SM max), so at each __syncthreads() the whole SM waits for the
//   slowest warp with no other block to switch to. v3 has each thread compute several
//   outputs so one shared-memory value feeds several FMAs from registers.

#include "kernels.h"
#include "../common.h"

namespace {

constexpr int TILE = 32;

__global__ void sgemm_v2_smem_tiling(int M, int N, int K, const float* __restrict__ A,
                                     const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    for (int t = 0; t < K; t += TILE) {
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

}  // namespace

void launch_v2_smem_tiling(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    sgemm_v2_smem_tiling<<<grid, block>>>(M, N, K, A, B, C);
}

KernelResources resources_v2_smem_tiling() { return kernel_resources(sgemm_v2_smem_tiling, TILE * TILE); }
