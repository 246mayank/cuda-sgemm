// v12_dispatch.cu — one entry point that picks between v9 (autotuned tiles) and v11 (split-K).
//
// WHAT CHANGED vs v9 / v11
//   No new device code. v9 picks a tile shape per problem size from the tuning study; v11 splits
//   K when the grid is too small to fill the GPU. Their measured results show each wins in a
//   different regime:
//     - v11 wins when its 64x128 grid has less than one full wave of blocks (fewer than
//       SMs x 4 slots, 4 = its blocks per SM): 1024^3 and 128 x 4096 x 4096 in the final run.
//     - v9 wins for tiny problems, where its small tiles fill the GPU without the cost of
//       split-K's memset and atomics (256^3, 512^3), and whenever the grid is already full.
//   v12 encodes exactly that: split-K if blocks(64x128) < SMs x 4 and cbrt(M*N*K) >= 768,
//   otherwise v9.
//
// HONEST MEASUREMENT
//   The rule was read off one benchmark run (results/sgemm_Tesla_T4_2026-10-12_170137.csv and
//   shapes_sgemm_..._170421.csv), so it is validated only by measuring v12 in a separate run.
//   It is a two-line heuristic, not a fitted model, and it is known to pick v9 on one shape
//   (512 x 4096 x 4096) where v11 measured better.

#include "kernels.h"

#include <cmath>

void launch_v12_dispatch(int M, int N, int K, const float* A, const float* B, float* C) {
    static int sms = 0;
    if (sms == 0) CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int blocks = ceil_div(M, 64) * ceil_div(N, 128);  // v11's block tile
    const int slots = sms * 4;                              // v11's blocks per SM
    const double size = std::cbrt((double)M * N * K);
    if (blocks < slots && size >= 768.0)
        launch_v11_splitk(M, N, K, A, B, C);
    else
        launch_v9_autotuned(M, N, K, A, B, C);
}

// Resources of the split-K configuration (the v9 side reports its own via v9_autotuned).
KernelResources resources_v12_dispatch() { return resources_v11_splitk(); }
