// kernels.h — host-side entry points, one pair per kernel file in src/kernels/.
// All compute C = A * B for row-major FP32 matrices: A is MxK, B is KxN, C is MxN.
// Pointers are device pointers. Launchers do not synchronize.
// resources_*() reports registers, spills, shared memory and blocks per SM of the kernel.
#pragma once

#include "../common.h"

void launch_v0_naive(int M, int N, int K, const float* A, const float* B, float* C);
void launch_v1_coalesced(int M, int N, int K, const float* A, const float* B, float* C);
void launch_v2_smem_tiling(int M, int N, int K, const float* A, const float* B, float* C);
void launch_v3_1d_blocktiling(int M, int N, int K, const float* A, const float* B, float* C);
void launch_v4_2d_blocktiling(int M, int N, int K, const float* A, const float* B, float* C);
void launch_v5_vectorized(int M, int N, int K, const float* A, const float* B, float* C);
void launch_v6_double_buffered(int M, int N, int K, const float* A, const float* B, float* C);

KernelResources resources_v0_naive();
KernelResources resources_v1_coalesced();
KernelResources resources_v2_smem_tiling();
KernelResources resources_v3_1d_blocktiling();
KernelResources resources_v4_2d_blocktiling();
KernelResources resources_v5_vectorized();
KernelResources resources_v6_double_buffered();
