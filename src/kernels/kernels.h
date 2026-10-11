// kernels.h — host-side launchers, one per kernel file in src/kernels/.
// All compute C = A * B for row-major FP32 matrices: A is MxK, B is KxN, C is MxN.
// Pointers are device pointers. Launchers do not synchronize.
#pragma once

void launch_v0_naive(int M, int N, int K, const float* A, const float* B, float* C);
void launch_v1_coalesced(int M, int N, int K, const float* A, const float* B, float* C);
