// kernels.h — host-side entry points, one pair per kernel file in src/kernels/.
// All compute C = A * B for row-major FP32 matrices: A is MxK, B is KxN, C is MxN.
// Pointers are device pointers. Launchers do not synchronize.
// resources_*() reports registers, spills, shared memory and blocks per SM of the kernel.
#pragma once

#include "../common.h"

#include <cuda_fp16.h>

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
void launch_v7_conflict_free(int M, int N, int K, const float* A, const float* B, float* C);
KernelResources resources_v7_conflict_free();

// Tuning study (bench --tune): the v7 kernel instantiated with different tile shapes.
struct TuneConfig {
    const char* name;  // v7_<BM>x<BN>x<BK>_<TM>x<TN>
    void (*launch)(int, int, int, const float*, const float*, float*);
    KernelResources (*resources)();
};
const std::vector<TuneConfig>& v7_tune_configs();

void launch_v8_warptiling(int M, int N, int K, const float* A, const float* B, float* C);
KernelResources resources_v8_warptiling();
const std::vector<TuneConfig>& v8_tune_configs();
const std::vector<TuneConfig>& v10_tune_configs();  // tuning study only

void launch_v9_autotuned(int M, int N, int K, const float* A, const float* B, float* C);
KernelResources resources_v9_autotuned();

// Tensor-core track (bench --tc): FP16 inputs, FP32 accumulation, WMMA. Sizes: multiples of 128.
void launch_tc0_wmma_naive(int M, int N, int K, const half* A, const half* B, float* C);
void launch_tc1_wmma_smem(int M, int N, int K, const half* A, const half* B, float* C);
KernelResources resources_tc0_wmma_naive();
KernelResources resources_tc1_wmma_smem();
void launch_tc2_wmma_db(int M, int N, int K, const half* A, const half* B, float* C);
KernelResources resources_tc2_wmma_db();
