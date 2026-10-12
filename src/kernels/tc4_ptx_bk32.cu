// tc4_ptx_bk32.cu — tensor-core track, step 4: tc3's PTX path with tc1's 32-deep K tile.
//
// WHAT CHANGED vs tc3
//   Same ldmatrix.x4 / mma.sync.m16n8k8 inner loop, same 128x128 block of 8 warps (64x32
//   each), same fragment layouts. The K schedule goes back to tc1's: one shared buffer with a
//   32-deep K tile and two barriers per tile, so each barrier is followed by two 16-deep MMA
//   steps (64 mma.sync per warp) instead of one.
//
// WHY
//   The measurements isolate the cause. tc2 (WMMA, K tile 16, double buffered) was slower than
//   tc1 (WMMA, K tile 32) at N >= 1024, and tc3 (PTX, tc2's schedule) roughly matched tc2. So
//   the PTX path was not the problem; the shallow tile was. tc3 also hit the 128-register cap
//   and spilled, because the prefetched tile sat in registers. tc4 drops the prefetch
//   registers and doubles the work between barriers.
//   A tile rows are 32 + 8 = 40 halves (80 bytes): the 8 row addresses of an ldmatrix phase
//   start 20 words apart, i.e. at banks 0, 20, 8, 28, 16, 4, 24, 12, each covering 4 banks:
//   all 32 banks once, no conflict (checked in the CPU emulation, like tc3).
//
// WHAT THE LIMIT IS NOW
//   Global->shared copies are not overlapped with MMA (Turing has no cp.async; overlapping
//   through registers is what cost tc3 its spill). Sizes: multiples of 128; sm_75+.

#include "kernels.h"

#include <cstdint>

namespace {

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 64 x 32 per warp
constexpr int FM = WM / 16, FN = WN / 8;             // 4 x 4 MMA tiles per warp
constexpr int THREADS = WARPS_M * WARPS_N * 32;      // 256
constexpr int PAD = 8;
constexpr int LDA = BK + PAD, LDB = BN + PAD;        // 40, 136 halves
constexpr int A_PER = BM * BK / 8 / THREADS, B_PER = BK * BN / 8 / THREADS;  // 2, 2 chunks
static_assert(A_PER * THREADS * 8 == BM * BK && B_PER * THREADS * 8 == BK * BN, "chunk mapping");

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const half* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const half* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
#endif
}

__device__ __forceinline__ void mma_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b));
#endif
}

__global__ void __launch_bounds__(THREADS, 2)
tc4_ptx_bk32_kernel(int M, int N, int K, const half* __restrict__ A, const half* __restrict__ B,
                    float* __restrict__ C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    __shared__ __align__(16) half As[BM * LDA];
    __shared__ __align__(16) half Bs[BK * LDB];

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int wr = (warp / WARPS_N) * WM, wc = (warp % WARPS_N) * WN;
    const int lrow = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int lcol = (lane >> 4) * 8;

    float acc[FM][FN][4];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

    for (int t = 0; t < K; t += BK) {
#pragma unroll
        for (int p = 0; p < A_PER; ++p) {
            const int c = tid + p * THREADS;
            const int r = c / (BK / 8), kc = (c % (BK / 8)) * 8;
            *reinterpret_cast<uint4*>(&As[r * LDA + kc]) =
                *reinterpret_cast<const uint4*>(&A[(size_t)(row0 + r) * K + t + kc]);
        }
#pragma unroll
        for (int p = 0; p < B_PER; ++p) {
            const int c = tid + p * THREADS;
            const int r = c / (BN / 8), cc = (c % (BN / 8)) * 8;
            *reinterpret_cast<uint4*>(&Bs[r * LDB + cc]) =
                *reinterpret_cast<const uint4*>(&B[(size_t)(t + r) * N + col0 + cc]);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[FM][4];
#pragma unroll
            for (int i = 0; i < FM; ++i)
                ldmatrix_x4(af[i], &As[(wr + i * 16 + lrow) * LDA + kk + lcol]);
            uint32_t bf[FN / 2][4];
#pragma unroll
            for (int p = 0; p < FN / 2; ++p)
                ldmatrix_x4_trans(bf[p], &Bs[(kk + lrow) * LDB + wc + p * 16 + lcol]);
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
#pragma unroll
                for (int i = 0; i < FM; ++i)
#pragma unroll
                    for (int j = 0; j < FN; ++j)
                        mma_16816(acc[i][j], af[i][2 * ks], af[i][2 * ks + 1], bf[j / 2][(j % 2) * 2 + ks]);
        }
        __syncthreads();
    }

    const int g = lane / 4, tg = lane % 4;
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            const int r = row0 + wr + i * 16 + g, c = col0 + wc + j * 8 + 2 * tg;
            *reinterpret_cast<float2*>(&C[(size_t)r * N + c]) = make_float2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<float2*>(&C[(size_t)(r + 8) * N + c]) = make_float2(acc[i][j][2], acc[i][j][3]);
        }
#endif
}

}  // namespace

void launch_tc4_ptx_bk32(int M, int N, int K, const half* A, const half* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    tc4_ptx_bk32_kernel<<<grid, THREADS>>>(M, N, K, A, B, C);
}

KernelResources resources_tc4_ptx_bk32() { return kernel_resources(tc4_ptx_bk32_kernel, THREADS); }
