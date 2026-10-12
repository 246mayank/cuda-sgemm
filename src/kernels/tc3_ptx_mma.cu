// tc3_ptx_mma.cu — tensor-core track, step 3: hand-written PTX (ldmatrix + mma.sync) instead of WMMA.
//
// WHAT CHANGED vs tc2
//   Same 128x128 block tile, 8 warps of 64x32, 16-deep double-buffered K tiles, padded shared
//   tiles, FP16 inputs with FP32 accumulation. What changes is how warps talk to the tensor cores:
//     WMMA (tc1/tc2): wmma::load_matrix_sync + wmma::mma_sync on opaque 16x16x16 fragments.
//     tc3: the instructions WMMA hides, written in inline PTX:
//       ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16 — one instruction loads four 8x8 FP16
//         matrices from shared memory straight into the register layout the MMA expects; each
//         lane supplies one row address. .trans transposes on the fly, which turns row-major B
//         rows into the column-major B fragment the MMA wants.
//       mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 — Turing's native tensor-core
//         instruction: a 16x8x8 tile per warp, FP32 accumulate.
//   Fragment layouts (PTX ISA), with g = lane / 4 and t = lane % 4:
//     A 16x8:  regs {A[g][2t..2t+1], A[g+8][2t..2t+1]}   <- ldmatrix x4 rows 0-7 / 8-15
//     B 8x8:   reg  {B[2t..2t+1][g]}                     <- ldmatrix .trans
//     C 16x8:  {C[g][2t..2t+1], C[g+8][2t..2t+1]}
//
// WHY IT SHOULD BE FASTER
//   Control. With WMMA the compiler decides how fragments are loaded (generic shared loads,
//   several per fragment); here one ldmatrix fills a whole A fragment pair for two k-steps.
//   Per 16-deep K step a warp issues 4 + 2 ldmatrix instructions for 32 mma.sync, and the
//   accumulators for its 64x32 tile (64 floats per thread) stay in named registers. This is
//   how CUTLASS reaches cuBLAS-class tensor-core performance.
//
// VERIFIED BEFORE THE GPU RUN
//   A CPU emulator of ldmatrix and mma.sync written from the PTX ISA fragment definitions ran
//   this kernel's exact index arithmetic over a full 128x128 block: result equal to A*B to
//   rounding, and the ldmatrix row addresses hit no bank conflicts (24- and 136-half strides).
//
// WHAT THE LIMIT IS NOW
//   No cp.async on Turing (global->shared copies go through registers), one k-step per barrier,
//   and a single tile shape. Sizes: multiples of 128. Requires sm_75+ (m16n8k8 f16 MMA and
//   ldmatrix); on older GPUs the kernel is empty.

#include "kernels.h"

#include <cstdint>

namespace {

constexpr int BM = 128, BN = 128, BK = 16;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 64 x 32 per warp
constexpr int FM = WM / 16, FN = WN / 8;             // 4 (m16) x 4 (n8) MMA tiles per warp
constexpr int THREADS = WARPS_M * WARPS_N * 32;      // 256
constexpr int PAD = 8;
constexpr int LDA = BK + PAD, LDB = BN + PAD;        // 24, 136 halves
static_assert(BM * BK / 8 == THREADS && BK * BN / 8 == THREADS, "one 16-byte chunk of each tile per thread");

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
tc3_ptx_mma_kernel(int M, int N, int K, const half* __restrict__ A, const half* __restrict__ B,
                   float* __restrict__ C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    __shared__ __align__(16) half As[2][BM * LDA];
    __shared__ __align__(16) half Bs[2][BK * LDB];

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int wr = (warp / WARPS_N) * WM, wc = (warp % WARPS_N) * WN;
    const int ar = tid / (BK / 8), ak = (tid % (BK / 8)) * 8;  // global->shared chunk of A
    const int br = tid / (BN / 8), bc = (tid % (BN / 8)) * 8;  // and of B
    // ldmatrix row address of this lane: matrix (lane >> 3), row (lane & 7) inside it.
    const int lrow = ((lane >> 3) & 1) * 8 + (lane & 7);  // 0..15
    const int lcol = (lane >> 4) * 8;                      // 0 or 8

    float acc[FM][FN][4];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

    uint4 ra = *reinterpret_cast<const uint4*>(&A[(size_t)(row0 + ar) * K + ak]);
    uint4 rb = *reinterpret_cast<const uint4*>(&B[(size_t)br * N + col0 + bc]);
    *reinterpret_cast<uint4*>(&As[0][ar * LDA + ak]) = ra;
    *reinterpret_cast<uint4*>(&Bs[0][br * LDB + bc]) = rb;
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        const bool has_next = t + BK < K;
        if (has_next) {
            ra = *reinterpret_cast<const uint4*>(&A[(size_t)(row0 + ar) * K + t + BK + ak]);
            rb = *reinterpret_cast<const uint4*>(&B[(size_t)(t + BK + br) * N + col0 + bc]);
        }

        // A: per m16 block, x4 = {rows 0-7 k 0-7, rows 8-15 k 0-7, rows 0-7 k 8-15, rows 8-15 k 8-15}
        uint32_t af[FM][4];
#pragma unroll
        for (int i = 0; i < FM; ++i)
            ldmatrix_x4(af[i], &As[cur][(wr + i * 16 + lrow) * LDA + lcol]);
        // B: per pair of n8 blocks, .trans x4 = {k 0-7 n 0-7, k 8-15 n 0-7, k 0-7 n 8-15, k 8-15 n 8-15}
        uint32_t bf[FN / 2][4];
#pragma unroll
        for (int p = 0; p < FN / 2; ++p)
            ldmatrix_x4_trans(bf[p], &Bs[cur][lrow * LDB + wc + p * 16 + lcol]);

#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j)
                    mma_16816(acc[i][j], af[i][2 * ks], af[i][2 * ks + 1], bf[j / 2][(j % 2) * 2 + ks]);

        if (has_next) {
            *reinterpret_cast<uint4*>(&As[cur ^ 1][ar * LDA + ak]) = ra;
            *reinterpret_cast<uint4*>(&Bs[cur ^ 1][br * LDB + bc]) = rb;
        }
        __syncthreads();
        cur ^= 1;
    }

    // acc[i][j] = {C[g][2t], C[g][2t+1], C[g+8][2t], C[g+8][2t+1]} of MMA tile (i, j).
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

void launch_tc3_ptx_mma(int M, int N, int K, const half* A, const half* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    tc3_ptx_mma_kernel<<<grid, THREADS>>>(M, N, K, A, B, C);
}

KernelResources resources_tc3_ptx_mma() { return kernel_resources(tc3_ptx_mma_kernel, THREADS); }
