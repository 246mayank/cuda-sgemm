// v5_vectorized.cu — v4 with 128-bit (float4) memory accesses and the A tile transposed.
//
// WHAT CHANGED vs v4
//   Same 128x128x8 block tile, same 256 threads, same 8x8 outputs per thread, same k-order
//   of the sum. Only how data moves:
//     - Global loads are float4 (LDG.128): each thread loads one float4 of the A tile and
//       one float4 of the B tile per phase, instead of 4 + 4 scalar loads.
//     - The A tile is stored transposed in shared memory, As[k][m] instead of As[m][k], so
//       the 8 A values a thread needs for one k are contiguous.
//     - Shared-memory reads into registers are float4 (LDS.128): 2 for A and 2 for B per k,
//       instead of 8 + 8 scalar reads.
//
// WHY IT SHOULD BE FASTER
//   Fewer instructions for the same bytes. Per phase each thread issues 2 global loads
//   instead of 8, and per k it issues 4 shared loads instead of 16, so even more issue
//   slots go to FFMA. The transpose is what makes the A reads vectorizable: in v4 a thread's
//   8 A values for one k were 8 floats apart (As[m][k]); now they are adjacent.
//
// ALIGNMENT AND EDGES
//   A float4 load needs a 16-byte aligned address. For row-major A, A[r*K + c] is aligned
//   when K % 4 == 0 and c % 4 == 0 (cudaMalloc returns >= 256-byte aligned memory); same
//   for B with N. For sizes like 1023, or chunks that cross the matrix edge, the loader
//   falls back to 4 guarded scalar loads. The float4 / scalar choice is per chunk, so
//   correctness never depends on the size.
//
// WHAT THE LIMIT IS NOW
//   - The transposed store into As (4 scalar STS per thread) has 2-way bank conflicts:
//     threads tid and tid+1 write rows of A 4 floats apart in the k dimension, i.e.
//     As[k][m] and As[k+4][m] = 4*128 floats apart, the same bank. It happens once per
//     phase, so it costs far less than conflicts in the inner loop would.
//   - Global-to-shared copies and the math are still serialized by __syncthreads(): a block
//     cannot load the next tile while computing this one (no double buffering), and each
//     warp's work is not arranged to match the hardware's warp scheduling (warp tiling, v6).

#include "kernels.h"

namespace {

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256
static_assert(BM * BK / 4 == THREADS && BK * BN / 4 == THREADS, "one float4 of each tile per thread");

// 4 consecutive floats p[0..3] from row `r` (of `rows`), starting at column c (of `cols`).
// Uses one 128-bit load when the chunk is fully in bounds and aligned, else guarded scalars.
__device__ __forceinline__ float4 load4(const float* __restrict__ p, int r, int c, int rows,
                                        int cols, bool vec_ok) {
    if (vec_ok && r < rows && c + 3 < cols)
        return *reinterpret_cast<const float4*>(p + (size_t)r * cols + c);
    float4 v;
    const bool rr = r < rows;
    const float* q = p + (size_t)r * cols + c;
    v.x = (rr && c + 0 < cols) ? q[0] : 0.f;
    v.y = (rr && c + 1 < cols) ? q[1] : 0.f;
    v.z = (rr && c + 2 < cols) ? q[2] : 0.f;
    v.w = (rr && c + 3 < cols) ? q[3] : 0.f;
    return v;
}

__global__ void __launch_bounds__(THREADS, 2)
sgemm_v5_vectorized(int M, int N, int K, const float* __restrict__ A,
                    const float* __restrict__ B, float* __restrict__ C) {
    __shared__ __align__(16) float As[BK * BM];  // transposed: [BK][BM]
    __shared__ __align__(16) float Bs[BK * BN];  // [BK][BN]

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);
    // Load mapping, in float4 chunks: A tile is 128 rows x 2 chunks, B tile is 8 rows x 32.
    const int aRow = tid / (BK / 4), aCol = (tid % (BK / 4)) * 4;
    const int bRow = tid / (BN / 4), bCol = (tid % (BN / 4)) * 4;
    // Uniform across the grid: can rows of A / B be read as aligned float4 at all?
    const bool vecA = (K % 4) == 0, vecB = (N % 4) == 0;

    float acc[TM][TN] = {{0.f}};
    float regM[TM], regN[TN];

    for (int t = 0; t < K; t += BK) {
        const float4 a = load4(A, row0 + aRow, t + aCol, M, K, vecA);
        As[(aCol + 0) * BM + aRow] = a.x;
        As[(aCol + 1) * BM + aRow] = a.y;
        As[(aCol + 2) * BM + aRow] = a.z;
        As[(aCol + 3) * BM + aRow] = a.w;
        const float4 b = load4(B, t + bRow, col0 + bCol, K, N, vecB);
        *reinterpret_cast<float4*>(&Bs[bRow * BN + bCol]) = b;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float4 m0 = *reinterpret_cast<const float4*>(&As[k * BM + threadRow * TM]);
            const float4 m1 = *reinterpret_cast<const float4*>(&As[k * BM + threadRow * TM + 4]);
            const float4 n0 = *reinterpret_cast<const float4*>(&Bs[k * BN + threadCol * TN]);
            const float4 n1 = *reinterpret_cast<const float4*>(&Bs[k * BN + threadCol * TN + 4]);
            regM[0] = m0.x; regM[1] = m0.y; regM[2] = m0.z; regM[3] = m0.w;
            regM[4] = m1.x; regM[5] = m1.y; regM[6] = m1.z; regM[7] = m1.w;
            regN[0] = n0.x; regN[1] = n0.y; regN[2] = n0.z; regN[3] = n0.w;
            regN[4] = n1.x; regN[5] = n1.y; regN[6] = n1.z; regN[7] = n1.w;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row0 + threadRow * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = col0 + threadCol * TN + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
    }
}

}  // namespace

void launch_v5_vectorized(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_v5_vectorized<<<grid, THREADS>>>(M, N, K, A, B, C);
}

KernelResources resources_v5_vectorized() { return kernel_resources(sgemm_v5_vectorized, THREADS); }
