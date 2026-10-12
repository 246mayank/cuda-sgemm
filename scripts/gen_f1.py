"""Generates src/kernels/f1_fused_epilogue.cu from src/kernels/v14_prefetch.cu.

The GEMM part is v14's code byte for byte; only the template header, the kernel signature (a
bias pointer), the store path (bias + activation before the store) and the launchers change.
Every replacement is asserted, so a change in v14 that breaks the generation fails loudly.
Usage: python scripts/gen_f1.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "src" / "kernels" / "v14_prefetch.cu").read_text(encoding="utf-8")
body = src[src.index("namespace {"):src.index("}  // namespace") + len("}  // namespace")]

T7 = "template <int BM, int BN, int BK, int TM, int TN, int MINB_, int SWZ>"
T8 = "template <int BM, int BN, int BK, int TM, int TN, int MINB_, int SWZ, int ACT>"


def rep(a, b, count=1):
    global body
    n = body.count(a)
    assert n == count, (a[:70], n)
    body = body.replace(a, b)


# kernel: extra template parameter and bias argument
rep(T7 + "\n__global__ void __launch_bounds__(Cfg<BM, BN, BK, TM, TN>::THREADS, MINB_)\nsgemm_v14(",
    T8 + "\n__global__ void __launch_bounds__(Cfg<BM, BN, BK, TM, TN>::THREADS, MINB_)\nsgemm_f1(")
rep("         float* __restrict__ C) {\n    using G = Cfg<BM, BN, BK, TM, TN>;",
    "         const float* __restrict__ bias, float* __restrict__ C) {\n    using G = Cfg<BM, BN, BK, TM, TN>;")
# store path: bias per column + activation, in registers, before the store
rep("                const float4 v = make_float4(acc[si * 4 + i][sj * 4 + 0]",
    "                float4 v = make_float4(acc[si * 4 + i][sj * 4 + 0]")
rep("                float* out = C + (size_t)r * N + c;\n",
    """                // Fused epilogue: C = act(A*B + bias[col]) applied in registers before the store,
                // so C is written once and never read back.
                v.x = epilogue<ACT>(v.x + (c + 0 < N ? bias[c + 0] : 0.f));
                v.y = epilogue<ACT>(v.y + (c + 1 < N ? bias[c + 1] : 0.f));
                v.z = epilogue<ACT>(v.z + (c + 2 < N ? bias[c + 2] : 0.f));
                v.w = epilogue<ACT>(v.w + (c + 3 < N ? bias[c + 3] : 0.f));
                float* out = C + (size_t)r * N + c;
""")
# launchers
rep(T7 + "\nvoid launch_cfg(int M, int N, int K, const float* A, const float* B, float* C) {",
    T8 + "\nvoid launch_cfg(int M, int N, int K, const float* A, const float* B, const float* bias,\n                float* C) {")
rep("    sgemm_v14<BM, BN, BK, TM, TN, MINB_, SWZ><<<grid, Cfg<BM, BN, BK, TM, TN>::THREADS>>>(M, N, K, A, B, C);",
    "    sgemm_f1<BM, BN, BK, TM, TN, MINB_, SWZ, ACT>\n        <<<grid, Cfg<BM, BN, BK, TM, TN>::THREADS>>>(M, N, K, A, B, bias, C);")
rep(T7 + "\nKernelResources resources_cfg() {\n    return kernel_resources(sgemm_v14<BM, BN, BK, TM, TN, MINB_, SWZ>,",
    T8 + "\nKernelResources resources_cfg() {\n    return kernel_resources(sgemm_f1<BM, BN, BK, TM, TN, MINB_, SWZ, ACT>,")
assert "sgemm_v14" not in body
# activation helper, placed before the kernel template
rep("__device__ __forceinline__ float4 load4(",
    """// ACT: 0 = bias only, 1 = ReLU, 2 = GELU (tanh approximation, as in BERT/GPT-2).
template <int ACT>
__device__ __forceinline__ float epilogue(float x) {
    if (ACT == 1) return fmaxf(x, 0.f);
    if (ACT == 2) return 0.5f * x * (1.f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    return x;
}

__device__ __forceinline__ float4 load4(""")

header = """// f1_fused_epilogue.cu — fused GEMM epilogue: C = act(A*B + bias), bias broadcast per column.
//
// WHAT CHANGED vs v14 (64x128x8 block tile, 8x8 per thread, MINB 4, no swizzle)
//   The GEMM is v14's code (generated from it by scripts/gen_f1.py). Only the output stage
//   changes: before each float4 store, every element gets its column's bias added and the
//   activation applied (ReLU, or GELU with the tanh approximation), still in registers.
//
// WHY IT SHOULD BE FASTER THAN cuBLAS + A SEPARATE KERNEL
//   Every transformer layer computes act(X*W + b). Unfused, cuBLAS writes C, then an
//   elementwise kernel reads C back, adds the bias, applies the activation and writes C again:
//   an extra M*N*8 bytes of DRAM traffic plus a second launch. Fused, that work happens on
//   values already in registers. The saving is fixed per output element, so it matters most
//   when the GEMM itself is short (small K), as in the FFN shapes of a transformer.
//
// WHAT THE LIMIT IS NOW
//   The GEMM part is unchanged (v14's limits). The comparison is against cuBLAS's SGEMM plus our
//   own elementwise kernel, not against cuBLASLt's fused epilogues.

#include "kernels.h"

"""

footer = """

void launch_f1_bias_relu(int M, int N, int K, const float* A, const float* B, const float* bias,
                         float* C) {
    launch_cfg<64, 128, 8, 8, 8, 4, 0, 1>(M, N, K, A, B, bias, C);
}

void launch_f1_bias_gelu(int M, int N, int K, const float* A, const float* B, const float* bias,
                         float* C) {
    launch_cfg<64, 128, 8, 8, 8, 4, 0, 2>(M, N, K, A, B, bias, C);
}

KernelResources resources_f1_bias_gelu() { return resources_cfg<64, 128, 8, 8, 8, 4, 0, 2>(); }
"""
(ROOT / "src" / "kernels" / "f1_fused_epilogue.cu").write_text(header + body + footer,
                                                               encoding="utf-8", newline="\n")
print("wrote src/kernels/f1_fused_epilogue.cu")
