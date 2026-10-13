# cuda-sgemm

![CI](https://github.com/246mayank/cuda-sgemm/actions/workflows/ci.yml/badge.svg)

FP32 matrix multiply (SGEMM) in CUDA C++, optimised one idea at a time on an NVIDIA Tesla T4,
with every kernel checked against cuBLAS, profiled with Nsight Compute, and every number in
this README generated from the CSVs in [`results/`](results/).

**Result:** the final kernel, `v12_dispatch` (per-size tuned tiles, with split-K for grids that
can't fill the GPU), reaches **{{pct:v12_dispatch:4096}} of cuBLAS at N = 4096** and **beats cuBLAS
at N = 256 ({{pct:v12_dispatch:256}})**. The best single configuration measured
{{best_pct:4096}} at N = 4096. Split-K lifts a
skinny transformer shape (128 × 4096 × 4096) from {{shape_pct:v9_autotuned:128:4096:4096}} to
**{{shape_pct:v11_splitk:128:4096:4096}}**. {{npass}}/{{nrows}} kernel/size results pass the
cuBLAS check, including the awkward sizes 1000 and 1023.

![GFLOP/s vs size](figures/gflops_vs_size.png)

## The progression

% of cuBLAS, timed interleaved with cuBLAS (same thermal state), from one benchmark session:

{{table:progression}}

v0 isn't run above N = 2048 (2 s per launch). Raw GFLOP/s and every other size are in the
[full analysis](docs/ANALYSIS.md).

## Key findings

- **Bank conflicts were the hidden limiter.** Nsight Compute counted
  {{conflicts_per_lds:v6_double_buffered}} conflicts per shared load in v4–v6. A new thread
  layout (v7) removed them: {{pspeedup:v7_conflict_free:v6_double_buffered:4096}} faster. A
  small Python model of the bank rules predicted the profiler's count exactly.
- **Occupancy plus register prefetch closed the rest.** Capping registers at 128 to fit a fourth
  block per SM, prefetching the next k-step's operands into registers (v14), and choosing the
  tile per size from the average of several tuning runs took the kernel from
  {{pct:v7_conflict_free:4096}} (v7) to {{pct:v12_dispatch:4096}} (v12) at N = 4096.
- **Two textbook optimisations did nothing, and the machine code shows why.** The compiler had
  already vectorised v4's shared loads (v5), and it already turns WMMA into `LDSM`/`HMMA`, so
  hand-written PTX tensor-core code (tc3, tc4) didn't beat it.
- **The T4 is power-capped at {{power_limit}}.** Clocks sag under load, so the harness times
  cuBLAS interleaved with every kernel, logs clock and power, and was cross-checked with the
  clock locked.

## Also in this repo

- **Tensor cores (FP16 in, FP32 accumulate):** {{tc_best_tflops:2048}} TFLOP/s with
  {{tc_best:2048}}, {{tc_best_pct:2048}} of `cublasGemmEx`.
  A tile sweep (tc5) reaches {{tctune_best_pct:4096}} of `cublasGemmEx` at N = 4096.
- **Fused epilogue (GEMM + bias + GELU/ReLU):** up to {{epi_best}} faster than cuBLAS followed by a
  separate bias + activation kernel.
- **Parallel reduction study:** r0 → r5 reaches {{red_gbps:r5_grid_stride}} GB/s,
  {{red_pct_cub:r5_grid_stride}} of CUB.
- **Roofline, tuning sweep, real-world shapes, SASS analysis, methodology:** see the
  **[full analysis →](docs/ANALYSIS.md)**.
- **Study guide** for every technique used here: [docs/LEARNING.md](docs/LEARNING.md).

## Reproduce

Open [`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb) in Google Colab with a
**T4 GPU** and run it; it builds everything, runs every benchmark, and downloads `results/`.
Then, locally:

```
pip install -r requirements.txt
python scripts/plot.py
python scripts/make_tables.py
```

Anywhere with CUDA: `make` then `./bench` (or `./bench --tc`, `--tune`, `--shapes`, `./reduce`).
Tests (no GPU needed; they replay every kernel's index math on the CPU): `python -m pytest tests`.

## Credits

The kernel sequence v0–v5 follows Simon Boehm's
[How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM). The
implementation, the T4 measurements, the harness, the Nsight Compute and SASS analysis, the
later kernels (v6–v12, tensor-core track, reduction) and the write-up are my own.
