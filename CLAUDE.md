# cuda-sgemm — step-by-step SGEMM optimization on a T4, with a performance analysis

## Goal
Take single-precision matrix multiply (C = A·B, row-major, FP32) from a naive CUDA kernel
to within reach of cuBLAS, one optimization at a time. For **every** step, *measure* the gain
and *explain why it happened* in terms of memory coalescing, data reuse, occupancy and
instruction mix.
The deliverable is the analysis as much as the code.

Target reader: an NVIDIA DevTech / Compute Architecture engineer. They care about
"implement and evaluate parallel algorithms and CUDA C/C++ kernels … use profiling and
benchmarking tools to measure performance, validate improvements, and explain results."

## Hard rules (non-negotiable)
1. **No number is ever typed by hand.** Every figure in the README, plots and tables is generated
   from `results/*.csv` by a script. If a number isn't in a CSV, it doesn't appear.
2. **Every kernel is checked against cuBLAS** on every run (max relative error with a tolerance).
   A kernel that fails the check is reported as FAILED and its timing is not plotted.
3. **One kernel per file, one idea per kernel.** Each version changes exactly one thing from the
   previous one, so the speedup can be attributed.
4. **Record the environment** with every result: GPU name, SM count, clocks, CUDA version,
   driver, date. Colab GPUs throttle, so take the median of N runs and report the spread.
5. If something is not measured (e.g. Nsight Compute is blocked on Colab), say so in the README.
   Never imply it was measured.
6. Mayank must be able to explain every kernel. Each kernel file starts with a comment block:
   *what changed vs the previous version, why it should be faster, what the limit is now.*

## Hardware / environment
- No local GPU. Develop locally with Claude Code; **run on Google Colab (Tesla T4, sm_75)**,
  with Kaggle (T4 / P100) as a fallback.
- Build with `nvcc -O3 -arch=sm_75 -lcublas`. Keep `-arch` configurable via the Makefile.
- `notebooks/run_on_colab.ipynb`: clone the repo → `nvidia-smi` → `make` → run the benchmark →
  `ncu` DRAM-byte measurement (it worked on the Colab T4 on 2026-10-11, see
  results/ncu_probe_Tesla_T4_2026-10-11.txt; if a later session blocks it, record that) → zip `results/`.
  Results then get committed from the local machine.

## Repository layout
```
src/
  kernels/
    v0_naive.cu            one thread per C element; threadIdx.x -> row, so A loads and C stores are
                           uncoalesced (B is a warp-wide broadcast)
    v1_coalesced.cu        threadIdx.x -> col: B loads and C stores contiguous, A broadcast
    v2_smem_tiling.cu      32x32 shared-memory tiles, __syncthreads between phases
    v3_1d_blocktiling.cu   each thread computes a column of TM outputs (register reuse)
    v4_2d_blocktiling.cu   each thread computes a TMxTN outer-product tile
    v5_vectorized.cu       float4 global loads, transposed A tile in smem
    (stretch) v6_warptiling.cu, v7_wmma_fp16.cu (tensor cores; separate FP16 track)
  bench.cu                 harness: allocate, init, run each kernel + cuBLAS, verify, time, CSV
  common.h                 CUDA_CHECK, timer, device query, verify()
  bandwidth.cu             measured DRAM bandwidth (copy kernel) for the roofline
scripts/
  plot.py                  GFLOP/s vs size, % of cuBLAS, roofline
  make_tables.py           README tables generated from the CSVs
results/                   CSVs + environment JSON (committed)
figures/                   generated PNGs (committed)
notebooks/run_on_colab.ipynb
README.md
```

## Benchmark harness (`bench.cu`)
- Sizes: square 256, 512, 1024, 2048, 4096, plus **non-multiples (1000, 1023)** to test
  boundary handling.
- Per kernel per size: 3 warm-ups, then 20 timed runs with CUDA events; record the median, min and p90.
- GFLOP/s = 2·M·N·K / time.
- Verify against `cublasSgemm` (mind the row-major vs column-major transpose trick; document it).
- Output: `results/sgemm_<gpu>_<date>.csv` with columns
  `kernel,M,N,K,median_ms,min_ms,p90_ms,gflops,pct_cublas,max_rel_err,status`
- `results/env_<gpu>_<date>.json`: from `cudaGetDeviceProperties` + `nvidia-smi --query-gpu`.

## Roofline
- Memory roof: **measured** bandwidth from `bandwidth.cu` (report it next to the 320 GB/s spec).
- Compute roof: FP32 peak computed from SM count × FP32 lanes/SM × boost clock × 2, labeled
  "theoretical"; also show cuBLAS as the "practical" ceiling.
- Arithmetic intensity per kernel: from Nsight Compute DRAM bytes if available, otherwise from an
  **analytical traffic model** derived per kernel (show the derivation in the README; label it
  "modelled, not measured").

## README structure
1. One-paragraph result: final kernel = X% of cuBLAS at 4096 on a T4 (generated).
2. Results table + GFLOP/s chart + roofline (generated).
3. One section per kernel: the change → why → measured effect → new bottleneck.
4. Methodology: timing, verification, variance, what wasn't measurable on Colab.
5. How to reproduce (Colab link + commands).

## Plan, in order (stop at any step and the repo is still complete and honest)
1. common.h, bench.cu with cuBLAS + v0; verify passes; CSV written. **First Colab run.**
2. v1, v2. Second Colab run. Plot script.
3. v3, v4. Bandwidth kernel. Roofline.
4. v5. README with generated tables.
5. Stretch: v6 warp tiling, or a parallel-reduction side study (shared mem → warp shuffle).

## Environment facts (for Claude Code)
- OS: Windows 11 Home Single Language 10.0.26200 (shells: PowerShell 5.1 and Git Bash). There is NO NVIDIA GPU on this machine.
- nvcc locally: NOT available. Every summary must say
  "not compiled locally — compile on Colab".
- Kernels run only on Colab (T4, sm_75) or Kaggle (T4 sm_75 / P100 sm_60). Results come back as CSVs
  in results/. Never state that a kernel ran, passed, or achieved a number unless it is in those CSVs.
- Python: use the venv at .venv for scripts/.
- Mayank reviews and pushes every commit himself. Never push.
- End every step with: files changed, what Mayank should check, and the exact Colab cells to run.
