# cuda-sgemm: SGEMM from a naive kernel toward cuBLAS on an NVIDIA T4

Single-precision matrix multiply (C = A·B, row-major FP32) written in CUDA C++ and optimized
one idea at a time. Each step is measured on a Tesla T4, checked against cuBLAS, and explained
in terms of memory coalescing, data reuse, occupancy and instruction mix. **Every number in
this README is generated from the CSVs in [`results/`](results/)** by
[`scripts/make_tables.py`](scripts/make_tables.py); nothing is typed by hand.

**Result.** The fastest kernel, `v5_vectorized`, reaches **{{gflops:v5_vectorized:4096}} GFLOP/s
at N = 4096, {{pct:v5_vectorized:4096}} of cuBLAS** measured in the same thermal state, up from
{{gflops:v0_naive:2048}} GFLOP/s for the naive kernel ({{speedup:v5_vectorized:v0_naive:2048}}
at N = 2048). All {{npass}} of {{nrows}} kernel/size results pass the cuBLAS check (largest
normwise relative error {{maxerr}}, tolerance {{tol}}), including the non-multiple sizes
1000 and 1023.

## Results

![GFLOP/s vs size](figures/gflops_vs_size.png)

{{table:headline}}

{{table:all_sizes}}

![% of cuBLAS](figures/pct_cublas.png)

### Roofline

![Roofline](figures/roofline.png)

The memory roof is the **measured** DRAM bandwidth ({{bw}} GB/s, {{bw_pct}} of the
{{bw_theo}} GB/s spec, from [`src/bandwidth.cu`](src/bandwidth.cu)). The compute roof is the
theoretical FP32 peak, {{peak}} GFLOP/s (40 SMs × 64 FP32 lanes × 1590 MHz boost × 2 for
FMA), labelled as theoretical. cuBLAS is drawn as the practical ceiling. Each kernel's
arithmetic intensity is **measured**: FLOPs divided by the DRAM bytes Nsight Compute counted
for that kernel at N = 2048. The vertical line is the best possible intensity at that size,
{{ideal_ai}} FLOP/byte, if A and B were read from DRAM exactly once and C written once.

{{table:ncu}}

{{table:resources}}

## The kernels, one idea at a time

Each kernel file starts with a comment block: what changed from the previous version, why it
should be faster, and what limits it now. Below, the same story with the measurements.

### v0: naive, one thread per output ([`v0_naive.cu`](src/kernels/v0_naive.cu))

Thread (x, y) of a 32×32 block computes C[row][col] with **row from `threadIdx.x`**. A warp is
32 consecutive `threadIdx.x`, so one warp covers 32 *rows* and one column. Its loads of
`A[row][k]` hit 32 addresses K floats apart: 32 separate 32-byte sectors to deliver 128 useful
bytes (uncoalesced). Its loads of `B[k][col]` are one address for the whole warp (a
broadcast), and the final stores to C are strided like A.

Measured: {{gflops:v0_naive:2048}} GFLOP/s at N = 2048, {{pct:v0_naive:2048}} of cuBLAS.
Nsight Compute counts {{dram_mb:v0_naive}} MB of DRAM traffic at N = 2048, which is
{{traffic_x:v0_naive}} the minimum needed, and {{sectors_per_req:v0_naive}} sectors per global
load request.

**A side finding: N = 1023 is faster than N = 1024.** v0 runs at {{gflops:v0_naive:1023}}
GFLOP/s at N = 1023 but {{gflops:v0_naive:1024}} at N = 1024, and this reproduced in every
run. At N = 1024 the 32 rows a warp reads are exactly 4 KB apart, a power-of-two stride, so
they map onto the same cache sets and DRAM partitions and evict each other. At 1023 the stride
is odd and the accesses spread out. The counters agree:

{{table:v0stride}}

### v1: coalesced ([`v1_coalesced.cu`](src/kernels/v1_coalesced.cu))

**One change:** `threadIdx.x` now picks the *column*. Same loop, same FLOPs, same loads. A warp
now covers one row and 32 adjacent columns, so `B[k][col]` is 32 consecutive floats (one
128-byte, 4-sector access), `A[row][k]` becomes the broadcast, and the C stores are contiguous.

Measured: **{{speedup:v1_coalesced:v0_naive:2048}} faster than v0** at N = 2048
({{gflops:v1_coalesced:2048}} GFLOP/s). Global sectors per request drop from
{{sectors_per_req:v0_naive}} to {{sectors_per_req:v1_coalesced}}.

New limit: every FMA still needs two loads from the memory system, and each element of A and B
is requested N times. Only the caches stand between that and DRAM.

### v2: shared-memory tiling ([`v2_smem_tiling.cu`](src/kernels/v2_smem_tiling.cu))

The block copies a 32×32 tile of A and of B into shared memory (an on-chip scratchpad the
program manages explicitly, 64 KB per SM on a T4), then each thread does its 32 multiply-adds
from there. Each element is fetched from global memory once per block instead of once per
thread: global load instructions drop 32×. Two `__syncthreads()` per tile: the first so no
thread reads the tile before it's fully written, the second so no thread overwrites it for the
next tile while others are still reading. Edge tiles are zero-padded, so 1000 and 1023 need no
special case.

Measured: **{{speedup:v2_smem_tiling:v1_coalesced:4096}} faster than v1** at N = 4096
({{gflops:v2_smem_tiling:4096}} GFLOP/s, {{pct:v2_smem_tiling:4096}} of cuBLAS).
DRAM traffic at N = 2048 falls to {{dram_mb:v2_smem_tiling}} MB ({{ai:v2_smem_tiling}}
FLOP/byte).

Why only that much: every FMA still needs two shared-memory loads
({{lds_per_ffma:v2_smem_tiling}} shared-load instructions per warp-level FFMA, measured), so
the SM spends as many issue slots moving data as computing. And 1024 threads per block means
{{blocks:v2_smem_tiling}} block per SM, so at each `__syncthreads()` the whole SM waits for its
slowest warp.

### v3: 1D register tiling ([`v3_1d_blocktiling.cu`](src/kernels/v3_1d_blocktiling.cu))

Each thread now computes **8 outputs in a column**, keeping 8 accumulators in registers. Per k
it loads one B value and reuses it for 8 FMAs. Block tile 64×64, depth 8, 512 threads, so
{{blocks:v3_1d_blocktiling}} blocks fit on an SM. Per tile phase a block loads 1024 floats for
32,768 FMAs, twice v2's reuse.

Measured: {{gflops:v3_1d_blocktiling:4096}} GFLOP/s at N = 4096,
**{{speedup:v3_1d_blocktiling:v2_smem_tiling:4096}} vs v2**. Shared loads per FFMA:
{{lds_per_ffma:v3_1d_blocktiling}} (v2: {{lds_per_ffma:v2_smem_tiling}}).
{{regs:v3_1d_blocktiling}} registers per thread, {{spill:v3_1d_blocktiling}} bytes spilled.

### v4: 2D register tiling ([`v4_2d_blocktiling.cu`](src/kernels/v4_2d_blocktiling.cu))

Each thread computes an **8×8 tile**: per k it loads 8 A values and 8 B values into registers
and does their 64-FMA outer product, so there are 16 shared loads per 64 FMAs. Block tile
128×128, 256 threads, 64 accumulators per thread. `__launch_bounds__(256, 2)` asks the compiler
to fit two blocks per SM (at most 128 registers per thread): fewer warps than v3, but each has
far more independent FMAs, so latency is hidden by instruction-level parallelism rather than by
more threads.

Measured: {{gflops:v4_2d_blocktiling:4096}} GFLOP/s at N = 4096,
**{{speedup:v4_2d_blocktiling:v3_1d_blocktiling:4096}} vs v3**,
{{pct:v4_2d_blocktiling:4096}} of cuBLAS. {{regs:v4_2d_blocktiling}} registers per thread,
{{spill:v4_2d_blocktiling}} bytes spilled, {{blocks:v4_2d_blocktiling}} blocks per SM
({{occ:v4_2d_blocktiling}} theoretical occupancy). Shared loads per FFMA:
{{lds_per_ffma:v4_2d_blocktiling}}; shared-load bank conflicts:
{{bank_conflicts:v4_2d_blocktiling}}.

### v5: vectorized loads ([`v5_vectorized.cu`](src/kernels/v5_vectorized.cu))

Same tiling as v4. Global loads become 128-bit `float4` (one instruction instead of four), and
the A tile is stored **transposed** in shared memory, so a thread's 8 A values for one k are
contiguous and can be read with two 128-bit loads, as can its 8 B values. When a row can't be
read as aligned `float4` (K or N not a multiple of 4, e.g. 1023) or a chunk crosses the matrix
edge, the loader falls back to guarded scalar loads.

Measured: {{gflops:v5_vectorized:4096}} GFLOP/s at N = 4096,
**{{speedup:v5_vectorized:v4_2d_blocktiling:4096}} vs v4**, {{pct:v5_vectorized:4096}} of
cuBLAS, {{pct_peak:v5_vectorized:4096}} of the theoretical FP32 peak. Shared loads per FFMA:
{{lds_per_ffma:v5_vectorized}}.

**What separates v5 from cuBLAS:** no double buffering (the next tile can't load while this one
is computed), no warp-level tiling, and one tile shape for every size. At N = 256 a 128×128
tile leaves most of the 40 SMs idle, which is why the small sizes trail cuBLAS the most.

## Methodology

- **Timing.** CUDA events around each launch; 3 warm-ups, then 20 timed runs. Reported:
  median (GFLOP/s uses it), min and p90. GFLOP/s = 2·N³ / median time.
- **Correctness.** Every kernel is checked against `cublasSgemm` on every run and at every size:
  normwise relative error max|C − C_ref| / max|C_ref| ≤ {{tol}}. Before each check C is filled
  with NaN, so a kernel that skips elements can't pass on stale data. A kernel that fails at any
  size is marked FAILED and left out of every figure.
- **Row-major vs cuBLAS's column-major.** A row-major matrix is its transpose in column-major,
  so row-major C = A·B is computed as column-major Cᵀ = Bᵀ·Aᵀ: swap the operands and M/N in the
  `cublasSgemm` call. No data is transposed or copied.
- **Throttling, and paired cuBLAS timing.** The T4 is capped at 70 W, and its SM clock drops
  under sustained load; the environment record below shows the clock at the start and end of the
  run. A cuBLAS time taken at the start of a size and a kernel time taken a minute later aren't
  comparable, so each timed kernel run is immediately followed by a timed cuBLAS run, and
  "% of cuBLAS" uses that paired median. The standalone cuBLAS row in the tables is kept for
  transparency; it can be lower than the paired figure ({{cublas_paired:4096}} GFLOP/s at
  N = 4096) because it ran on a hotter GPU.
- **Run-to-run variation.** Separate Colab sessions get different GPUs and thermal states (table
  below), so the headline numbers all come from a single run in one fresh session.
- **Nsight Compute** worked on Colab's T4 (often it is blocked with `ERR_NVGPUCTRPERM`). It
  profiles with locked clocks and flushed caches, so only its counters are used here, never its
  timings. The measured counters are DRAM bytes, cache hit rates, sectors per request, shared
  bank conflicts and instruction counts.
- **Registers, spills and occupancy** are read from the compiled binary at run time
  (`cudaFuncGetAttributes`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`).
- **Not measured:** power per kernel, and clocks during each individual kernel (only at the
  start and end of a run).

{{table:variability}}

Environment of the headline run:

{{table:env}}

Measured DRAM bandwidth:

{{table:bandwidth}}

Source files for every number above:

{{table:sources}}

## Reproduce

1. Open [`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb) in Google Colab
   (File → Open notebook → GitHub → `246mayank/cuda-sgemm`) and select a **T4 GPU** runtime.
2. Run the cells: they clone the repo, `make` (default `ARCH=sm_75`), run `./bench` and
   `./bandwidth`, collect Nsight Compute counters, and download `results/` as a zip.
3. Locally, unzip into `results/` and regenerate the figures and this README:
   ```
   python scripts/plot.py
   python scripts/make_tables.py
   ```
   (`pip install -r requirements.txt` first.)

Build anywhere with CUDA: `make` (or `make ARCH=sm_60` for a P100), then `./bench` from the
repo root. `./bench --kernel v4 --size 2048` runs a single kernel and size.

## Credits

The kernel sequence follows Simon Boehm's worklog
[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM).
The implementation, the T4 measurements, the harness (paired cuBLAS timing, verification,
generated reporting), the Nsight Compute analysis and the write-up are my own.
