# cuda-sgemm: SGEMM from a naive kernel toward cuBLAS on an NVIDIA T4

Single-precision matrix multiply (C = A·B, row-major FP32) written in CUDA C++ and optimized
one idea at a time. Each step is measured on a Tesla T4, checked against cuBLAS, profiled with
Nsight Compute, and explained in terms of memory coalescing, data reuse, occupancy and
instruction mix. **Every number in this README is generated from the CSVs in
[`results/`](results/)** by [`scripts/make_tables.py`](scripts/make_tables.py); none is typed
by hand.

**Result.** The fastest kernel at N = 4096, {{best_kernel:4096}}, reaches **{{best_gflops:4096}}
GFLOP/s, {{best_pct:4096}} of cuBLAS** timed in the same thermal state. At N = 2048 the fastest is
{{best_kernel:2048}} ({{best_pct:2048}}), {{best_speedup_vs:v0_naive:2048}} the naive kernel. All {{npass}} of {{nrows}} kernel/size results pass the cuBLAS check
(largest normwise relative error {{maxerr}}, tolerance {{tol}}), including the non-multiple
sizes 1000 and 1023.

Main findings:

- **Register tiling is where the speed comes from.** Coalescing and shared-memory tiling
  (v1, v2) leave the kernel at {{pct:v2_smem_tiling:4096}} of cuBLAS. Making each thread compute a
  column of 8 outputs (v3) and then an 8×8 tile (v4) is what gets it to {{pct:v4_2d_blocktiling:4096}}: shared-memory loads per
  FMA drop from {{lds_per_ffma:v2_smem_tiling}} to {{lds_per_ffma:v4_2d_blocktiling}}, and
  DRAM traffic falls to {{traffic_x:v4_2d_blocktiling}} the minimum.
- **Two textbook optimizations did little here, and the counters show why.** The compiler had
  already turned v4's shared-memory reads into 128-bit loads, so hand-vectorizing (v5) changed
  nothing measurable. Two resident blocks per SM already overlap one block's loads with the
  other's math, so double buffering (v6) adds only {{pspeedup:v6_double_buffered:v5_vectorized:4096}} at N = 4096.
- **The T4 is power-limited.** Under a dense FFMA load the SM clock drops far below its
  {{boost_mhz}} MHz boost. To compare kernels fairly, the harness times cuBLAS interleaved with every
  kernel, and records the clock and power after each measurement.

## Results

![GFLOP/s vs size](figures/gflops_vs_size.png)

{{table:headline}}

{{table:all_sizes}}

![% of cuBLAS](figures/pct_cublas.png)

### Roofline

![Roofline](figures/roofline.png)

- **Memory roof:** the **measured** DRAM bandwidth, {{bw}} GB/s ({{bw_pct}} of the
  {{bw_theo}} GB/s spec), from [`src/bandwidth.cu`](src/bandwidth.cu).
- **Compute roof:** the theoretical FP32 peak, {{peak}} GFLOP/s ({{sm_count}} SMs × 64 FP32 lanes ×
  {{boost_mhz}} MHz boost × 2 for FMA). It is labelled theoretical because the T4 can't hold that clock
  under load (see Methodology).
- **Practical ceiling:** cuBLAS.
- **Each kernel's arithmetic intensity is measured:** 2·N³ FLOPs divided by the DRAM bytes
  Nsight Compute counted for that kernel at N = 2048.
- The roofs meet at {{ridge}} FLOP/byte. The vertical line is the best possible intensity at
  this size, {{ideal_ai}} FLOP/byte, which is A and B read once and C written once.

How to read it: v0–v2 sit far *below* the DRAM roof, so DRAM bandwidth isn't what limits them.
They are limited inside the SM, by how many load instructions the L1/shared-memory path can
serve. v4–v6 sit to the *right* of the ridge ({{ai:v4_2d_blocktiling}}–{{ai:v6_double_buffered}}
FLOP/byte), so DRAM is no longer the limit at all. What remains is how fast the SM can issue
FMAs at the clock it can sustain.

{{table:ncu}}

How to read the counter columns:
- "Global sectors per load request" is 32-byte sectors per warp-wide load instruction. 4 is
  ideal for a 32-bit load and 16 for a 128-bit load (v5, v6).
- "Shared loads per FFMA" counts warp-level shared-memory load instructions per warp-level FMA
  instruction.

{{table:resources}}

## The kernels, one idea at a time

Each kernel file starts with a comment block: what changed from the previous version, why it
should be faster, and what limits it now. Below, the same story with the measurements.
Comparisons between kernels use the **throttle-normalised speedup**, the ratio of their
"% of cuBLAS". It cancels out clock changes between the two measurements.

### v0: naive, one thread per output ([`v0_naive.cu`](src/kernels/v0_naive.cu))

Thread (x, y) of a 32×32 block computes C[row][col], with **row taken from `threadIdx.x`**.
A warp is 32 consecutive `threadIdx.x`, so one warp covers 32 *rows* and one column:
- Its loads of `A[row][k]` hit 32 addresses K floats apart: 32 separate 32-byte sectors for 128
  useful bytes. That's uncoalesced; ncu counts {{sectors_per_req:v0_naive}} sectors per request.
- Its loads of `B[k][col]` are a single address for the whole warp, a broadcast.
- Its stores to C are strided like A.

Measured: {{gflops:v0_naive:2048}} GFLOP/s at N = 2048, {{pct:v0_naive:2048}} of cuBLAS.
It moves {{dram_mb:v0_naive}} MB through DRAM, {{traffic_x:v0_naive}} the minimum.

**A side finding: v0 is {{sizeratio:v0_naive:1023:1024}} faster at N = 1023 than at N = 1024.** v0 reaches {{gflops:v0_naive:1023}}
GFLOP/s at N = 1023 but only {{gflops:v0_naive:1024}} at N = 1024, and this reproduced in every
run. The obvious explanation would be extra cache misses from the power-of-two stride. The
counters rule that out: DRAM traffic, L1 and L2 hit rates and sector counts are all
essentially the same at both sizes, yet the kernel takes far longer at 1024.

{{table:v0stride}}

So the same accesses, hitting in L1 just as often, are served more slowly. With a 4 KB stride
(1024 floats), the 32 addresses of one warp request all map to the same L1 cache *set*, so the
lookups can't proceed in parallel; an odd stride spreads them across sets. That is my working
explanation. The L1 set-conflict counter that would confirm it directly wasn't collected in
this run.

### v1: coalesced ([`v1_coalesced.cu`](src/kernels/v1_coalesced.cu))

**One change:** `threadIdx.x` now picks the *column*. Same loop, same FLOPs, same number of
loads. A warp now covers one row and 32 adjacent columns:
- `B[k][col]` is 32 consecutive floats, one 128-byte access.
- `A[row][k]` becomes the broadcast.
- The C stores are contiguous.

Measured: **{{speedup:v1_coalesced:v0_naive:2048}} faster than v0** at N = 2048
({{gflops:v1_coalesced:2048}} GFLOP/s). Sectors per request drop from {{sectors_per_req:v0_naive}}
to {{sectors_per_req:v1_coalesced}}.

New limit: every FMA still needs two global load instructions. They mostly hit in L1
({{l1hit:v1_coalesced}}), but the load/store pipeline, not the FMA units, sets the pace.

### v2: shared-memory tiling ([`v2_smem_tiling.cu`](src/kernels/v2_smem_tiling.cu))

The block copies a 32×32 tile of A and of B into shared memory, an on-chip scratchpad the
program manages explicitly (64 KB per SM on a T4). Each thread then does its 32 multiply-adds
from there. Each element is fetched from global memory once per block instead of once per
thread, so global load instructions drop 32×.

There are two `__syncthreads()` per tile:
1. The first, so no thread reads the tile before it's fully written.
2. The second, so no thread overwrites it with the next tile while others are still reading.

Edge tiles are zero-padded, so sizes 1000 and 1023 need no special case.

Measured: **{{pspeedup:v2_smem_tiling:v1_coalesced:4096}} vs v1** at N = 4096
({{gflops:v2_smem_tiling:4096}} GFLOP/s, {{pct:v2_smem_tiling:4096}} of cuBLAS).

What the counters say:
- **DRAM traffic barely changed** ({{dram_mb:v1_coalesced}} → {{dram_mb:v2_smem_tiling}} MB).
  v1 was already getting its reuse from L1. The gain is in *instructions*: global loads became
  shared loads. ncu counts {{lds_per_ffma:v2_smem_tiling}} shared loads per FMA, not the naive
  2, because the compiler reads each thread's A row with 128-bit loads (4 values at once).
- The new limit is still about 1 shared load per FMA. On top of that, 1024 threads per block
  means {{blocks:v2_smem_tiling}} block per SM, so at every `__syncthreads()` the whole SM waits
  for its slowest warp.

### v3: 1D register tiling ([`v3_1d_blocktiling.cu`](src/kernels/v3_1d_blocktiling.cu))

Each thread now computes **8 outputs in a column**, keeping 8 accumulators in registers. Per k
it loads one B value and reuses it for 8 FMAs. The block tile is 64×64, 8 deep, with 512
threads, so {{blocks:v3_1d_blocktiling}} blocks fit on an SM. Per tile phase a block loads 1024
floats for 32,768 FMAs, twice v2's reuse.

Measured: **{{pspeedup:v3_1d_blocktiling:v2_smem_tiling:4096}} vs v2** at N = 4096
({{gflops:v3_1d_blocktiling:4096}} GFLOP/s).
- Shared loads per FMA fall to {{lds_per_ffma:v3_1d_blocktiling}}.
- DRAM traffic falls to {{dram_mb:v3_1d_blocktiling}} MB, thanks to the larger tile.
- {{regs:v3_1d_blocktiling}} registers per thread, {{spill:v3_1d_blocktiling}} bytes spilled.

### v4: 2D register tiling ([`v4_2d_blocktiling.cu`](src/kernels/v4_2d_blocktiling.cu))

Each thread computes an **8×8 tile**. Per k it loads 8 A values and 8 B values into registers
and does their 64-FMA outer product. The block tile is 128×128 with 256 threads, and each thread
holds 64 accumulators.

`__launch_bounds__(256, 2)` makes the compiler fit two blocks per SM, which caps it at 128
registers per thread. It uses {{regs:v4_2d_blocktiling}}, with {{spill:v4_2d_blocktiling}}
bytes spilled. That leaves {{occ:v4_2d_blocktiling}} theoretical occupancy
({{achieved_occ:v4_2d_blocktiling}} achieved): fewer warps than v3, but each has 64 independent
FMAs per k step, so latency is hidden by instruction-level parallelism instead of by more
threads.

Measured: **{{pspeedup:v4_2d_blocktiling:v3_1d_blocktiling:4096}} vs v3**,
{{gflops:v4_2d_blocktiling:4096}} GFLOP/s, {{pct:v4_2d_blocktiling:4096}} of cuBLAS at N = 4096.
- Shared loads per FMA: {{lds_per_ffma:v4_2d_blocktiling}}, which is 4 per 64 FMAs. That means
  the compiler already uses 128-bit shared loads.
- DRAM traffic is {{dram_mb:v4_2d_blocktiling}} MB ({{ai:v4_2d_blocktiling}} FLOP/byte, past the
  roofline ridge).
- ncu reports {{bank_conflicts:v4_2d_blocktiling}} shared-memory bank conflicts at N = 2048.

### v5: vectorized loads ([`v5_vectorized.cu`](src/kernels/v5_vectorized.cu))

Same tiling as v4. Global loads become 128-bit `float4`, and the A tile is stored **transposed**
in shared memory, so a thread's 8 A values for one k are contiguous. When a row can't be read as
aligned `float4` (K or N not a multiple of 4, as at 1023) or a chunk crosses the matrix edge, the
loader falls back to guarded scalar loads.

Measured: **{{pspeedup:v5_vectorized:v4_2d_blocktiling:4096}} vs v4** at N = 4096 and
{{pspeedup:v5_vectorized:v4_2d_blocktiling:2048}} at N = 2048, so no real change.

The counters explain why:
- Shared loads per FMA are identical ({{lds_per_ffma:v5_vectorized}} vs
  {{lds_per_ffma:v4_2d_blocktiling}}). The compiler had already vectorized v4's shared reads.
- The bank-conflict count is unchanged ({{bank_conflicts:v5_vectorized}}).
- Global loads were never the bottleneck. Per 8-deep tile, a thread does 512 FMAs and only 8
  global loads, so cutting those 8 to 2 saves almost nothing.
- At N = 1023 v5 is slower than v4 ({{pct:v5_vectorized:1023}} vs {{pct:v4_2d_blocktiling:1023}}
  of cuBLAS): every load takes the guarded scalar path, which costs more than v4's plain loads.

This is a negative result, kept on purpose: measure before assuming an optimization helps.

### v6: double buffering ([`v6_double_buffered.cu`](src/kernels/v6_double_buffered.cu))

Same tiles and loads as v5, with two shared-memory buffers. Each thread issues its global loads
for tile t+1 *before* computing tile t, and stores them into the other buffer afterwards, with
one `__syncthreads()` per tile instead of two. The cost is 16 KB of shared memory per block
(2 blocks still fit) and {{regs:v6_double_buffered}} registers, still within the 128 cap and
with {{spill:v6_double_buffered}} bytes spilled.

Measured: **{{pspeedup:v6_double_buffered:v5_vectorized:4096}} vs v5** at N = 4096
({{gflops:v6_double_buffered:4096}} GFLOP/s, {{pct:v6_double_buffered:4096}} of cuBLAS) and
{{pspeedup:v6_double_buffered:v5_vectorized:2048}} at N = 2048. A small gain: with 2 blocks
per SM, the hardware was already overlapping one block's loads with the other's math.

### v7: conflict-free shared memory ([`v7_conflict_free.cu`](src/kernels/v7_conflict_free.cu))

**Diagnosis.** In v4–v6, every shared-load instruction had exactly two bank conflicts. Shared
memory has 32 banks of 4 bytes, and a 128-bit load is served in 4 phases of 8 threads. In v6,
the 8 threads of a phase read B at columns 0, 8, 16, …, 56. Those 16-byte slots are 32 bytes
apart, so threads 0 and 4, 1 and 5, and so on collide on the same banks.

**The fix changes only which outputs a thread owns.** Each thread now holds four 4×4 pieces 64
rows/columns apart, instead of one 8×8 square. A phase then reads columns 0, 4, …, 28: 128
contiguous bytes, which is conflict-free. The transposed A tile is padded from 128 to 132 floats
per row, so its stores no longer collide either. The output strips are now contiguous, so C is
written with 128-bit stores.

Measured: **{{pspeedup:v7_conflict_free:v6_double_buffered:4096}} vs v6** at N = 4096
({{gflops:v7_conflict_free:4096}} GFLOP/s, {{pct:v7_conflict_free:4096}} of cuBLAS) and
{{pspeedup:v7_conflict_free:v6_double_buffered:2048}} at N = 2048. Bank conflicts at N = 2048:
{{bank_conflicts:v7_conflict_free}} on loads and {{bank_conflicts_st:v7_conflict_free}} on stores,
against {{bank_conflicts:v6_double_buffered}} and {{bank_conflicts_st:v6_double_buffered}} for v6.
<!-- FINALIZE AFTER RUN: one sentence interpreting the v7 speedup given 0 conflicts. -->

**The conflict counts were predicted before the run.**
[`scripts/bank_model.py`](scripts/bank_model.py) replays each kernel's shared-memory addresses
for one warp and applies the bank rules. For v5 and v6 it reproduces Nsight Compute's counts
exactly:

{{table:bankmodel}}

### v8: warp tiling ([`v8_warptiling.cu`](src/kernels/v8_warptiling.cu))

v8 adds a level between the block and the thread, the way CUTLASS and cuBLAS organise SGEMM:
block tile → **warp tile** → thread tile.
- The 8 warps of a block each own a contiguous 64×32 piece of the 128×128 tile.
- Inside it, each thread computes 2×2 pieces of 4×4.

Per k step a thread loads exactly what it did in v7 (8 A + 8 B values). But the warp as a whole
asks shared memory for **96 distinct floats instead of 144**: more lanes share each value. That
cuts shared-memory bandwidth per FMA by a third. Reads stay conflict-free (see the model table
above).

Measured: **{{pspeedup:v8_warptiling:v7_conflict_free:4096}} vs v7** at N = 4096
({{gflops:v8_warptiling:4096}} GFLOP/s, {{pct:v8_warptiling:4096}} of cuBLAS) and
{{pspeedup:v8_warptiling:v7_conflict_free:2048}} at N = 2048. {{regs:v8_warptiling}} registers
per thread, {{spill:v8_warptiling}} bytes spilled.
<!-- FINALIZE AFTER RUN: interpretation. -->

### Tuning study: one kernel, many tile shapes

The v7 and v8 kernels are C++ templates over their tile sizes, so `./bench --tune` can run 18
configurations of the same code at every size, all verified against cuBLAS:
- **v7:** 12 block/thread tile shapes;
- **v8:** 6 warp-tile shapes.

The number to watch is **waves**. With {{sm_count}} SMs × (blocks per SM) slots, a grid of B blocks
runs in B / slots waves. A fractional last wave leaves SMs idle, and fewer than one wave means
SMs are idle the whole time.
- At N = 256 a 128×128 tile gives {{tune_waves:v7_128x128x8_8x8:256}} waves.
- A 32×32 tile gives {{tune_waves:v7_32x32x8_4x4:256}} waves.

{{table:tune}}

<!-- FINALIZE AFTER RUN: what the best-per-size pattern shows. Caveat: picking the best of many
noisy measurements biases the winner upward (winner's curse). -->

### What the compiler actually emitted (SASS)

"The compiler already vectorised v4's shared loads" is checked in the machine code: `cuobjdump
-sass` on the built binary, counting instructions per kernel:

{{table:sass}}

<!-- FINALIZE AFTER RUN: point to v4's LDS.128 count, v5's LDG.128, v7's STG.128. -->

### What still separates the best kernel from cuBLAS

- **The clock.** The fastest single timed run of {{best_kernel:4096}} at N = 4096 is far above
  its median; the T4 throttles to stay within {{power_limit}} (table in Methodology).
- **Occupancy.** 64 accumulators per thread cap v4–v8 at {{occ:v4_2d_blocktiling}} theoretical
  occupancy.
- **Tile choice per size.** At N = 256, v3's smaller tile beats every 128×128 kernel
  ({{gflops:v3_1d_blocktiling:256}} vs {{gflops:v6_double_buffered:256}} GFLOP/s for v6), which is
  what the tuning study quantifies.

## Methodology

- **Timing.** CUDA events around each launch: {{warmups}} warm-ups, then {{reps}} timed runs. Reported are the
  median (GFLOP/s uses it), the min and the p90. GFLOP/s = 2·N³ / median time.
- **Correctness.** Every kernel is checked against `cublasSgemm` on every run, at every size.
  The check is normwise relative error, max|C − C_ref| / max|C_ref| ≤ {{tol}}. Before each check
  C is filled with NaN, so a kernel that skips elements can't pass on stale data. A kernel that
  fails at any size is marked FAILED and left out of every figure.
- **Row-major vs cuBLAS's column-major.** A row-major matrix read as column-major is its
  transpose. So row-major C = A·B is computed as column-major Cᵀ = Bᵀ·Aᵀ, by swapping the
  operands and M/N in the `cublasSgemm` call. No data is transposed or copied.
- **Throttling and paired cuBLAS timing.** The T4 is capped at {{power_limit}}, and its SM clock drops
  under sustained load. A cuBLAS time taken at the start of a size and a kernel time taken a
  minute later aren't comparable. So each timed kernel run is immediately followed by a timed
  cuBLAS run, and "% of cuBLAS" uses that paired median. The standalone cuBLAS row in the
  tables is kept for transparency; it can differ from the paired figure ({{cublas_paired:4096}}
  GFLOP/s at N = 4096) because it ran in a different thermal state. The clock table follows
  the list.
- **Run-to-run variation.** Separate Colab sessions get different GPUs and thermal states
  (table below), so every headline number comes from a single run in one session.
- **Nsight Compute** worked on Colab's T4 (it is often blocked with `ERR_NVGPUCTRPERM`). ncu
  locks clocks and flushes caches between passes, so only its counters are used here, never
  its timings. Counters come from a separate session right after the benchmark run, with the
  same code.
- **Registers, spills and occupancy** are read from the compiled binary at run time
  (`cudaFuncGetAttributes`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`).
- **Not measured:** the clock *during* each kernel (only sampled right after), per-kernel
  energy, and the L1 set-conflict counter for the v0 stride effect.

{{table:clocks}}

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
2. Run the cells. They clone the repo, run `make` (default `ARCH=sm_75`), run `./bench` and
   `./bandwidth`, collect Nsight Compute counters, and download `results/` as a zip.
3. Locally, unzip into `results/` and regenerate the figures and this README:
   ```
   pip install -r requirements.txt
   python scripts/plot.py
   python scripts/make_tables.py
   ```

Build anywhere with CUDA: run `make` (or `make ARCH=sm_60` for a P100), then `./bench` from the
repo root. `./bench --kernel v4 --size 2048` runs a single kernel at a single size.

## Credits

The kernel sequence v0–v5 follows Simon Boehm's worklog
[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM).
My own contributions:
- the implementation and the T4 measurements;
- the harness (verification, paired cuBLAS timing, clock and power logging, generated
  reporting);
- the Nsight Compute analysis, including the negative results for v5 and v6;
- the write-up.
