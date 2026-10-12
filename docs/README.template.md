# cuda-sgemm: SGEMM from a naive kernel toward cuBLAS on an NVIDIA T4

Single-precision matrix multiply (C = A·B, row-major FP32) written in CUDA C++ and optimized
one idea at a time. Each step is measured on a Tesla T4, checked against cuBLAS, profiled with
Nsight Compute, and explained in terms of memory coalescing, data reuse, occupancy and
instruction mix. **Every number in this README is generated from the CSVs in
[`results/`](results/)** by [`scripts/make_tables.py`](scripts/make_tables.py); none is typed
by hand.

**Result.** The best FP32 kernel at N = 4096, {{best_kernel:4096}}, reaches **{{best_pct:4096}} of
cuBLAS** ({{best_gflops:4096}} GFLOP/s). At N = 256 the autotuned dispatcher `v9_autotuned` **beats
cuBLAS: {{pct:v9_autotuned:256}}**. On a skinny transformer shape (128 × 4096 × 4096), split-K
lifts the result from {{shape_pct:v9_autotuned:128:4096:4096}} to
{{shape_pct:v11_splitk:128:4096:4096}}. Kernels are ranked by % of cuBLAS timed in the same thermal state, because raw
GFLOP/s on a power-capped T4 depends on how hot the GPU was at the time. At N = 2048 the best,
{{best_kernel:2048}} ({{best_pct:2048}}), is {{best_speedup_vs:v0_naive:2048}} the naive kernel. A
separate FP16 **tensor-core** track reaches **{{tc_best_tflops:2048}} TFLOP/s** with
{{tc_best:2048}} ({{tc_best_pct:2048}} of `cublasGemmEx`). All {{npass}} of {{nrows}} kernel/size results pass the cuBLAS check
(largest normwise relative error {{maxerr}}, tolerance {{tol}}), including the non-multiple
sizes 1000 and 1023.

Main findings:

- **Register tiling is where the speed comes from.** Coalescing and shared-memory tiling
  (v1, v2) leave the kernel at {{pct:v2_smem_tiling:4096}} of cuBLAS. Making each thread compute a
  column of 8 outputs (v3) and then an 8×8 tile (v4) is what gets it to {{pct:v4_2d_blocktiling:4096}}: shared-memory loads per
  FMA drop from {{lds_per_ffma:v2_smem_tiling}} to {{lds_per_ffma:v4_2d_blocktiling}}, and
  DRAM traffic falls to {{traffic_x:v4_2d_blocktiling}} the minimum.
- **Bank conflicts were the hidden limiter.** Nsight Compute found {{conflicts_per_lds:v6_double_buffered}} shared-memory bank
  conflicts per shared load in v4–v6. Remapping which outputs each thread owns (v7) cut them
  {{conflict_ratio:v6_double_buffered:v7_conflict_free}}, for a
  {{pspeedup:v7_conflict_free:v6_double_buffered:4096}} speedup: from {{pct:v6_double_buffered:4096}}
  to {{pct:v7_conflict_free:4096}} of cuBLAS at N = 4096. A small Python model of the bank
  rules predicted v6's conflict count exactly.
- **Two textbook optimizations did little here, and the counters show why.** The machine code
  shows the compiler had already turned v4's shared-memory reads into 128-bit loads, so
  hand-vectorizing (v5) changed nothing measurable. Two resident blocks per SM already overlap
  one block's loads with the other's math, so double buffering (v6) adds only
  {{pspeedup:v6_double_buffered:v5_vectorized:4096}} at N = 4096.
- **Tile shape matters as much as any single optimization.** A sweep over {{tune_count}} tile
  configurations of the same kernel finds {{tune_best:4096}} at {{tune_best_pct:4096}} of cuBLAS
  at N = 4096. At N = 256, small tiles beat cuBLAS outright ({{tune_best_pct:256}} for
  {{tune_best:256}}): there the question is how many of the {{sm_count}} SMs get any work at all.
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

So the same accesses, hitting in L1 just as often, are served more slowly. My first hypothesis
was L1 cache *set* conflicts. With a 4 KB stride, the 32 addresses of one warp request map to
the same set, and an odd stride would spread them out. I measured it, and **the data rejects
it**: N = 1023 has *more* L1 set conflicts ({{stride:l1tex__t_set_conflicts_pipe_lsu_mem_global_op_ld.sum:1023}})
than N = 1024 ({{stride:l1tex__t_set_conflicts_pipe_lsu_mem_global_op_ld.sum:1024}}), yet runs
faster. The cause is still open. The next candidates are contention for L2 slices or DRAM
channels: a power-of-two stride sends every request of a warp to the same address-interleaved
partition. None of the counters collected here isolates that.

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
The shared-load conflict count fell {{conflict_ratio:v6_double_buffered:v7_conflict_free}}, and
that alone bought the largest step since v4: the FFMA share of instructions, shared loads per
FMA and DRAM traffic are essentially unchanged from v6 (table above). What changed is that each
shared load now takes one pass instead of three.

**The conflict counts were predicted before the run.**
[`scripts/bank_model.py`](scripts/bank_model.py) replays each kernel's shared-memory addresses
for one warp and applies the bank rules.
- For v5 and v6 it reproduces Nsight Compute's counts exactly.
- For v7 and v8 it predicted zero. ncu measures a small residual of v6's count
  that the model doesn't explain ({{conflict_pct:v7_conflict_free:v6_double_buffered}} for v7, {{conflict_pct:v8_warptiling:v6_double_buffered}} for v8), so it is reported here, not hidden.

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
Warp tiling bought nothing measurable on top of v7. That's consistent with v7's counters: once
the conflicts were gone, shared-memory bandwidth was no longer the limit. So asking for fewer
distinct floats per warp (96 instead of 144) relieved a resource that wasn't the bottleneck. On
tensor-core GPUs warp tiling matters more, because the warp tile is the unit that feeds `mma`
instructions.

### Tuning study: one kernel, many tile shapes

The v7 and v8 kernels are C++ templates over their tile sizes, so `./bench --tune` can run 18
configurations ({{tune_count}} in total) of the same code at every size, all verified against cuBLAS:
- **v7:** 12 block/thread tile shapes;
- **v8:** 6 warp-tile shapes.

The number to watch is **waves**. With {{sm_count}} SMs × (blocks per SM) slots, a grid of B blocks
runs in B / slots waves. A fractional last wave leaves SMs idle, and fewer than one wave means
SMs are idle the whole time.
- At N = 256 a 128×128 tile gives {{tune_waves:v7_128x128x8_8x8:256}} waves.
- A 32×32 tile gives {{tune_waves:v7_32x32x8_4x4:256}} waves.

{{table:tune}}

What the sweep shows:
- **Large sizes:** `v7_64x128x8_8x8` beats the default 128×128 tile
  ({{tune_pct:v7_64x128x8_8x8:4096}} vs {{tune_pct:v7_128x128x8_8x8:4096}} at N = 4096). With
  128 threads per block, 3 blocks fit on an SM instead of 2.
- **Deeper K tiles (BK = 16) are consistently worse.** They double shared memory per block, so
  fewer blocks fit on an SM, and that outweighs halving the barriers.
- **Small sizes:** what matters is filling the SMs. At N = 256 the default tile makes
  {{tune_waves:v7_128x128x8_8x8:256}} waves (most SMs idle), while a 32×32 tile makes
  {{tune_waves:v7_32x32x8_4x4:256}}. The small-tile configs are the ones that beat cuBLAS there.
- **Caveat:** picking the best of 18 noisy measurements biases the winner's number upward
  (the "winner's curse"). The per-size winners should be re-measured on their own before
  quoting them as one kernel's result.

### v10: occupancy by launch bounds, and threadblock swizzle

The sweep above says the best shape wins on **occupancy**: with 128-thread blocks, 3 fit on an SM
instead of 2. [`v10_tuned.cu`](src/kernels/v10_tuned.cu) is v7's kernel (generated from it) with
two more template knobs, swept in the same study:
- **`MINB`**, the minimum-blocks-per-SM hint of `__launch_bounds__`. It caps registers per
  thread at 65536 / (MINB × threads).
- **`SWZ`**, a threadblock swizzle. The grid is walked in groups of SWZ block-columns, so blocks
  resident at the same time share B column-panels in L2.

Results for the 64×128 tile at N = 4096:

| config | registers | blocks/SM | % of cuBLAS |
|---|---|---|---|
| `v10_64x128x8_8x8_b2_s0` (= v7's best shape) | {{tune_res:v10_64x128x8_8x8_b2_s0:regs_per_thread}} | {{tune_res:v10_64x128x8_8x8_b2_s0:blocks_per_sm}} | {{tune_pct:v10_64x128x8_8x8_b2_s0:4096}} |
| `v10_64x128x8_8x8_b4_s0` (128-register cap) | {{tune_res:v10_64x128x8_8x8_b4_s0:regs_per_thread}} | {{tune_res:v10_64x128x8_8x8_b4_s0:blocks_per_sm}} | {{tune_pct:v10_64x128x8_8x8_b4_s0:4096}} |
| `v10_64x128x8_8x8_b2_s8` (swizzle only) | {{tune_res:v10_64x128x8_8x8_b2_s8:regs_per_thread}} | {{tune_res:v10_64x128x8_8x8_b2_s8:blocks_per_sm}} | {{tune_pct:v10_64x128x8_8x8_b2_s8:4096}} |
| `v10_64x128x8_8x8_b4_s4` (cap + swizzle 4) | {{tune_res:v10_64x128x8_8x8_b4_s4:regs_per_thread}} | {{tune_res:v10_64x128x8_8x8_b4_s4:blocks_per_sm}} | {{tune_pct:v10_64x128x8_8x8_b4_s4:4096}} |
| `v10_64x128x8_8x8_b4_s8` (cap + swizzle 8) | {{tune_res:v10_64x128x8_8x8_b4_s8:regs_per_thread}} | {{tune_res:v10_64x128x8_8x8_b4_s8:blocks_per_sm}} | {{tune_pct:v10_64x128x8_8x8_b4_s8:4096}} |
| `v10_64x128x8_8x8_b4_s16` (cap + swizzle 16) | {{tune_res:v10_64x128x8_8x8_b4_s16:regs_per_thread}} | {{tune_res:v10_64x128x8_8x8_b4_s16:blocks_per_sm}} | {{tune_pct:v10_64x128x8_8x8_b4_s16:4096}} |

- **Capping registers worked.** At 128 registers per thread the compiler still didn't spill
  ({{tune_res:v10_64x128x8_8x8_b4_s0:local_bytes_per_thread}} bytes of local memory), and a
  fourth block per SM gives more warps to hide shared-load and barrier latency.
- **The swizzle results scatter in both directions** across group widths, by about as much as
  the same configuration moves between tuning runs. Within this noise there's no reliable swizzle
  effect. That's consistent with the roofline: these kernels sit far to the right of the ridge,
  so L2 and DRAM traffic aren't what limits them.
- **Pushing further spilled.** A 102-register cap (MINB = 5) and the 96×128 and 64×192 shapes
  spill ({{tune_res:v10_64x128x8_8x8_b5_s0:local_bytes_per_thread}},
  {{tune_res:v10_96x128x8_8x8_b3_s0:local_bytes_per_thread}} and
  {{tune_res:v10_64x192x8_8x8_b3_s0:local_bytes_per_thread}} bytes per thread) and lose.

### What the compiler actually emitted (SASS)

"The compiler already vectorised v4's shared loads" is checked in the machine code: `cuobjdump
-sass` on the built binary, counting instructions per kernel:

{{table:sass}}

Three claims made earlier are confirmed in the machine code:
- **v4** already reads shared memory only with `LDS.128`, though its source uses scalar reads.
  That is why v5 gained nothing.
- **v5** is where `LDG.128` (128-bit global loads) first appears.
- **v7** writes C with `STG.128`.

v2's static shared-loads-per-FMA ratio ({{sass:v2_smem_tiling:LDS}} loads for
{{sass:v2_smem_tiling:FFMA}} FMAs) matches the dynamic ratio ncu measured
({{lds_per_ffma:v2_smem_tiling}}).

### v9: autotuned dispatch ([`v9_autotuned.cu`](src/kernels/v9_autotuned.cu))

v9 adds no new device code. [`scripts/gen_dispatch.py`](scripts/gen_dispatch.py) reads the tuning
CSV ({{dispatch_source}}) and writes a table: for each tuned size, the configuration with the best
% of cuBLAS among those that passed everywhere. At run time v9 picks the entry closest to the
problem size. The choices it makes:
- {{dispatch:256}} at N = 256;
- {{dispatch:1024}} at N = 1024;
- {{dispatch:4096}} at N = 4096.

Because the best of many noisy measurements is biased upward, v9 is **measured again in its own
benchmark run**, and only that number is reported:
- **{{pct:v9_autotuned:4096}} of cuBLAS at N = 4096**, {{pspeedup:v9_autotuned:v7_conflict_free:4096}}
  vs v7;
- **{{pct:v9_autotuned:256}} at N = 256**, where it beats cuBLAS ({{gflops:v9_autotuned:256}} vs
  {{gflops:cublas:256}} GFLOP/s).
**Winner's-curse check.** At N = 4096 the independent measurement
({{pct:v9_autotuned:4096}}) matches what the tuning run recorded for that configuration
({{dispatch_pct:4096}}). At N = 256 it comes in lower ({{pct:v9_autotuned:256}} vs
{{dispatch_pct:256}}): small kernels take tens of microseconds, so their measurements are
noisier, and picking the best of 18 of them inflates the winner the most. v9 still beats cuBLAS
there. At N = 1024 the dispatched configuration ({{pct:v9_autotuned:1024}}) is slightly below the
default v7 ({{pct:v7_conflict_free:1024}}) in this run, for the same reason.

### Tensor-core track (separate precision)

The FP32 series above multiplies FP32 inputs on the FP32 units. A T4 also has **tensor cores**:
8 per SM, each doing a 4×4×4 FP16 matrix multiply-accumulate per clock. That's 8× the SM's FP32
FMA rate. They're programmed through WMMA (`<mma.h>`): a warp loads 16×16 "fragments" of A and B
and issues `mma_sync`.

This is a different numerical problem: A and B are rounded to FP16, accumulation stays FP32. So
this track is **verified and timed against `cublasGemmEx` with the same FP16 inputs**, never
against SGEMM, and it never appears in the FP32 tables. Two kernels:
- [`tc0_wmma_naive.cu`](src/kernels/tc0_wmma_naive.cu): one warp per 16×16 output tile, with
  fragments loaded straight from global memory. This is v1's problem again: no reuse.
- [`tc1_wmma_smem.cu`](src/kernels/tc1_wmma_smem.cu): a 128×128 block tile staged in shared
  memory, with rows padded 8 halves to spread banks. Each of 8 warps owns 64×32 = 4×2 fragments.

{{table:tc}}

- [`tc2_wmma_db.cu`](src/kernels/tc2_wmma_db.cu): tc1 plus double buffering. The next K tile's
  global loads are in flight while the tensor cores work, with two shared buffers and one
  barrier per tile. The K tile is cut from 32 to 16 so both buffers still let 2 blocks share an
  SM.

At N = 4096, `tc1` runs **{{tc_speedup:tc1_wmma_smem:tc0_wmma_naive:4096}} faster than `tc0`** and
{{tc_vs_fp32:tc1_wmma_smem:4096}} the throughput of the fastest FP32 kernel, at
{{tc_pct:tc1_wmma_smem:4096}} of `cublasGemmEx`. `tc2` vs `tc1`:
{{tc_speedup:tc2_wmma_db:tc1_wmma_smem:2048}} at N = 2048 ({{tc_pct:tc2_wmma_db:2048}} of
`cublasGemmEx`), with the tensor pipe busy {{tcpipe:tc2_wmma_db}} of the time.
**Double buffering only paid off at small sizes.**
- At N = 512, tc2 beats tc1: {{tc_pct:tc2_wmma_db:512}} vs {{tc_pct:tc1_wmma_smem:512}} of
  `cublasGemmEx`.
- From N = 1024 up it's slower: {{tc_pct:tc2_wmma_db:2048}} vs {{tc_pct:tc1_wmma_smem:2048}} at
  N = 2048. To fit two buffers and still keep 2 blocks per SM, tc2 halves the K tile. That
  halves the MMAs between barriers and doubles the load rounds per unit of K, and once the grid
  fills the GPU this costs more than the overlap saves.
- Tensor pipe busy: {{tcpipe:tc2_wmma_db}} for tc2 vs {{tcpipe:tc1_wmma_smem}} for tc1. This is
  another measured negative result, kept for the same reason as v5.

- [`tc3_ptx_mma.cu`](src/kernels/tc3_ptx_mma.cu): the same tiling as tc2, written without WMMA.
  It uses the PTX instructions WMMA hides: `ldmatrix.sync…x4` (one instruction loads four 8×8
  FP16 matrices into the exact register layout the MMA expects; `.trans` handles B) and
  `mma.sync.aligned.m16n8k8`, Turing's native tensor-core instruction. Its fragment index math
  was checked before the GPU run against a CPU emulation of both instructions, written from the
  PTX ISA's layout definitions.

`tc3` vs `tc2`: {{tc_speedup:tc3_ptx_mma:tc2_wmma_db:2048}} at N = 2048
({{tc_pct:tc3_ptx_mma:2048}} of `cublasGemmEx`), tensor pipe busy {{tcpipe:tc3_ptx_mma}}.
tc3 passes the check at every size, so the hand-written fragment layouts are right on real
hardware. It is **not faster than tc2**, which has the same tiling: tc3 changed the instructions,
not the schedule, and the schedule is what held tc2 back. It also reached the 128-register cap
and spilled {{tc_res:tc3_ptx_mma:local_bytes_per_thread}} bytes per thread, because the
prefetched tile is held in registers.

- [`tc4_ptx_bk32.cu`](src/kernels/tc4_ptx_bk32.cu): tc3's PTX inner loop with tc1's schedule
  (32-deep K tile, single buffer, two 16-deep MMA steps per barrier, no prefetch registers).
  This separates the two effects: instruction path (PTX vs WMMA) and schedule (K depth).

`tc4`: {{tc_pct:tc4_ptx_bk32:2048}} of `cublasGemmEx` at N = 2048 (tc1:
{{tc_pct:tc1_wmma_smem:2048}}), {{tc_tflops:tc4_ptx_bk32:2048}} TFLOP/s, spills
{{tc_res:tc4_ptx_bk32:local_bytes_per_thread}} bytes per thread.
**Hand-written PTX did not beat WMMA, and the machine code shows why.** The compiler already
lowers tc1's WMMA calls to the same instructions: its SASS contains
{{sass:tc1_wmma_smem:LDSM}} `LDSM` (the machine form of `ldmatrix`) and
{{sass:tc1_wmma_smem:HMMA}} `HMMA` (tensor-core MMA) instructions. Writing them by hand bought
control but no new capability on Turing, and tc4 also spills at the 128-register cap. The
tensor-core track therefore stops at tc1 as its best kernel. The lesson generalises: read the
SASS before assuming hand-written assembly will win.
The counters (N = 2048) show it's the same reuse story as v1 → v2, now on tensor cores:
- **`tc0`** moves {{tcdram_mb:tc0_wmma_naive}} MB through DRAM, and its tensor pipe is busy only
  {{tcpipe:tc0_wmma_naive}} of the time: the tensor cores mostly wait for data.
- **`tc1`** stages tiles in shared memory, cutting traffic to {{tcdram_mb:tc1_wmma_smem}} MB, and
  the tensor pipe is busy {{tcpipe:tc1_wmma_smem}} of the time.
- **The remaining gap to `cublasGemmEx`:** one shared buffer with two barriers per K step (no
  double buffering), and fragment loads through generic shared loads instead of `ldmatrix`.
- **Precision cost of FP16 inputs** (vs FP32 SGEMM on the same data): {{tc_prec:2048}} normwise
  relative error at N = 2048. That's acceptable for many workloads, but not interchangeable with
  SGEMM.

### Real-world shapes

Square matrices are the textbook case. Transformer layers multiply rectangles: M = tokens, and
N and K are model or feed-forward widths. `./bench --shapes` runs v4, v7 and v9 on such shapes.
v9's dispatch table was tuned on squares only, so this tests how well the tuning generalises.

{{table:shapes}}
What generalises:
- **Wide shapes.** v9, tuned only on squares, reaches {{shape_pct:v9_autotuned:4096:1024:4096}}
  on 4096 × 1024 × 4096 and {{shape_pct:v9_autotuned:2048:768:768}} on 2048 × 768 × 768. Shapes
  this small finish in tens of microseconds, so their percentages move several points between
  sessions; compare them within one table, not across runs.
- **Small K.** With K = 128 it is level with cuBLAS ({{shape_pct:v9_autotuned:4096:4096:128}}).

What doesn't:
- **Skinny M.** At M = 128, v9 drops to {{shape_pct:v9_autotuned:128:4096:4096}}. A 64-row tile
  gives just 2 block-rows × 32 block-columns = 64 blocks for 40 SMs × 4 slots, so most of the
  GPU idles.

**v11: split-K** ([`v11_splitk.cu`](src/kernels/v11_splitk.cu)) fixes exactly that case.
- Blocks along a third grid dimension each sum one slice of K, and the slices are combined with
  `atomicAdd`.
- The launcher picks enough slices for about two waves of blocks, and 1 slice (exactly v10)
  when the grid already fills the GPU.
- Skinny M = 128: {{shape_pct:v11_splitk:128:4096:4096}} of cuBLAS (v9:
  {{shape_pct:v9_autotuned:128:4096:4096}}).

### What still separates the best kernel from cuBLAS

- **The clock.** The fastest single timed run of each kernel at N = 4096 is far above its
  median; the T4 throttles to stay within {{power_limit}} (table in Methodology).
- **Occupancy.** 64 accumulators per thread cap the 8×8-per-thread kernels at
  {{occ:v4_2d_blocktiling}} theoretical occupancy; the tuning winner gets 3 blocks per SM by
  using 128-thread blocks.
- **What cuBLAS-class kernels add:** per-architecture hand-tuned SASS, larger K pipelines, and
  heuristics over many more shapes than 18.

## Side study: parallel reduction

A second classic kernel, measured the same way: sum 64 Mi floats, one idea per step
([`src/reduce.cu`](src/reduce.cu)).

| Kernel | What changes |
|---|---|
| `r0` | interleaved tree in shared memory, with divergent warps |
| `r1` | contiguous active threads, but bank-conflicting indices |
| `r2` | sequential addressing: no divergence, no conflicts |
| `r3` | add two elements during the load |
| `r4` | finish the last 32 values with warp shuffles |
| `r5` | grid-stride `float4` loads into a register sum, one atomic per block |

A reduction does one add per 4-byte element, so it is **bandwidth-bound**. The yardstick is
therefore the measured DRAM bandwidth and CUB's `DeviceReduce::Sum`, not FLOPs. Each result is
checked against a double-precision host sum.

{{table:reduce}}
- **The big steps:** `r3` (adding during the load, {{red_speedup:r3_first_add:r2_sequential}} vs
  r2) and `r4` (finishing with warp shuffles, {{red_speedup:r4_warp_shuffle:r3_first_add}} vs
  r3). Both cut the work done per byte loaded, not the bytes themselves.
- **`r5` reaches {{red_gbps:r5_grid_stride}} GB/s, {{red_pct_cub:r5_grid_stride}} of CUB.** That's
  above the copy kernel's figure, because a copy has to read *and* write while a reduction only
  reads.
- **Divergence vs bank conflicts:** `r1` is *slower* than `r0` here
  ({{red_speedup:r1_strided_index:r0_interleaved}}): its strided shared-memory indices cost
  more than the warp divergence they remove.

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
  its timings. Counters, SASS, tuning and the locked run come from a session right after the
  headline benchmark, built from the same commit.
- **Registers, spills and occupancy** are read from the compiled binary at run time
  (`cudaFuncGetAttributes`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`).
- **Clock-locked cross-check.** Colab allowed `nvidia-smi -lgc 900,900`, so the whole benchmark
  was also run with the SM clock locked. The relative results hold:
  {{locked_pct:v7_conflict_free:4096}} of cuBLAS for v7 at N = 4096 with the clock locked,
  vs {{pct:v7_conflict_free:4096}} unlocked. At N = 4096 the power cap still pulled the clock
  below the lock (table below), because a lock can't override the 70 W limit.
- **Not measured:** the clock *during* each kernel (only sampled right after) and per-kernel
  energy.

{{table:clocks}}

{{table:locked}}

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
