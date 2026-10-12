# Learning guide: everything after v3

This is a study guide for the part of the project after v3 (1D register tiling). Each topic
covers four things:

1. the concept from first principles;
2. where it lives in our code;
3. what our measurements showed;
4. the interview questions you're likely to get, with short model answers.

**Every number here comes from `results/*.csv` or the generated `README.md`.** v0–v6 numbers
in sections 1–4 come from the earlier run `sgemm_Tesla_T4_2026-10-12_112528.csv`. v7, v8, the
tuning study, SASS, the set-conflict test and the clock-locked run come from
`sgemm_Tesla_T4_2026-10-12_153929.csv` and the `*_2026-10-12_154*` files. Anything about v9 or
the tensor-core kernels is marked **pending Colab run 7** until that run exists.

Quick reminders about the T4 (sm_75), from `results/env_*.json`:

| Resource | Per SM | Whole GPU |
|---|---|---|
| SMs | — | 40 |
| FP32 lanes | 64 | — |
| Registers (32-bit) | 65,536 | — |
| Shared memory | 64 KB | — |
| Max threads | 1,024 (= 32 warps) | — |
| L2 cache | — | 4 MB |
| Boost clock | 1,590 MHz | — |
| Theoretical FP32 peak | — | 40 × 64 × 1.59 GHz × 2 = 8,141 GFLOP/s |
| DRAM bandwidth (spec / measured) | — | 320 GB/s / 234 GB/s |

---

## 1. 2D register tiling and the outer product (v4)

### The concept
Matrix multiply is a sum of **outer products**. For each k, take a column of A and a row of B;
their outer product adds to every element of the C tile. A thread that owns a TM×TN block of C
does, for every k:

```
   load TM values of A (a column slice)   a0 a1 ... a7
   load TN values of B (a row slice)      b0 b1 ... b7
   acc[i][j] += a[i] * b[j]   for all i, j  ->  TM*TN FMAs
```

With TM = TN = 8 that's **16 loads for 64 FMAs, 0.25 loads per FMA**. The values are held in
registers, the fastest storage on the GPU. Compare:
- v2 (1 output per thread): about 2 loads per FMA in source;
- v3 (8×1 strip): 9 loads per 8 FMAs = 1.125.

That ratio is **arithmetic intensity at the register level**: how much math each value
fetched from shared memory pays for.

### In our code
`src/kernels/v4_2d_blocktiling.cu`:
- 128×128 block tile, BK = 8, 256 threads, each holding `acc[8][8]` (64 accumulators).
- `regM[8]` and `regN[8]` are filled from shared memory once per k, then a doubly unrolled
  8×8 FMA loop runs.

### Why 64 accumulators means 128 registers and 50% occupancy
The registers a thread needs: 64 accumulators + 16 operands (8 A + 8 B) + addresses, loop
counters and pointers, so on the order of 100+.

`__launch_bounds__(256, 2)` tells the compiler: "blocks of 256 threads, and I want at least 2
blocks per SM". The compiler must therefore fit 2 × 256 = 512 threads in 65,536 registers,
which caps it at **128 registers per thread**. Measured: **128 registers, 0 bytes spilled**
(`kernels_*.csv`).

With 2 blocks × 256 threads = 512 threads = 16 warps out of 32, **theoretical occupancy is
50%**. ncu measured **47% achieved**.

### ILP vs TLP: why 50% occupancy is fine here
GPUs hide latency in two ways:
- **TLP (thread-level parallelism):** many warps, so when one stalls another issues.
- **ILP (instruction-level parallelism):** one warp has many *independent* instructions in a
  row, so it can keep issuing while earlier results are still in flight.

v4's inner loop has 64 independent FMAs per k step. Even with only 16 warps per SM, each warp
has plenty to issue. v3 had 100% occupancy but far less work per load.

### What we measured
- v4 is **1.98x faster than v3** at N = 4096 (throttle-normalised), at 2,716 GFLOP/s and
  58.5% of cuBLAS.
- **Shared loads per FFMA: measured 0.062**, not the 0.25 the source suggests. 0.0625 = 4/64:
  the compiler packs the 8 A values and the 8 B values into **128-bit loads** (4 floats each),
  so it's 4 load instructions per 64 FMAs, not 16. (Section 7 shows how to confirm this in SASS.)
- DRAM traffic: 224 MB at N = 2048, which is 4.5x the minimum, 76.6 FLOP/byte.

### Interview questions
- **Why does an 8×8 thread tile beat an 8×1 strip?** Loads grow as TM + TN while FMAs grow as
  TM × TN. 8×1 is 9 loads per 8 FMAs; 8×8 is 16 loads per 64 FMAs.
- **Why not 16×16 per thread?** 256 accumulators alone exceed the 255-register limit, so it
  would spill to local memory (which lives in DRAM/L1) and lose everything.
- **What does `__launch_bounds__(256, 2)` do?** It caps registers per thread so that 2 blocks
  of 256 threads fit per SM: 65,536 / 512 = 128.
- **Your occupancy dropped from 100% (v3) to 50% (v4), yet it got 2x faster. Why?** Latency
  is hidden by ILP (64 independent FMAs per step) instead of TLP, and loads per FMA fell 6x
  (0.375 → 0.062 measured).

---

## 2. Occupancy in depth

### The concept
**Occupancy = resident warps per SM ÷ maximum warps per SM** (32 on a T4). A block is resident
only if the SM has room for all of it. Three resources limit how many blocks fit; the smallest
limit wins:

```
blocks/SM = min( 1024 / threads_per_block,                      # thread slots
                 65536 / (regs_per_thread * threads_per_block),  # register file
                 65536 B / smem_per_block,                       # shared memory
                 16 )                                            # hardware block limit
```

Worked examples using our measured resources:

| Kernel | Threads | Regs/thread | Smem/block | Thread limit | Register limit | Smem limit | → blocks/SM | Warps |
|---|---|---|---|---|---|---|---|---|
| v2 | 1024 | 42 | 8 KB | 1 | 1 | 8 | **1** | 32/32 = 100% |
| v3 | 512 | 64 | 4 KB | 2 | 65536 / 32768 = 2 | — | **2** | 32/32 = 100% |
| v4 | 256 | 128 | 8 KB | 4 | 65536 / 32768 = 2 | 8 | **2** | 16/32 = 50% |
| v6 | 256 | 127 | 16 KB | — | 2 | 4 | **2** | 16/32 = 50% |

(In real hardware, registers are allocated in chunks, so the exact limit can differ slightly.
That's why we ask the runtime with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` rather than
computing it by hand.)

**Theoretical vs achieved.** "Theoretical" is what fits. "Achieved"
(`sm__warps_active.avg.pct_of_peak_sustained_active`) is the average actually resident over the
kernel's life. It's lower because of the tail (the last blocks finishing) and ramp-up.
Measured: v4–v6 theoretical 50%, achieved 47%; v2 theoretical 100%, achieved 100%.

### Why lower occupancy can be faster
Occupancy is a means, not a goal. You need *enough* parallelism to hide latency, and each extra
warp costs registers that could hold accumulators. v4 trades half the warps for 8x more reuse
per warp and wins. (Vasily Volkov's well-known talk "Better performance at lower occupancy"
makes exactly this point.)

### In our code
- `src/common.h`: `kernel_resources()` calls `cudaFuncGetAttributes` (registers, spills,
  static smem) and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
- `bench` writes these to `results/kernels_*.csv` on every run.

### Interview questions
- **How do you compute occupancy?** Take the minimum of the thread, register, shared-memory
  and block-count limits, then multiply by warps per block and divide by 32.
- **What does `localSizeBytes > 0` mean?** Spills: registers that didn't fit were moved to
  local memory, which is slow. Ours are all 0.
- **Is higher occupancy always better?** No. It's only needed to hide latency; ILP can do the
  same job, and registers spent on extra warps can't be spent on reuse.
- **Why is achieved occupancy below theoretical?** Tail effects, uneven block finish times,
  and launch ramp-up.

---

## 3. Vectorized memory access (v5), and a null result

### The concept
A load instruction can move 32, 64 or 128 bits per thread: `LDG.E` vs `LDG.E.128` for global
memory, `LDS` vs `LDS.U.128` for shared memory, `STS.128` for shared stores. One 128-bit
instruction moves 4 floats, so it uses 4x fewer instructions for the same bytes.

**Alignment rule:** a 128-bit access must be at an address that's a multiple of 16 bytes. For
row-major A, `A[r*K + c]` is 16-byte aligned when `K % 4 == 0` and `c % 4 == 0` (the
`cudaMalloc` base is 256-byte aligned).

### In our code
`src/kernels/v5_vectorized.cu`:
- `load4()` does one `float4` load when the chunk is in bounds and aligned, otherwise 4
  guarded scalar loads.
- The A tile is stored **transposed** (`As[k][m]`), so a thread's 8 A values for one k are
  adjacent and can be read with two `float4` loads.

### What we measured: the null result
- **v5 vs v4: 1.00x at N = 4096** (throttle-normalised). No gain.
- Shared loads per FFMA: **0.062 in v4 and 0.062 in v5**, identical. The compiler had *already*
  generated 128-bit shared loads for v4's contiguous reads.
- Global loads were never the bottleneck. Per 8-deep tile, a thread does 512 FMAs and only
  8 global loads (v4) versus 2 (v5). Saving 6 instructions out of more than 512 is invisible.
- **At N = 1023 v5 is worse**: 53.1% of cuBLAS vs v4's 60.0%. 1023 isn't a multiple of 4, so
  every load takes the guarded scalar path. That's more instructions and branches than v4's
  plain loads.
- Global sectors per request jump from 3.9 (v4) to 15.8 (v5). That isn't a regression: a
  128-bit warp request covers 512 bytes = 16 sectors, which is ideal for that width.

**Lesson (a great interview story):** "I applied a textbook optimization, measured no change,
and used the counters to explain why: the compiler had already vectorized the hot loads, and
the loads I changed weren't on the critical path."

### Interview questions
- **When does `float4` help?** When load *instruction count* or memory transaction width is
  the bottleneck, e.g. memory-bound copy kernels. Not when FMAs dominate the instruction count.
- **What breaks vectorized loads?** Misalignment (sizes not a multiple of 4, odd offsets),
  which forces a scalar fallback.
- **How did you know the compiler already vectorized v4?** The shared-load-instruction counter
  per FFMA matched v5 exactly, and the SASS dump confirms it: all 32 of v4's shared loads are
  `LDS.128`, even though its C++ source reads scalars.
- **Why transpose A in shared memory?** So that one thread's 8 A values for a given k are
  contiguous, which makes them readable with 128-bit loads.

---

## 4. Double buffering / software pipelining (v6)

### The concept
A tile loop without pipelining looks like this:

```
load tile t  ->  sync  ->  compute tile t  ->  sync  ->  load tile t+1 -> ...
   (wait ~hundreds of cycles for DRAM/L2)
```

With **double buffering** there are two shared-memory buffers, and global loads for the
*next* tile are issued *before* the math on the current one:

```
issue loads(t+1) into registers
compute tile t from buffer cur         <- loads are in flight during these 512 FMAs
store registers into buffer cur^1
sync; cur ^= 1
```

**Why one barrier is enough.** In iteration i, threads read `cur` and write `cur^1`. The barrier
at the end of iteration i−1 guarantees two things:
- everyone has finished *reading* `cur^1` (it was `cur` in i−1) before anyone overwrites it;
- the writes into `cur` from i−1 are visible.

### In our code
`src/kernels/v6_double_buffered.cu`: `As[2][…]`, `Bs[2][…]`, a prologue loads tile 0, and the
`na`/`nb` registers hold the prefetched tile.

### What we measured
- **v6 vs v5: 1.04x at N = 4096**, giving 2,884 GFLOP/s and **60.4% of cuBLAS**, the best
  kernel so far. 63.4% at N = 2048.
- Cost: shared memory doubles to 16,384 bytes per block, and registers go to 127 (still under
  the cap, 0 spilled). 2 blocks per SM still fit.
- **Why the gain is small:** with 2 blocks per SM, while one block waits at its barrier for
  loads, the other block is already computing. The hardware was already overlapping, so
  explicit prefetch helps only a little.

### Interview questions
- **What does double buffering hide?** Global-memory latency, by overlapping the next tile's
  loads with the current tile's math.
- **Why only one `__syncthreads()`?** Reads and writes go to different buffers. One barrier per
  iteration orders "finished reading" before "start overwriting".
- **Why was your gain only about 4%?** Two resident blocks per SM already overlapped one
  block's loads with the other's compute.
- **What does it cost?** 2x shared memory and extra staging registers, which can reduce
  occupancy if either limit is hit.
- **What's the hardware version of this on newer GPUs?** `cp.async` (Ampere and later) and TMA
  (Hopper) copy global→shared asynchronously without staging through registers. The T4
  (Turing) has neither.

---

## 5. Shared-memory bank conflicts (v7)

### The concept
Shared memory is split into **32 banks**, each 4 bytes wide. Byte address `a` lives in bank
`(a / 4) % 32`. Each bank can serve one 4-byte word per cycle.

- If the threads of a request hit **32 different banks**, the request takes 1 cycle
  (1 "wavefront").
- If two threads hit the **same bank at different addresses**, that's a **bank conflict**,
  and the request is replayed.
- If they hit the **same address**, it's a **broadcast**, which is free.

Wider accesses are split into phases:
- 32-bit: the whole warp at once.
- 64-bit: 2 phases of 16 threads.
- **128-bit: 4 phases of 8 threads.** Each phase moves 8 × 16 B = 128 B, exactly one pass over
  all 32 banks, *if* the 8 slots are distinct.

```
banks:   |0..3|4..7|8..11|12..15|16..19|20..23|24..27|28..31|   (each 16-byte slot = 4 banks)
good:     t0   t1    t2     t3     t4     t5     t6     t7      8 consecutive float4 -> 1 cycle
v4-v6:    t0,t4      t1,t5         t2,t6         t3,t7          stride-8 floats -> 2 threads/slot -> 2 cycles
```

### Why v4–v6 conflict
A thread reads B columns `tc*8 … tc*8+7`. In one phase, threads tc = 0..7 start at float
offsets 0, 8, 16, …, 56, i.e. 32 bytes apart. That's 256 bytes of span folded onto 128 bytes of
banks, so threads 0 and 4 share banks, and every phase takes 2 wavefronts.

**Measured:** v4, v5 and v6 each executed **16,777,216 shared-load instructions** with
**33,554,432 bank conflicts, exactly 2 per instruction**, and v5/v6 didn't change it
(`ncu_smem_*.csv`).

The transposed-A *store* in v5 and v6 adds **2,097,152 store conflicts**. Rows 4 apart in k
are 4 × 128 = 512 floats apart, and 512 % 32 = 0, so they hit the same bank.

### The v7 fix (`src/kernels/v7_conflict_free.cu`)
1. **Remap the fragment.** Thread (tr, tc) now owns two 4-wide column strips, `tc*4` and
   `64 + tc*4` (and likewise two row strips). In one phase, threads tc = 0..7 read float
   offsets 0, 4, 8, …, 28: **8 consecutive 16-byte slots, 128 contiguous bytes, conflict-free.**
   A reads are one address per phase, a broadcast.
2. **Pad the transposed A tile from 128 to 132 floats per row.** Rows 4 apart are then
   4 × 132 = 528 floats apart, and 528 % 32 = 16, so they land 16 banks apart and don't
   conflict. Why 132 and not 129? Because every row must start on a **16-byte boundary** so
   the `float4` reads stay legal. 132 × 4 B = 528 B, which is a multiple of 16; 129 wouldn't be.
3. Bonus: each thread's 4 output columns are now contiguous, so C is written with 128-bit
   stores.

**Result (measured):**
- Shared-load bank conflicts fell from **33,554,432 to 14,653** at N = 2048, a **2,290x** drop.
  Store conflicts fell from 2,097,152 to 130,291.
- v7 reaches **81.9% of cuBLAS at N = 4096** (3,177 GFLOP/s), a **1.39x** throttle-normalised
  speedup over v6. It's the biggest single step since v4.
- Nothing else moved: shared loads per FMA (0.062) and FFMA share (87%) are the same as v6, and
  DRAM traffic only dips (207 → 184 MB). The gain is purely that each shared load now takes one
  pass instead of three.
- The analytical model (section 14) predicted **exactly 0**. ncu shows a small residual (0.04%
  of v6's count) that the model doesn't explain. Report it; don't round it to zero.

### Interview questions
- **What is a bank conflict?** Two threads of the same request accessing different addresses
  in the same bank; the request is serialized into multiple wavefronts.
- **Is two threads reading the same address a conflict?** No, it's a broadcast.
- **How do 128-bit accesses change the analysis?** They're served in 4 phases of 8 threads, so
  only threads within the same quarter-warp can conflict with each other.
- **How do you fix conflicts?** Remap which thread reads what (make each phase contiguous), or
  pad the array stride so that colliding rows shift banks. Swizzling (XOR-ing the index) is
  the padding-free alternative.
- **How did you measure it?** With
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (and `_st`) in Nsight Compute,
  divided by the shared-load instruction count.

---

## 6. Autotuning with C++ templates, and wave quantization

### Templates
In `sgemm_v7<BM, BN, BK, TM, TN>` the tile sizes are **compile-time constants**. That lets the
compiler fully unroll every loop and keep `acc[TM][TN]` in registers; runtime-sized arrays
would be forced into local memory. Each configuration is a separate compiled kernel, listed
in `v7_tune_configs()`. `./bench --tune` runs all of them at every size, checked against
cuBLAS like everything else.

### Wave quantization
The GPU runs blocks in **waves**: one wave = SMs × blocks per SM. For v4–v7 at 2 blocks per
SM, that's 40 × 2 = 80 slots.

| Size | 128×128 tiles | Waves | Effect |
|---|---|---|---|
| N = 2048 | 16 × 16 = 256 blocks | 256 / 80 = 3.2 | The last wave runs at 20% occupancy, a "tail" |
| N = 256 | 2 × 2 = **4 blocks** | 4 / 80 = 0.05 | 76 of 80 slots sit idle |

At N = 256 that's why v3 (64×64 tiles, 16 blocks) beats v6 there: 484 vs 232 GFLOP/s,
measured. Small sizes want small tiles; large sizes want big tiles (more reuse). That's
exactly what tuning per size exploits.

### Winner's curse
If you measure 12 noisy configurations and pick the fastest, the winner's number is biased
upward, because part of why it won is luck. The honest fixes: re-measure the chosen
configuration in a separate run, or report the spread across configs.

**Results (measured, `results/tune_sgemm_Tesla_T4_2026-10-12_154334.csv`, 126 of 126
config/size results pass):**
- **Large N:** the best config is `v7_64x128x8_8x8` (128 threads, 3 blocks/SM) at **88.6% of
  cuBLAS at N = 4096** (87.8% at 2048). The default 128×128 tile gets 82.6% in the same run.
  More resident blocks beat a bigger tile here.
- **Small N:** at N = 256, small tiles **beat cuBLAS outright**. Best is `v7_64x64x8_4x4` at
  126.5%; three other small-tile configs also exceed 100%. The default tile makes 0.1 waves
  there, so most SMs sit idle.
- **BK = 16 is consistently worse** (e.g. 72% vs 83% at N = 4096 for 128×128). Doubling shared
  memory per block cuts resident blocks more than halving the barriers helps.
- These are best-of-18 numbers, so they carry the winner's curse (see section 16). That's why
  v9 is re-measured independently.

### Interview questions
- **Why templates rather than runtime parameters?** Full unrolling, and register-resident
  accumulators.
- **What is wave quantization?** Performance loss when the block count isn't a multiple of the
  number of concurrent slots, so the last wave underutilizes the GPU.
- **How would you pick a tile size for an unseen shape?** From a tuning table keyed on size
  (what cuBLAS's heuristics do), or by modelling waves × per-block efficiency.
- **What's the risk in picking the best of N measurements?** Selection bias (winner's curse).
  Confirm with an independent re-run.

---

## 7. SASS vs PTX, and reading machine code

### The concept
- **PTX** is NVIDIA's virtual ISA: portable and not what actually runs.
- **SASS** is the real machine code for a specific architecture (sm_75 here). `ptxas`
  translates PTX to SASS, and that's where register allocation and many optimizations happen.
- `cuobjdump -sass bench` disassembles every kernel in the binary.

Opcodes you'll see:

| Opcode | Meaning |
|---|---|
| `FFMA` | FP32 fused multiply-add |
| `LDG.E` / `LDG.E.128` | global load, 32-bit / 128-bit |
| `LDS` / `LDS.U.128` | shared load, 32-bit / 128-bit |
| `STS` / `STS.128` | shared store |
| `STG.E` / `STG.E.128` | global store |
| `BAR.SYNC` | `__syncthreads()` |

**Static vs dynamic counts.**
- *Static*: how many times an opcode appears in the code (from SASS).
- *Dynamic*: how many times it executed (from ncu, e.g. `smsp__inst_executed`).

Because our inner loops are fully unrolled, the static mix of the loop body mirrors the dynamic
ratio.

### In our project
`cuobjdump -sass bench`, with opcodes counted per kernel (README "What the compiler actually
emitted"). Measured, static counts:
- **v4:** 32 shared loads, all 32 of them `LDS.128`, from scalar C++. That is the direct proof
  behind v5's null result.
- **v5:** where `LDG.128` (128-bit global loads) first appears.
- **v7:** writes C with 16 `STG.128` stores.
- **v2:** 40 `LDS` (8 of them `.128`) for 32 `FFMA`, a static ratio of 1.25 that matches the
  1.250 shared loads per FMA ncu measured at run time.

### Interview questions
- **Difference between PTX and SASS?** PTX is a virtual, forward-compatible IR; SASS is the
  native per-architecture code produced by `ptxas`. Profilers and performance analysis
  ultimately care about SASS.
- **How do you check that the compiler vectorized a load?** Look for `.128` loads in
  `cuobjdump -sass`, or compare instruction counters in ncu.
- **Why might the source and the SASS disagree on instruction count?** Unrolling,
  vectorization, constant folding and predication.

---

## 8. The roofline in depth

### The concept
Performance ≤ min(compute peak, bandwidth × arithmetic intensity), where AI = FLOPs ÷ DRAM
bytes. On a log-log plot that's a slanted "memory roof" meeting a flat "compute roof" at the
**ridge point** = peak ÷ bandwidth.

- Measured roof: 8,141 GFLOP/s ÷ 234 GB/s ≈ **35 FLOP/byte** (the README's ridge).
- Spec roof: 8,141 ÷ 320 ≈ 25.

We use the **measured** bandwidth from `src/bandwidth.cu` (a 512 MiB copy, far larger than
L2): 234 GB/s via `cudaMemcpy`, 73% of the 320 GB/s spec. A kernel can't beat what a plain
copy achieves.

**Minimum-traffic line.** At N = 2048, if A and B were read once and C written once:
3 × 2048² × 4 B gives AI = 2 × 2048³ / (12 × 2048²) = **341 FLOP/byte**. That's the best any
kernel could do.

### Our kernels on it (N = 2048, AI measured from ncu DRAM bytes)
| Kernel(s) | FLOP/byte | Where on the plot |
|---|---|---|
| v0 | 7.6 | Below the roof, not on it |
| v1 / v2 | 10.2 / 10.3 | Below the roof |
| v3 | 25.6 | Below the roof |
| v4 / v5 / v6 | 76.6 / 81.4 / 82.8 | **Right of the ridge**: DRAM can't be the limit |

**Why v0–v2 sit *below* the DRAM roof yet aren't DRAM-bound.** At 10 FLOP/byte the roof allows
10 × 234 ≈ 2,340 GFLOP/s, but v2 gets 849. Something inside the SM limits them first: the
number of load instructions the L1/shared path can serve. Being under the roof only means
"not limited by *this* resource". It doesn't tell you what the limit *is*.

### Interview questions
- **What's the ridge point?** Peak FLOP/s ÷ bandwidth, the AI where a kernel moves from
  memory-bound to compute-bound.
- **A kernel sits far below both roofs: what do you conclude?** Neither DRAM nor peak FP32 is
  the bottleneck. Look at latency, occupancy, L1/shared throughput, instruction mix or
  conflicts.
- **Why use measured rather than spec bandwidth?** Spec bandwidth isn't achievable; a roof
  you can't reach overstates the headroom.
- **How did you get AI?** FLOPs divided by `dram__bytes_read.sum + dram__bytes_write.sum` from
  Nsight Compute. It's measured, not modelled.

---

## 9. Cache set-associativity and the 1023 vs 1024 effect

### The concept
A cache is split into **sets**. An address maps to one set (from some of its address bits),
and each set holds a few lines (its "ways"). Addresses separated by a large **power-of-two
stride** map to the *same set*: they share the bits that pick the set. Many simultaneous
accesses to one set compete for that set, even when every one of them is a hit.

### Our case
v0's warp reads 32 rows of A, K floats apart. At **K = 1024 the stride is 4 KB**, a power of
two, so all 32 addresses map to the same L1 set. At K = 1023 the stride is odd, which spreads
them across sets.

Measured:
- v0 runs at 187 GFLOP/s at N = 1023 versus 60 at N = 1024, **3.1x**, and this reproduced in
  every run.
- Nsight Compute at both sizes shows nearly identical counts, yet the kernel takes 29.2 ms at
  1023 and 96.2 ms at 1024 (ncu, locked clocks):

| | N = 1023 | N = 1024 |
|---|---|---|
| DRAM read | 220 MB | 200 MB |
| L1 hit rate | 99.1% | 99.2% |
| L2 hit rate | 83.3% | 82.9% |
| Global load sectors | 1,104.1 M | 1,107.3 M |

So it isn't misses: the same hits are simply served slower. The first **hypothesis** was
serialization from L1 set conflicts in the tag-lookup stage. We measured the counter that
tests it (`l1tex__t_set_conflicts_pipe_lsu_mem_global_op_ld`), and **the data refutes it**:

| | N = 1023 | N = 1024 |
|---|---|---|
| L1 tag-stage set conflicts | 262.3 M | 234.9 M |
| kernel time (ncu) | 29.1 ms | 96.2 ms |

The *faster* size has *more* set conflicts. So set conflicts aren't the cause.

**What's left (untested):** contention for L2 slices or DRAM channels ("partition camping").
Memory is address-interleaved across slices and channels, and a 4 KB power-of-two stride can
send every request of a warp to the same one. None of the counters we collected isolates that.

**Why a refuted hypothesis is a strength in an interview.** It shows the scientific loop
interviewers look for: observe, hypothesize, pick the counter that could prove you wrong,
measure, update. Saying "I thought X; I measured the counter that tests X; it disagreed; here's
what's left" is more credible than a confident story nobody checked. It also shows you know the
difference between "consistent with" and "confirmed".

### Interview questions
- **Why do power-of-two sizes often run slower?** Strided accesses alias into the same cache
  sets, memory channels or banks.
- **Hit rates are identical but it's 3x slower; how is that possible?** Hit rate counts
  outcomes, not throughput. Conflicts in the lookup stage serialize hits.
- **How would you fix it in a real kernel?** Pad the leading dimension (e.g. 1024 → 1028), or
  access memory in a coalesced way so the stride never appears inside a warp. That's what v1
  does.

---

## 10. Power and thermal throttling, and benchmarking methodology

### The concept
A T4 has a **70 W** board limit. Under dense FFMA load it would exceed that at its 1,590 MHz
boost clock, so firmware lowers the clock. Throughput then depends on the clock, and the clock
depends on temperature and on what ran just before.

### What we measured (N = 4096, clock sampled right after each kernel's timed runs)
- **Clocks of 750–1,125 MHz**, with power at 58–69 W.
- **v6's fastest single run reached 5,015 GFLOP/s, against a median of 2,884.** The first runs
  happen before the clock drops.
- The cuBLAS standalone row ranged from 4,465 to 6,921 GFLOP/s at N = 2048 across 4 Colab
  sessions, a **55% spread**.

### Our defences (`src/bench.cu`)
1. **Paired cuBLAS timing.** Every timed kernel run is immediately followed by a timed cuBLAS
   run, and % of cuBLAS = median paired cuBLAS time ÷ median kernel time. Both see the same
   thermal state.
2. **Throttle-normalised speedup.** Compare kernels by the ratio of their % of cuBLAS, not by
   raw GFLOP/s measured minutes apart. Example from run 3: at N = 1024, raw GFLOP/s made v5
   look 11% slower than v4, but cuBLAS had slowed just as much during v5's slot. Normalised,
   they were equal (56.9% vs 56.7%).
3. **Median, not mean or min.** The median resists outliers. The min is the
   least-throttled run, interesting but not sustainable. p90 shows the spread.
4. **Clock and power logged per row**, plus a start/end snapshot in `env_*.json`.
5. **Headline numbers from a single session.**

**Clock locking.** `nvidia-smi -lgc <min>,<max>` pins the clock, which is how professional
benchmarks get stable numbers. It needs administrator rights on the GPU, which cloud
notebooks often don't grant. **Colab allowed it** (`results/clock_lock_2026-10-12_154252.txt`),
so the whole benchmark was rerun with the clock locked at 900 MHz. The relative results held:
v7 got **82.5% of cuBLAS at N = 4096 locked vs 81.9% unlocked**. At N = 4096 the power cap still
pulled the clock below the lock, because a clock lock can't override the 70 W limit.

### Interview questions
- **Why not just report the best run?** It's not sustainable throughput; under a power cap the
  steady-state clock is what users get.
- **How do you compare two kernels on a throttling GPU?** Interleave each with a reference
  (paired timing), compare their ratios, and log the clocks.
- **Your cuBLAS number varied 55% between sessions. Is the data useless?** No. Within a
  session the paired ratios are stable; that's why the headline uses % of cuBLAS from one run.
- **How else could you stabilize?** Lock the clocks, warm up to thermal steady state, or cool
  down between runs.

---

## 11. Nsight Compute metrics we used

| Metric | What it counts | How to read it |
|---|---|---|
| `dram__bytes_read.sum` / `dram__bytes_write.sum` | Bytes moved to and from DRAM | Divide FLOPs by their sum to get measured AI. v0 moved 2,258 MB at N = 2048 (44.9x the minimum); v6 moved 207 MB (4.1x). |
| `l1tex__t_sector_hit_rate.pct` | % of L1 sector lookups that hit | High (95–99%) in v0/v1 because of caching; about 0% in v2/v3 because data is staged in shared memory exactly once. |
| `lts__t_sector_hit_rate.pct` | L2 hit rate | 80% for v4–v6: big tiles mean nearby blocks share A/B rows and columns in L2. |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` ÷ `l1tex__t_requests_…` | 32-byte sectors per warp load request | 4 is ideal for 32-bit loads (32 × 4 B = 128 B), 16 for 128-bit. v0 is 16.5 (scattered 32-bit), v1 is 2.5 (broadcast and coalesced). |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy | 47% for v4–v6, against 50% theoretical. |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (`_st`) | Extra shared-memory wavefronts due to conflicts | 33,554,432 loads (2 per instruction) for v4–v6; 2,097,152 stores for v5/v6. |
| `smsp__inst_executed_op_shared_ld.sum` | Warp-level shared-load instructions | Divide by warp-level FFMAs to get loads per FMA: 1.25 (v2), 0.375 (v3), 0.062 (v4–v6). |
| `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | Thread-level FFMAs executed | ÷ 32 = warp-level FFMAs, 268,435,456 for every kernel at N = 2048. Same math, as expected. |
| `smsp__inst_executed.sum` | All warp-level instructions executed | **FFMA share = FFMA ÷ all**: v0 about 22% (1,247 M instructions), v2 about 33%, v3 about 53%, v4 about 83%, v6 about 87% (308 M). |

The FFMA share is the clearest single picture of the project: we removed instruction overhead
until nearly 9 in 10 issued instructions are math. The remaining gap to cuBLAS is therefore
mostly *stalls* (conflicts, latency, a throttled clock), not wasted instructions.

---

## 12. Cheat sheet

### One sentence per kernel
- **v0 naive:** one thread per output, `threadIdx.x` → row, so A loads are uncoalesced
  (16.5 sectors per request). 62 GFLOP/s at N = 2048.
- **v1 coalesced:** `threadIdx.x` → column, so a warp reads contiguous B: 9.1x faster than v0
  at N = 2048.
- **v2 shared-memory tiling:** 32×32 tiles in shared memory with 2 barriers per tile. Global
  load instructions drop 32x, but DRAM traffic barely changes because v1 already reused data
  via L1. 1.23x vs v1.
- **v3 1D register tiling:** 8 outputs per thread in a column, so one B value feeds 8 FMAs.
  0.375 shared loads per FMA, 2.26x vs v2.
- **v4 2D register tiling:** an 8×8 outer product per thread, 128 registers, 50% occupancy,
  0.062 loads per FMA. 1.98x vs v3.
- **v5 vectorized:** `float4` loads and transposed A. A null result (1.00x), because the
  compiler had already vectorized v4 and global loads were off the critical path.
- **v6 double-buffered:** prefetches the next tile during the math with one barrier per tile.
  1.04x; about 60% of cuBLAS at N = 4096.
- **v7 conflict-free:** a fragment remap plus padding 128 → 132 cuts bank conflicts 2,290x
  (33.5M → 14.7K) and gives a 1.39x speedup: **81.9% of cuBLAS at N = 4096** (3,177 GFLOP/s).
- **v8 warp tiling:** block → warp (64×32) → thread tile; 96 instead of 144 distinct shared
  floats per warp per k. 81.4% of cuBLAS (0.99x vs v7): no gain once the conflicts were gone.
- **v9 autotuned:** dispatches each size to its best measured config from the tuning study
  (table generated by `scripts/gen_dispatch.py`). Re-measured independently: pending Colab run 7.
- **v10 tuning knobs:** v7's code with `__launch_bounds__` MINB and a threadblock swizzle as
  template parameters. MINB 4 on the 64×128 tile gives 128 registers, 4 blocks/SM and 0 spills:
  90.4% of cuBLAS at N = 4096 (v7 at the same shape: 88.4%). The swizzle is about neutral.
  Round 2: pending tune3.
- **tc0 WMMA naive:** one warp per 16×16 tile on the tensor cores, FP16 in, FP32 accumulate,
  fragments loaded straight from global memory. Pending Colab run 7.
- **tc1 WMMA shared memory:** 128×128 block tile staged in padded shared memory, 8 warps of 64×32
  (4×2 fragments each). Pending Colab run 7.

### Five numbers to remember (`results/`, run of 2026-10-12 15:39 UTC unless noted)
1. **3,177 GFLOP/s = 81.9% of cuBLAS** at N = 4096 (v7); 82.1% at N = 2048. That's **58.1x**
   the naive kernel at N = 2048.
2. **62 of 62** kernel/size results pass the cuBLAS check (max relative error 2.2e-6).
3. Bank conflicts **33,554,432 → 14,653** (v6 → v7, 2,290x), predicted to the exact count for
   v5/v6 by `scripts/bank_model.py`.
4. Tuning study: **88.6% of cuBLAS** at N = 4096 (`v7_64x128x8_8x8`), and **126.5%** at N = 256
   (small tiles beat cuBLAS). These are best-of-18 numbers.
5. DRAM traffic **44.9x → 3.7x** the minimum (v0 → v7); measured bandwidth **235 GB/s**; FFMA
   share of instructions **22% → 87%**.

---

## 13. Warp tiling (v8)

### The concept
A GPU schedules **warps**, not threads, and a warp's shared-memory request is served as one
unit. So the useful question isn't "how many values does a thread load per k?" but "how many
*distinct* values does a warp ask for?". Lanes asking for the same value get it broadcast for
free.

High-performance SGEMM (CUTLASS, cuBLAS) uses three levels of tiling:

```
block tile 128x128   (shared memory; one per thread block)
  └─ warp tile 64x32 (8 warps arranged 2 x 4)
       └─ thread tile: 2 x 2 pieces of 4x4 = 64 outputs per thread
```

### In our code
[`src/kernels/v8_warptiling.cu`](../src/kernels/v8_warptiling.cu). Each thread computes its
position in two steps: `warp = tid / 32` picks the warp tile, and `lane = tid % 32` picks the
spot inside it (`trw = lane / 4`, `tcw = lane % 4`). The global→shared copy is identical to v7;
only the compute mapping changes.

### The arithmetic (checked by CPU emulation)
Distinct floats one warp requests from shared memory per k step:

| | A floats | B floats | total |
|---|---|---|---|
| v7 (16×2 thread grid, pieces 64 apart) | 16 | 128 | **144** |
| v8 (8×4 lanes inside a 64×32 warp tile) | 64 | 32 | **96** |

Same FMAs, a third less shared-memory traffic. v7 is lopsided (16 vs 128) because a warp spans
just 2 thread-rows but all 16 thread-columns. A square-ish warp tile balances the two.

### What we measured
v8 reaches **81.4% of cuBLAS at N = 4096 (0.99x vs v7)**: no gain. Shared loads per FMA
(0.062), FFMA share (87%) and occupancy (50%) are identical to v7. The explanation that fits
the data: once v7 removed the bank conflicts, shared-memory bandwidth stopped being the
bottleneck. Asking for fewer distinct floats per warp then relieves a resource nobody was
waiting on. This is another null result worth telling. On tensor-core GPUs warp tiling matters
much more, because the warp tile is exactly what feeds the `mma` instructions (section 15).

### Interview questions
- **Why tile at the warp level if each thread's loads are unchanged?** Shared-memory bandwidth
  is consumed per warp request, by distinct addresses. A compact warp tile turns more of each
  request into broadcasts, so fewer bytes cross the shared-memory crossbar per FMA.
- **How does this map to CUTLASS?** CUTLASS names the same three levels "threadblock tile",
  "warp tile" and "thread (instruction) tile". On tensor-core GPUs the warp tile is exactly what
  a warp feeds to `mma` instructions.
- **What constrains the warp tile shape?** It must split into exactly 32 lane positions
  (`(WSUBM/4) * (WSUBN/4) == 32`), the block tile must be a whole number of warp tiles, and the
  accumulators (64 per thread here) must fit the register budget.

---

## 14. Predicting bank conflicts analytically

### The idea
Bank conflicts are deterministic: they depend only on which addresses each lane touches in one
instruction. So they can be computed without a GPU. Replay the kernel's index arithmetic for
one warp, group addresses into phases, and count the extra cycles.

### The rules we model
- 32 banks, each 4 bytes wide. Bank = (word address) mod 32.
- A 32-bit access is served for all 32 lanes together; a 128-bit access in **4 phases of 8
  lanes**.
- In one phase, lanes reading the *same* word get a broadcast (free). Two *different* words in
  the same bank cost one extra cycle each.

### Result
[`scripts/bank_model.py`](../scripts/bank_model.py) predicts **33,554,432** load conflicts and
**2,097,152** store conflicts for v5 and v6 at N = 2048. Those are exactly the Nsight Compute
counts. The scaling: 64 extra cycles per warp per tile × 8 warps × 256 tiles × 256 blocks. For
v7 and v8 the model predicts 0. ncu measured 14,653 (v7) and 27,469 (v8) load conflicts:
0.04% and 0.08% of v6's count. The model explains 99.9%+ of the effect, and the residual is
reported as unexplained, not rounded away.

### Interview questions
- **How would you check a shared-memory layout before running it?** Write down each lane's
  addresses for one instruction, compute `address mod 32` per phase, and look for two different
  addresses in one bank. That's exactly what the script does.
- **Why is padding 128 → 132 enough for the transposed A store?** Two threads that collided
  wrote rows 4 apart in k, i.e. 4×128 = 512 words apart, and 512 mod 32 = 0 means the same bank.
  With a stride of 132 they're 528 apart, and 528 mod 32 = 16: a different bank. 132 is still a
  multiple of 4, so float4 reads stay 16-byte aligned.
- **Why don't broadcasts conflict?** The bank delivers one word per cycle to every lane that
  asked for that word, so many lanes reading one address cost one access.

---

## 15. Tensor cores and WMMA

### The concept
Tensor cores are separate matrix units inside each SM. On a T4 (Turing) each SM has **8 tensor
cores**. Each computes a **4×4×4 FP16 matrix multiply-accumulate per clock**, i.e. 64 FMAs. All
64 FP32 CUDA cores of an SM together also do 64 FMAs per clock, so the tensor cores give **8x
the FMA rate** of the regular cores, but only for small matrix tiles with FP16 (or other
low-precision) inputs.

You don't program them per thread. A whole warp cooperates through the **WMMA** API
(`#include <mma.h>`, namespace `nvcuda::wmma`):

```
wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a;   // a 16x16 tile of A
wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> b;
wmma::fragment<wmma::accumulator, 16,16,16, float> acc;
wmma::fill_fragment(acc, 0.f);
wmma::load_matrix_sync(a, ptrA, ldA);   // the warp loads a 16x16 tile, stride ldA
wmma::load_matrix_sync(b, ptrB, ldB);
wmma::mma_sync(acc, a, b, acc);         // acc += a * b on the tensor cores (4096 FMAs)
wmma::store_matrix_sync(ptrC, acc, ldC, wmma::mem_row_major);
```

A **fragment** is a tile distributed across the 32 lanes' registers in an *unspecified* layout.
You never index it per element; you only load, multiply and store it as a whole.
`row_major` / `col_major` tells the load how the tile sits in memory. Our A and B are
row-major, so both use `row_major` fragments.

**Rules that bite:**
- `ldm` (the row stride) must be a multiple of 8 halves (16 bytes).
- The pointer passed to `load/store_matrix_sync` must be **32-byte (256-bit) aligned**.
- Matrix dimensions must be multiples of 16, or you handle the edges yourself. Our tc kernels
  run only sizes that are multiples of 128.
- `_sync` means every lane of the warp must call it together. Never put it behind a branch
  that splits a warp.

### Why FP16 inputs with FP32 accumulation
FP16 has a 10-bit mantissa, so each *input* is rounded to about 3 significant digits. But the
product of two FP16 numbers is exact in FP32, and summing thousands of them in FP32 avoids the
catastrophic error FP16 accumulation would cause. This "mixed precision" is how deep-learning
training uses tensor cores.

### Why the reference is `cublasGemmEx` with the same FP16 inputs
Correctness checks must compare like with like. Our tc kernels see FP16-rounded A and B, so
comparing them to FP32 SGEMM would flag the *input rounding* as a kernel bug. The reference is
therefore `cublasGemmEx(..., CUDA_R_16F inputs, CUDA_R_32F output, CUBLAS_COMPUTE_32F)` on the
very same FP16 buffers, and "% of cuBLAS" means % of that call. The precision cost of FP16
inputs (FP16-input result vs FP32 SGEMM on the same data) is measured *separately* and reported
as a property of the format, not of our kernels (`results/tc_precision_*.csv`).

### tc0 vs tc1 (`src/kernels/tc0_wmma_naive.cu`, `tc1_wmma_smem.cu`)
- **tc0:** each warp owns one 16×16 tile of C and loads its A and B fragments **straight from
  global memory** at every k step. That's the tensor-core equivalent of v1: no reuse, so the
  tensor cores mostly wait for data.
- **tc1:** a 128×128 block tile, 8 warps in a 2×4 grid, each warp owning 64×32 = 4×2 fragments
  (8 accumulators). Per 32-deep k step, the block copies the A and B tiles into **shared
  memory** with 16-byte vector loads. Each A fragment then feeds 2 MMAs and each B fragment 4.
  Rows are **padded by 8 halves** (16 bytes = 4 banks), so the 16 rows a fragment load touches
  don't land on the same banks. The padded strides (40, 136) keep the 32-byte alignment WMMA
  needs.

### What CUTLASS adds on top
- **`ldmatrix`:** a dedicated instruction that loads fragments from shared memory in the exact
  register layout the MMA wants, bank-conflict-free with a swizzled layout.
- **Swizzled shared-memory layouts:** XOR-permuted addresses instead of padding, so no memory is
  wasted.
- **Multistage pipelines:** on Ampere and later, `cp.async` copies global → shared without
  passing through registers, with 3–5 stages in flight. That's double buffering generalised.
  Turing (our T4) has no `cp.async`.
- **Raw `mma.sync` PTX** instead of WMMA, for full control of the fragment layout.

### Results
**Pending Colab run 7** (`bench --tc`): TFLOP/s and % of `cublasGemmEx` for tc0 and tc1, the
FP16 precision cost, and tensor-pipe utilisation from ncu.

### Interview questions
- **Why can't you just compare the tensor-core kernel with SGEMM?** Different inputs: FP16
  rounding changes the answer by roughly 1e-3 relative, which would look like a bug. Compare
  with `cublasGemmEx` on the same FP16 inputs, and report the precision cost separately.
- **What is a WMMA fragment?** A warp-wide, opaque, register-resident 16×16 tile. The layout is
  implementation-defined, so you only load, mma and store it as a unit.
- **Why does tc0 underperform?** No reuse: every warp reloads its fragments from global memory
  every k step, so the tensor cores are starved, exactly like v1 for FP32.
- **Why pad the shared tiles by 8 halves?** The fragment load reads 16 rows at the tile's
  stride. With a 64- or 256-byte stride many rows map to the same banks; 16 bytes of padding
  shifts each row by 4 banks while keeping 32-byte alignment and an `ldm` that's a multiple of 8.
- **What would you do next on an A100?** `cp.async` multistage pipelines, `ldmatrix` with
  swizzled layouts, larger warp tiles, and TF32 or BF16 tensor cores for FP32-like range.

---

## 16. Autotuned dispatch and the winner's curse (v9)

### How the table is built
1. `./bench --tune` measures 18 configurations of the v7/v8 templates at every size, each
   verified against cuBLAS and timed with paired cuBLAS runs.
2. [`scripts/gen_dispatch.py`](../scripts/gen_dispatch.py) reads the newest tuning CSV.
   - It drops any configuration that failed the check at *any* size.
   - For each tuned size it picks the config with the highest % of paired cuBLAS.
   - It writes `src/kernels/v9_dispatch_table.inc`: lines like `{4096, "v7_64x128x8_8x8"}`,
     with a comment naming the CSV.
3. [`src/kernels/v9_autotuned.cu`](../src/kernels/v9_autotuned.cu) `#include`s the table. At
   run time it computes s = cbrt(M·N·K), the side of the equivalent cube, and picks the entry
   whose size is closest to s *in log scale*. N = 1023 maps to the 1023 entry, N = 3000 to 4096.

No device code is new. v9 is a policy, the same idea as cuBLAS's internal heuristics and
`cublasLt`'s algorithm selection (`cublasLtMatmulAlgoGetHeuristic`), which choose a kernel per
problem shape. The difference is that our table comes from our own measurements on this GPU.

### The winner's curse
Each config's measured speed = true speed + noise. Pick the maximum of 18 noisy values and you
preferentially pick configs whose noise happened to be positive. So **the winner's measured
number overstates its true speed**, and the more configs you try, the bigger the bias. The fix
is to re-measure the chosen configuration in an independent run and report only that number.
That's why bench measures v9 again and the README quotes v9 only from that measurement.

### Results
**Pending Colab run 7.** The tuning run itself showed best-per-size values of 126.5% (N = 256)
to 88.6% (N = 4096) of cuBLAS; expect v9's independent numbers to come in somewhat lower.

### Interview questions
- **Why not just use the best config everywhere?** No single tile wins at every size. Small N
  needs small tiles to fill 40 SMs (wave quantization), and large N needs reuse and enough
  resident blocks.
- **What is the winner's curse here?** Selection bias: the max of noisy measurements is
  optimistic, so you re-measure the winner independently.
- **How does cuBLAS do this?** Internal heuristics pick a kernel per shape and GPU; cublasLt
  exposes the candidates and their heuristic ranking, and you can benchmark them yourself.
- **How would you handle non-square or new shapes?** Tune on a grid of (M, N, K) buckets, or fall
  back to a cost model (waves × per-tile time) for shapes outside the grid.

---

## 17. Occupancy by launch bounds, and threadblock swizzling (v10)

Source: [`src/kernels/v10_tuned.cu`](../src/kernels/v10_tuned.cu). Data: round 1 of the v10 study,
`results/tune_sgemm_Tesla_T4_2026-10-12_161520.csv` and `results/tune_kernels_…_161520.csv`.
Round 2 is pending tune3.

v10 is v7's device code, generated from it so the inner loop is identical, with two extra
template parameters. It exists only for the tuning study; v9 can dispatch to its configs.

### Knob 1: `__launch_bounds__(THREADS, MINB)`
**The concept.** The compiler picks how many registers each thread gets. More registers mean
fewer spills, but fewer warps fit on an SM: an SM has 65,536 registers, shared by all resident
threads. `__launch_bounds__(THREADS, MINB)` tells the compiler "this kernel is launched with
`THREADS` threads per block, and I want at least `MINB` blocks resident per SM". On sm_75 that
caps registers per thread at

```
65536 / (MINB × THREADS)      e.g. 65536 / (4 × 128) = 128
```

(the hardware also rounds allocations, so the real cap can be a little lower). If the kernel
needs more registers than the cap, the compiler **spills**: it keeps some values in local
memory, which is slow, off-chip memory cached in L1/L2. A spill shows up in two places:
- at compile time, `nvcc -Xptxas -v` prints "bytes spill stores/loads" (our Makefile passes it);
- at run time, `cudaFuncGetAttributes(...).localSizeBytes`, which bench writes as
  `local_bytes_per_thread` in `results/*kernels_*.csv`.

**What our data shows** (64×128 block tile, 8×8 per thread, 128 threads per block):

| config | MINB | registers/thread | spill bytes | blocks/SM | warps/SM | % of cuBLAS at N = 4096 |
|---|---|---|---|---|---|---|
| `v7_64x128x8_8x8` | 2 (v7 default) | 137 | 0 | 3 | 12 | 88.4% |
| `v10_64x128x8_8x8_b2_s0` | 2 | 137 | 0 | 3 | 12 | 89.0% |
| `v10_64x128x8_8x8_b4_s0` | 4 | 128 | 0 | 4 | 16 | **90.4%** |

Asking for 4 blocks pushed the compiler from 137 to 128 registers **without any spills**, so a
fourth block fits on each SM. The same config is also the best at N = 2048 (89.3%).
- All v10 configs have 0 spill bytes.
- Of the 33 round-1 configs, only `v8_128x128x16_w64x32_2x2` spills (48 bytes per thread, at
  128 registers). It's also one of the weakest v8 shapes, a live example of the trade-off.

**Why more resident warps help even though each thread already has lots of ILP.** One thread's
64 accumulators give it plenty of independent FMAs. But there are also stretches where every
warp of a block waits together:
- at `__syncthreads()`;
- for a shared load (`LDS.128`) to return;
- for the next tile's global loads.

With 3 blocks per SM, the scheduler has 12 warps to pick from; with 4, it has 16. The extra
block keeps the FMA pipes busy while another block sits at a barrier. ILP hides latency
*within* a warp; occupancy hides it *across* warps. You need enough of both.

### Knob 2: threadblock swizzle (`SWZ`)
**The concept.** Blocks are launched roughly in order of their linear index
`blockIdx.y * gridDim.x + blockIdx.x`, i.e. row by row. So the blocks resident at the same time
cover a few block-rows but **every** block-column, and therefore every column-panel of B. A
swizzle remaps the block index so that resident blocks cover a compact patch instead. They then
share both A row-panels and B column-panels, and those panels stay in L2.

**The exact remapping in v10** (groups of `SWZ` block-columns, all rows of a group first):

```
lin   = blockIdx.y * gridDim.x + blockIdx.x
group = lin / (SWZ * gridDim.y)
first = group * SWZ                          // first block-column of the group
cols  = min(gridDim.x - first, SWZ)          // last group may be narrower
in    = lin - group * SWZ * gridDim.y
by    = in / cols
bx    = first + in % cols
```

It is a bijection: every block is still computed exactly once. This was checked on ragged grids
such as 5×7 and 33×9 before running on the GPU.

**What our data shows: about neutral.** With the swizzle on, the same 64×128 config gets
- 89.8% at MINB 2, against 89.0% without;
- 87.5% at MINB 4, against 90.4% without;
- 78.4% for 128×128 at MINB 2, against 82.5% for v7's unswizzled 128×128.

Differences of a few points in both directions are within run-to-run noise. **This is an
explained null result.** The roofline already showed v7-class kernels sit to the *right* of the
ridge, so DRAM bandwidth isn't their limit and better L2 reuse has little to buy. A swizzle
pays off when a kernel is close to the memory roof: larger matrices relative to L2, lower
arithmetic intensity, or GPUs with more SMs (more resident blocks) and relatively less L2 per
SM.

### How to read a tuning sweep
- **Change one knob at a time** against a fixed baseline (here `v10_…_b2_s0` = v7's code and
  shape), so a difference can be attributed.
- **Trust wins that are consistent across sizes and neighbouring configs**, not a single cell.
  MINB 4 helps at both N = 2048 and N = 4096; the swizzle's effect changes sign between
  configurations, which is what noise looks like.
- **Small N is noisy.** At N = 256 a kernel runs for tens of microseconds, so launch overhead and
  timer resolution matter, and the "best" config there is the least trustworthy.
- **Winner's curse.** The best of 33 noisy measurements is biased upward. Re-measure the winner
  on its own (v9 does) before quoting it.
- **Correctness first.** All 231 of 231 configuration/size results passed the cuBLAS check. A
  tuning sweep that let a fast-but-wrong config win would be worse than useless. Correctness is
  checked for every point, so the tuner can never pick such a config.

### Interview questions
- **Why can fewer registers per thread make a kernel faster?** If the reduction lets more blocks
  fit on an SM without spilling (here 137 → 128 registers, 3 → 4 blocks), the scheduler has more
  warps to hide barrier and memory latency. It backfires once the compiler starts spilling to
  local memory.
- **How do you know a config spilled?** `nvcc -Xptxas -v` reports spill stores/loads per kernel,
  and `cudaFuncGetAttributes().localSizeBytes > 0` at run time. We record the latter as
  `local_bytes_per_thread`. In round 1 it's 0 for every v10 config and 48 bytes for
  `v8_128x128x16_w64x32_2x2`. Nsight Compute's local-memory traffic counters would show the
  dynamic cost; we didn't collect them.
- **When would a threadblock swizzle help?** When the kernel is memory-bound or close to it and
  the working set of concurrently resident blocks overflows L2. Then grouping blocks so they
  share A and B panels raises the L2 hit rate. Here the kernels are past the ridge, so it was
  neutral.
- **What does `__launch_bounds__` not do?** It doesn't change the launch configuration or
  guarantee residency. It's a contract that lets the compiler budget registers; shared memory
  and threads per SM can still limit blocks per SM.
- **Occupancy vs ILP: which matters?** Both. ILP keeps one warp's pipeline full between its own
  dependent instructions; occupancy covers latencies that stall a whole warp or block (barriers,
  memory). Our best config raised occupancy from 12 to 16 warps per SM while keeping 64
  independent accumulators per thread.

---

## Further learning
- Simon Boehm, *How to Optimize a CUDA Matmul Kernel*: https://siboehm.com/articles/22/CUDA-MMM
- NVIDIA, *An Even Easier Introduction to CUDA*: https://developer.nvidia.com/blog/even-easier-introduction-cuda/
- CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Nsight Compute Profiling Guide (metric definitions): https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- GPU MODE lectures (index with video links): https://github.com/gpu-mode/lectures
