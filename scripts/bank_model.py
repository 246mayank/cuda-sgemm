"""Analytical shared-memory bank-conflict model for the explicitly vectorized kernels (v5-v8).

For one warp and one BK-deep tile, it lists the shared-memory word addresses every lane touches
in every shared load/store instruction of the kernel's inner loop and tile copy (the same index
arithmetic as the .cu files), then applies the hardware rule:

  * 32-bit accesses are served for all 32 lanes at once; 128-bit accesses in 4 phases of 8
    lanes (each phase moves at most 128 bytes);
  * within one phase, lanes reading the same word are a broadcast (free); two different words
    in the same bank (word address mod 32) cost one extra cycle each.

Extra cycles per warp per tile, times warps x tiles x blocks, gives the predicted total at size
N, written to results/model_bank_conflicts.csv. make_tables.py puts it next to Nsight Compute's
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum, so the model can be checked
against the profiler. It is a model of our access patterns, not a measurement; v2-v4 are left
out because the compiler, not the source, chose their vector widths (see the SASS table).

Usage: .venv/Scripts/python scripts/bank_model.py [--n 2048]
"""
import argparse
import csv

from results_io import RESULTS


def phase_cycles(lane_words, lanes_per_phase):
    """Extra cycles for one warp-wide shared access. lane_words[l] = words lane l touches."""
    extra = 0
    for p in range(0, 32, lanes_per_phase):
        banks = {}
        for lane in range(p, p + lanes_per_phase):
            for w in lane_words[lane]:
                banks.setdefault(w % 32, set()).add(w)
        extra += max((len(v) for v in banks.values()), default=1) - 1
    return extra


def vec4(base):
    return [base + q for q in range(4)]


# ---------------------------------------------------------------------------- access patterns
# Each returns (loads, stores): lists of (lanes_per_phase, fn(lane_tid) -> words) for one warp
# and one tile. tid is the thread index inside the block.

def v5_v6(BK=8, BM=128, BN=128, TM=8, TN=8):
    TC = BN // TN
    loads, stores = [], []
    for k in range(BK):
        for half in (0, 4):
            loads.append((8, lambda t, k=k, h=half: vec4(k * BM + (t // TC) * TM + h)))      # A
            loads.append((8, lambda t, k=k, h=half: vec4(k * BN + (t % TC) * TN + h)))      # B
    for q in range(4):  # transposed A: 4 scalar stores
        stores.append((32, lambda t, q=q: [((t % (BK // 4)) * 4 + q) * BM + t // (BK // 4)]))
    stores.append((8, lambda t: vec4((t // (BN // 4)) * BN + (t % (BN // 4)) * 4)))       # B
    return loads, stores


def v7(BK=8, BM=128, BN=128, TM=8, TN=8):
    TR, TC, BMP = BM // TM, BN // TN, BM + 4
    loads, stores = [], []
    for k in range(BK):
        for s in range(TM // 4):
            loads.append((8, lambda t, k=k, s=s: vec4(k * BMP + s * 4 * TR + (t // TC) * 4)))
        for s in range(TN // 4):
            loads.append((8, lambda t, k=k, s=s: vec4(k * BN + s * 4 * TC + (t % TC) * 4)))
    for q in range(4):
        stores.append((32, lambda t, q=q: [((t % (BK // 4)) * 4 + q) * BMP + t // (BK // 4)]))
    stores.append((8, lambda t: vec4((t // (BN // 4)) * BN + (t % (BN // 4)) * 4)))
    return loads, stores


def v8(BK=8, BM=128, BN=128, WM=64, WN=32, MI=2, NI=2):
    WARPS_N, BMP = BN // WN, BM + 4
    WSUBM, WSUBN = WM // MI, WN // NI
    LN = WSUBN // 4

    def parts(t):
        warp, lane = t // 32, t % 32
        return (warp // WARPS_N) * WM, (warp % WARPS_N) * WN, lane // LN, lane % LN

    loads, stores = [], []
    for k in range(BK):
        for wm in range(MI):
            loads.append((8, lambda t, k=k, wm=wm: vec4(k * BMP + parts(t)[0] + wm * WSUBM + parts(t)[2] * 4)))
        for wn in range(NI):
            loads.append((8, lambda t, k=k, wn=wn: vec4(k * BN + parts(t)[1] + wn * WSUBN + parts(t)[3] * 4)))
    for q in range(4):
        stores.append((32, lambda t, q=q: [((t % (BK // 4)) * 4 + q) * BMP + t // (BK // 4)]))
    stores.append((8, lambda t: vec4((t // (BN // 4)) * BN + (t % (BN // 4)) * 4)))
    return loads, stores


KERNELS = {  # name: (pattern, threads per block, BM, BN, BK)
    "v5_vectorized": (v5_v6, 256, 128, 128, 8),
    "v6_double_buffered": (v5_v6, 256, 128, 128, 8),
    "v7_conflict_free": (v7, 256, 128, 128, 8),
    "v8_warptiling": (v8, 256, 128, 128, 8),
}


def predict(name, n):
    pattern, threads, BM, BN, BK = KERNELS[name]
    loads, stores = pattern()
    ld = st = 0
    for w0 in range(0, threads, 32):
        lanes = range(w0, w0 + 32)
        ld += sum(phase_cycles([f(t) for t in lanes], lpp) for lpp, f in loads)
        st += sum(phase_cycles([f(t) for t in lanes], lpp) for lpp, f in stores)
    blocks = (n // BM) * (n // BN)
    tiles = n // BK
    return ld * tiles * blocks, st * tiles * blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=2048)
    n = ap.parse_args().n
    out = RESULTS / "model_bank_conflicts.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "N", "predicted_ld_conflicts", "predicted_st_conflicts"])
        for name in KERNELS:
            ld, st = predict(name, n)
            w.writerow([name, n, ld, st])
            print(f"{name:20s} N={n}: load conflicts {ld:,}  store conflicts {st:,}")
    print("wrote", out.relative_to(RESULTS.parent))


if __name__ == "__main__":
    main()
