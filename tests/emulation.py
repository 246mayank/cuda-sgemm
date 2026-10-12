"""CPU emulation of the kernels' index arithmetic (no GPU needed).

Each function replays, for every block and thread, exactly the index maps used in the .cu
files, with shared memory as a numpy array, and returns the C it would write plus a count of
how often each output was written. These are the checks that were run before every GPU run
during development; the tests make them permanent.
"""
import numpy as np


def _ld(X, r, c):
    R, Cc = X.shape
    return X[r, c] if r < R and c < Cc else 0.0


def phase_conflicts(word_addrs_per_lane, lanes_per_phase):
    """Extra cycles for one warp-wide shared access: lanes are served in phases of
    `lanes_per_phase`; two different 4-byte words in one bank cost one extra cycle each."""
    extra = 0
    for p in range(0, 32, lanes_per_phase):
        banks = {}
        for lane in range(p, p + lanes_per_phase):
            for w in word_addrs_per_lane[lane]:
                banks.setdefault(w % 32, set()).add(w)
        extra += max((len(v) for v in banks.values()), default=1) - 1
    return extra


# ------------------------------------------------------------------ v7 / v10 (thread strips)

def v7_gemm(A, B, cfg):
    """v7_conflict_free.cu (and v10/v14, same maps): thread (tr, tc) owns TM/4 x TN/4 strips of
    4x4 outputs, 4*TR rows / 4*TC columns apart; A tile transposed with row stride BM + 4."""
    BM, BN, BK, TM, TN = cfg
    M, K = A.shape
    N = B.shape[1]
    TR, TC = BM // TM, BN // TN
    T, BMP = TR * TC, BM + 4
    C = np.full((M, N), np.nan)
    writes = np.zeros((M, N), int)
    for by in range(-(-M // BM)):
        for bx in range(-(-N // BN)):
            r0, c0 = by * BM, bx * BN
            acc = np.zeros((T, TM, TN))
            for t in range(0, K, BK):
                As = np.zeros(BK * BMP)
                Bs = np.zeros(BK * BN)
                for c in range(BM * BK // 4):
                    r, kc = c // (BK // 4), (c % (BK // 4)) * 4
                    for q in range(4):
                        As[(kc + q) * BMP + r] = _ld(A, r0 + r, t + kc + q)
                for c in range(BK * BN // 4):
                    r, cc = c // (BN // 4), (c % (BN // 4)) * 4
                    for q in range(4):
                        Bs[r * BN + cc + q] = _ld(B, t + r, c0 + cc + q)
                for tid in range(T):
                    tr, tc = tid // TC, tid % TC
                    for k in range(BK):
                        a = [As[k * BMP + s * 4 * TR + tr * 4 + q] for s in range(TM // 4) for q in range(4)]
                        b = [Bs[k * BN + s * 4 * TC + tc * 4 + q] for s in range(TN // 4) for q in range(4)]
                        acc[tid] += np.outer(a, b)
            for tid in range(T):
                tr, tc = tid // TC, tid % TC
                for si in range(TM // 4):
                    for i in range(4):
                        r = r0 + si * 4 * TR + tr * 4 + i
                        for sj in range(TN // 4):
                            for j in range(4):
                                c = c0 + sj * 4 * TC + tc * 4 + j
                                if r < M and c < N:
                                    C[r, c] = acc[tid, si * 4 + i, sj * 4 + j]
                                    writes[r, c] += 1
    return C, writes


def v7_read_conflicts(cfg):
    """Extra shared-load cycles per BK tile summed over the warps of one block (v7 layout)."""
    BM, BN, BK, TM, TN = cfg
    TR, TC = BM // TM, BN // TN
    T, BMP = TR * TC, BM + 4
    total = 0
    for w0 in range(0, T, 32):
        lanes = range(w0, w0 + 32)
        for k in range(BK):
            for s in range(TM // 4):
                total += phase_conflicts([[k * BMP + s * 4 * TR + (l // TC) * 4 + q for q in range(4)] for l in lanes], 8)
            for s in range(TN // 4):
                total += phase_conflicts([[k * BN + s * 4 * TC + (l % TC) * 4 + q for q in range(4)] for l in lanes], 8)
    return total


def v6_b_read_conflicts():
    """v6's layout (thread owns 8 adjacent columns): extra B-read cycles per warp per 8-deep tile."""
    extra = 0
    for k in range(8):
        for half in (0, 4):
            extra += phase_conflicts([[k * 128 + (l % 16) * 8 + half + q for q in range(4)] for l in range(32)], 8)
    return extra


# ------------------------------------------------------------------ v8 / v13 (warp tiles)

def v8_gemm(A, B, cfg):
    """v8_warptiling.cu (and v13): block -> warp tile WMxWN -> MI x NI pieces of 4x4 per lane."""
    BM, BN, BK, WM, WN, MI, NI = cfg
    M, K = A.shape
    N = B.shape[1]
    WARPS_N = BN // WN
    T = (BM // WM) * WARPS_N * 32
    WSUBM, WSUBN = WM // MI, WN // NI
    LN = WSUBN // 4
    assert (WSUBM // 4) * LN == 32
    BMP = BM + 4

    def thread(tid):
        warp, lane = tid // 32, tid % 32
        return (warp // WARPS_N) * WM, (warp % WARPS_N) * WN, lane // LN, lane % LN

    C = np.full((M, N), np.nan)
    writes = np.zeros((M, N), int)
    for by in range(-(-M // BM)):
        for bx in range(-(-N // BN)):
            r0, c0 = by * BM, bx * BN
            acc = np.zeros((T, MI * 4, NI * 4))
            for t in range(0, K, BK):
                As = np.zeros(BK * BMP)
                Bs = np.zeros(BK * BN)
                for c in range(BM * BK // 4):
                    r, kc = c // (BK // 4), (c % (BK // 4)) * 4
                    for q in range(4):
                        As[(kc + q) * BMP + r] = _ld(A, r0 + r, t + kc + q)
                for c in range(BK * BN // 4):
                    r, cc = c // (BN // 4), (c % (BN // 4)) * 4
                    for q in range(4):
                        Bs[r * BN + cc + q] = _ld(B, t + r, c0 + cc + q)
                for tid in range(T):
                    wr, wc, trw, tcw = thread(tid)
                    for k in range(BK):
                        a = [As[k * BMP + wr + wm * WSUBM + trw * 4 + q] for wm in range(MI) for q in range(4)]
                        b = [Bs[k * BN + wc + wn * WSUBN + tcw * 4 + q] for wn in range(NI) for q in range(4)]
                        acc[tid] += np.outer(a, b)
            for tid in range(T):
                wr, wc, trw, tcw = thread(tid)
                for wm in range(MI):
                    for i in range(4):
                        r = r0 + wr + wm * WSUBM + trw * 4 + i
                        for wn in range(NI):
                            for j in range(4):
                                c = c0 + wc + wn * WSUBN + tcw * 4 + j
                                if r < M and c < N:
                                    C[r, c] = acc[tid, wm * 4 + i, wn * 4 + j]
                                    writes[r, c] += 1
    return C, writes


def v8_read_conflicts(cfg):
    BM, BN, BK, WM, WN, MI, NI = cfg
    WARPS_N = BN // WN
    T = (BM // WM) * WARPS_N * 32
    WSUBM, WSUBN = WM // MI, WN // NI
    LN = WSUBN // 4
    BMP = BM + 4

    def thread(tid):
        warp, lane = tid // 32, tid % 32
        return (warp // WARPS_N) * WM, (warp % WARPS_N) * WN, lane // LN, lane % LN

    total = 0
    for w0 in range(0, T, 32):
        lanes = [thread(t) for t in range(w0, w0 + 32)]
        for k in range(BK):
            for wm in range(MI):
                total += phase_conflicts([[k * BMP + wr + wm * WSUBM + trw * 4 + q for q in range(4)]
                                          for wr, wc, trw, tcw in lanes], 8)
            for wn in range(NI):
                total += phase_conflicts([[k * BN + wc + wn * WSUBN + tcw * 4 + q for q in range(4)]
                                          for wr, wc, trw, tcw in lanes], 8)
    return total


# ------------------------------------------------------------------ tc3 / tc4 (PTX fragments)

def ldmatrix_x4(smem, row_addr, trans):
    """PTX ldmatrix .m8n8 .x4 .b16: lanes 8i..8i+7 give matrix i's row addresses; lane T gets
    (row T/4, cols 2(T%4), +1) of each matrix, or with .trans (rows 2(T%4), +1; col T/4)."""
    regs = [[None] * 32 for _ in range(4)]
    for i in range(4):
        mat = np.array([smem[row_addr[8 * i + r]: row_addr[8 * i + r] + 8] for r in range(8)])
        for T in range(32):
            if not trans:
                regs[i][T] = (mat[T // 4, 2 * (T % 4)], mat[T // 4, 2 * (T % 4) + 1])
            else:
                regs[i][T] = (mat[2 * (T % 4), T // 4], mat[2 * (T % 4) + 1, T // 4])
    return regs


def mma_m16n8k8(c, a0, a1, b):
    """PTX mma.m16n8k8.row.col: rebuild A (16x8) and B (8x8) from the lanes' fragments."""
    A = np.zeros((16, 8))
    B = np.zeros((8, 8))
    for T in range(32):
        g, t = T // 4, T % 4
        A[g, 2 * t], A[g, 2 * t + 1] = a0[T]
        A[g + 8, 2 * t], A[g + 8, 2 * t + 1] = a1[T]
        B[2 * t, g], B[2 * t + 1, g] = b[T]
    D = A @ B
    for T in range(32):
        g, t = T // 4, T % 4
        c[T][0] += D[g, 2 * t]
        c[T][1] += D[g, 2 * t + 1]
        c[T][2] += D[g + 8, 2 * t]
        c[T][3] += D[g + 8, 2 * t + 1]


def tc_ptx_block(A, B, BK):
    """One 128x128 block of tc3 (BK = 16) or tc4 (BK = 32): 8 warps of 64x32, 4x4 MMA tiles each.
    Returns C (128x128) and the extra ldmatrix cycles summed over the block per K tile."""
    BM = BN = 128
    WARPS_N, WM, WN = 4, 64, 32
    FM, FN = WM // 16, WN // 8
    LDA, LDB = BK + 8, BN + 8
    K = A.shape[1]
    acc = {(w, i, j): [[0.0] * 4 for _ in range(32)] for w in range(8) for i in range(FM) for j in range(FN)}
    conflicts = 0
    for t in range(0, K, BK):
        As = np.zeros(BM * LDA)
        Bs = np.zeros(BK * LDB)
        for r in range(BM):
            As[r * LDA: r * LDA + BK] = A[r, t:t + BK]
        for k in range(BK):
            Bs[k * LDB: k * LDB + BN] = B[t + k, :]
        for w in range(8):
            wr, wc = (w // WARPS_N) * WM, (w % WARPS_N) * WN
            for kk in range(0, BK, 16):
                af, bf = [], []
                for i in range(FM):
                    addr = [(wr + i * 16 + ((l >> 3) & 1) * 8 + (l & 7)) * LDA + kk + (l >> 4) * 8 for l in range(32)]
                    conflicts += sum(phase_conflicts([[a // 2 + q for q in range(4)] for a in addr[p:p + 8]] + [[]] * 24, 32)
                                     for p in range(0, 32, 8))
                    af.append(ldmatrix_x4(As, addr, trans=False))
                for p in range(FN // 2):
                    addr = [(kk + ((l >> 3) & 1) * 8 + (l & 7)) * LDB + wc + p * 16 + (l >> 4) * 8 for l in range(32)]
                    conflicts += sum(phase_conflicts([[a // 2 + q for q in range(4)] for a in addr[s:s + 8]] + [[]] * 24, 32)
                                     for s in range(0, 32, 8))
                    bf.append(ldmatrix_x4(Bs, addr, trans=True))
                for ks in range(2):
                    for i in range(FM):
                        for j in range(FN):
                            mma_m16n8k8(acc[(w, i, j)], af[i][2 * ks], af[i][2 * ks + 1], bf[j // 2][(j % 2) * 2 + ks])
    C = np.full((BM, BN), np.nan)
    for w in range(8):
        wr, wc = (w // WARPS_N) * WM, (w % WARPS_N) * WN
        for i in range(FM):
            for j in range(FN):
                for T in range(32):
                    g, tg = T // 4, T % 4
                    r, c = wr + i * 16 + g, wc + j * 8 + 2 * tg
                    C[r, c], C[r, c + 1], C[r + 8, c], C[r + 8, c + 1] = acc[(w, i, j)][T]
    return C, conflicts


# ------------------------------------------------------------------ split-K and swizzle

def splitk_slices(M, N, K, BK=8, sms=40, minb=4, bm=64, bn=128):
    """v11/v15 launcher heuristic: returns (kchunk, slices)."""
    cd = lambda a, b: -(-a // b)
    blocks = cd(N, bn) * cd(M, bm)
    slots = sms * minb
    s = min(8, (2 * slots + blocks - 1) // blocks)
    s = max(1, min(s, K // 256))
    kchunk = cd(cd(K, s), BK) * BK
    return kchunk, cd(K, kchunk)


def swizzle(bxi, byi, gx, gy, S):
    """v10 threadblock swizzle: groups of S block-columns, all rows of a group first."""
    lin = byi * gx + bxi
    group = lin // (S * gy)
    first = group * S
    cols = min(gx - first, S)
    i = lin - group * S * gy
    return first + i % cols, i // cols
