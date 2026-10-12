"""Index-map tests: every kernel's tile/thread/fragment arithmetic, replayed on the CPU."""
import numpy as np
import pytest

import emulation as em
from bank_model import predict

RNG = np.random.default_rng(0)


def _check(C, writes, A, B):
    assert not np.isnan(C).any(), "an output element was never written"
    assert (writes == 1).all(), "an output element was written more than once"
    assert np.abs(C - A @ B).max() < 1e-10


# v7 / v10 / v14 thread-strip maps: (BM, BN, BK, TM, TN)
V7_CFGS = [(128, 128, 8, 8, 8), (64, 128, 8, 8, 8), (128, 64, 8, 8, 8), (64, 64, 8, 4, 4),
           (32, 32, 8, 4, 4), (64, 128, 4, 8, 8), (96, 128, 8, 8, 8)]


@pytest.mark.parametrize("cfg", V7_CFGS)
def test_v7_index_map(cfg):
    BM, BN, BK, _, _ = cfg
    M, N, K = BM + 5, BN - 3, 2 * BK + 3  # awkward: partial tiles in every dimension
    A, B = RNG.uniform(-1, 1, (M, K)), RNG.uniform(-1, 1, (K, N))
    _check(*em.v7_gemm(A, B, cfg), A, B)


@pytest.mark.parametrize("cfg", [c for c in V7_CFGS if c[2] == 8])
def test_v7_layout_is_conflict_free(cfg):
    assert em.v7_read_conflicts(cfg) == 0


# v8 / v13 warp-tile maps: (BM, BN, BK, WM, WN, WMITER, WNITER)
V8_CFGS = [(128, 128, 8, 64, 32, 2, 2), (128, 128, 8, 32, 64, 2, 2), (64, 128, 8, 32, 64, 2, 2),
           (64, 128, 8, 64, 32, 2, 2), (64, 128, 8, 32, 64, 1, 4), (128, 64, 8, 64, 32, 2, 2),
           (64, 64, 8, 32, 32, 1, 2)]


@pytest.mark.parametrize("cfg", V8_CFGS)
def test_v8_index_map(cfg):
    BM, BN, BK = cfg[:3]
    M, N, K = BM - 1, BN + 3, BK + 5
    A, B = RNG.uniform(-1, 1, (M, K)), RNG.uniform(-1, 1, (K, N))
    _check(*em.v8_gemm(A, B, cfg), A, B)


@pytest.mark.parametrize("cfg", V8_CFGS)
def test_v8_layout_is_conflict_free(cfg):
    assert em.v8_read_conflicts(cfg) == 0


def test_v6_layout_has_the_measured_conflicts():
    # 64 extra cycles per warp per tile; x 8 warps x 256 tiles x 256 blocks = ncu's 33,554,432
    assert em.v6_b_read_conflicts() == 64


def test_bank_model_matches_nsight_compute():
    # Nsight Compute, N = 2048: v5/v6 measured 33,554,432 load and 2,097,152 store conflicts.
    for k in ("v5_vectorized", "v6_double_buffered"):
        assert predict(k, 2048) == (33_554_432, 2_097_152)
    for k in ("v7_conflict_free", "v8_warptiling"):
        assert predict(k, 2048) == (0, 0)


@pytest.mark.parametrize("bk", [16, 32])  # tc3 (16-deep K tile) and tc4 (32-deep)
def test_tc_ptx_fragments(bk):
    K = 2 * bk
    A, B = RNG.uniform(-1, 1, (128, K)), RNG.uniform(-1, 1, (K, 128))
    C, conflicts = em.tc_ptx_block(A, B, bk)
    assert not np.isnan(C).any()
    assert np.abs(C - A @ B).max() < 1e-10
    assert conflicts == 0, "ldmatrix row addresses must hit 8 distinct 16-byte bank groups"


SPLITK_SHAPES = [(128, 4096, 4096), (4096, 4096, 128), (1023, 1023, 1023), (1024, 1024, 1024),
                 (256, 256, 256), (512, 4096, 4096), (100, 77, 3001), (1, 1, 257), (64, 128, 2049)]


@pytest.mark.parametrize("shape", SPLITK_SHAPES)
def test_splitk_slices_cover_k_once(shape):
    M, N, K = shape
    BK = 8
    kchunk, s = em.splitk_slices(M, N, K, BK)
    assert kchunk % BK == 0
    covered = np.zeros(K, int)
    for z in range(s):
        kb, ke = z * kchunk, min(K, (z + 1) * kchunk)
        assert kb < K, "no empty slice"
        for t in range(kb, ke, BK):  # the kernel's tile loop: tiles must not straddle a slice
            hi = min(t + BK, K)
            assert hi <= ke or ke == K
            covered[t:hi] += 1
    assert (covered == 1).all()


@pytest.mark.parametrize("grid", [(32, 64), (64, 32), (8, 8), (5, 7), (33, 9), (1, 4), (9, 1)])
@pytest.mark.parametrize("S", [4, 8, 16])
def test_swizzle_is_a_bijection(grid, S):
    gx, gy = grid
    seen = {em.swizzle(x, y, gx, gy, S) for y in range(gy) for x in range(gx)}
    assert seen == {(x, y) for y in range(gy) for x in range(gx)}
