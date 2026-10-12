"""Reporting tests: the README/ANALYSIS renderer must fill every token from the CSVs."""
import json

import pandas as pd
import pytest

import make_tables
import results_io


def test_committed_results_render_without_gaps():
    """Every {{token}} in both templates resolves from the committed results/ (no 'n/a')."""
    if not list(results_io.RESULTS.glob("sgemm_*.csv")):
        pytest.skip("no committed results")
    data = make_tables.Data(None)
    for template, _, _ in make_tables.OUTPUTS:
        _, missing = make_tables.render(data, template.read_text(encoding="utf-8"))
        assert missing == [], f"{template.name}: {missing}"


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """A tiny results/ directory: one run with cuBLAS and two kernels at two sizes."""
    rows = []
    for n in (256, 512):
        rows.append(dict(kernel="cublas", M=n, N=n, K=n, median_ms=1.0, min_ms=0.9, p90_ms=1.1,
                         gflops=1000.0, pct_cublas=100.0, max_rel_err=0.0, status="REF",
                         cublas_paired_median_ms=None, sm_clock_mhz_after=1500, power_w_after=60.0))
        for k, pct, st in (("v1_coalesced", 10.0, "PASS"), ("v7_conflict_free", 80.0, "PASS")):
            rows.append(dict(kernel=k, M=n, N=n, K=n, median_ms=10.0, min_ms=9.0, p90_ms=11.0,
                             gflops=pct * 10, pct_cublas=pct, max_rel_err=1e-6, status=st,
                             cublas_paired_median_ms=1.0, sm_clock_mhz_after=1400, power_w_after=65.0))
    stamp = "Tesla_T4_2099-01-01_000000"
    pd.DataFrame(rows).to_csv(tmp_path / f"sgemm_{stamp}.csv", index=False)
    env = {"date_utc": "2099-01-01T00:00:00Z", "gpu_name": "Tesla T4", "compute_capability": "7.5",
           "sm_count": 40, "sm_clock_max_khz": 1590000, "mem_clock_max_khz": 5001000,
           "mem_bus_width_bits": 256, "l2_cache_bytes": 4194304, "smem_per_sm_bytes": 65536,
           "cuda_runtime_version": 13000, "nvcc_version": "13.0.88", "compiled_arch": "sm_75",
           "cublas_version": 130101, "warmup_runs": 3, "timed_runs": 20,
           "tolerance_max_rel_err": 1e-4, "input_distribution": "uniform[-1,1]",
           "nvidia_smi_start": "name, driver_version, power.limit [W]\nTesla T4, 580, 70.00 W",
           "nvidia_smi_end": "name, driver_version, power.limit [W]\nTesla T4, 580, 70.00 W"}
    (tmp_path / f"env_{stamp}.json").write_text(json.dumps(env))
    monkeypatch.setattr(results_io, "RESULTS", tmp_path)
    monkeypatch.setattr(make_tables, "RESULTS", tmp_path)
    monkeypatch.setattr(make_tables, "newest", lambda pat: (sorted(tmp_path.glob(pat)) or [None])[-1])
    return make_tables.Data(tmp_path / f"sgemm_{stamp}.csv")


def test_synthetic_tables(synthetic):
    d = synthetic
    for name in ("headline", "all_sizes", "progression", "env"):
        assert d.table(name)
    assert d.scalar("pct", ["v7_conflict_free", "512"]) == "80.0%"
    assert d.scalar("speedup", ["v7_conflict_free", "v1_coalesced", "256"]) == "8.0x"
    assert d.scalar("npass", []) == "4"


def test_missing_data_is_reported_not_invented(synthetic):
    text, missing = make_tables.render(synthetic, "x {{pct:v99_nope:512}} y")
    assert "n/a" in text and len(missing) == 1


def test_failed_kernels_are_marked(synthetic, tmp_path):
    p = next(tmp_path.glob("sgemm_*.csv"))
    df = pd.read_csv(p)
    df.loc[(df.kernel == "v1_coalesced") & (df.M == 512), "status"] = "FAILED"
    df.to_csv(p, index=False)
    d = make_tables.Data(p)
    assert "FAILED" in d.table("all_sizes")
    _, missing = make_tables.render(d, "{{pct:v1_coalesced:512}}")
    assert missing, "a FAILED result must not be quoted as a number"
