"""Shared loading for plot.py and make_tables.py.

Every published number comes from one benchmark run: a results/sgemm_<gpu>_<stamp>.csv and
its matching results/env_<gpu>_<stamp>.json. By default the newest run (by timestamp in the
file name) is used; pass --csv to pick another.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Kernel order for tables and fixed color slots for plots. Colors follow the kernel, so
# adding a kernel later never repaints the existing ones.
KERNEL_ORDER = ["v0_naive", "v1_coalesced", "v2_smem_tiling", "v3_1d_blocktiling",
                "v4_2d_blocktiling", "v5_vectorized", "v6_double_buffered", "v7_conflict_free",
                "v8_warptiling", "v9_autotuned"]


def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--csv", type=Path, help="results CSV (default: newest results/sgemm_*.csv)")
    return p.parse_args()


def load_run(csv_path=None):
    """Returns (df, env, csv_path). df has every row; filter on status yourself."""
    if csv_path is None:
        runs = sorted(RESULTS.glob("sgemm_*.csv"))
        if not runs:
            raise SystemExit("no results/sgemm_*.csv found")
        csv_path = runs[-1]
    csv_path = Path(csv_path)
    env_path = csv_path.with_name(csv_path.name.replace("sgemm_", "env_", 1)).with_suffix(".json")
    if not env_path.exists():
        raise SystemExit(f"missing environment record {env_path.name} for {csv_path.name}")
    df = pd.read_csv(csv_path)
    env = json.loads(env_path.read_text(encoding="utf-8"))
    return df, env, csv_path


def kernels_in(df):
    """Non-cuBLAS kernels present in df, in KERNEL_ORDER (unknown names appended)."""
    names = [k for k in df["kernel"].unique() if k != "cublas"]
    return sorted(names, key=lambda k: KERNEL_ORDER.index(k) if k in KERNEL_ORDER else 99)


def peak_fp32_gflops(env):
    """Theoretical FP32 peak = SMs x FP32 lanes/SM x max SM clock x 2 (FMA)."""
    lanes = {"7.5": 64, "7.0": 64, "6.0": 64, "8.0": 64, "8.6": 128, "8.9": 128}
    cc = env["compute_capability"]
    if cc not in lanes:
        return None
    return env["sm_count"] * lanes[cc] * env["sm_clock_max_khz"] * 1e3 * 2 / 1e9


def newest(pattern):
    """Newest results file matching pattern (file names end in a sortable UTC stamp)."""
    files = sorted(RESULTS.glob(pattern))
    return files[-1] if files else None


def load_resources(csv_path):
    """results/kernels_<gpu>_<stamp>.csv written by the same bench run, or None."""
    p = Path(csv_path).with_name(Path(csv_path).name.replace("sgemm_", "kernels_", 1))
    return (pd.read_csv(p), p) if p.exists() else (None, None)


def load_bandwidth():
    p = newest("bandwidth_*.csv")
    return (pd.read_csv(p), p) if p else (None, None)


TC_KERNELS = ["tc0_wmma_naive", "tc1_wmma_smem", "tc2_wmma_db", "tc3_ptx_mma"]
V7_DEFAULT = (128, 128, 8, 8, 8)
V8_DEFAULT = (128, 128, 8, 64, 32, 2, 2)


def bench_name(kernel_symbol):
    """Kernel symbol (demangled like 'sgemm_v7<128, 128, 8, 8, 8>(int, ...)' or mangled like
    '_ZN..sgemm_v7ILi128ELi128ELi8ELi8ELi8EEE..') -> bench name ('v7_conflict_free',
    'v7_64x64x8_4x4', 'v2_smem_tiling'), or None if it isn't one of ours."""
    import re
    s = str(kernel_symbol)
    m = re.search(r"sgemm_v7(?:<\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*>|"
                  r"ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E)", s)
    if m:
        dims = tuple(int(g) for g in m.groups() if g is not None)
        return "v7_conflict_free" if dims == V7_DEFAULT else "v7_{}x{}x{}_{}x{}".format(*dims)
    m = re.search(r"sgemm_v8(?:<\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*>|"
                  r"ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E)", s)
    if m:
        d = tuple(int(g) for g in m.groups() if g is not None)
        return ("v8_warptiling" if d == V8_DEFAULT
                else "v8_{}x{}x{}_w{}x{}_{}x{}".format(*d))
    m = re.search(r"sgemm_v10(?:<\s*" + r",\s*".join([r"(\d+)"] * 7) + r"\s*>|"
                  + r"ILi(\d+)E" + "".join([r"Li(\d+)E"] * 6) + ")", s)
    if m:
        d = tuple(int(g) for g in m.groups() if g is not None)
        return "v10_{}x{}x{}_{}x{}_b{}_s{}".format(*d)
    for name in TC_KERNELS:  # tensor-core track: symbol is <name>_kernel
        if name in s:
            return name
    for name in sorted(KERNEL_ORDER, key=len, reverse=True):  # known names first
        if "sgemm_" + name in s:
            return name
    m = re.search(r"sgemm_(v\d+_[A-Za-z0-9_]+?)\(", s)  # demangled, unknown name
    return m.group(1) if m else None


def parse_sass(path):
    """`cuobjdump -sass bench` output (optionally .gz) -> static instruction counts per kernel.

    Static = instructions in the compiled code, not executed counts. The fully unrolled inner
    loop dominates, so ratios such as LDS per FFMA reflect the inner loop.
    """
    import gzip
    import re
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        text = f.read()
    rows, cur = {}, None
    instr = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)")
    for line in text.splitlines():
        if "Function :" in line:
            cur = bench_name(line.split("Function :", 1)[1].strip())
            if cur is not None:
                rows.setdefault(cur, {})
            continue
        if cur is None:
            continue
        m = instr.search(line)
        if not m:
            continue
        op = m.group(1)
        if op.startswith("NOP"):
            continue
        c = rows[cur]
        c["total"] = c.get("total", 0) + 1
        for key, test in (("FFMA", op.startswith("FFMA")),
                          ("LDS", op.startswith("LDS")),
                          ("LDS.128", op.startswith("LDS") and ".128" in op),
                          ("LDG", op.startswith("LDG")),
                          ("LDG.128", op.startswith("LDG") and ".128" in op),
                          ("STS", op.startswith("STS")),
                          ("STS.128", op.startswith("STS") and ".128" in op),
                          ("STG", op.startswith("STG")),
                          ("STG.128", op.startswith("STG") and ".128" in op),
                          ("BAR", op.startswith("BAR"))):
            if test:
                c[key] = c.get(key, 0) + 1
    df = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
    return df


def parse_ncu(path):
    """Nsight Compute --csv log -> DataFrame indexed by kernel, one column per metric.

    Each kernel is launched more than once under ncu (warm-up + timed rep); the last launch
    is used. Kernel names are reduced to the bench name (sgemm_v2_smem_tiling -> v2_smem_tiling).
    Returns None if the file has no metric rows (e.g. ncu was blocked or a metric was invalid).
    """
    import io
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return None
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df["kernel"] = df["Kernel Name"].map(bench_name)
    df["value"] = pd.to_numeric(df["Metric Value"].astype(str).str.replace(",", ""),
                                errors="coerce")
    df = df.dropna(subset=["kernel"]).sort_values("ID")
    return df.groupby(["kernel", "Metric Name"])["value"].last().unstack()


def load_ncu(kind):
    """Newest results/ncu_<kind>_*.csv parsed, plus its path; (None, None) if absent."""
    p = newest(f"ncu_{kind}_*.csv")
    if p is None:
        return None, None
    return parse_ncu(p), p


def nvsmi_row(snapshot):
    """Parses the nvidia-smi CSV snapshot (header line + one data line) into a dict."""
    lines = [l for l in snapshot.splitlines() if l.strip()]
    if len(lines) < 2:
        return {}
    keys = [k.strip() for k in lines[0].split(",")]
    vals = [v.strip() for v in lines[1].split(",")]
    return dict(zip(keys, vals))
