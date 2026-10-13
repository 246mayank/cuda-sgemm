"""Renders README.md and docs/ANALYSIS.md from their templates in docs/, filling every number from results/.

The template is ordinary Markdown with {{tokens}}. Each token is replaced by a value read
from the CSVs, so no number in the README is typed by hand (CLAUDE.md rule 1). A token whose
data is missing renders as "n/a" and is listed on stdout, and the script exits non-zero, so
a gap can't slip into the README unnoticed.

Tokens (K = kernel name such as v2_smem_tiling, N = square size):
  {{table:headline}} {{table:all_sizes}} {{table:env}} {{table:resources}} {{table:ncu}}
  {{table:bandwidth}} {{table:v0stride}} {{table:variability}} {{table:sources}}
  {{gflops:K:N}}        median GFLOP/s                     (K may be "cublas")
  {{pct:K:N}}           % of paired cuBLAS
  {{speedup:K1:K2:N}}   GFLOP/s of K1 / GFLOP/s of K2, e.g. "9.4x"
  {{cublas_paired:N}}   median over kernels of the paired cuBLAS GFLOP/s
  {{pct_peak:K:N}}      GFLOP/s as % of theoretical FP32 peak
  {{maxerr}}            largest max_rel_err over all PASS rows;  {{tol}} the tolerance
  {{npass}} / {{nrows}} PASS rows / kernel rows in the run
  {{peak}}              theoretical FP32 GFLOP/s
  {{bw}} {{bw_pct}} {{bw_theo}}   measured DRAM GB/s (best method median), % of spec, spec
  {{regs:K}} {{blocks:K}} {{occ:K}} {{spill:K}} {{smem:K}}   from results/kernels_*.csv
  {{dram_mb:K}} {{ai:K}} {{traffic_x:K}}   ncu DRAM MB, FLOP/byte, traffic / minimum traffic
  {{ncu:dram|mem|smem:METRIC:K}}           raw ncu metric
  {{lds_per_ffma:K}} {{bank_conflicts:K}} {{sectors_per_req:K}} {{l1hit:K}} {{l2hit:K}}
  {{achieved_occ:K}}
  {{stride:METRIC:N}}   v0 ncu metric at N = 1023 or 1024
  {{var:K:N}}           run-to-run range of median GFLOP/s across all runs, "min-max (n runs)"
  {{ideal_ai}}          FLOP / minimum DRAM bytes at the ncu size
  {{ridge}}             FLOP/byte where the measured DRAM roof meets the FP32 peak
  {{gflops_min:K:N}}    GFLOP/s of the fastest single timed run
  {{pspeedup:K1:K2:N}}  throttle-normalised speedup: ratio of the two % of paired cuBLAS
  {{table:clocks}}      SM clock / power sampled after each kernel at the largest size

Usage: .venv/Scripts/python scripts/make_tables.py [--csv results/sgemm_....csv]
"""
import re
import sys

import pandas as pd

from results_io import (RESULTS, ROOT, kernels_in, load_bandwidth, load_ncu, load_resources,
                        load_run, nvsmi_row, parse_args, parse_ncu, parse_sass, newest,
                        peak_fp32_gflops)

# (template, output, relink): relink rewrites repo-relative links for files under docs/.
OUTPUTS = [(ROOT / "docs" / "README.template.md", ROOT / "README.md", False),
           (ROOT / "docs" / "ANALYSIS.template.md", ROOT / "docs" / "ANALYSIS.md", True)]
IDEAS = {  # one line per kernel for the README's progression table (text, not data)
    "v0_naive": "one thread per output, uncoalesced loads",
    "v1_coalesced": "threads remapped so a warp reads contiguous memory",
    "v2_smem_tiling": "32×32 tiles staged in shared memory",
    "v3_1d_blocktiling": "8 outputs per thread (register reuse)",
    "v4_2d_blocktiling": "8×8 outputs per thread",
    "v5_vectorized": "128-bit loads, transposed A tile (null result)",
    "v6_double_buffered": "load next tile during math",
    "v7_conflict_free": "thread layout with zero bank conflicts",
    "v8_warptiling": "warp-level tile (null result)",
    "v9_autotuned": "per-size tile chosen from a tuning sweep",
    "v11_splitk": "split-K when the grid can't fill the GPU",
    "v12_dispatch": "picks v9 or v11 by grid fill",
    "v15_splitk_det": "split-K with a fixed-order sum (bit-reproducible)",
}
NCU_N = 2048  # size profiled by the ncu cells in the notebook


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def kname(k):
    return "cuBLAS" if k == "cublas" else f"`{k}`"


class Missing(Exception):
    pass


class Data:
    def __init__(self, csv_path):
        self.df, self.env, self.csv_path = load_run(csv_path)
        self.res, self.res_path = load_resources(self.csv_path)
        self.bw, self.bw_path = load_bandwidth()
        self.ncu, self.ncu_paths = {}, {}
        for kind in ("dram", "mem", "smem"):
            self.ncu[kind], self.ncu_paths[kind] = load_ncu(kind)
        self.stride = {}
        for n in (1023, 1024):
            p = newest(f"ncu_v0stride_{n}_*.csv")
            self.stride[n] = (parse_ncu(p), p) if p else (None, None)
            p = newest(f"ncu_v0set_{n}_*.csv")  # L1 set-conflict counters (run 5 onwards)
            self.stride[f"set{n}"] = (parse_ncu(p), p) if p else (None, None)
        self.peak = peak_fp32_gflops(self.env)
        p = newest("sass_*.txt.gz")
        self.sass, self.sass_path = (parse_sass(p), p) if p else (None, None)
        # Tuning study and clock-locked runs: separate file prefixes, newest of each.
        self.tune, self.tune_res, self.tune_path = None, None, None
        p = newest("tune_sgemm_*.csv")
        if p is not None:
            self.tune, self.tune_path = pd.read_csv(p), p
            rp = p.with_name(p.name.replace("tune_sgemm_", "tune_kernels_", 1))
            self.tune_res = pd.read_csv(rp) if rp.exists() else None
        self.tc, self.tc_path, self.tc_prec = None, None, None
        p = newest("tc_sgemm_*.csv")
        if p is not None:
            self.tc, self.tc_path = pd.read_csv(p), p
            pp = p.with_name(p.name.replace("tc_sgemm_", "tc_precision_", 1))
            self.tc_prec = pd.read_csv(pp) if pp.exists() else None
        self.locked, self.locked_path = None, None
        p = newest("locked*_sgemm_*.csv")
        if p is not None:
            self.locked, self.locked_path = pd.read_csv(p), p

    # ------------------------------------------------------------- lookups
    def row(self, k, n):
        r = self.df[(self.df.kernel == k) & (self.df.M == int(n))]
        if r.empty:
            raise Missing(f"{k} at N={n} not in {self.csv_path.name}")
        r = r.iloc[0]
        if r.status == "FAILED":
            raise Missing(f"{k} at N={n} FAILED the check")
        return r

    def ncu_key(self, k):
        """v9 has no kernel of its own: under ncu it runs as the config it dispatches to at NCU_N."""
        if k in ("v9_autotuned", "v12_dispatch"):  # at NCU_N the grid is full: v12 runs v9
            try:
                return self.scalar("dispatch", [str(NCU_N)]).strip("`")
            except Missing:
                return k
        return k

    def ncu_val(self, kind, metric, k):
        t = self.ncu.get(kind)
        k = self.ncu_key(k)
        if t is None or k not in t.index or metric not in t.columns or pd.isna(t.loc[k, metric]):
            raise Missing(f"ncu {kind} {metric} for {k}")
        return float(t.loc[k, metric])

    def res_val(self, k, col):
        if self.res is None or k not in set(self.res.kernel):
            raise Missing(f"resources for {k}")
        return self.res[self.res.kernel == k].iloc[0][col]

    def dram_bytes(self, k):
        return (self.ncu_val("dram", "dram__bytes_read.sum", k)
                + self.ncu_val("dram", "dram__bytes_write.sum", k))

    def bw_best(self):
        if self.bw is None:
            raise Missing("bandwidth_*.csv")
        return self.bw.gbps_median.max(), self.bw.theoretical_gbps.iloc[0]

    def cublas_paired(self, n):
        d = self.df[(self.df.M == int(n)) & self.df.cublas_paired_median_ms.notna()] \
            if "cublas_paired_median_ms" in self.df else self.df.iloc[0:0]
        if d.empty:
            raise Missing(f"paired cuBLAS at N={n}")
        return 2.0 * int(n) ** 3 / (d.cublas_paired_median_ms.median() * 1e6)

    # ------------------------------------------------------------- scalar tokens
    def scalar(self, name, args):
        f2 = lambda x: f"{x:,.0f}"
        if name == "gflops":
            return f2(self.row(args[0], args[1]).gflops)
        if name == "pct":
            return f"{self.row(args[0], args[1]).pct_cublas:.1f}%"
        if name == "speedup":
            a, b = self.row(args[0], args[2]).gflops, self.row(args[1], args[2]).gflops
            return f"{a / b:.1f}x"
        if name in ("best_kernel", "best_gflops", "best_pct"):  # {{best_*:N}} by % of paired cuBLAS
            n = int(args[0])
            d = self.df[(self.df.M == n) & (self.df.status == "PASS")]
            if d.empty:
                raise Missing(f"no PASS rows at N={n}")
            b = d.loc[d.pct_cublas.idxmax()]
            return {"best_kernel": f"`{b.kernel}`", "best_gflops": f2(b.gflops),
                    "best_pct": f"{b.pct_cublas:.1f}%"}[name]
        if name == "best_speedup_vs":  # {{best_speedup_vs:K:N}} best kernel GFLOP/s / K's at N
            n = int(args[1])
            d = self.df[(self.df.M == n) & (self.df.status == "PASS")]
            b = d.loc[d.pct_cublas.idxmax()]
            return f"{b.gflops / self.row(args[0], n).gflops:.1f}x"
        if name == "sizeratio":  # same kernel, GFLOP/s at N1 / GFLOP/s at N2
            a, b = self.row(args[0], args[1]).gflops, self.row(args[0], args[2]).gflops
            return f"{a / b:.1f}x"
        if name == "warmups":
            return str(self.env["warmup_runs"])
        if name == "reps":
            return str(self.env["timed_runs"])
        if name == "boost_mhz":
            return str(self.env["sm_clock_max_khz"] // 1000)
        if name == "sm_count":
            return str(self.env["sm_count"])
        if name == "power_limit":
            return nvsmi_row(self.env["nvidia_smi_start"]).get("power.limit [W]", "n/a")
        if name == "ridge":  # FLOP/byte where the measured DRAM roof meets the FP32 peak
            return f"{self.peak / self.bw_best()[0]:.0f}"
        if name == "gflops_min":  # from the fastest single timed run (least throttled)
            r = self.row(args[0], args[1])
            return f2(2.0 * int(args[1]) ** 3 / (r.min_ms * 1e6))
        if name == "pspeedup":  # throttle-normalised: ratio of the two kernels' % of paired cuBLAS
            a, b = self.row(args[0], args[2]).pct_cublas, self.row(args[1], args[2]).pct_cublas
            return f"{a / b:.2f}x"
        if name == "cublas_paired":
            return f2(self.cublas_paired(args[0]))
        if name == "pct_peak":
            return f"{100 * self.row(args[0], args[1]).gflops / self.peak:.1f}%"
        if name == "maxerr":
            ok = self.df[(self.df.status == "PASS")]
            return f"{ok.max_rel_err.max():.1e}"
        if name == "tol":
            return f"{self.env['tolerance_max_rel_err']:.0e}"
        if name == "npass":
            return str(int((self.df.status == "PASS").sum()))
        if name == "nrows":
            return str(int((self.df.kernel != "cublas").sum()))
        if name == "peak":
            return f2(self.peak)
        if name == "bw":
            return f2(self.bw_best()[0])
        if name == "bw_theo":
            return f2(self.bw_best()[1])
        if name == "bw_pct":
            b, t = self.bw_best()
            return f"{100 * b / t:.0f}%"
        if name in ("regs", "blocks", "spill", "smem"):
            col = {"regs": "regs_per_thread", "blocks": "blocks_per_sm",
                   "spill": "local_bytes_per_thread", "smem": "static_smem_bytes"}[name]
            return str(int(self.res_val(args[0], col)))
        if name == "occ":
            return f"{self.res_val(args[0], 'theoretical_occupancy_pct'):.0f}%"
        if name == "dram_mb":
            return f2(self.dram_bytes(args[0]) / 1e6)
        if name == "ai":
            return f"{2.0 * NCU_N ** 3 / self.dram_bytes(args[0]):.1f}"
        if name == "ideal_ai":
            return f"{2.0 * NCU_N ** 3 / (3 * NCU_N * NCU_N * 4):.0f}"
        if name == "traffic_x":
            return f"{self.dram_bytes(args[0]) / (3 * NCU_N * NCU_N * 4):.1f}x"
        if name == "ncu":
            return f"{self.ncu_val(args[0], args[1], args[2]):,.4g}"
        if name == "lds_per_ffma":
            return f"{self.lds_per_ffma(args[0]):.3f}"
        if name == "bank_conflicts_st":
            return f2(self.ncu_val("smem", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
                                   args[0]))
        if name == "ffma_share":
            ffma_warp = self.ncu_val("smem", "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                                     args[0]) / 32
            return f"{100 * ffma_warp / self.ncu_val('smem', 'smsp__inst_executed.sum', args[0]):.0f}%"
        if name == "sass":  # {{sass:K:COLUMN}}
            if self.sass is None or args[0] not in self.sass.index:
                raise Missing(f"sass for {args[0]}")
            return f"{int(self.sass.loc[args[0]].get(args[1], 0)):,}"
        if name == "tune_best":  # {{tune_best:N}} -> config name with best % at N
            t = self.tune_rows()
            g = t[(t.M == int(args[0])) & (t.status == "PASS")]
            if g.empty:
                raise Missing(f"tuning at N={args[0]}")
            return f"`{g.loc[g.pct_cublas.idxmax()].kernel}`"
        if name == "tune_best_pct":
            t = self.tune_rows()
            g = t[(t.M == int(args[0])) & (t.status == "PASS")]
            if g.empty:
                raise Missing(f"tuning at N={args[0]}")
            return f"{g.pct_cublas.max():.1f}%"
        if name == "tune_pct":  # {{tune_pct:CONFIG:N}}
            t = self.tune_rows()
            g = t[(t.kernel == args[0]) & (t.M == int(args[1]))]
            if g.empty:
                raise Missing(f"tuning {args[0]} at N={args[1]}")
            return f"{g.pct_cublas.iloc[0]:.1f}%"
        if name == "tune_waves":  # {{tune_waves:CONFIG:N}}
            t = self.tune_rows()
            g = t[(t.kernel == args[0]) & (t.M == int(args[1]))]
            if g.empty or "waves" not in g:
                raise Missing(f"waves {args[0]} at N={args[1]}")
            return f"{g.waves.iloc[0]:.1f}"
        if name in ("tc_best", "tc_best_tflops", "tc_best_pct"):  # best tensor-core kernel at N
            if self.tc is None:
                raise Missing("tc_sgemm_*.csv")
            d = self.tc[(self.tc.M == int(args[0])) & (self.tc.status == "PASS")]
            if d.empty:
                raise Missing(f"tc at N={args[0]}")
            b = d.loc[d.pct_cublas.idxmax()]
            return {"tc_best": f"`{b.kernel}`", "tc_best_tflops": f"{b.gflops / 1000:.1f}",
                    "tc_best_pct": f"{b.pct_cublas:.1f}%"}[name]
        if name == "tune_res":  # {{tune_res:CONFIG:COLUMN}} from tune_kernels_*.csv
            if self.tune_res is None or args[0] not in set(self.tune_res.kernel):
                raise Missing(f"tune resources for {args[0]}")
            return str(self.tune_res[self.tune_res.kernel == args[0]].iloc[0][args[1]])
        if name == "tc_res":  # {{tc_res:K:COLUMN}} from tc_kernels_*.csv of the newest tc run
            p = self.tc_path.with_name(self.tc_path.name.replace("tc_sgemm_", "tc_kernels_", 1)) \
                if self.tc_path else None
            if p is None or not p.exists():
                raise Missing("tc_kernels_*.csv")
            d = pd.read_csv(p)
            r = d[d.kernel == args[0]]
            if r.empty:
                raise Missing(f"tc resources for {args[0]}")
            return str(r.iloc[0][args[1]])
        if name == "shape_pct":  # {{shape_pct:K:M:N:Kdim}} % of paired cuBLAS on a shape
            p = newest("shapes_sgemm_*.csv")
            if p is None:
                raise Missing("shapes_sgemm_*.csv")
            d = pd.read_csv(p)
            r = d[(d.kernel == args[0]) & (d.M == int(args[1])) & (d.N == int(args[2]))
                  & (d.K == int(args[3]))]
            if r.empty or r.status.iloc[0] == "FAILED":
                raise Missing(f"shape {args}")
            return f"{r.pct_cublas.iloc[0]:.1f}%"
        if name in ("red_gbps", "red_pct_cub", "red_pct_bw", "red_speedup"):
            r = self.reduce_row(args[0])
            if name == "red_gbps":
                return f"{r.gbps:.0f}"
            if name == "red_pct_cub":
                return f"{r.pct_of_cub:.0f}%"
            if name == "red_pct_bw":
                return f"{100 * r.gbps / self.bw_best()[0]:.0f}%"
            return f"{r.gbps / self.reduce_row(args[1]).gbps:.1f}x"  # {{red_speedup:K1:K2}}
        if name in ("tctune_best", "tctune_best_pct"):  # {{tctune_best:N}} best tc5 config at N
            d = self.tctune()
            g = d[(d.kernel != "cublas") & (d.M == int(args[0])) & (d.status == "PASS")]
            if g.empty:
                raise Missing(f"tctune at N={args[0]}")
            b = g.loc[g.pct_cublas.idxmax()]
            return f"`{b.kernel}`" if name == "tctune_best" else f"{b.pct_cublas:.1f}%"
        if name in ("epi_lt_speedup", "epi_lt_best", "epi_lt_worst"):  # cuBLASLt fused epilogue
            d, _ = self.epi()
            if "lt_status" not in d.columns:
                raise Missing("cuBLASLt columns in epi_*.csv")
            d = d[d.lt_status == "PASS"]
            if d.empty:
                raise Missing("cuBLASLt PASS rows")
            if name == "epi_lt_speedup":
                r = d[(d.act == args[0]) & (d.M == int(args[1])) & (d.N == int(args[2]))
                      & (d.K == int(args[3]))]
                if r.empty:
                    raise Missing(f"cuBLASLt epilogue {args}")
                return f"{r.lt_speedup.iloc[0]:.2f}x"
            b = d.loc[d.lt_speedup.idxmax() if name == "epi_lt_best" else d.lt_speedup.idxmin()]
            return f"{b.lt_speedup:.2f}x ({b.act}, {b.M} × {b.N} × {b.K})"
        if name in ("epi_speedup", "epi_best", "epi_worst"):  # {{epi_speedup:ACT:M:N:K}}, {{epi_best}}
            d, _ = self.epi()
            d = d[d.status == "PASS"]
            if name == "epi_best":
                if d.empty:
                    raise Missing("epilogue PASS rows")
                b = d.loc[d.speedup.idxmax()]
                return f"{b.speedup:.2f}x ({b.act}, {b.M} × {b.N} × {b.K})"
            if name == "epi_worst":
                if d.empty:
                    raise Missing("epilogue PASS rows")
                b = d.loc[d.speedup.idxmin()]
                return f"{b.speedup:.2f}x ({b.act}, {b.M} × {b.N} × {b.K})"
            r = d[(d.act == args[0]) & (d.M == int(args[1])) & (d.N == int(args[2]))
                  & (d.K == int(args[3]))]
            if r.empty:
                raise Missing(f"epilogue {args}")
            return f"{r.speedup.iloc[0]:.2f}x"
        if name == "tc_tflops":  # {{tc_tflops:K:N}} (K may be cublas = cublasGemmEx)
            return f"{self.tc_row(args[0], args[1]).gflops / 1000:.1f}"
        if name == "tc_pct":
            return f"{self.tc_row(args[0], args[1]).pct_cublas:.1f}%"
        if name == "tc_speedup":  # {{tc_speedup:K1:K2:N}}
            a, b = self.tc_row(args[0], args[2]).gflops, self.tc_row(args[1], args[2]).gflops
            return f"{a / b:.1f}x"
        if name == "tc_vs_fp32":  # {{tc_vs_fp32:K:N}} tensor-core GFLOP/s / best FP32 kernel's
            n = int(args[1])
            d = self.df[(self.df.M == n) & (self.df.status == "PASS")]
            return f"{self.tc_row(args[0], n).gflops / d.gflops.max():.1f}x"
        if name == "tc_prec":  # {{tc_prec:N}}
            if self.tc_prec is None:
                raise Missing("tc_precision_*.csv")
            r = self.tc_prec[self.tc_prec.N == int(args[0])]
            if r.empty:
                raise Missing(f"tc precision at N={args[0]}")
            return f"{r.max_rel_err_fp16_inputs_vs_fp32_sgemm.iloc[0]:.1e}"
        if name == "locked_clock":
            if self.locked is None or "sm_clock_mhz_after" not in self.locked:
                raise Missing("locked run")
            return f"{self.locked.sm_clock_mhz_after.median():.0f}"
        if name == "locked_pct":  # {{locked_pct:K:N}}
            if self.locked is None:
                raise Missing("locked run")
            d = self.locked
            r = d[(d.kernel == args[0]) & (d.M == int(args[1]))]
            if r.empty:
                raise Missing(f"locked {args[0]} {args[1]}")
            return f"{r.pct_cublas.iloc[0]:.1f}%"
        if name == "bank_conflicts":
            return f2(self.ncu_val("smem", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
                                   args[0]))
        if name == "sectors_per_req":
            return f"{self.sectors_per_req(args[0]):.1f}"
        if name == "l1hit":
            return f"{self.ncu_val('mem', 'l1tex__t_sector_hit_rate.pct', args[0]):.1f}%"
        if name == "l2hit":
            return f"{self.ncu_val('mem', 'lts__t_sector_hit_rate.pct', args[0]):.1f}%"
        if name == "achieved_occ":
            return f"{self.ncu_val('mem', 'sm__warps_active.avg.pct_of_peak_sustained_active', args[0]):.0f}%"
        if name == "stride":
            v = None
            for key in (int(args[1]), f"set{args[1]}"):
                t, _ = self.stride[key]
                if t is not None and "v0_naive" in t.index and args[0] in t.columns:
                    v = float(t.loc["v0_naive", args[0]])
            if v is None:
                raise Missing(f"v0 stride {args[0]} at {args[1]}")
            if "bytes" in args[0]:
                return f"{v / 1e6:,.0f} MB"
            if args[0].endswith(".pct"):
                return f"{v:.1f}%"
            if "duration" in args[0]:
                return f"{v / 1e6:.1f} ms"
            return f"{v / 1e6:,.1f} M"
        if name in ("tune_mean", "tune_runs"):  # {{tune_mean:CONFIG:N}} mean over ALL tuning runs
            vals = []
            for p in sorted(RESULTS.glob("tune_sgemm_*.csv")):
                d = pd.read_csv(p)
                r = d[(d.kernel == args[0]) & (d.M == int(args[1])) & (d.status == "PASS")]
                if not r.empty:
                    vals.append(float(r.pct_cublas.iloc[0]))
            if not vals:
                raise Missing(f"tuning runs for {args[0]} at N={args[1]}")
            return f"{sum(vals) / len(vals):.1f}%" if name == "tune_mean" else str(len(vals))
        if name == "dispatch_source":  # which tuning CSV v9's table was generated from
            inc = ROOT / "src" / "kernels" / "v9_dispatch_table.inc"
            files = re.findall(r"results/(\S+?\.csv)", inc.read_text(encoding="utf-8")) if inc.exists() else []
            if not files:
                raise Missing("v9_dispatch_table.inc")
            if len(files) == 1:
                return f"`results/{files[0]}`"
            return f"averaged over {len(files)} tuning runs, `results/tune_sgemm_*.csv`"
        if name in ("tcpipe", "tcdram_mb"):  # tensor-core ncu counters at N = 2048
            kind = "tcpipe" if name == "tcpipe" else "tcdram"
            t, _ = load_ncu(kind)
            if t is None or args[0] not in t.index:
                raise Missing(f"ncu {kind} for {args[0]}")
            r = t.loc[args[0]]
            if name == "tcpipe":
                return f"{r['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active']:.1f}%"
            return f"{(r['dram__bytes_read.sum'] + r['dram__bytes_write.sum']) / 1e6:,.0f}"
        if name == "dispatch_pct":  # {{dispatch_pct:N}} the tuning-run % recorded in the table
            inc = ROOT / "src" / "kernels" / "v9_dispatch_table.inc"
            for line in inc.read_text(encoding="utf-8").splitlines():
                m = re.match(r'\{(\d+), "[^"]+"\},\s*//\s*(?:mean\s+)?([\d.]+)%', line)
                if m and int(m.group(1)) == int(args[0]):
                    return f"{float(m.group(2)):.1f}%"
            raise Missing(f"dispatch pct for N={args[0]}")
        if name == "dispatch":  # {{dispatch:N}} config v9 uses at size N
            inc = ROOT / "src" / "kernels" / "v9_dispatch_table.inc"
            for line in inc.read_text(encoding="utf-8").splitlines():
                m = re.match(r'\{(\d+), "([^"]+)"\}', line)
                if m and int(m.group(1)) == int(args[0]):
                    return f"`{m.group(2)}`"
            raise Missing(f"dispatch entry for N={args[0]}")
        if name == "conflicts_per_lds":  # {{conflicts_per_lds:K}}
            c = self.ncu_val("smem", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", args[0])
            return f"{c / self.ncu_val('smem', 'smsp__inst_executed_op_shared_ld.sum', args[0]):.1f}"
        if name == "conflict_pct":  # {{conflict_pct:K1:K2}} K1 load conflicts as % of K2's
            m = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
            return f"{100 * self.ncu_val('smem', m, args[0]) / self.ncu_val('smem', m, args[1]):.2f}%"
        if name == "tune_count":
            t = self.tune_rows()
            return str(t.kernel.nunique())
        if name == "conflict_ratio":  # {{conflict_ratio:K1:K2}} K1 load conflicts / K2's
            m = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
            return f"{self.ncu_val('smem', m, args[0]) / self.ncu_val('smem', m, args[1]):,.0f}x"
        if name == "var":
            lo, hi, n = self.variability(args[0], int(args[1]))
            return f"{lo:,.0f}-{hi:,.0f} GFLOP/s ({n} runs)"
        raise Missing(f"unknown token {name}")

    def lds_per_ffma(self, k):
        lds = self.ncu_val("smem", "smsp__inst_executed_op_shared_ld.sum", k)
        ffma_warp = self.ncu_val("smem", "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", k) / 32
        return lds / ffma_warp

    def sectors_per_req(self, k):
        return (self.ncu_val("mem", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", k)
                / self.ncu_val("mem", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", k))

    def variability(self, k, n):
        vals = []
        for p in sorted(RESULTS.glob("sgemm_*.csv")):
            d = pd.read_csv(p)
            r = d[(d.kernel == k) & (d.M == n) & d.status.isin(["PASS", "REF"])]
            if not r.empty:
                vals.append(float(r.gflops.iloc[0]))
        if not vals:
            raise Missing(f"variability {k} {n}")
        return min(vals), max(vals), len(vals)

    # ------------------------------------------------------------- tables
    def table(self, name):
        return getattr(self, "t_" + name)()

    def t_headline(self):
        df = self.df
        size = df.M.max()
        d = df[df.M == size]
        rows = []
        for k in ["cublas"] + kernels_in(d):
            r = d[d.kernel == k].iloc[0]
            if r.status == "FAILED":
                rows.append([kname(k), "FAILED", "", "", "", f"{r.max_rel_err:.1e}", "FAILED"])
                continue
            rows.append([kname(k), f"{r.gflops:,.0f}", f"{r.median_ms:.2f}",
                         f"{r.min_ms:.2f} – {r.p90_ms:.2f}",
                         "100% (ref)" if k == "cublas" else f"{r.pct_cublas:.1f}%",
                         "reference" if k == "cublas" else f"{r.max_rel_err:.1e}", r.status])
        out = (f"**N = {size}** (M = N = K), GFLOP/s from the median of the timed runs:\n\n"
               + md_table(["kernel", "GFLOP/s", "median ms", "min – p90 ms", "% cuBLAS",
                           "max rel err", "check"], rows))
        skipped = [k for k in kernels_in(df) if k not in set(d.kernel)]
        if skipped:
            out += ("\n\nNot run at this size: " + ", ".join(kname(k) for k in skipped)
                    + " (largest size run: "
                    + ", ".join(str(df[df.kernel == k].M.max()) for k in skipped) + ").")
        return out

    def t_all_sizes(self):
        df = self.df
        sizes = sorted(df.M.unique())
        rows = []
        for k in ["cublas"] + kernels_in(df):
            cells = []
            for s in sizes:
                r = df[(df.kernel == k) & (df.M == s)]
                cells.append("" if r.empty else "FAILED" if r.status.iloc[0] == "FAILED"
                             else f"{r.gflops.iloc[0]:,.0f}")
            rows.append([kname(k)] + cells)
        return md_table(["GFLOP/s"] + [f"N={s}" for s in sizes], rows)

    def t_env(self):
        env = self.env
        start, end = nvsmi_row(env["nvidia_smi_start"]), nvsmi_row(env["nvidia_smi_end"])
        cuda = lambda v: f"{v // 1000}.{(v % 1000) // 10}"
        sm = lambda r: (f"{r.get('clocks.current.sm [MHz]', 'n/a')}, "
                        f"{r.get('temperature.gpu', 'n/a')} C, {r.get('power.draw [W]', 'n/a')} "
                        f"(limit {r.get('power.limit [W]', 'n/a')})")
        return "\n".join([
            f"- GPU: **{env['gpu_name']}** (compute capability {env['compute_capability']}), "
            f"{env['sm_count']} SMs, max SM clock {env['sm_clock_max_khz'] // 1000} MHz, "
            f"L2 {env['l2_cache_bytes'] // 2**20} MB, shared memory {env['smem_per_sm_bytes'] // 1024} KB/SM",
            f"- CUDA runtime {cuda(env['cuda_runtime_version'])}, nvcc {env['nvcc_version']} "
            f"(`-O3 -arch={env['compiled_arch']}`), driver {start.get('driver_version', 'n/a')}, "
            f"cuBLAS {env['cublas_version']}",
            f"- Run {env['date_utc']}: {env['warmup_runs']} warm-ups + {env['timed_runs']} timed "
            f"runs per point, inputs {env['input_distribution']}",
            f"- SM clock, temperature, power at start: {sm(start)}; at end: {sm(end)}",
        ])

    def t_resources(self):
        if self.res is None:
            raise Missing("kernels_*.csv")
        rows = []
        for _, r in self.res.iterrows():
            ach = ""
            try:
                ach = self.scalar("achieved_occ", [r.kernel])
            except Missing:
                ach = "n/a"
            rows.append([kname(r.kernel), r.threads_per_block, r.regs_per_thread,
                         r.local_bytes_per_thread, f"{r.static_smem_bytes:,}", r.blocks_per_sm,
                         f"{r.warps_per_sm}/{r.max_warps_per_sm} ({r.theoretical_occupancy_pct:.0f}%)",
                         ach])
        return md_table(["kernel", "threads/block", "registers/thread", "spill bytes/thread",
                         "static smem bytes", "blocks/SM", "warps/SM (theoretical occupancy)",
                         "achieved occupancy (ncu)"], rows)

    def t_ncu(self):
        if self.ncu.get("dram") is None:
            raise Missing("ncu_dram_*.csv")
        rows = []
        for k in kernels_in(self.df):
            if self.ncu_key(k) not in self.ncu["dram"].index:
                continue
            cells = [kname(k)]
            for tok in ("dram_mb", "traffic_x", "ai", "sectors_per_req", "l1hit", "l2hit",
                        "lds_per_ffma", "bank_conflicts", "bank_conflicts_st", "ffma_share"):
                try:
                    cells.append(self.scalar(tok, [k]))
                except Missing:
                    cells.append("n/a")
            rows.append(cells)
        return (f"Nsight Compute counters at N = {NCU_N} (last of 2 profiled launches per kernel):\n\n"
                + md_table(["kernel", "DRAM MB moved", "vs minimum", "FLOP per DRAM byte",
                            "global sectors per load request", "L1 hit", "L2 hit",
                            "shared loads per FFMA (warp-level)", "shared-load bank conflicts",
                            "shared-store bank conflicts", "FFMA share of instructions"],
                           rows))

    def t_progression(self):
        df = self.df
        big = int(df.M.max())
        mid = 2048 if 2048 in set(df.M) else big
        rows = []
        for k in kernels_in(df):
            if k not in IDEAS:
                continue

            def pct(n):
                r = df[(df.kernel == k) & (df.M == n)]
                return "" if r.empty else ("FAILED" if r.status.iloc[0] == "FAILED"
                                            else f"{r.pct_cublas.iloc[0]:.0f}%")
            rows.append([f"`{k}`", IDEAS[k], pct(mid), pct(big)])
        return md_table(["kernel", "idea", f"% of cuBLAS, N={mid}", f"% of cuBLAS, N={big}"], rows)

    def t_bankmodel(self):
        p = RESULTS / "model_bank_conflicts.csv"
        if not p.exists():
            raise Missing("model_bank_conflicts.csv (run scripts/bank_model.py)")
        m = pd.read_csv(p)
        rows = []
        for _, r in m.iterrows():
            def meas(metric):
                try:
                    return f"{self.ncu_val('smem', metric, r.kernel):,.0f}"
                except Missing:
                    return "not profiled"
            ld = meas("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
            st = meas("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum")
            rows.append([kname(r.kernel), f"{r.predicted_ld_conflicts:,}", ld,
                         f"{r.predicted_st_conflicts:,}", st])
        return (f"Shared-memory bank conflicts at N = {m.N.iloc[0]}: the analytical model in "
                "[`scripts/bank_model.py`](scripts/bank_model.py) (lane addresses from the kernel "
                "source + the 32-bank / 8-lanes-per-128-bit-phase rule) next to Nsight Compute's "
                "counters:\n\n"
                + md_table(["kernel", "model: load", "ncu: load", "model: store", "ncu: store"],
                           rows))

    def t_sass(self):
        if self.sass is None:
            raise Missing("sass_*.txt.gz")
        cols = ["total", "FFMA", "LDS", "LDS.128", "LDG", "LDG.128", "STS", "STS.128", "STG",
                "STG.128", "BAR"]
        rows = []
        for k in kernels_in(self.df):
            if k not in self.sass.index:
                continue
            r = self.sass.loc[k]
            cells = [kname(k)] + [f"{int(r.get(c, 0)):,}" for c in cols]
            ffma, lds = r.get("FFMA", 0), r.get("LDS", 0)
            cells.append(f"{lds / ffma:.3f}" if ffma else "n/a")
            rows.append(cells)
        return ("Static instruction counts in each kernel's compiled machine code (SASS, from "
                "`cuobjdump -sass`). Static means instructions present in the binary, not "
                "executed; the fully unrolled inner loop dominates, so the last column tracks the "
                "inner loop's shared-load to FMA ratio. `.128` = 128-bit (vector) access.\n\n"
                + md_table(["kernel"] + cols + ["LDS per FFMA (static)"], rows))

    def tune_rows(self):
        """Tuning results joined with resources: one row per (config, size), PASS only."""
        if self.tune is None:
            raise Missing("tune_sgemm_*.csv")
        t = self.tune[self.tune.kernel != "cublas"].copy()
        dims = t.kernel.str.extract(r"v\d+_(\d+)x(\d+)x(\d+)_").astype(float)
        t[["BM", "BN", "BK"]] = dims.values
        if self.tune_res is not None:
            t = t.merge(self.tune_res, on="kernel", how="left")
            slots = self.env["sm_count"] * t.blocks_per_sm
            blocks = (-(-t.M // t.BM)) * (-(-t.N // t.BN))
            t["waves"] = blocks / slots
        return t

    def t_tune(self):
        t = self.tune_rows()
        sizes = sorted(t.M.unique())
        failed = sorted(set(t[t.status == "FAILED"].kernel))
        ok = t[~t.kernel.isin(failed)]
        rows = []
        for k, g in ok.groupby("kernel", sort=False):
            r0 = g.iloc[0]
            cells = [f"`{k}`", int(r0.threads_per_block) if "threads_per_block" in g else "",
                     int(r0.regs_per_thread) if "regs_per_thread" in g else "",
                     int(r0.blocks_per_sm) if "blocks_per_sm" in g else ""]
            for s in sizes:
                rr = g[g.M == s]
                cells.append(f"{rr.pct_cublas.iloc[0]:.0f}%" if not rr.empty else "")
            rows.append(cells)
        best = []
        for s in sizes:
            g = ok[ok.M == s]
            if g.empty:
                continue
            b = g.loc[g.pct_cublas.idxmax()]
            best.append(f"N = {s}: `{b.kernel}` ({b.pct_cublas:.0f}%, {b.waves:.1f} waves)"
                        if "waves" in g else f"N = {s}: `{b.kernel}` ({b.pct_cublas:.0f}%)")
        note = ("" if not failed else
                "\n\nFailed the cuBLAS check (excluded): " + ", ".join(f"`{k}`" for k in failed))
        return ("% of paired cuBLAS for each tile configuration (v7_<BM>x<BN>x<BK>_<TM>x<TN>: "
                "block tile, then thread tile; v8_<BM>x<BN>x<BK>_w<WM>x<WN>_<WMITER>x<WNITER>: "
                "block tile, warp tile, pieces per thread), from "
                f"`results/{self.tune_path.name}`:\n\n"
                + md_table(["config", "threads", "registers", "blocks/SM"]
                           + [f"N={s}" for s in sizes], rows)
                + "\n\nBest configuration per size: " + "; ".join(best) + "." + note)

    def tc_row(self, k, n):
        if self.tc is None:
            raise Missing("tc_sgemm_*.csv")
        r = self.tc[(self.tc.kernel == k) & (self.tc.M == int(n))]
        if r.empty or r.status.iloc[0] == "FAILED":
            raise Missing(f"tc {k} at N={n}")
        return r.iloc[0]

    def t_tc(self):
        if self.tc is None:
            raise Missing("tc_sgemm_*.csv")
        d = self.tc
        sizes = sorted(d.M.unique())
        rows = []
        for k in ["cublas"] + [x for x in d.kernel.unique() if x != "cublas"]:
            cells = ["`cublasGemmEx` (FP16 in, FP32 out)" if k == "cublas" else f"`{k}`"]
            for s in sizes:
                r = d[(d.kernel == k) & (d.M == s)]
                if r.empty:
                    cells.append("")
                elif r.status.iloc[0] == "FAILED":
                    cells.append("FAILED")
                else:
                    r = r.iloc[0]
                    cells.append(f"{r.gflops / 1000:.1f}" + ("" if k == "cublas"
                                                             else f" ({r.pct_cublas:.0f}%)"))
            rows.append(cells)
        prec = ""
        if self.tc_prec is not None:
            prec = ("\n\nPrecision cost of FP16 inputs (not a kernel error): normwise relative "
                    "difference between the FP16-input result and FP32 SGEMM on the same data: "
                    + "; ".join(f"N = {int(r.N)}: {r.max_rel_err_fp16_inputs_vs_fp32_sgemm:.1e}"
                                for _, r in self.tc_prec.iterrows()) + ".")
        return (f"TFLOP/s (and % of `cublasGemmEx`, timed interleaved), from "
                f"`results/{self.tc_path.name}`. Every kernel is checked against `cublasGemmEx` "
                f"with the same FP16 inputs (largest observed error {d.max_rel_err.max():.1e})."
                f"\n\n"
                + md_table(["kernel"] + [f"N={s}" for s in sizes], rows) + prec)

    def t_reduce(self):
        p = newest("reduce_Tesla*.csv") or newest("reduce_*.csv")
        if p is None:
            raise Missing("reduce_*.csv")
        d = pd.read_csv(p)
        bw, spec = self.bw_best()
        rows = []
        for _, r in d.iterrows():
            rows.append([f"`{r.kernel}`" if r.kernel != "cub" else "CUB `DeviceReduce::Sum`",
                         f"{r.median_ms:.3f}", f"{r.gbps:.0f}", f"{100 * r.gbps / spec:.0f}%",
                         "100% (ref)" if r.kernel == "cub" else f"{r.pct_of_cub:.0f}%",
                         f"{r.rel_err:.1e}", r.status])
        return (f"Sum of {int(d.n.iloc[0]):,} floats ({int(d.n.iloc[0]) * 4 // 2**20} MiB), from "
                f"`results/{p.name}`. Bandwidth = input bytes / median time, compared with the "
                f"{spec:.0f} GB/s DRAM spec. (The measured copy bandwidth, {bw:.0f} GB/s, isn't the "
                f"right roof here: a copy reads and writes, while a reduction only reads, and a "
                f"read-only stream gets closer to the spec.)\n\n") + md_table(
            ["kernel", "median ms", "GB/s", "% of DRAM spec", "% of CUB", "relative error",
             "check"], rows)

    def reduce_row(self, k):
        p = newest("reduce_Tesla*.csv") or newest("reduce_*.csv")
        if p is None:
            raise Missing("reduce_*.csv")
        d = pd.read_csv(p)
        r = d[d.kernel == k]
        if r.empty:
            raise Missing(f"reduce {k}")
        return r.iloc[0]

    def epi(self):
        p = newest("epi_Tesla*.csv") or newest("epi_[!e]*.csv")
        if p is None:
            raise Missing("epi_*.csv")
        return pd.read_csv(p), p

    def t_epilogue(self):
        d, p = self.epi()
        has_lt = "lt_status" in d.columns
        rows = []
        for _, r in d.iterrows():
            ok = r.status == "PASS"
            row = [r.act, f"{r.M} × {r.N} × {r.K}", f"{r.unfused_median_ms:.3f}",
                   f"{r.fused_median_ms:.3f}" if ok else "FAILED",
                   f"{r.speedup:.2f}x" if ok else "", f"{r.max_rel_err:.1e}", r.status]
            if has_lt:
                lok = r.lt_status == "PASS"
                row += [f"{r.lt_median_ms:.3f}" if lok else r.lt_status,
                        f"{r.lt_speedup:.2f}x" if lok else "", r.lt_status]
            rows.append(row)
        head = ["activation", "M × N × K", "unfused ms", "f1 fused ms", "f1 speedup",
                "max rel err (f1 vs unfused)", "f1 check"]
        if has_lt:
            head += ["cuBLASLt fused ms", "cuBLASLt speedup", "cuBLASLt check"]
        return ("act(A·B + bias), unfused = `cublasSgemm` followed by a separate bias + activation "
                "kernel, f1 = this repo's fused kernel"
                + (", cuBLASLt = one `cublasLtMatmul` with its built-in bias + activation epilogue"
                   if has_lt else "")
                + "; speedup = unfused time ÷ fused time; median times, timed interleaved, from "
                f"`results/{p.name}`:\n\n" + md_table(head, rows))

    def t_shapes(self):
        p = newest("shapes_sgemm_*.csv")
        if p is None:
            raise Missing("shapes_sgemm_*.csv")
        d = pd.read_csv(p)
        rows = []
        for (m, n, k), g in d.groupby(["M", "N", "K"], sort=False):
            cell = lambda name: (lambda r: "" if r.empty else ("FAILED" if r.status.iloc[0] == "FAILED"
                                  else f"{r.pct_cublas.iloc[0]:.0f}%"))(g[g.kernel == name])
            cub = g[g.kernel == "cublas"]
            rows.append([f"{m} × {n} × {k}", f"{cub.gflops.iloc[0]:,.0f}" if not cub.empty else "",
                         cell("v4_2d_blocktiling"), cell("v7_conflict_free"), cell("v9_autotuned"),
                         cell("v11_splitk"), cell("v12_dispatch"), cell("v15_splitk_det")])
        return (f"M × N × K (M = tokens; N, K = layer widths), cuBLAS GFLOP/s and each kernel's % of "
                f"paired cuBLAS, from `results/{p.name}`:\n\n"
                + md_table(["shape", "cuBLAS GFLOP/s", "v4", "v7", "v9", "v11", "v12", "v15"], rows))

    def tctune(self):
        p = newest("tctune_sgemm_*.csv")
        if p is None:
            raise Missing("tctune_sgemm_*.csv")
        self.tctune_path = p
        return pd.read_csv(p)

    def t_tctune(self):
        d = self.tctune()
        sizes = sorted(d.M.unique())
        failed = sorted(set(d[d.status == "FAILED"].kernel))
        rows = []
        for k, g in d[(d.kernel != "cublas") & ~d.kernel.isin(failed)].groupby("kernel", sort=False):
            cells = [f"`{k}`"]
            for s in sizes:
                r = g[g.M == s]
                cells.append(f"{r.pct_cublas.iloc[0]:.0f}%" if not r.empty else "")
            rows.append(cells)
        note = ("" if not failed else
                "\n\nFailed the check (excluded): " + ", ".join(f"`{k}`" for k in failed))
        return ("% of `cublasGemmEx` (same FP16 inputs, timed interleaved) for each tensor-core tile "
                "configuration (tc5_<BM>x<BN>x<BK>_w<WARPS_M>x<WARPS_N>_b<MINB>), from "
                f"`results/{self.tctune_path.name}`:\n\n"
                + md_table(["config"] + [f"N={s}" for s in sizes], rows) + note)

    def t_locked(self):
        if self.locked is None:
            raise Missing("locked*_sgemm_*.csv")
        d = self.locked
        size = d.M.max()
        rows = []
        for k in ["cublas"] + kernels_in(d):
            r = d[(d.kernel == k) & (d.M == size)]
            if r.empty or r.status.iloc[0] == "FAILED":
                continue
            r = r.iloc[0]
            rows.append([kname(k), f"{r.gflops:,.0f}",
                         "100% (ref)" if k == "cublas" else f"{r.pct_cublas:.1f}%",
                         f"{r.median_ms:.2f}", f"{r.min_ms:.2f} – {r.p90_ms:.2f}",
                         f"{r.sm_clock_mhz_after:.0f}" if "sm_clock_mhz_after" in d else "n/a"])
        return (f"Clock-locked run (`results/{self.locked_path.name}`), N = {size}:\n\n"
                + md_table(["kernel", "GFLOP/s", "% cuBLAS", "median ms", "min – p90 ms",
                            "SM clock after (MHz)"], rows))

    def t_bandwidth(self):
        if self.bw is None:
            raise Missing("bandwidth_*.csv")
        rows = [[f"`{r.method}`", f"{r.gbps_median:.1f}", f"{r.gbps_best:.1f}",
                 f"{100 * r.gbps_median / r.theoretical_gbps:.0f}%"] for _, r in self.bw.iterrows()]
        return (f"Buffer {self.bw.buffer_bytes.iloc[0] // 2**20} MiB in + out (far larger than L2); "
                f"spec {self.bw.theoretical_gbps.iloc[0]:.0f} GB/s:\n\n"
                + md_table(["method", "GB/s (median)", "GB/s (best run)", "% of spec"], rows))

    def t_v0stride(self):
        mets = [("gpu__time_duration.sum", "kernel time (ncu, locked clocks)",
                 lambda v: f"{v / 1e6:.1f} ms"),
                ("dram__bytes_read.sum", "DRAM read", lambda v: f"{v / 1e6:,.0f} MB"),
                ("l1tex__t_sector_hit_rate.pct", "L1 hit rate", lambda v: f"{v:.1f}%"),
                ("lts__t_sector_hit_rate.pct", "L2 hit rate", lambda v: f"{v:.1f}%"),
                ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "global load sectors",
                 lambda v: f"{v / 1e6:,.1f} M")]
        mets += [("l1tex__t_set_accesses_pipe_lsu_mem_global_op_ld.sum", "L1 tag-stage set accesses",
                  lambda v: f"{v / 1e6:,.1f} M"),
                 ("l1tex__t_set_conflicts_pipe_lsu_mem_global_op_ld.sum", "L1 tag-stage set conflicts",
                  lambda v: f"{v / 1e6:,.1f} M")]
        rows = []
        for m, label, fmt in mets:
            cells = [label]
            for n in (1023, 1024):
                val = None
                for key in (n, f"set{n}"):
                    t, _ = self.stride[key]
                    if t is not None and "v0_naive" in t.index and m in t.columns:
                        val = fmt(float(t.loc["v0_naive", m]))
                cells.append(val or "n/a")
            if any(c != "n/a" for c in cells[1:]):
                rows.append(cells)
        if not rows:
            raise Missing("ncu_v0stride_*.csv")
        return md_table(["v0_naive", "N = 1023", "N = 1024"], rows)

    def t_clocks(self):
        df = self.df
        if "sm_clock_mhz_after" not in df.columns or df.sm_clock_mhz_after.isna().all():
            raise Missing("sm_clock_mhz_after column (added after run 3)")
        size = df.M.max()
        lanes = self.peak / (self.env["sm_clock_max_khz"] / 1e6)  # GFLOP/s per GHz at peak
        rows = []
        for k in ["cublas"] + kernels_in(df):
            r = df[(df.kernel == k) & (df.M == size)]
            if r.empty or r.status.iloc[0] == "FAILED":
                continue
            r = r.iloc[0]
            clk = r.sm_clock_mhz_after
            rows.append([kname(k), f"{r.gflops:,.0f}", f"{clk:.0f}", f"{r.power_w_after:.1f}",
                         f"{100 * r.gflops / (lanes * clk / 1000):.0f}%"])
        return (f"Throttling at N = {size}: SM clock and board power sampled by `nvidia-smi` right "
                f"after each kernel's timed runs (one sample, not an average), and GFLOP/s as a "
                f"fraction of the FP32 peak *at that clock*. Max boost clock is "
                f"{self.env['sm_clock_max_khz'] // 1000} MHz; the power limit is "
                f"{nvsmi_row(self.env['nvidia_smi_start']).get('power.limit [W]', 'n/a')}.\n\n"
                + md_table(["kernel", "GFLOP/s", "SM clock after (MHz)", "power after (W)",
                            "% of peak at that clock"], rows))

    def t_variability(self):
        runs = sorted(RESULTS.glob("sgemm_*.csv"))
        n = NCU_N
        rows = []
        for k in ["cublas"] + kernels_in(self.df):
            try:
                lo, hi, cnt = self.variability(k, n)
            except Missing:
                continue
            rows.append([kname(k), cnt, f"{lo:,.0f}", f"{hi:,.0f}", f"{100 * (hi - lo) / lo:.0f}%"])
        return (f"Median GFLOP/s at N = {n} in each of the {len(runs)} committed runs "
                f"(separate Colab sessions):\n\n"
                + md_table(["kernel", "runs", "lowest", "highest", "spread"], rows))

    def t_sources(self):
        files = [self.csv_path, self.csv_path.with_name(self.csv_path.name.replace("sgemm_", "env_", 1)
                                                         ).with_suffix(".json"),
                 self.res_path, self.bw_path, *self.ncu_paths.values(),
                 *(p for _, p in self.stride.values()), self.sass_path, self.tune_path,
                 self.locked_path, self.tc_path, RESULTS / "model_bank_conflicts.csv",
                 newest("reduce_Tesla*.csv"), newest("shapes_sgemm_*.csv"), newest("epi_Tesla*.csv")]
        files = [p for p in files if p is None or p.exists()]
        return "\n".join(f"- `results/{p.name}`" for p in files if p is not None)


TOKEN = re.compile(r"\{\{([^{}]+)\}\}")


def render(data, text):
    missing = []

    def sub(m):
        parts = m.group(1).strip().split(":")
        try:
            if parts[0] == "table":
                return data.table(parts[1])
            return data.scalar(parts[0], parts[1:])
        except Missing as e:
            missing.append(f"{m.group(0)}: {e}")
            return "n/a"

    return TOKEN.sub(sub, text), missing


def main():
    args = parse_args(__doc__)
    data = Data(args.csv)
    missing = []
    for template, out, relink in OUTPUTS:
        text, miss = render(data, template.read_text(encoding="utf-8"))
        if relink:  # links emitted by generated tables are repo-relative
            text = re.sub(r"\]\((?!\.\./)(src|figures|results|scripts|notebooks)/", r"](../\1/", text)
        banner = (f"<!-- GENERATED by scripts/make_tables.py from {template.relative_to(ROOT).as_posix()} "
                  "and results/*.csv. Edit the template, not this file. -->\n\n")
        out.write_text(banner + text, encoding="utf-8")
        missing += [f"{out.name}: {m}" for m in miss]
        print(f"rendered {out.relative_to(ROOT).as_posix()} from {data.csv_path.name}")
    if missing:
        print(f"{len(missing)} token(s) had no data and render as n/a:")
        for m in missing:
            print("  ", m)
        sys.exit(1)


if __name__ == "__main__":
    main()
