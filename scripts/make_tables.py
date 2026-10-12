"""Renders README.md from docs/README.template.md, filling every number from results/.

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
                        load_run, nvsmi_row, parse_args, parse_ncu, newest, peak_fp32_gflops)

TEMPLATE = ROOT / "docs" / "README.template.md"
README = ROOT / "README.md"
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
        self.peak = peak_fp32_gflops(self.env)

    # ------------------------------------------------------------- lookups
    def row(self, k, n):
        r = self.df[(self.df.kernel == k) & (self.df.M == int(n))]
        if r.empty:
            raise Missing(f"{k} at N={n} not in {self.csv_path.name}")
        r = r.iloc[0]
        if r.status == "FAILED":
            raise Missing(f"{k} at N={n} FAILED the check")
        return r

    def ncu_val(self, kind, metric, k):
        t = self.ncu.get(kind)
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
            t, _ = self.stride[int(args[1])]
            if t is None or "v0_naive" not in t.index or args[0] not in t.columns:
                raise Missing(f"v0 stride {args[0]} at {args[1]}")
            v = float(t.loc["v0_naive", args[0]])
            return f"{v / 1e6:,.0f} MB" if "bytes" in args[0] else (
                f"{v:.1f}%" if args[0].endswith(".pct") else f"{v:,.4g}")
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
            if k not in self.ncu["dram"].index:
                continue
            cells = [kname(k)]
            for tok in ("dram_mb", "traffic_x", "ai", "sectors_per_req", "l1hit", "l2hit",
                        "lds_per_ffma", "bank_conflicts"):
                try:
                    cells.append(self.scalar(tok, [k]))
                except Missing:
                    cells.append("n/a")
            rows.append(cells)
        return (f"Nsight Compute counters at N = {NCU_N} (last of 2 profiled launches per kernel):\n\n"
                + md_table(["kernel", "DRAM MB moved", "vs minimum", "FLOP per DRAM byte",
                            "global sectors per load request", "L1 hit", "L2 hit",
                            "shared loads per FFMA (warp-level)", "shared load bank conflicts"],
                           rows))

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
        rows = []
        for m, label, fmt in mets:
            cells = [label]
            for n in (1023, 1024):
                t, _ = self.stride[n]
                cells.append(fmt(float(t.loc["v0_naive", m]))
                             if t is not None and "v0_naive" in t.index and m in t.columns
                             else "n/a")
            rows.append(cells)
        if all(c == "n/a" for r in rows for c in r[1:]):
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
                 *(p for _, p in self.stride.values())]
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
    text, missing = render(data, TEMPLATE.read_text(encoding="utf-8"))
    banner = ("<!-- GENERATED by scripts/make_tables.py from docs/README.template.md and "
              "results/*.csv. Edit the template, not this file. -->\n\n")
    README.write_text(banner + text, encoding="utf-8")
    print(f"rendered README.md from {data.csv_path.name}")
    if missing:
        print(f"{len(missing)} token(s) had no data and render as n/a:")
        for m in missing:
            print("  ", m)
        sys.exit(1)


if __name__ == "__main__":
    main()
