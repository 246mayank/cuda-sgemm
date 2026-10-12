"""Figures from one benchmark run (see results_io.py for which run).

  figures/gflops_vs_size.png   GFLOP/s vs square size (powers of two), one line per kernel,
                               cuBLAS as a dashed reference
  figures/pct_cublas.png       % of cuBLAS at the largest size, one bar per kernel
  figures/roofline.png         roofline at N = 2048: measured DRAM roof (bandwidth.cu),
                               theoretical FP32 roof, cuBLAS as the practical ceiling, each
                               kernel placed at its MEASURED arithmetic intensity (Nsight
                               Compute DRAM bytes) and its benchmark GFLOP/s. Skipped, with a
                               message, if the ncu or bandwidth results are missing.

A kernel that FAILED the cuBLAS check at any size is left out of every figure, and the
failing sizes are listed in the figure note.
Usage: .venv/Scripts/python scripts/plot.py [--csv results/sgemm_....csv]
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results_io import (FIGURES, KERNEL_ORDER, kernels_in, load_bandwidth, load_ncu, load_run,
                        parse_args, peak_fp32_gflops)

# Reference categorical palette (light mode), fixed slot per kernel in KERNEL_ORDER.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
COLOR = dict(zip(KERNEL_ORDER, SLOTS))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REF = "#52514e"  # cuBLAS reference line: neutral ink, not a category color

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False,
})


def passing(df):
    """Rows of kernels that passed at every size they ran, plus cuBLAS."""
    bad = set(df[df.status == "FAILED"].kernel)
    return df[~df.kernel.isin(bad)]


def failure_note(fig, df):
    f = df[df.status == "FAILED"]
    if f.empty:
        return
    parts = [f"{k} (N = {', '.join(str(m) for m in sorted(g.M))})" for k, g in f.groupby("kernel")]
    fig.text(0.01, 0.01, "Not plotted, failed the cuBLAS check: " + "; ".join(parts),
             fontsize=8, color=INK2)


def subtitle(env, csv_path):
    return (f"{env['gpu_name']}, {env['sm_count']} SMs, CUDA {env['nvcc_version']}, "
            f"{env['date_utc'][:10]}; median of {env['timed_runs']} runs; "
            f"source: {csv_path.name}")


def gflops_vs_size(df, env, csv_path):
    ok = passing(df)
    sizes = sorted(s for s in ok.M.unique() if s & (s - 1) == 0)  # powers of two only
    fig, ax = plt.subplots(figsize=(8, 5))

    ref = ok[(ok.kernel == "cublas") & ok.M.isin(sizes)].sort_values("M")
    ax.plot(ref.M, ref.gflops, color=REF, lw=2, ls="--", marker="o", ms=5, label="cuBLAS")
    ax.annotate("cuBLAS", (ref.M.iloc[-1], ref.gflops.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK2)

    for k in kernels_in(ok):
        d = ok[(ok.kernel == k) & ok.M.isin(sizes)].sort_values("M")
        if d.empty:
            continue
        ax.plot(d.M, d.gflops, color=COLOR.get(k, INK2), lw=2, marker="o", ms=5, label=k)
        ax.annotate(k, (d.M.iloc[-1], d.gflops.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlim(sizes[0] / 1.2, sizes[-1] * 2.2)  # room for end labels
    ax.set_xlabel("matrix size N (M = N = K)")
    ax.set_ylabel("GFLOP/s (log scale)")
    ax.set_title("SGEMM throughput vs size", loc="left", fontsize=13, color=INK, pad=18)
    ax.text(0, 1.01, subtitle(env, csv_path), transform=ax.transAxes, fontsize=8, color=INK2)
    ax.legend(loc="upper left", fontsize=9)
    failure_note(fig, df)
    fig.tight_layout()
    out = FIGURES / "gflops_vs_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def pct_cublas(df, env, csv_path):
    size = df.M.max()
    d = passing(df)
    d = d[(d.M == size) & (d.kernel != "cublas")]
    ks = kernels_in(d)
    vals = [float(d[d.kernel == k].pct_cublas.iloc[0]) for k in ks]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(ks) + 1.6))
    y = range(len(ks))[::-1]
    ax.barh(list(y), vals, height=0.6, color=SLOTS[0])
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.1f}%", va="center", color=INK)
    ax.set_yticks(list(y), ks)
    ax.set_xlim(0, max(100, max(vals) * 1.15) if vals else 100)
    ax.set_xlabel(f"% of cuBLAS GFLOP/s at {size}x{size}x{size}")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Fraction of cuBLAS at N = {size}", loc="left", fontsize=13, color=INK, pad=18)
    ax.text(0, 1.02, subtitle(env, csv_path), transform=ax.transAxes, fontsize=8, color=INK2)
    failure_note(fig, df)
    fig.tight_layout()
    out = FIGURES / "pct_cublas.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


ROOFLINE_N = 2048  # size profiled by the ncu cells in the notebook


def roofline(df, env, csv_path):
    ncu, ncu_path = load_ncu("dram")
    bw, bw_path = load_bandwidth()
    peak = peak_fp32_gflops(env)
    if ncu is None or bw is None or peak is None:
        missing = [n for n, v in (("ncu_dram_*.csv", ncu), ("bandwidth_*.csv", bw),
                                  ("FP32 lanes for this GPU", peak)) if v is None]
        print("roofline skipped, missing:", ", ".join(missing))
        return None

    n = ROOFLINE_N
    flop = 2.0 * n ** 3
    ok = passing(df)
    at_n = ok[ok.M == n]
    mem_roof = bw.gbps_median.max()                  # best measured method, median of its runs
    theo_bw = bw.theoretical_gbps.iloc[0]
    paired = at_n.cublas_paired_median_ms.dropna() if "cublas_paired_median_ms" in at_n else []
    cublas = (flop / (paired.median() * 1e6) if len(paired)
              else at_n[at_n.kernel == "cublas"].gflops.iloc[0])
    ideal_ai = flop / (3 * n * n * 4)                # A, B read once and C written once

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = [0.05, 2000]
    import numpy as np
    xs = np.logspace(np.log10(x[0]), np.log10(x[1]), 200)
    ax.plot(xs, np.minimum(xs * mem_roof, peak), color=INK, lw=2)
    ax.plot(xs, np.minimum(xs * theo_bw, peak), color=INK2, lw=1, ls=":")
    ax.axhline(cublas, color=REF, lw=1.5, ls="--")
    ax.axvline(ideal_ai, color=GRID, lw=1.5)
    ax.text(x[1], peak * 1.08, f"FP32 peak {peak:,.0f} GFLOP/s (theoretical)  ", ha="right",
            va="bottom", color=INK2, fontsize=8)
    ax.text(x[1], cublas * 0.92, f"cuBLAS {cublas:,.0f} GFLOP/s at N = {n} (practical)  ",
            ha="right", va="top", color=INK2, fontsize=8)
    ax.text(0.3, 0.3 * mem_roof * 0.55, f"measured DRAM roof\n{mem_roof:.0f} GB/s", ha="left",
            va="top", color=INK, fontsize=8)
    ax.text(0.07, 0.07 * theo_bw * 2.4, f"spec {theo_bw:.0f} GB/s (dotted)", ha="left",
            va="bottom", color=INK2, fontsize=8)
    ax.text(ideal_ai * 0.93, 14, "minimum DRAM traffic\n(each matrix moved once)", ha="right",
            va="bottom", color=INK2, fontsize=8)

    placed = []
    for k in kernels_in(at_n):
        if k not in ncu.index:
            continue
        r = ncu.loc[k]
        moved = r.get("dram__bytes_read.sum", np.nan) + r.get("dram__bytes_write.sum", np.nan)
        if not moved > 0:
            continue
        ai = flop / moved
        g = float(at_n[at_n.kernel == k].gflops.iloc[0])
        ax.scatter([ai], [g], s=60, color=COLOR.get(k, INK2), edgecolor=SURFACE, lw=2, zorder=3,
                   label=k)
        ax.annotate(k, (ai, g), xytext=(7, -3), textcoords="offset points", color=INK2,
                    fontsize=8)
        placed.append(k)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x)
    ax.set_ylim(10, peak * 3)
    ax.set_xlabel("arithmetic intensity (FLOP per DRAM byte, measured with Nsight Compute)")
    ax.set_ylabel("GFLOP/s (log scale)")
    ax.set_title(f"Roofline, N = {n}", loc="left", fontsize=13, color=INK, pad=30)
    ax.text(0, 1.01, f"{subtitle(env, csv_path)}\nDRAM bytes: {ncu_path.name}; "
            f"bandwidth: {bw_path.name}", transform=ax.transAxes, fontsize=7, color=INK2)
    if placed:
        ax.legend(loc="upper left", fontsize=8)
    failure_note(fig, df)
    fig.tight_layout()
    out = FIGURES / "roofline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    args = parse_args(__doc__)
    df, env, csv_path = load_run(args.csv)
    FIGURES.mkdir(exist_ok=True)
    for out in (gflops_vs_size(df, env, csv_path), pct_cublas(df, env, csv_path),
                roofline(df, env, csv_path)):
        if out:
            print("wrote", out.relative_to(FIGURES.parent))


if __name__ == "__main__":
    main()
