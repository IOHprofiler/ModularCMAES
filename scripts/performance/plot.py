"""Figures and tables for PERFORMANCE.md from the output of benchmark.py and threads.py.

    python scripts/performance/plot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Other implementations: one colour each; modcma in a strong colour
IMPLEMENTATIONS = {
    "modcma": ("modcma (C++)", "#d62728", "o"),
    "modcma (Eigen only)": ("modcma (C++, Eigen only, no LAPACK)", "#ff9896", "o"),
    "pycma": ("pycma", "#1f77b4", "s"),
    "cmaes": ("cmaes (Optuna)", "#2ca02c", "^"),
    "evosax": ("evosax (JAX, jit + scan)", "#9467bd", "D"),
    "evotorch": ("EvoTorch (PyTorch)", "#8c564b", "v"),
}

MATRIX = {
    "modcma": "COVARIANCE (default)",
    "modcma (Eigen only)": "COVARIANCE, Eigen only (no LAPACK)",
    "modcma (fast sampling)": "COVARIANCE, fast_sampling",
    "modcma COVARIANCE_NO_EIGV": "COVARIANCE_NO_EIGV",
    "modcma MATRIX": "MATRIX (MA-ES)",
    "modcma CHOLESKY": "CHOLESKY",
    "modcma CMSA": "CMSA",
    "modcma NATURAL_GRADIENT": "NATURAL_GRADIENT (xNES)",
    "modcma SEPARABLE": "SEPARABLE",
    "modcma NONE": "NONE (step size only)",
}
MATRIX_COLORS = ["#d62728", "#ff9896", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b",
                 "#e377c2", "#7f7f7f", "#bcbd22"]


# The modcma variant the others are compared with (the default: LAPACK from d = 120)
BASELINE = "modcma"


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["implementation", "dim"])["us_per_generation"]
    return pd.DataFrame(
        {"median": g.median(), "q1": g.quantile(0.25), "q3": g.quantile(0.75), "n": g.size()}
    ).reset_index()


def line(ax, s, name, label, color, marker, **kw):
    part = s[s.implementation == name].sort_values("dim")
    if part.empty:
        return
    ax.plot(part.dim, part["median"], marker=marker, color=color, label=label, lw=2, ms=6, **kw)
    ax.fill_between(part.dim, part.q1, part.q3, color=color, alpha=0.15, lw=0)


def style(ax, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    dims = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.3)
    ax.set_xlabel("dimension $d$")
    ax.set_title(title, fontsize=12, loc="left")


def plot_implementations(s: pd.DataFrame, out: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    for name, (label, color, marker) in IMPLEMENTATIONS.items():
        line(left, s, name, label, color, marker)
    style(left, "Time per generation (ellipsoid, default $\\lambda$, one thread)")
    left.set_ylabel("µs per generation (median, IQR shaded)")
    left.legend(frameon=False)

    base = s[s.implementation == BASELINE].set_index("dim")["median"]
    for name, (label, color, marker) in IMPLEMENTATIONS.items():
        if name.startswith("modcma"):
            continue
        other = s[s.implementation == name].set_index("dim")["median"]
        ratio = (other / base).dropna()
        right.plot(ratio.index, ratio.values, marker=marker, color=color, label=label, lw=2, ms=6)
    right.axhline(1.0, color="#d62728", lw=1.5, ls="--", label="modcma")
    style(right, "How many times slower than modcma")
    right.set_ylabel("time per generation / modcma")
    right.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_matrix(s: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (name, label), color in zip(MATRIX.items(), MATRIX_COLORS):
        line(ax, s, name, label, color, "o", ls="-" if name == "modcma" else "--")
    style(ax, "modcma matrix adaptations: time per generation")
    ax.set_ylabel("µs per generation (median, IQR shaded)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_threads(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(df.threads, df.speedup, marker="o", color="#d62728", lw=2, label="modcma, free-threaded 3.14t")
    ax.plot(df.threads, df.threads, color="grey", ls=":", label="linear")
    ax.set_xlabel("threads (one optimizer each)")
    ax.set_ylabel("throughput / one thread")
    ax.set_title("Optimizers in parallel threads, no GIL", fontsize=12, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fmt(us: float) -> str:
    if not np.isfinite(us):
        return "–"
    if us >= 1e6:
        return f"{us / 1e6:.2f} s"
    if us >= 1e3:
        return f"{us / 1e3:.1f} ms"
    return f"{us:.1f} µs" if us < 10 else f"{us:.0f} µs"


def table(s: pd.DataFrame, names: dict, dims) -> str:
    rows = ["| " + " | ".join(["implementation"] + [f"d={d}" for d in dims]) + " |",
            "|---" * (len(dims) + 1) + "|"]
    for name, label in names.items():
        part = s[s.implementation == name].set_index("dim")["median"]
        if part.empty:
            continue
        cells = [fmt(part.get(d, np.nan)) for d in dims]
        if name == BASELINE:
            cells = [f"**{c}**" for c in cells]
        rows.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(rows)


def speedup_table(s: pd.DataFrame, dims) -> str:
    base = s[s.implementation == BASELINE].set_index("dim")["median"]
    rows = ["| time / time of modcma | " + " | ".join(f"d={d}" for d in dims) + " |",
            "|---" * (len(dims) + 1) + "|"]
    for name, (label, *_rest) in IMPLEMENTATIONS.items():
        if name.startswith("modcma"):
            continue
        other = s[s.implementation == name].set_index("dim")["median"]
        cells = []
        for d in dims:
            r = other.get(d, np.nan) / base.get(d, np.nan)
            cells.append("–" if not np.isfinite(r) else (f"{r:.0f}×" if r >= 10 else f"{r:.1f}×"))
        rows.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--timing", default=str(HERE / "results" / "timing.csv"))
    parser.add_argument("--threads", default=str(HERE / "results" / "threads.csv"))
    parser.add_argument("--figures", default=str(HERE / "figures"))
    args = parser.parse_args()

    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    s = summary(pd.read_csv(args.timing))
    plot_implementations(s, figures / "implementations.png")
    plot_matrix(s, figures / "matrix_adaptations.png")

    dims = [2, 10, 50, 100, 200, 500, 1000]
    tables = ["## Implementations (µs per generation, median)", "",
              table(s, {k: v[0] for k, v in IMPLEMENTATIONS.items()}, dims), "",
              "## Speed-up", "", speedup_table(s, dims), "",
              "## Matrix adaptations (µs per generation, median)", "",
              table(s, MATRIX, dims), "",
              "## Python objective: vectorized vs per point (µs per generation, median)", "",
              table(s, {"modcma": "vectorized (one call per generation)",
                        "modcma (per-point callback)": "one call per point"}, dims), ""]
    if Path(args.threads).exists():
        threads = pd.read_csv(args.threads)
        plot_threads(threads, figures / "threads.png")
        rows = ["| threads | generations/s | speed-up |", "|---|---|---|"]
        rows += [f"| {r.threads} | {r.generations_per_second:,.0f} | {r.speedup:.2f}× |"
                 for r in threads.itertuples()]
        tables += ["## Threads", "", "\n".join(rows), ""]
    (HERE / "results" / "tables.md").write_text("\n".join(tables))
    print("\n".join(tables))


if __name__ == "__main__":
    main()
