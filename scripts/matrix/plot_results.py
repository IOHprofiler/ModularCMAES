"""Create the matrix-adaptation figures used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from iohinspector import DataManager

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]

METHODS = {
    "CHOLESKY": ("Cholesky", "#4C72B0"),
    "CMSA": ("CMSA", "#55A868"),
    "COVARIANCE": ("Covariance", "#C44E52"),
    "MATRIX": ("Matrix", "#8172B3"),
    "NATURAL_GRADIENT": ("Natural Gradient", "#CCB974"),
    "NONE": ("None", "#64B5CD"),
    # The enum was historically exposed with this spelling.
    "SEPERABLE": ("Separable", "#937860"),
    "SEPARABLE": ("Separable", "#937860"),
    "pycma": ("Pycma", "#000000"),
}


def plot_runtime(
    modcma_csv: Path,
    pycma_csv: Path,
    output: Path,
) -> None:
    """Plot median runtime and median time per adaptation update."""
    stats = pd.concat(
        [pd.read_csv(modcma_csv), pd.read_csv(pycma_csv)],
        ignore_index=True,
    )

    figure, (total_ax, update_ax) = plt.subplots(1, 2, figsize=(13, 6))

    for method, group in stats.groupby("method", sort=True):
        if method not in METHODS:
            print(f"Skipping unknown method in timing data: {method}")
            continue

        label, color = METHODS[method]
        by_dim = group.groupby("dim")
        summary = by_dim["time"].agg(
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        updated = group[group["n_updates"] > 0]
        per_update = (
            (updated["time"] / updated["n_updates"]).groupby(updated["dim"]).median()
        )
        marker = "o" if method == "pycma" else "^"

        total_ax.errorbar(
            summary.index,
            summary["median"],
            yerr=np.vstack(
                (
                    summary["median"] - summary["q1"],
                    summary["q3"] - summary["median"],
                )
            ),
            label=label,
            marker=marker,
            markersize=13,
            linestyle="dashed",
            alpha=0.8,
            linewidth=2,
            color=color,
        )
        if not per_update.empty:
            update_ax.plot(
                per_update.index,
                per_update,
                label=label,
                marker=marker,
                markersize=13,
                linestyle="dashed",
                alpha=0.8,
                linewidth=2,
                color=color,
            )

    for axis in (total_ax, update_ax):
        axis.grid(which="major")
        axis.set_yscale("log", base=10)
        axis.set_xscale("log", base=10)
        axis.tick_params(axis="both", which="major", labelsize=16)
        axis.legend(fontsize=16, ncol=1, fancybox=True, shadow=True)
        axis.set_xlabel(r"dimensionality $d$", fontsize=22)

    total_ax.set_ylabel("Time total [s]", fontsize=22)
    update_ax.set_ylabel("Time per update [s]", fontsize=22)
    figure.tight_layout()
    figure.savefig(output, dpi=500)
    plt.close(figure)


def ert(runs: pl.DataFrame, target: float) -> float:
    """Compute fixed-target ERT from an IOHinspector overview selection."""
    total_evals = 0
    n_successes = 0
    for row in runs.iter_rows(named=True):
        total_evals += row["evals"]
        n_successes += row["best_y"] <= target
    return float("inf") if n_successes == 0 else total_evals / n_successes


def load_complete_overview(data_dir: Path, repetitions: int) -> pl.DataFrame:
    """Load IOH data and retain complete method/function/dimension groups."""
    manager = DataManager()
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir():
            manager.add_folder(str(folder))

    keys = ["algorithm_name", "function_id", "dimension"]
    completed = (
        manager.overview.group_by(keys).len().filter(pl.col("len") == repetitions)
    )
    return manager.overview.join(completed, on=keys, how="inner")


def plot_bbob(
    data_dir: Path,
    output: Path,
    dims: tuple[int, ...],
    repetitions: int,
    target: float,
) -> None:
    """Plot ERT/dimension for all 24 BBOB functions."""
    overview = load_complete_overview(data_dir, repetitions)
    available = set(overview["algorithm_name"].unique().to_list())

    figure, axes = plt.subplots(5, 5, figsize=(14, 15), sharex="col")
    axes = axes.ravel()

    for fid, axis in zip(range(1, 25), axes):
        function_data = overview.filter(pl.col("function_id") == fid)
        plotted = False

        for method, (label, color) in METHODS.items():
            if method not in available:
                continue
            method_data = function_data.filter(pl.col("algorithm_name") == method)
            values = np.array(
                [
                    ert(
                        method_data.filter(pl.col("dimension") == dim),
                        target,
                    )
                    / dim
                    for dim in dims
                ]
            )
            mask = np.isfinite(values)
            if not mask.any():
                continue
            marker = "o" if method == "pycma" else "^"
            axis.plot(
                np.asarray(dims)[mask],
                values[mask],
                label=label,
                marker=marker,
                markersize=12,
                linestyle="dashed",
                alpha=0.8,
                linewidth=2,
                color=color,
            )
            plotted = True

        axis.grid(which="both", axis="x")
        axis.grid(which="major", axis="y")
        axis.set_xscale("log", base=2)
        axis.set_xticks(dims, dims)
        axis.tick_params(axis="both", which="both", labelsize=12)
        if plotted:
            axis.set_yscale("log", base=10)
        else:
            axis.set_ylim(0, 1)
            axis.text(
                0.5,
                0.5,
                "No finite ERT",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        function_name = (
            function_data["function_name"][0] if len(function_data) else "missing data"
        )
        axis.text(
            0.01,
            0.99,
            f"$f_{{{fid}}}$ ({function_name})",
            transform=axis.transAxes,
            bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.5},
            ha="left",
            va="top",
            fontsize=10,
        )

        if fid == 11:
            axis.set_ylabel("Expected Running Time (ERT) / $d$", fontsize=16)
        if fid == 23:
            axis.set_xlabel(r"Dimensionality $d$", fontsize=16)

    axes[24].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[24].legend(
        handles,
        labels,
        loc="center",
        fancybox=True,
        shadow=True,
        fontsize=11,
    )
    figure.subplots_adjust(hspace=0.05, wspace=0.2)
    figure.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create runtime and BBOB matrix-adaptation figures."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
    )
    parser.add_argument(
        "--timing-modcma",
        type=Path,
        default=SCRIPT_DIR / "time_stats.csv",
    )
    parser.add_argument(
        "--timing-pycma",
        type=Path,
        default=SCRIPT_DIR / "time_stats_pycma.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--dims", nargs="+", type=int, default=(2, 3, 5, 10, 20, 40))
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--target", type=float, default=1e-8)
    parser.add_argument(
        "--figure",
        choices=("all", "runtime", "bbob"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.figure in ("all", "runtime"):
        plot_runtime(
            args.timing_modcma.resolve(),
            args.timing_pycma.resolve(),
            output_dir / "time_matrix_adaptation.png",
        )

    if args.figure in ("all", "bbob"):
        plot_bbob(
            args.data_dir.resolve(),
            output_dir / "bbob_matrix_adaptation.png",
            tuple(args.dims),
            args.repetitions,
            args.target,
        )


if __name__ == "__main__":
    main()
