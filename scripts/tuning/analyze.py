"""Validate SMAC configurations and create the ERT and SHAP figures."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import ioh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostRegressor

from modcma import c_maes

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_COLUMNS = [
    "active",
    "elitist",
    "matrix_adaptation",
    "mirrored",
    "orthogonal",
    "repelling_type",
    "restart_strategy",
    "sample_transformation",
    "sampler",
    "sequential_selection",
    "ssa",
    "threshold_convergence",
    "weights",
]
MODULE_DEFAULTS = {
    "active": False,
    "elitist": False,
    "matrix_adaptation": "COVARIANCE",
    "mirrored": "NONE",
    "orthogonal": False,
    "repelling_type": "NONE",
    "restart_strategy": "NONE",
    "sample_transformation": "GAUSSIAN",
    "sampler": "UNIFORM",
    "sequential_selection": False,
    "ssa": "CSA",
    "threshold_convergence": False,
    "weights": "DEFAULT",
}


def find_runhistory(data_dir: Path, fid: int, dim: int) -> Path:
    """Resolve exactly one paper-configuration run history."""
    function_dir = data_dir / f"BBOB_F{fid}_{dim}D_LRFalseTrue"
    matches = sorted(function_dir.glob("*/*/runhistory.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one runhistory below {function_dir}, found {len(matches)}. "
            "Keep each reproduction run in a separate data directory."
        )
    return matches[0]


def load_runhistories(
    data_dir: Path,
    dim: int,
) -> dict[int, dict]:
    """Load one SMAC run history for every BBOB function."""
    runs = {}
    for fid in range(1, 25):
        filename = find_runhistory(data_dir, fid, dim)
        with filename.open(encoding="utf-8") as handle:
            runs[fid] = json.load(handle)
    return runs


def configuration_costs(run_data: dict) -> dict[str, list[float]]:
    """Collect finite SMAC costs by configuration ID."""
    costs = defaultdict(list)
    for record in run_data["data"]:
        cost = float(record["cost"])
        if np.isfinite(cost):
            costs[str(record["config_id"])].append(cost)
    return dict(costs)


def top_configurations(
    run_data: dict,
    min_runs: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Return the best sufficiently evaluated configurations by mean cost."""
    ranked = [
        (config_id, float(np.mean(costs)))
        for config_id, costs in configuration_costs(run_data).items()
        if len(costs) >= min_runs
    ]
    ranked.sort(key=lambda item: item[1])
    if len(ranked) < top_k:
        raise RuntimeError(
            f"Only {len(ranked)} configurations have at least {min_runs} runs; "
            f"cannot select the requested top {top_k}."
        )
    return ranked[:top_k]


def evaluate_configuration(
    config: dict,
    fid: int,
    dim: int,
    instance: int,
    seed: int,
    target: float,
    budget_per_dimension: int,
) -> tuple[int, bool]:
    """Evaluate one configuration on one held-out BBOB instance."""
    run_seed = seed + instance
    np.random.seed(run_seed)
    c_maes.utils.set_seed(run_seed)

    problem = ioh.get_problem(fid, instance, dim)
    settings = c_maes.settings_from_dict(
        dim,
        **config,
        lb=problem.bounds.lb,
        ub=problem.bounds.ub,
    )
    settings.modules.center_placement = c_maes.options.CenterPlacement.UNIFORM
    settings.budget = dim * budget_per_dimension
    settings.target = problem.optimum.y + target

    optimizer = c_maes.ModularCMAES(settings)
    optimizer.run(problem)
    return problem.state.evaluations, bool(problem.state.final_target_found)


def fixed_target_ert(results: list[tuple[int, bool]]) -> tuple[float, int]:
    """Return ERT and success count for evaluation-count/success pairs."""
    successes = sum(success for _, success in results)
    total_evaluations = sum(evaluations for evaluations, _ in results)
    ert = float("inf") if successes == 0 else total_evaluations / successes
    return ert, successes


def validate_candidates(
    runhistories: dict[int, dict],
    dim: int,
    min_runs: int,
    top_k: int,
    validation_start: int,
    validation_runs: int,
    seed: int,
    target: float,
    budget_per_dimension: int,
) -> pd.DataFrame:
    """Validate the top SMAC candidates and select one per function."""
    selected_rows = []

    for fid in range(1, 25):
        run_data = runhistories[fid]
        candidates = top_configurations(run_data, min_runs, top_k)
        validations = []

        for config_id, mean_cost in candidates:
            config = run_data["configs"][config_id].copy()
            results = [
                evaluate_configuration(
                    config,
                    fid,
                    dim,
                    instance,
                    seed,
                    target,
                    budget_per_dimension,
                )
                for instance in range(
                    validation_start,
                    validation_start + validation_runs,
                )
            ]
            ert, successes = fixed_target_ert(results)
            validations.append(
                {
                    "fid": fid,
                    "cid": config_id,
                    "ert": ert,
                    "sr": successes,
                    "mean_cost": mean_cost,
                    **config,
                }
            )
            print(
                f"f{fid}, config {config_id}: "
                f"ERT={ert:.3f}, successes={successes}/{validation_runs}",
                flush=True,
            )

        selected_rows.append(min(validations, key=lambda row: row["ert"]))

    return pd.DataFrame(selected_rows)


def evaluate_defaults(
    dim: int,
    validation_start: int,
    validation_runs: int,
    seed: int,
    target: float,
    budget_per_dimension: int,
) -> list[float]:
    """Compute the default ModCMA ERT row on the validation instances."""
    values = []

    for fid in range(1, 25):
        results = []
        for instance in range(
            validation_start,
            validation_start + validation_runs,
        ):
            run_seed = seed + instance
            np.random.seed(run_seed)
            c_maes.utils.set_seed(run_seed)

            problem = ioh.get_problem(fid, instance, dim)
            modules = c_maes.parameters.Modules()
            modules.center_placement = c_maes.options.CenterPlacement.UNIFORM
            settings = c_maes.Settings(
                dim,
                modules=modules,
                budget=dim * budget_per_dimension,
                target=problem.optimum.y + target,
                lb=problem.bounds.lb,
                ub=problem.bounds.ub,
            )
            optimizer = c_maes.ModularCMAES(settings)
            optimizer.run(problem)
            results.append(
                (
                    problem.state.evaluations,
                    bool(problem.state.final_target_found),
                )
            )

        ert, successes = fixed_target_ert(results)
        values.append(ert)
        print(
            f"default f{fid}: ERT={ert:.3f}, "
            f"successes={successes}/{validation_runs}",
            flush=True,
        )

    return values


def flatten_runs(runhistories: dict[int, dict]) -> pd.DataFrame:
    """Convert SMAC run histories to configuration-performance rows."""
    rows = []
    for fid, run_data in sorted(runhistories.items()):
        for record in run_data["data"]:
            config_id = str(record["config_id"])
            row = dict(record.get("additional_info", {}))
            row.update(
                {
                    "fid": fid,
                    "cid": config_id,
                    "cost": float(record["cost"]),
                    "time": record["time"],
                }
            )
            row.update(run_data["configs"][config_id])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmap(
    data: pd.DataFrame,
    output: Path,
    title: str,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    formatter,
    minimize: bool,
    font_size: int = 10,
) -> None:
    """Create an annotated heatmap with bold per-column optima."""
    height = 15 if len(data) > 10 else max(2.5, len(data) * 0.8)
    figure, axis = plt.subplots(figsize=(13, height))
    image = axis.imshow(
        data.values.astype(float),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    axis.set_xticks(np.arange(data.shape[1]), data.columns, fontsize=14)
    axis.set_yticks(np.arange(data.shape[0]), data.index, fontsize=14)

    for row_index in range(data.shape[0]):
        for column_index in range(data.shape[1]):
            column = data.columns[column_index]
            value = float(data.iloc[row_index, column_index])
            optimum = data[column].min() if minimize else data[column].max()
            weight = "bold" if value == optimum else "normal"
            red, green, blue, _ = image.cmap(image.norm(value))
            brightness = 0.299 * red + 0.587 * green + 0.114 * blue
            color = "white" if brightness < 0.5 and np.isfinite(value) else "black"
            axis.text(
                column_index,
                row_index,
                formatter(value),
                ha="center",
                va="center",
                color=color,
                fontdict={"size": font_size, "weight": weight},
            )

    axis.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_title(title, fontsize=15)

    color_axis = axis.inset_axes([0.74, 1.01, 0.22, 0.04])
    colorbar = figure.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_ticks([])

    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def ert_formatter(value: float, budget: int) -> str:
    if np.isinf(value):
        return r"$\infty$"
    if value > budget:
        return f">{budget // 1000}k"
    if value > 1_000:
        return f"{value / 1_000:.1f}k"
    return str(int(value))


def create_ert_figures(
    selected: pd.DataFrame,
    default_ert: list[float],
    reference_csv: Path,
    output_dir: Path,
    dim: int,
    budget_per_dimension: int,
) -> None:
    """Create the main-text and appendix ERT figures."""
    reference = pd.read_csv(reference_csv).set_index("algorithm_name")
    reference = reference.apply(pd.to_numeric, errors="coerce")
    reference.columns = reference.columns.astype(str)

    function_columns = [str(fid) for fid in range(1, 25)]
    tuned = selected.set_index("fid").reindex(range(1, 25))["ert"].to_numpy()
    summary = pd.DataFrame(
        [default_ert, tuned],
        columns=function_columns,
        index=["Default", "Tuned"],
    )
    summary.loc["bbob2009"] = reference.min(axis=0).reindex(function_columns)

    full = pd.concat([summary.iloc[:2], reference])
    budget = dim * budget_per_dimension
    formatter = lambda value: ert_formatter(value, budget)

    plot_heatmap(
        summary,
        output_dir / "ert_heatmap_small.pdf",
        "ERT per function",
        vmin=0,
        vmax=budget,
        cmap="inferno_r",
        formatter=formatter,
        minimize=True,
    )
    plot_heatmap(
        full,
        output_dir / "ert_heatmap_all.pdf",
        "ERT per function",
        vmin=0,
        vmax=budget,
        cmap="inferno_r",
        formatter=formatter,
        minimize=True,
    )


def readable_option(value: str) -> str:
    replacements = {
        "True": "On",
        "False": "Off",
        "NATURAL_GRADIENT": "Nat. Grad.",
        "DOUBLE_WEIBULL": "dWeibull",
    }
    if value in replacements:
        return replacements[value]
    if len(value) < 6 and value not in {"NONE", "SOBOL", "EQUAL"}:
        return value
    return value.replace("_", " ").title().replace("Scaled", "")


def compute_shap(
    configurations: pd.DataFrame,
    output_dir: Path,
    iterations: int,
    seed: int,
) -> None:
    """Fit per-function CatBoost models and create SHAP figures."""
    categorical = [
        column for column in MODULE_COLUMNS if column in configurations.columns
    ]
    data = configurations.copy()
    for column in categorical:
        data[column] = data[column].fillna(MODULE_DEFAULTS[column]).astype(str)

    long_tables = []
    for fid in range(1, 25):
        function_data = data.query("fid == @fid")
        function_data = function_data[np.isfinite(function_data["cost"])]
        features = function_data[categorical]
        costs = function_data["cost"].to_numpy()

        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.03,
            depth=10,
            loss_function="RMSE",
            l2_leaf_reg=10,
            random_strength=2,
            bagging_temperature=1,
            random_seed=seed,
            verbose=False,
        )
        model.fit(features, costs, cat_features=categorical)
        shap_values = shap.TreeExplainer(model).shap_values(features)
        for index, module in enumerate(categorical):
            long_tables.append(
                pd.DataFrame(
                    {
                        "fid": fid,
                        "module": module,
                        "option": features[module].to_numpy(),
                        "shap": shap_values[:, index],
                    }
                )
            )
        print(f"SHAP f{fid} complete", flush=True)

    shap_data = pd.concat(long_tables, ignore_index=True)
    global_data = (
        shap_data.groupby(["module", "option"], as_index=False)
        .agg(shap=("shap", "median"))
        .sort_values(["module", "shap"], ascending=[True, False])
    )
    global_data["key"] = (
        global_data["module"].astype(str) + "___" + global_data["option"].astype(str)
    )

    figure, axis = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=global_data,
        x="key",
        y="shap",
        hue="module",
        ax=axis,
        legend=False,
    )
    axis.set_xticks(
        axis.get_xticks(),
        [readable_option(value) for value in global_data["option"].astype(str)],
        fontsize=9,
        rotation=90,
    )
    axis.grid(axis="y")
    axis.set_xlabel(None)
    axis.set_ylabel("Median SHAP value")

    maximum = global_data["shap"].max()
    minimum = global_data["shap"].min()
    margin = max(abs(maximum), abs(minimum)) * 0.15
    axis.set_ylim(minimum - margin, maximum + margin)
    start = 0
    groups = global_data.groupby("module", sort=False).size()
    for group_index, (module, size) in enumerate(groups.items()):
        end = start + size - 1
        center = (start + end) / 2
        label = (
            "Step-size\nAdaptation"
            if module == "ssa"
            else module.replace("_", "\n").title()
        )
        vertical_position = maximum if group_index % 2 == 0 else minimum
        axis.text(
            center,
            vertical_position,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none"},
        )
        if group_index < len(groups) - 1:
            axis.axvline(end + 0.5, color="black", linewidth=3, alpha=0.25)
        start += size

    figure.tight_layout()
    figure.savefig(output_dir / "median_shap.pdf")
    plt.close(figure)

    for module in categorical:
        table = (
            shap_data.query("module == @module")
            .pivot_table(
                index="fid",
                columns="option",
                values="shap",
                aggfunc="median",
            )
            .T
        )
        limit = max(
            0.1,
            np.ceil(max(abs(table.min().min()), table.max().max()) * 10) / 10,
        )
        plot_heatmap(
            table,
            output_dir / f"{module}_shap_heat.pdf",
            (
                "Step-Size Adaptation"
                if module == "ssa"
                else module.replace("_", " ").title()
            ),
            vmin=-limit,
            vmax=limit,
            cmap="coolwarm",
            formatter=lambda value: f"{value:.2f}",
            minimize=False,
            font_size=12,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SMAC results and create ERT and SHAP figures."
    )
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--reference-ert", type=Path, default=SCRIPT_DIR / "ERT_BBOB.csv"
    )
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--min-runs", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--validation-start", type=int, default=100)
    parser.add_argument("--validation-runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--target", type=float, default=1e-8)
    parser.add_argument("--budget-per-dimension", type=int, default=10_000)
    parser.add_argument("--catboost-iterations", type=int, default=2_000)
    parser.add_argument("--skip-shap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runhistories = load_runhistories(args.data_dir.resolve(), args.dim)
    selected = validate_candidates(
        runhistories,
        args.dim,
        args.min_runs,
        args.top_k,
        args.validation_start,
        args.validation_runs,
        args.seed,
        args.target,
        args.budget_per_dimension,
    )
    selected.to_csv(output_dir / "configs_selected.csv", index=False)

    configurations = flatten_runs(runhistories)
    configurations.to_csv(output_dir / "configs_all.csv", index=False)

    defaults = evaluate_defaults(
        args.dim,
        args.validation_start,
        args.validation_runs,
        args.seed,
        args.target,
        args.budget_per_dimension,
    )
    create_ert_figures(
        selected,
        defaults,
        args.reference_ert.resolve(),
        output_dir,
        args.dim,
        args.budget_per_dimension,
    )

    if not args.skip_shap:
        compute_shap(
            configurations,
            output_dir,
            args.catboost_iterations,
            args.seed,
        )


if __name__ == "__main__":
    main()
